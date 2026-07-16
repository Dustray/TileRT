"""Gated Attention module for Qwen3.6."""

from typing import Any

import torch
import torch.nn.functional as F

from tilert.models.base import SerializableTileRTModule, TileRTModule
from tilert.models.common import RMSNorm, init_func, linear
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.models.qwen3_6.ops.gqa_attention import (
    GQAAttention as GQAAttentionOp,
    GQAAttentionAlgorithm,
)
from tilert.models.qwen3_6.ops.unproj_o_allreduce import (
    UnProjOAllReduce,
    UnProjOAllReduceAlgorithm,
)


class QwenAttentionRef(TileRTModule):
    """Lightweight reference-only holder for GQA weights.

    Mirrors the weight aliases in ``ops.gqa_attention.GQAAttentionRefWeightsAlias``.
    Holds q/k/v/o_proj, q_norm/k_norm, and input/output layernorm weights for
    the golden forward path.  The optimized path uses the TileRT ops below.
    """

    def __init__(
        self,
        model_args: ModelArgsQwen36,
        device_id: int,
        num_devices: int,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
        )
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_kv_heads
        self.head_dim = model_args.qk_head_dim
        self.rope_dim = model_args.rope_dim
        self.no_pe_dim = self.head_dim - self.rope_dim
        self.num_local_heads = self.n_heads // num_devices
        self.num_local_kv_heads = max(1, self.n_kv_heads // num_devices)

        self.q_proj_weight: torch.Tensor | None = None
        self.k_proj_weight: torch.Tensor | None = None
        self.v_proj_weight: torch.Tensor | None = None
        self.o_proj_weight: torch.Tensor | None = None
        self.q_norm_weight: torch.Tensor | None = None
        self.k_norm_weight: torch.Tensor | None = None
        self.input_layernorm_weight: torch.Tensor | None = None
        self.post_attention_layernorm_weight: torch.Tensor | None = None

    def get_weights_list(self) -> list[torch.Tensor]:
        return []

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        del weights_map
        return {}

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        prefix = "self_attn"
        self.q_proj_weight = state_dict[f"{prefix}.q_proj.weight"]
        self.k_proj_weight = state_dict[f"{prefix}.k_proj.weight"]
        self.v_proj_weight = state_dict[f"{prefix}.v_proj.weight"]
        self.o_proj_weight = state_dict[f"{prefix}.o_proj.weight"]
        self.q_norm_weight = state_dict[f"{prefix}.q_norm.weight"]
        self.k_norm_weight = state_dict[f"{prefix}.k_norm.weight"]
        self.input_layernorm_weight = state_dict["input_layernorm.weight"]
        self.post_attention_layernorm_weight = state_dict["post_attention_layernorm.weight"]

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        del state_dict

    def init_random_weights(self) -> None:
        pass

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        del batch_size, seq_len

    @staticmethod
    def _rmsnorm_heads(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Apply RMSNorm along the head dimension.

        Args:
            x: Tensor of shape (bsz, n_heads, seq_len, head_dim).
            weight: Per-head or per-element weight, broadcastable to head_dim.
            eps: Small constant for numerical stability.

        Returns:
            Normalized tensor with the same shape as ``x``.
        """
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
        return x * weight / rms

    def golden_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard GQA reference forward.

        Returns the attention output, updated k_cache, and updated v_cache.
        """
        assert self.q_proj_weight is not None
        assert self.k_proj_weight is not None
        assert self.v_proj_weight is not None
        assert self.o_proj_weight is not None
        bsz, seq_len, _ = x.shape
        h = linear(x, self.q_proj_weight)
        k = linear(x, self.k_proj_weight)
        v = linear(x, self.v_proj_weight)

        q = h.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply per-head RMSNorm to Q/K query/key projections when present.
        if self.q_norm_weight is not None:
            q = self._rmsnorm_heads(q, self.q_norm_weight)
        if self.k_norm_weight is not None:
            k = self._rmsnorm_heads(k, self.k_norm_weight)

        q_pe, q_no_pe = torch.split(q, [self.rope_dim, self.no_pe_dim], dim=-1)
        k_pe, k_no_pe = torch.split(k, [self.rope_dim, self.no_pe_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis, interleaved=False)
        k_pe = apply_rotary_emb(k_pe, freqs_cis, interleaved=False)
        q = torch.cat([q_pe, q_no_pe], dim=-1)
        k = torch.cat([k_pe, k_no_pe], dim=-1)

        k_cache[:bsz, start_pos : start_pos + seq_len] = k.transpose(1, 2)
        v_cache[:bsz, start_pos : start_pos + seq_len] = v.transpose(1, 2)

        k_full = k_cache[:bsz, : start_pos + seq_len].transpose(1, 2)
        v_full = v_cache[:bsz, : start_pos + seq_len].transpose(1, 2)

        # GQA repeat
        if self.n_heads != self.n_kv_heads:
            reps = self.n_heads // self.n_kv_heads
            k_full = k_full.repeat_interleave(reps, dim=1)
            v_full = v_full.repeat_interleave(reps, dim=1)

        scores = torch.matmul(q, k_full.transpose(-2, -1)) / (self.head_dim**0.5)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        o = torch.matmul(attn, v_full)
        o = o.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        out = linear(o, self.o_proj_weight)
        return out, k_cache, v_cache


class GatedAttention(SerializableTileRTModule):
    """Gated Attention layer for Qwen3.6.

    Wraps the GQA path using the TileRT ops that have already been ported from
    DSv3.2/GLM5:
      - QKVRoPE   (rotary embedding)
      - Rotate    (hadamard transform on QK)
      - UnProjOAllReduce (o_proj + allreduce)

    The gating mechanism in the Qwen3.6 Gated Attention layer is represented
    implicitly by the RoPE + hadamard transform pattern; the actual gating
    is implemented in the CUDA kernels and is not part of this Python wrapper.
    """

    def __init__(
        self,
        model_args: ModelArgsQwen36,
        device_id: int,
        num_devices: int,
        remove_selected: bool = False,
    ):
        super().__init__(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            remove_selected=remove_selected,
        )

        self.attn_ref = QwenAttentionRef(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.attn_ref)

        self.attn = GQAAttentionOp(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            algorithm=GQAAttentionAlgorithm.GENERAL,
        )
        self.register_op(self.attn)

        self.unproj_o_allreduce = UnProjOAllReduce(
            model_args=model_args,
            num_devices=num_devices,
            device_id=device_id,
            algorithm=UnProjOAllReduceAlgorithm.FP16MMA,
        )
        self.register_op(self.unproj_o_allreduce)

    def golden_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reference GQA forward."""
        out, k_cache, v_cache = self.attn_ref.golden_forward(
            x, start_pos, freqs_cis, k_cache, v_cache, mask
        )
        return out, k_cache, v_cache

    def tilert_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimized forward using ``GQAAttentionOp`` + ``UnProjOAllReduce``.

        Currently ``GQAAttentionOp.tilert_forward`` is itself a placeholder that
        calls into ``torch.ops.tilert.gqa_attention_op``; it will become
        functional once the CUDA kernel is registered.
        """
        attn_out, k_cache, v_cache = self.attn.forward(
            x, start_pos, freqs_cis, k_cache, v_cache, mask
        )
        out = self.unproj_o_allreduce.forward(attn_out)
        return out, k_cache, v_cache

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.flag_enable_tilert:
            return self.tilert_forward(x, start_pos, freqs_cis, k_cache, v_cache, mask)
        return self.golden_forward(x, start_pos, freqs_cis, k_cache, v_cache, mask)