"""Gated Attention module for Qwen3.6."""

from typing import Any

import torch
import torch.nn.functional as F

from tilert import logger
from tilert.models.base import SerializableTileRTModule, TileRTModule
from tilert.models.common import RMSNorm, init_func, linear
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.models.qwen3_6.modules.moe import QwenMoeBlock
from tilert.models.utils import apply_mrope_embed, apply_rotary_emb
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

        logger.info(f'[{__file__.split(chr(47))[-1]}] QwenAttentionRef.__init__')
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
        # EP8: replicate full attention weights on every device.
        self.num_local_heads = self.n_heads
        self.num_local_kv_heads = self.n_kv_heads

        self.q_proj_weight: torch.Tensor | None = None
        self.k_proj_weight: torch.Tensor | None = None
        self.v_proj_weight: torch.Tensor | None = None
        self.o_proj_weight: torch.Tensor | None = None
        self.q_norm_weight: torch.Tensor | None = None
        self.k_norm_weight: torch.Tensor | None = None
        self.input_layernorm_weight: torch.Tensor | None = None
        self.post_attention_layernorm_weight: torch.Tensor | None = None

    def get_weights_list(self) -> list[torch.Tensor]:

        logger.info(f'[{__file__.split(chr(47))[-1]}] QwenAttentionRef.get_weights_list')
        return []

    def get_tilert_weights_alias(self) -> list[str]:

        logger.info(f'[{__file__.split(chr(47))[-1]}] QwenAttentionRef.get_tilert_weights_alias')
        return self.get_ref_weights_alias()

    def get_ref_weights_alias(self) -> list[str]:

        logger.info(f'[{__file__.split(chr(47))[-1]}] QwenAttentionRef.get_ref_weights_alias')
        return ["input_layernorm.weight", 
                "post_attention_layernorm.weight", 
                "mlp.gate.weight",
                "mlp.experts.gate_up_proj",
                "mlp.shared_expert.gate_proj.weight",
                "mlp.shared_expert.up_proj.weight",
                "mlp.shared_expert_gate.weight",
                "mlp.experts.down_proj",
                "mlp.shared_expert.down_proj.weight"
                ]

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        logger.info(f'[{__file__.split(chr(47))[-1]}] QwenAttentionRef.device_sharding')
        del weights_map
        return {}

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:

        logger.info(f'[{__file__.split(chr(47))[-1]}] QwenAttentionRef.init_reference_weights')
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

        logger.info(f'[{__file__.split(chr(47))[-1]}] QwenAttentionRef.init_tilert_weights')
        del state_dict

    def init_random_weights(self) -> None:

        logger.info(f'[{__file__.split(chr(47))[-1]}] QwenAttentionRef.init_random_weights')
        pass

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:

        logger.info(f'[{__file__.split(chr(47))[-1]}] QwenAttentionRef.init_tilert_vars')
        del batch_size, seq_len

    def tilert_forward(self, *args: Any, **kwargs: Any) -> Any:

        logger.info(f'[{__file__.split(chr(47))[-1]}] QwenAttentionRef.tilert_forward')
        raise NotImplementedError("QwenAttentionRef is reference-only")

    @staticmethod
    def _rmsnorm_heads(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Apply per-head RMSNorm to Q/K projections.

        Matches Qwen3.5/3.6 convention: the checkpoint stores a weight that is
        added to 1.0 before scaling (``(1.0 + weight) * x / rms``).
        ``ops/gqa_attention.py`` and ``lynn-engine/engine/full_forward.py`` use
        the same formula.

        Args:
            x: Tensor of shape (bsz, n_heads, seq_len, head_dim).
            weight: Per-head or per-element weight, broadcastable to head_dim.
            eps: Small constant for numerical stability.

        Returns:
            Normalized tensor with the same shape as ``x``.

        """
        logger.info(f'[{__file__.split(chr(47))[-1]}] QwenAttentionRef._rmsnorm_heads')
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
        return x * (1.0 + weight) / rms

    def golden_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        mrope_embed: tuple[torch.Tensor, torch.Tensor],
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard GQA reference forward.

        Returns the attention output, updated k_cache, and updated v_cache.

        """
        logger.info(f'[{__file__.split(chr(47))[-1]}] QwenAttentionRef.golden_forward')
        assert self.q_proj_weight is not None
        assert self.k_proj_weight is not None
        assert self.v_proj_weight is not None
        assert self.o_proj_weight is not None
        bsz, seq_len, _ = x.shape
        h = linear(x, self.q_proj_weight)
        k = linear(x, self.k_proj_weight)
        v = linear(x, self.v_proj_weight)

        q = q.view(bsz, seq_len, self.num_local_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_local_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_local_kv_heads, self.head_dim).transpose(1, 2)

        # Apply per-head RMSNorm to Q/K query/key projections when present.
        if self.q_norm_weight is not None:
            q = self._rmsnorm_heads(q, self.q_norm_weight)
        if self.k_norm_weight is not None:
            k = self._rmsnorm_heads(k, self.k_norm_weight)

        freqs_cos, freqs_sin = mrope_embed
        # Slice the full tables to the current decode window.
        cur_cos = freqs_cos[start_pos : start_pos + seq_len]
        cur_sin = freqs_sin[start_pos : start_pos + seq_len]

        q_pe, q_no_pe = torch.split(q, [self.rope_dim, self.no_pe_dim], dim=-1)
        k_pe, k_no_pe = torch.split(k, [self.rope_dim, self.no_pe_dim], dim=-1)
        q_pe, k_pe = apply_mrope_embed(
            q_pe, k_pe, cur_cos, cur_sin, unsqueeze_dim=1
        )
        q = torch.cat([q_pe, q_no_pe], dim=-1)
        k = torch.cat([k_pe, k_no_pe], dim=-1)

        k_cache[:bsz, start_pos : start_pos + seq_len] = k.transpose(1, 2)
        v_cache[:bsz, start_pos : start_pos + seq_len] = v.transpose(1, 2)

        k_full = k_cache[:bsz, : start_pos + seq_len].transpose(1, 2)
        v_full = v_cache[:bsz, : start_pos + seq_len].transpose(1, 2)

        # GQA repeat
        if self.num_local_heads != self.num_local_kv_heads:
            reps = self.num_local_heads // self.num_local_kv_heads
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

        logger.info(f'[{__file__.split(chr(47))[-1]}] GatedAttention.__init__')
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
        # GQAAttention and UnProjOAllReduce both consume the same ``o_proj.weight``
        # from the checkpoint.  Retain the key so the latter can still read it.
        self.register_op(self.attn, retain_weights=True)

        self.unproj_o_allreduce = UnProjOAllReduce(
            model_args=model_args,
            num_devices=num_devices,
            device_id=device_id,
            algorithm=UnProjOAllReduceAlgorithm.FP16MMA,
        )
        self.register_op(self.unproj_o_allreduce)

        self.ffn = QwenMoeBlock(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
        )
        self.register_op(self.ffn)

        self.input_layernorm = RMSNorm(model_args.dim, eps=model_args.eps)
        self.post_attention_layernorm = RMSNorm(model_args.dim, eps=model_args.eps)

    def _ensure_weights(self, x: torch.Tensor) -> None:

        """Lazy initialize weights for sanity testing without a checkpoint."""
        # if self.attn.qkv_proj_weights is None:
        #     self.attn.init_random_weights(device=str(x.device))
        # if not self.ffn.moe.rmsnorm_expert_proj.is_ref_weights_init:
        #     self.ffn.init_random_weights(device=str(x.device))
        logger.info(f'[{__file__.split(chr(47))[-1]}] GatedAttention._ensure_weights')
        if self.input_layernorm.weight.device != x.device:
            self.input_layernorm.to(x.device)
        if self.post_attention_layernorm.weight.device != x.device:
            self.post_attention_layernorm.to(x.device)

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:

        """Load weights and also set the RMSNorm module weights from the checkpoint."""
        logger.debug(f"{self.op_name}: loading tilert weights + layernorms")
        super().init_tilert_weights(state_dict)
        self._load_layernorm_weights(state_dict)
        if self.attn.qkv_proj_weights is None:
            self.attn.init_reference_weights(state_dict)
        if not self.ffn.moe.rmsnorm_expert_proj.is_ref_weights_init:
            self.ffn.init_reference_weights(state_dict)

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:

        """Load reference weights and also set the RMSNorm module weights."""
        logger.debug(f"{self.op_name}: loading reference weights + layernorms")
        super().init_reference_weights(state_dict)
        # The GQA attention op currently only loads tilert-layout weights; it
        # needs the same reference weights as the reference holder so that
        # golden_forward can run without lazy random initialization.  Copy
        # them from the reference holder if it was populated.
        if self.attn_ref.q_proj_weight is not None:
            self.attn.qkv_proj_weights = torch.cat(
                [self.attn_ref.q_proj_weight, self.attn_ref.k_proj_weight, self.attn_ref.v_proj_weight], dim=0
            )
            self.attn.o_proj_weights = self.attn_ref.o_proj_weight
            self.attn.q_norm_weights = self.attn_ref.q_norm_weight
            self.attn.k_norm_weights = self.attn_ref.k_norm_weight
            self.attn.is_ref_weights_init = True
        self._load_layernorm_weights(state_dict)

    def _load_layernorm_weights(self, state_dict: dict[str, torch.Tensor]) -> None:

        """Copy checkpoint layernorm weights into the RMSNorm modules."""
        logger.info(f'[{__file__.split(chr(47))[-1]}] GatedAttention._load_layernorm_weights')
        if "input_layernorm.weight" in state_dict:
            self.input_layernorm.weight.data.copy_(state_dict["input_layernorm.weight"])
        if "post_attention_layernorm.weight" in state_dict:
            self.post_attention_layernorm.weight.data.copy_(state_dict["post_attention_layernorm.weight"])

    def golden_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        mrope_embed: tuple[torch.Tensor, torch.Tensor],
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """Reference GQA forward with residuals, norms, and MoE FFN."""
        logger.info(f'[{__file__.split(chr(47))[-1]}] GatedAttention.golden_forward')
        self._ensure_weights(x)

        norm_x = self.input_layernorm(x)
        attn_out, k_cache, v_cache = self.attn.golden_forward(
            norm_x, start_pos, mrope_embed, k_cache, v_cache, mask
        )
        h = x + attn_out

        norm_h = self.post_attention_layernorm(h)
        ffn_partial = self.ffn.golden_forward(norm_h)

        out = h + ffn_partial

        return out, k_cache, v_cache

    def tilert_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        mrope_embed: tuple[torch.Tensor, torch.Tensor],
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """Optimized forward using ``GQAAttentionOp`` + ``UnProjOAllReduce`` + ``QwenMoeBlock``."""
        logger.info(f'[{__file__.split(chr(47))[-1]}] GatedAttention.tilert_forward')
        attn_out, k_cache, v_cache = self.attn.forward(
            x, start_pos, mrope_embed, k_cache, v_cache, mask
        )
        out = self.unproj_o_allreduce.forward(attn_out)
        out = self.ffn.forward(out)
        return out, k_cache, v_cache

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        mrope_embed: tuple[torch.Tensor, torch.Tensor],
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        logger.info(f'[{__file__.split(chr(47))[-1]}] GatedAttention.forward')
        if self.flag_enable_tilert:
            return self.tilert_forward(x, start_pos, mrope_embed, k_cache, v_cache, mask)
        return self.golden_forward(x, start_pos, mrope_embed, k_cache, v_cache, mask)