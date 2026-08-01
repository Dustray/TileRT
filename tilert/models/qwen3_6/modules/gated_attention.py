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
        return []

    def get_tilert_weights_alias(self) -> list[str]:
        return []

    def get_ref_weights_alias(self) -> list[str]:
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

    def tilert_forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("QwenAttentionRef is reference-only")

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
        mrope_embed: tuple[torch.Tensor, torch.Tensor],
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
        if self.attn.qkv_proj_weights is None:
            self.attn.init_random_weights(device=str(x.device))
        if not self.ffn.moe.rmsnorm_expert_proj.is_ref_weights_init:
            self.ffn.init_random_weights(device=str(x.device))
        if self.input_layernorm.weight.device != x.device:
            self.input_layernorm.to(x.device)
        if self.post_attention_layernorm.weight.device != x.device:
            self.post_attention_layernorm.to(x.device)

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load weights and also set the RMSNorm module weights from the checkpoint."""
        logger.debug(f"{self.op_name}: loading tilert weights + layernorms")
        super().init_tilert_weights(state_dict)
        self._load_layernorm_weights(state_dict)

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
        logger.info(f"[GatedAttention.golden_forward_{self.device_id}] ENTRY: x.shape={x.shape}, start_pos={start_pos}, k_cache.shape={k_cache.shape}, v_cache.shape={v_cache.shape}")
        
        self._ensure_weights(x)

        # Pre-attention norm + GQA (o_proj applied internally) + residual.
        logger.info(f"[GatedAttention.golden_forward_{self.device_id}] Step1: input_layernorm")
        norm_x = self.input_layernorm(x)
        logger.info(f"[GatedAttention.golden_forward_{self.device_id}] Step2: GQA attention (attn.golden_forward)")
        attn_out, k_cache, v_cache = self.attn.golden_forward(
            norm_x, start_pos, mrope_embed, k_cache, v_cache, mask
        )
        logger.info(f"[GatedAttention.golden_forward_{self.device_id}] attn_out: shape={attn_out.shape}, k_cache.shape={k_cache.shape}, v_cache.shape={v_cache.shape}")
        
        h = x + attn_out
        logger.info(f"[GatedAttention.golden_forward_{self.device_id}] After residual: h.shape={h.shape}")

        # Post-attention norm + MoE FFN (partial TP8 sum) + all-reduce.
        logger.info(f"[GatedAttention.golden_forward_{self.device_id}] Step3: post_attention_layernorm")
        norm_h = self.post_attention_layernorm(h)
        logger.info(f"[GatedAttention.golden_forward_{self.device_id}] Step4: MoE FFN (ffn.golden_forward)")
        
        ffn_partial = self.ffn.golden_forward(norm_h)
        logger.info(
            f"[GatedAttention.golden_forward_{self.device_id}] ffn_partial: "
            f"shape={ffn_partial.shape} mean={ffn_partial.float().mean().item():.6f} "
            f"std={ffn_partial.float().std().item():.6f}"
        )

        if self.moe_sync_callback is not None:
            logger.info(f"[GatedAttention.golden_forward_{self.device_id}] Step5: MoE sync (all-reduce)")
            ffn_full = self.moe_sync_callback(ffn_partial)
            logger.info(
                f"[GatedAttention.golden_forward_{self.device_id}] ffn_full after sync: "
                f"shape={ffn_full.shape} mean={ffn_full.float().mean().item():.6f} "
                f"std={ffn_full.float().std().item():.6f}"
            )
        else:
            ffn_full = ffn_partial

        # Final residual uses the all-reduced (full) FFN output.
        out = h + ffn_full
        logger.info(f"[GatedAttention.golden_forward_{self.device_id}] EXIT: out.shape={out.shape}, mean={out.float().mean().item():.4f}")

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
        if self.flag_enable_tilert:
            return self.tilert_forward(x, start_pos, mrope_embed, k_cache, v_cache, mask)
        return self.golden_forward(x, start_pos, mrope_embed, k_cache, v_cache, mask)