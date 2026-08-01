"""GQA attention operation module for Qwen3.6.

This is a Python-side placeholder wrapper for the future CUDA kernel
``gqa_attention_op``.  It mirrors the structure used by ``flash_sparse_mla``
(from deepseek_v3_2) but drops all MLA-specific dimensions and uses the GQA
parameters defined in ``ModelArgsQwen36``.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from tilert import logger
from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.common import _safe_weight_dequant
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.utils import get_profile_log_tensor

__all__ = [
    "gqa_attention",
    "GQAAttentionAlgorithm",
    "GQAAttentionRefWeightsAlias",
    "GQAAttentionTilertWeightsAlias",
    "GQAAttentionWeightsConverter",
    "GQAAttention",
]


def gqa_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
    start_pos: torch.Tensor,
    profile_logs: torch.Tensor,
    model_arch: str,
    compute_kernel_type: str = "general",
) -> None:
    """GQA attention operation.

    Args:
        q: Query tensor (bsz, n_heads, seq, qk_head_dim).
        k_cache: Key cache (bsz, max_seq_len, n_kv_heads, qk_head_dim).
        v_cache: Value cache (bsz, max_seq_len, n_kv_heads, qk_head_dim).
        out: Output tensor (bsz, seq, dim).
        start_pos: Current start position (int32 scalar).
        profile_logs: Profile logs tensor.
        model_arch: Architecture string (e.g., "qwen3_6").
        compute_kernel_type: Kernel type ("general" for now).
    """
    torch.ops.tilert.gqa_attention_op(
        q,
        k_cache,
        v_cache,
        out,
        start_pos,
        profile_logs,
        model_arch,
        compute_kernel_type,
    )


@dataclass
class GQAAttentionRefWeightsAlias:
    """Reference weights alias for GQA attention (Qwen3.6 full_attention layer)."""

    key_prefix: str = "self_attn"

    @property
    def ref_tensor_alias(self) -> list[str]:
        return [
            f"{self.key_prefix}.q_proj.weight",
            f"{self.key_prefix}.k_proj.weight",
            f"{self.key_prefix}.v_proj.weight",
            f"{self.key_prefix}.o_proj.weight",
            f"{self.key_prefix}.q_norm.weight",
            f"{self.key_prefix}.k_norm.weight",
        ]

    def __call__(self) -> list[str]:
        return self.ref_tensor_alias


@dataclass
class GQAAttentionTilertWeightsAlias:
    """TileRT weights alias for GQA attention."""

    qkv_proj_weights = "qkv_proj_weights"
    o_proj_weights = "o_proj_weights"
    q_norm_weights = "q_norm_weights"
    k_norm_weights = "k_norm_weights"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [
            self.qkv_proj_weights,
            self.o_proj_weights,
            self.q_norm_weights,
            self.k_norm_weights,
        ]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class GQAAttentionAlgorithm(Enum):
    """GQA attention algorithm."""

    GENERAL = "general"


class GQAAttentionWeightsConverter(TilertWeightsConverter):
    """GQA attention weights converter.

    The checkpoint stores q/k/v/o_proj and the per-head q_norm/k_norm weights
    in plain bf16.  We keep the original layout; the CUDA kernel will apply
    RoPE and head-dim RMSNorm directly.
    """

    def convert_to_general(
        self, weights_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # The real converted checkpoint stores q/k/v/o_proj and q_norm/k_norm
        # as six separate tensors.  Qwen3.5-MoE full attention doubles the
        # q-projection output (query + gate), so ``q_proj.weight`` already
        # has shape ``[2 * n_heads * head_dim, dim]``.  Concatenate q, k, and v
        # along the output dimension to form the fused ``qkv_proj_weights``.
        if len(weights_list) == 6:
            q_proj_w, k_proj_w, v_proj_w, o_proj_w, q_norm_w, k_norm_w = weights_list
            qkv_proj_weights = torch.cat([q_proj_w, k_proj_w, v_proj_w], dim=0)
        else:
            qkv_proj_weights, o_proj_w, q_norm_w, k_norm_w = weights_list
        return qkv_proj_weights, o_proj_w, q_norm_w, k_norm_w


class GQAAttention(TileRTModule):
    """GQA attention op wrapper for Qwen3.6."""

    _SUPPORTED_ALGORITHMS = {
        "qwen3_6": [GQAAttentionAlgorithm.GENERAL],
    }

    def __init__(
        self,
        model_args: ModelArgsQwen36,
        device_id: int,
        num_devices: int,
        algorithm: GQAAttentionAlgorithm = GQAAttentionAlgorithm.GENERAL,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
        )
        self.algorithm = algorithm
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_kv_heads
        self.dim = model_args.dim
        self.head_dim = model_args.qk_head_dim
        self.v_head_dim = model_args.v_head_dim
        self.rope_dim = model_args.rope_dim
        # EP8: no tensor parallelism for attention weights.  Replicate the full
        # attention matrices on every device; only MoE experts are sharded.
        self.num_local_heads = self.n_heads
        self.num_local_kv_heads = self.n_kv_heads

        self.tilert_weights_alias = GQAAttentionTilertWeightsAlias()
        self.ref_weights_alias = GQAAttentionRefWeightsAlias()

        self.qkv_proj_weights: torch.Tensor | None = None
        self.o_proj_weights: torch.Tensor | None = None
        self.q_norm_weights: torch.Tensor | None = None
        self.k_norm_weights: torch.Tensor | None = None

        self.out: torch.Tensor | None = None
        self.profile_logs: torch.Tensor | None = None
        self.is_init = False

    @property
    def tensor_alias(self) -> list[str]:
        return list(self.ref_weights_alias())

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return list(self.tilert_weights_alias())

    def get_weights_list(self) -> list[torch.Tensor]:
        return [
            self.qkv_proj_weights,
            self.o_proj_weights,
            self.q_norm_weights,
            self.k_norm_weights,
        ]

    def device_sharding(
        self, weights_map: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Replicate full GQA reference weights on every device (EP8).

        Under EP8 attention weights are not tensor-parallel sharded; only the
        MoE experts are split across devices.  This method stacks the same full
        q/k/v/o_proj and q_norm/k_norm tensors ``num_devices`` times so that
        ``init_reference_weights`` can simply index ``[device_id]``.
        """
        prefix = self.ref_weights_alias.key_prefix
        logger.info(f"[device_sharding] key_prefix: {prefix}, num_devices: {self.num_devices}")
        
        q_w = weights_map[f"{prefix}.q_proj.weight"]
        k_w = weights_map[f"{prefix}.k_proj.weight"]
        v_w = weights_map[f"{prefix}.v_proj.weight"]
        o_w = weights_map[f"{prefix}.o_proj.weight"]
        q_norm_w = weights_map[f"{prefix}.q_norm.weight"]
        k_norm_w = weights_map[f"{prefix}.k_norm.weight"]

        qkv_proj_weights = torch.cat([q_w, k_w, v_w], dim=0)

        result = {
            self.tilert_weights_alias.qkv_proj_weights: torch.stack(
                [qkv_proj_weights for _ in range(self.num_devices)], dim=0
            ).contiguous(),
            self.tilert_weights_alias.o_proj_weights: torch.stack(
                [o_w for _ in range(self.num_devices)], dim=0
            ).contiguous(),
            self.tilert_weights_alias.q_norm_weights: torch.stack(
                [q_norm_w for _ in range(self.num_devices)], dim=0
            ).contiguous(),
            self.tilert_weights_alias.k_norm_weights: torch.stack(
                [k_norm_w for _ in range(self.num_devices)], dim=0
            ).contiguous(),
        }
        
        # Log sharding details
        logger.info(f"[device_sharding] key: {prefix}.q_proj.weight, original shape: {q_w.shape}, sharded shape: {result[self.tilert_weights_alias.qkv_proj_weights].shape}, dtype: {result[self.tilert_weights_alias.qkv_proj_weights].dtype}")
        logger.info(f"[device_sharding] key: {prefix}.o_proj.weight, original shape: {o_w.shape}, sharded shape: {result[self.tilert_weights_alias.o_proj_weights].shape}, dtype: {result[self.tilert_weights_alias.o_proj_weights].dtype}")
        logger.info(f"[device_sharding] key: {prefix}.q_norm.weight, original shape: {q_norm_w.shape}, sharded shape: {result[self.tilert_weights_alias.q_norm_weights].shape}, dtype: {result[self.tilert_weights_alias.q_norm_weights].dtype}")
        logger.info(f"[device_sharding] key: {prefix}.k_norm.weight, original shape: {k_norm_w.shape}, sharded shape: {result[self.tilert_weights_alias.k_norm_weights].shape}, dtype: {result[self.tilert_weights_alias.k_norm_weights].dtype}")
        
        return result

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        logger.debug(f"{self.op_name}: init_reference_weights on device {self.device_id}")
        sharded = self.device_sharding(state_dict)
        did = self.device_id
        self.qkv_proj_weights = sharded[self.tilert_weights_alias.qkv_proj_weights][did]
        self.o_proj_weights = sharded[self.tilert_weights_alias.o_proj_weights][did]
        self.q_norm_weights = sharded[self.tilert_weights_alias.q_norm_weights][did]
        self.k_norm_weights = sharded[self.tilert_weights_alias.k_norm_weights][did]
        self.is_ref_weights_init = True

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        logger.debug(f"{self.op_name}: init_tilert_weights on device {self.device_id}")
        # Prefer the six-tensor dot-weight checkpoint layout when present.
        ref_alias = self.ref_weights_alias()
        if all(alias in state_dict for alias in ref_alias):
            weights_list = [state_dict[alias] for alias in ref_alias]
        else:
            weights_list = [state_dict[alias] for alias in self.tilert_weights_alias()]
        converter = GQAAttentionWeightsConverter(self.model_args, self.num_devices)
        (
            self.qkv_proj_weights,
            self.o_proj_weights,
            self.q_norm_weights,
            self.k_norm_weights,
        ) = converter.dispatch(self.algorithm, weights_list)

    def init_tilert_vars(
        self, batch_size: int, seq_len: int, device: str = "cuda"
    ) -> None:
        self.out = torch.zeros(
            batch_size,
            seq_len,
            self.n_heads * self.v_head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        self.profile_logs = get_profile_log_tensor(device=device)
        self.is_init = True

    def init_random_weights(self, device: str = "cuda") -> None:
        logger.debug(f"{self.op_name}: init_random_weights on {device}")
        # Qwen3.5-MoE full attention doubles the q-projection for the gate.
        qkv_out = (
            self.n_heads * self.head_dim * 2
            + self.n_kv_heads * self.head_dim
            + self.n_kv_heads * self.v_head_dim
        )
        # Scale by 1/sqrt(fan_in) for stable 40-layer reference numerics.
        qkv_w = torch.randn(
            qkv_out,
            self.dim,
            dtype=torch.bfloat16,
            device=device,
        ) / (self.dim ** 0.5)
        o_w = torch.randn(
            self.dim,
            self.n_heads * self.v_head_dim,
            dtype=torch.bfloat16,
            device=device,
        ) / ((self.n_heads * self.v_head_dim) ** 0.5)
        # q_norm/k_norm are per-head scalars applied to each head individually.
        q_norm_w = torch.ones(
            self.head_dim,
            dtype=torch.float32,
            device=device,
        )
        k_norm_w = torch.ones(
            self.v_head_dim,
            dtype=torch.float32,
            device=device,
        )
        # EP8: reuse device_sharding (which now replicates full weights).
        q_split = self.n_heads * self.head_dim * 2
        state_dict = {
            f"{self.ref_weights_alias.key_prefix}.q_proj.weight": qkv_w[:q_split],
            f"{self.ref_weights_alias.key_prefix}.k_proj.weight": qkv_w[
                q_split : q_split + self.n_kv_heads * self.head_dim
            ],
            f"{self.ref_weights_alias.key_prefix}.v_proj.weight": qkv_w[
                q_split + self.n_kv_heads * self.head_dim :
            ],
            f"{self.ref_weights_alias.key_prefix}.o_proj.weight": o_w,
            f"{self.ref_weights_alias.key_prefix}.q_norm.weight": q_norm_w,
            f"{self.ref_weights_alias.key_prefix}.k_norm.weight": k_norm_w,
        }
        self.init_reference_weights(state_dict)

    def golden_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        mrope_embed: tuple[torch.Tensor, torch.Tensor],
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reference GQA forward implemented in PyTorch.

        This is a fallback used until the CUDA kernel is available.
        """
        logger.info(f"[GQAAttentionOp.golden_forward_{self.device_id}] ENTRY: x.shape={x.shape}, start_pos={start_pos}, k_cache.shape={k_cache.shape}")
        
        import torch.nn.functional as F

        assert self.qkv_proj_weights is not None
        assert self.o_proj_weights is not None
        bsz, seq_len, _ = x.shape
        logger.info(f"[GQAAttentionOp.golden_forward_{self.device_id}] Input: bsz={bsz}, seq_len={seq_len}, dim={x.shape[-1]}")

        # QKV projection
        logger.info(f"[GQAAttentionOp.golden_forward_{self.device_id}] Step1: QKV projection")
        qkv = x @ self.qkv_proj_weights.T
        logger.info(f"[GQAAttentionOp.golden_forward_{self.device_id}]   qkv.shape={qkv.shape}")
        
        # Qwen3.5-MoE full attention: q-projection is doubled; the second half is
        # the per-head gating signal.  Sizes now reflect the local shard.
        logger.info(f"[GQAAttentionOp.golden_forward_{self.device_id}] Step2: Split Q/K/V and gate")
        q_gate, k, v = torch.split(
            qkv,
            [
                self.num_local_heads * self.head_dim * 2,
                self.num_local_kv_heads * self.head_dim,
                self.num_local_kv_heads * self.v_head_dim,
            ],
            dim=-1,
        )
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        logger.info(f"[GQAAttentionOp.golden_forward_{self.device_id}]   q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}")

        # Keep (bsz, seq_len, n_local_heads, head_dim) for apply_rotary_emb.
        logger.info(f"[GQAAttentionOp.golden_forward_{self.device_id}] Step3: Reshape for multi-head")
        q = q.view(bsz, seq_len, self.num_local_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.num_local_kv_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.num_local_kv_heads, self.v_head_dim)
        logger.info(f"[GQAAttentionOp.golden_forward_{self.device_id}]   q.view={q.shape}, k.view={k.shape}, v.view={v.shape}")

        # Apply per-head RMSNorm on q/k.  q_norm/k_norm weights have shape
        # (head_dim,) and are broadcast across all heads.  Qwen3.5-MoE uses the
        # (1 + weight) convention.
        logger.info(f"[GQAAttentionOp.golden_forward_{self.device_id}] Step4: RMSNorm on Q/K")
        q_norm_w = self.q_norm_weights.view(1, 1, 1, self.head_dim)
        k_norm_w = self.k_norm_weights.view(1, 1, 1, self.v_head_dim)
        q = q * torch.rsqrt(q.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6) * (1.0 + q_norm_w)
        k = k * torch.rsqrt(k.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6) * (1.0 + k_norm_w)
        logger.info(f"[GQAAttentionOp.golden_forward_{self.device_id}]   After RMSNorm")

        # RoPE
        logger.info(f"[GQAAttentionOp.golden_forward_{self.device_id}] Step5: RoPE embedding")
        rope_dim = self.rope_dim
        no_pe_dim = self.head_dim - rope_dim
        q_pe, q_no_pe = torch.split(q, [rope_dim, no_pe_dim], dim=-1)
        k_pe, k_no_pe = torch.split(k, [rope_dim, no_pe_dim], dim=-1)

        from tilert.models.utils import apply_mrope_embed

        freqs_cos, freqs_sin = mrope_embed
        freqs_cos = freqs_cos[start_pos : start_pos + seq_len, :rope_dim]
        freqs_sin = freqs_sin[start_pos : start_pos + seq_len, :rope_dim]
        q_pe, k_pe = apply_mrope_embed(
            q_pe, k_pe, freqs_cos, freqs_sin, unsqueeze_dim=1
        )
        q = torch.cat([q_pe, q_no_pe], dim=-1)
        k = torch.cat([k_pe, k_no_pe], dim=-1)
        logger.info(f"[GQAAttentionOp.golden_forward_{self.device_id}]   After RoPE: q.shape={q.shape}")

        # Permute to (bsz, n_heads, seq_len, head_dim) for attention math.
        logger.info(f"[GQAAttentionOp.golden_forward_{self.device_id}] Step6: Permute and cache")
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Cache layout is (batch, seq_len, n_kv_heads, head_dim).
        k_cache[:bsz, start_pos : start_pos + seq_len] = k.transpose(1, 2)
        v_cache[:bsz, start_pos : start_pos + seq_len] = v.transpose(1, 2)

        k_full = k_cache[:bsz, : start_pos + seq_len].transpose(1, 2)
        v_full = v_cache[:bsz, : start_pos + seq_len].transpose(1, 2)

        if self.num_local_heads != self.num_local_kv_heads:
            reps = self.num_local_heads // self.num_local_kv_heads
            k_full = k_full.repeat_interleave(reps, dim=1)
            v_full = v_full.repeat_interleave(reps, dim=1)
        logger.info(f"[GQAAttentionOp.golden_forward_{self.device_id}]   k_full.shape={k_full.shape}, v_full.shape={v_full.shape}")

        # Attention computation
        logger.info(f"[GQAAttentionOp.golden_forward_{self.device_id}] Step7: Attention scores and softmax")
        scores = torch.matmul(q.float(), k_full.transpose(-2, -1).float()) / (self.head_dim**0.5)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        o = torch.matmul(attn, v_full.float())
        o = o.transpose(1, 2).contiguous()
        
        # Apply the per-head gate to the projected output.  The gate has the
        # same shape as the query (bsz, seq_len, n_local_heads, head_dim);
        # apply it element-wise (Qwen3.5-MoE attention gate).
        logger.info(f"[GQAAttentionOp.golden_forward_{self.device_id}] Step8: Apply gate and output projection")
        gate = gate.view(bsz, seq_len, self.num_local_heads, self.head_dim)
        out = o * torch.sigmoid(gate)
        out = out.view(bsz, seq_len, -1).to(x.dtype)
        out = out @ self.o_proj_weights.T
        logger.info(f"[GQAAttentionOp.golden_forward_{self.device_id}] EXIT: out.shape={out.shape}, k_cache.shape={k_cache.shape}, v_cache.shape={v_cache.shape}")

        # EP8: attention weights are replicated, so no all-reduce is needed.
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
        """Optimized forward placeholder."""
        logger.info(f"[GQAAttentionOp.tilert_forward_{self.device_id}] ENTRY: x.shape={x.shape}, start_pos={start_pos}")
        
        del mrope_embed, mask
        assert self.is_init
        assert self.out is not None
        assert self.profile_logs is not None
        
        logger.info(f"[GQAAttentionOp.tilert_forward_{self.device_id}] Calling CUDA kernel gqa_attention")
        gqa_attention(
            x,
            k_cache,
            v_cache,
            self.out,
            torch.tensor([start_pos], dtype=torch.int32, device=x.device),
            self.profile_logs,
            model_arch=self.model_args.arch_name,
        )
        logger.info(f"[GQAAttentionOp.tilert_forward_{self.device_id}] EXIT: out.shape={self.out.shape}, k_cache.shape={k_cache.shape}")
        
        return self.out, k_cache, v_cache

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
