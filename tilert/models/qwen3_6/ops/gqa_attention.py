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

from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.common import weight_dequant
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
        q_proj_w, k_proj_w, v_proj_w, o_proj_w, q_norm_w, k_norm_w = weights_list
        qkv_proj_weights = torch.cat([q_proj_w, k_proj_w, v_proj_w], dim=0)
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
        self.num_local_heads = self.n_heads // num_devices
        self.num_local_kv_heads = max(1, self.n_kv_heads // num_devices)

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
        """Shard GQA reference weights across devices.

        For the reference/golden path we split q/k/v/o_proj and q_norm/k_norm
        along the head dimension so each device owns ``num_local_heads`` Q
        heads and ``num_local_kv_heads`` KV heads.
        """
        prefix = self.ref_weights_alias.key_prefix
        q_w = weights_map[f"{prefix}.q_proj.weight"]
        k_w = weights_map[f"{prefix}.k_proj.weight"]
        v_w = weights_map[f"{prefix}.v_proj.weight"]
        o_w = weights_map[f"{prefix}.o_proj.weight"]
        q_norm_w = weights_map[f"{prefix}.q_norm.weight"]
        k_norm_w = weights_map[f"{prefix}.k_norm.weight"]

        q_per_dev = self.num_local_heads * self.head_dim
        kv_per_dev = self.num_local_kv_heads * self.v_head_dim
        qkv_parts = [
            q_w[i * q_per_dev : (i + 1) * q_per_dev]
            for i in range(self.num_devices)
        ]
        k_parts = [
            k_w[i * kv_per_dev : (i + 1) * kv_per_dev]
            for i in range(self.num_devices)
        ]
        v_parts = [
            v_w[i * kv_per_dev : (i + 1) * kv_per_dev]
            for i in range(self.num_devices)
        ]
        # o_proj input dim equals the Q head count * value_head_dim.
        o_per_dev = self.num_local_heads * self.v_head_dim
        o_parts = [
            o_w[:, i * o_per_dev : (i + 1) * o_per_dev]
            for i in range(self.num_devices)
        ]

        qkv_stacked = []
        for i in range(self.num_devices):
            qkv_stacked.append(
                torch.cat([qkv_parts[i], k_parts[i], v_parts[i]], dim=0)
            )

        # q_norm/k_norm are per-head weights; split accordingly.
        q_norm_parts = [
            q_norm_w[i * self.num_local_heads * self.head_dim : (i + 1) * self.num_local_heads * self.head_dim]
            for i in range(self.num_devices)
        ]
        k_norm_parts = [
            k_norm_w[i * self.num_local_kv_heads * self.v_head_dim : (i + 1) * self.num_local_kv_heads * self.v_head_dim]
            for i in range(self.num_devices)
        ]

        return {
            self.tilert_weights_alias.qkv_proj_weights: torch.stack(qkv_stacked, dim=0).contiguous(),
            self.tilert_weights_alias.o_proj_weights: torch.stack(o_parts, dim=0).contiguous(),
            self.tilert_weights_alias.q_norm_weights: torch.stack(q_norm_parts, dim=0).contiguous(),
            self.tilert_weights_alias.k_norm_weights: torch.stack(k_norm_parts, dim=0).contiguous(),
        }

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        sharded = self.device_sharding(state_dict)
        did = self.device_id
        self.qkv_proj_weights = sharded[self.tilert_weights_alias.qkv_proj_weights][did]
        self.o_proj_weights = sharded[self.tilert_weights_alias.o_proj_weights][did]
        self.q_norm_weights = sharded[self.tilert_weights_alias.q_norm_weights][did]
        self.k_norm_weights = sharded[self.tilert_weights_alias.k_norm_weights][did]

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
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
        qkv_out = self.n_heads * self.head_dim + 2 * self.n_kv_heads * self.v_head_dim
        qkv_w = torch.randn(
            qkv_out,
            self.dim,
            dtype=torch.bfloat16,
            device=device,
        )
        o_w = torch.randn(
            self.dim,
            self.n_heads * self.v_head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        q_norm_w = torch.randn(
            self.n_heads * self.head_dim,
            dtype=torch.float32,
            device=device,
        )
        k_norm_w = torch.randn(
            self.n_kv_heads * self.v_head_dim,
            dtype=torch.float32,
            device=device,
        )
        converter = GQAAttentionWeightsConverter(self.model_args, self.num_devices)
        (
            self.qkv_proj_weights,
            self.o_proj_weights,
            self.q_norm_weights,
            self.k_norm_weights,
        ) = converter.convert_to_general(
            [
                qkv_w,
                torch.empty(0, dtype=torch.bfloat16, device=device),
                torch.empty(0, dtype=torch.bfloat16, device=device),
                o_w,
                q_norm_w,
                k_norm_w,
            ]
        )

    def golden_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reference GQA forward implemented in PyTorch.

        This is a fallback used until the CUDA kernel is available.
        """
        import torch.nn.functional as F

        assert self.qkv_proj_weights is not None
        assert self.o_proj_weights is not None
        bsz, seq_len, _ = x.shape

        qkv = x @ self.qkv_proj_weights.T
        q, k, v = torch.split(
            qkv,
            [
                self.num_local_heads * self.head_dim,
                self.num_local_kv_heads * self.v_head_dim,
                self.num_local_kv_heads * self.v_head_dim,
            ],
            dim=-1,
        )

        q = q.view(bsz, seq_len, self.num_local_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_local_kv_heads, self.v_head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_local_kv_heads, self.v_head_dim).transpose(1, 2)

        # apply_rotary_emb expects (bsz, seq_len, n_heads, head_dim);
        # transpose so dim 1 is the sequence dimension, then restore.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        rope_dim = self.rope_dim
        no_pe_dim = self.head_dim - rope_dim
        q_pe, q_no_pe = torch.split(q, [rope_dim, no_pe_dim], dim=-1)
        k_pe, k_no_pe = torch.split(k, [rope_dim, no_pe_dim], dim=-1)

        from tilert.models.utils import apply_rotary_emb

        local_freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
        q_pe = apply_rotary_emb(q_pe, local_freqs_cis, interleaved=False)
        k_pe = apply_rotary_emb(k_pe, local_freqs_cis, interleaved=False)
        q = torch.cat([q_pe, q_no_pe], dim=-1)
        k = torch.cat([k_pe, k_no_pe], dim=-1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        k_cache[:bsz, start_pos : start_pos + seq_len] = k.transpose(1, 2)
        v_cache[:bsz, start_pos : start_pos + seq_len] = v.transpose(1, 2)

        k_full = k_cache[:bsz, : start_pos + seq_len].transpose(1, 2)
        v_full = v_cache[:bsz, : start_pos + seq_len].transpose(1, 2)

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
        out = o @ self.o_proj_weights.T
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
        """Optimized forward placeholder."""
        del freqs_cis, mask
        assert self.is_init
        assert self.out is not None
        assert self.profile_logs is not None
        gqa_attention(
            x,
            k_cache,
            v_cache,
            self.out,
            torch.tensor([start_pos], dtype=torch.int32, device=x.device),
            self.profile_logs,
            model_arch=self.model_args.arch_name,
        )
        return self.out, k_cache, v_cache

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
