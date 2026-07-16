"""Qwen3.6-35B-A3B model arguments and hyperparameters."""

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "ModelArgsQwen36",
]


@dataclass
class ModelArgsQwen36:
    """Model arguments for Qwen3.6-35B-A3B.

    Architecture:
        - 40 layers in a repeating pattern: 10 × [3 × DeltaNet → 1 × Gated Attention]
        - GQA (Grouped Query Attention) with 16 Q heads and 2 KV heads
        - Sparse MoE with 256 experts, 8 activated + 1 shared

    Key differences from DeepSeek-V3.2:
        - Uses GQA instead of MLA
        - Heterogeneous layer structure (DeltaNet + Gated Attention)
        - Much smaller MoE inter_dim (512 vs 2048)
    """

    arch_name: str = "qwen3_6"

    # Batch & Sequence
    max_batch_size: int = 1
    max_seq_len: int = 262144  # 256K native, up to 1M with YaRN

    # Data type
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: str | None = None

    # Model dimensions
    vocab_size: int = 248320
    dim: int = 2048  # Hidden dimension

    # RoPE
    rope_theta: float = 10000000.0  # 10M for YaRN

    # MLP / MoE
    inter_dim: int = 512       # Much smaller than DeepSeek's 2048

    # MoE configuration
    n_layers: int = 40
    n_routed_experts: int = 256
    n_activated_experts: int = 8
    n_shared_experts: int = 1
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "softmax"
    route_scale: float = 2.5

    # Layer structure: exact heterogeneous pattern from text_config.layer_types
    layer_types: list[str] | None = None  # e.g. ["linear_attention", ...]

    # Layer structure: 10 blocks of [3 DeltaNet + 1 Gated Attention]
    n_blocks: int = 10
    n_delta_per_block: int = 3  # DeltaNet layers per block
    n_gated_per_block: int = 1  # Gated Attention layers per block

    # Layer types
    n_delta_layers: int = 30   # Total DeltaNet layers (10 × 3)
    n_gated_layers: int = 10   # Total Gated Attention layers (10 × 1)

    # Full attention (Gated Attention / GQA)
    qk_head_dim: int = 256
    rope_dim: int = 64         # partial_rotary_factor * head_dim
    n_heads: int = 16
    n_kv_heads: int = 2

    # DeltaNet specific
    delta_q_heads: int = 16
    delta_kv_heads: int = 32   # V=32, QK=16
    delta_key_head_dim: int = 128
    delta_value_head_dim: int = 128
    delta_conv_kernel_dim: int = 4
    delta_conv_dim: int = 8192  # key_dim * 2 + value_dim
    delta_v_dim: int = 4096  # value_dim
    delta_gate_dim: int = 4096  # value_dim (z projection)
    delta_a_dim: int = 32  # num_v_heads decay gate projection a
    delta_b_dim: int = 32  # num_v_heads decay gate projection b

    # MTP
    n_mtp_layers: int = 1

    # Quantization / KV cache
    kv_cache_pad: int = 8
    block_size: int = 128
    eps: float = 1e-6