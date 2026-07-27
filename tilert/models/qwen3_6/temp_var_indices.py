"""Named indices for Qwen3.6-35B-A3B temporary variables.

Provides Python-side symbolic names for the fixed-size ``temp_vars`` tensor
list used by the Qwen3.6 CUDA-graph decode path.  The indices are kept as a
contiguous 0..N-1 enum so they can be validated against the backend once
``libtilert_qwen36.so`` is available.

Until the CUDA kernels land, these same slots are used by the Python reference
decode loop in ``QwenShowHandsLayer`` so that the layout stays aligned with
future kernel expectations.
"""

from enum import IntEnum

__all__ = [
    "QwenTempVarIdx",
    "TEMP_VARS_SIZE",
    "Idx",
    "validate_temp_vars_layout",
]


class QwenTempVarIdx(IntEnum):
    """Index constants for Qwen3.6 temp_vars."""

    # Main activation tensors.
    X = 0  # Input hidden state after embedding lookup [bsz, seq, dim] bf16.
    HIDDEN_RMSNORM = 1  # Output of input RMSNorm / residual RMSNorm [bsz, seq, dim] bf16.
    EMBEDDING_RMSNORM = 2  # Normalized hidden before head projection [bsz, seq, dim] bf16.

    # Attention / DeltaNet intermediate tensors.
    DELTA_OUT = 3  # Output of DeltaNet attention [bsz, seq, dim] bf16.
    GQA_OUT = 4  # Output of GQA attention [bsz, seq, dim] bf16.
    ROPE_FREQS = 5  # RoPE frequencies for GQA layers [seq, rope_dim*2] fp32.
    CUR_POS = 6  # Current decode position [bsz] int32.
    TOKEN_ID = 7  # Input token id(s) [bsz, seq, 1] int32.

    # MoE / MLP tensors.
    X_MLP_IN = 8  # Post-attention RMSNorm output [bsz, seq, dim] bf16.
    SCORES = 9  # Router scores before top-k [bsz, seq, n_routed_experts] fp32.
    SEL_PROBS = 10  # Selected expert probabilities [bsz, seq, n_activated] fp32.
    SEL_INDICES = 11  # Selected expert indices [bsz, seq, n_activated] int32.
    UP_GATE = 12  # Up * gate * SiLU output [bsz, seq, n_total, inter_dim] bf16.
    EXP_OUT = 13  # MoE/MLP output [bsz, seq, dim] bf16.

    # Head projection / sampling.
    LOGITS_OUT = 14  # Full logits replicated on this device [bsz, seq, vocab_size] fp32.
    TOKEN_OUT = 15  # Sampled next token [bsz, seq, 1] int32.
    SAMPLING_SEED = 16  # Request-level seed [bsz, seq] int64.
    SAMPLING_POSITIONS = 17  # Per-step position offsets [bsz, seq] int64.
    SAMPLING_CONFIG = 18  # [temperature, top_p, top_k_float, use_topp_flag] fp32.
    TOP_P_SCORES = 19  # Sampled token score [bsz, seq] fp32.
    TOP_P_DEBUG = 20  # Debug buffer for top-p kernel [bsz, seq, vocab_size] fp32.

    # Optional / reserved slots for future kernels.
    X_QUANT = 21  # Quantized activation [bsz, seq, dim] fp8.
    X_SCALE = 22  # Activation scale [bsz, seq, dim // block_size] fp32.
    MOE_UP_GATE = 23  # Extra workspace for fused MoE [bsz, seq, n_total, inter_dim] bf16.

    # MTP-related (reserved, Qwen3.6 currently has one MTP layer).
    DRAFT_TOKENS = 24
    PREDICTED_TOKENS = 25
    PREDICTED_HIDDEN = 26
    ACCEPTED_TOKENS = 27
    NEXT_DRAFT_TOKENS = 28
    MTP0_TOKEN_OUT = 29
    MTP0_EXP_OUT = 30
    LAST_HIDDEN_STATES = 31

    # Logprobs / debug.
    TOP_N_LOG_PROBS = 32  # [bsz, seq, 256] fp32.
    TOP_N_INDICES = 33  # [bsz, seq, 256] int32.
    LOGPROBS_FLAG = 34  # [1] int32.


TEMP_VARS_SIZE = 35

Idx = QwenTempVarIdx


def validate_temp_vars_layout() -> None:
    """Validate the temporary-variable index enum.

    Checks:
      1. Enum member count equals TEMP_VARS_SIZE.
      2. Indices are contiguous 0..TEMP_VARS_SIZE-1 with no gaps or duplicates.
      3. (If the backend is loaded) the backend temp_vars_size matches TEMP_VARS_SIZE.

    Raises:
      RuntimeError: If any validation check fails.
    """
    members = list(QwenTempVarIdx)

    if len(members) != TEMP_VARS_SIZE:
        raise RuntimeError(
            f"QwenTempVarIdx has {len(members)} members but TEMP_VARS_SIZE={TEMP_VARS_SIZE}"
        )

    indices = sorted(m.value for m in members)
    expected = list(range(TEMP_VARS_SIZE))
    if indices != expected:
        missing = set(expected) - set(indices)
        dupes = [i for i in indices if indices.count(i) > 1]
        raise RuntimeError(
            f"QwenTempVarIdx indices are not contiguous 0..{TEMP_VARS_SIZE - 1}. "
            f"Missing: {missing}, Duplicates: {set(dupes)}"
        )

    try:
        import torch

        cpp_size = torch.ops.tilert.qwen36_temp_vars_size()
        if cpp_size != TEMP_VARS_SIZE:
            raise RuntimeError(
                f"TEMP_VARS_SIZE={TEMP_VARS_SIZE} != backend temp_vars_size={cpp_size}"
            )
    except (AttributeError, RuntimeError):
        pass
