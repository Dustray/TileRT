"""Utility functions for tilert models."""

__all__ = [
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "precompute_mrope_embed",
    "apply_mrope_embed",
]

import math
from enum import IntEnum

import torch

_FACTOR_OVERRIDE_UNSET = object()
_THETA_OVERRIDE_UNSET = object()


def precompute_freqs_cis(  # type: ignore[no-untyped-def]
    args,
    *,
    factor_override=_FACTOR_OVERRIDE_UNSET,
    theta_override=_THETA_OVERRIDE_UNSET,
) -> torch.Tensor:
    """
    Pre-computes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.
        factor_override: If unset, ``args.rope_factor`` is used. Pass a
            numeric value to override the factor inline.
        theta_override: If unset, ``args.rope_theta`` is used. Pass a numeric
            value to override the rope base. ``None`` is rejected.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = getattr(args, "qk_rope_head_dim", getattr(args, "rope_dim", None))
    if dim is None:
        raise AttributeError("args must contain qk_rope_head_dim or rope_dim")
    seqlen = args.max_seq_len
    beta_fast = getattr(args, "beta_fast", 32)
    beta_slow = getattr(args, "beta_slow", 1)
    base = args.rope_theta if theta_override is _THETA_OVERRIDE_UNSET else theta_override
    factor = getattr(args, "rope_factor", None) if factor_override is _FACTOR_OVERRIDE_UNSET else factor_override

    def find_correction_dim(num_rotations: float, dim: int, base: float, max_seq_len: int) -> float:
        """
        Find correction dimension.

        Computes the correction dimension for a given number of rotations in the rotary positional
        embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(
        low_rot: float,
        high_rot: float,
        dim: int,
        base: float,
        max_seq_len: int,
    ) -> tuple[int, int]:
        """
        Find correction range.

        Computes the range of correction dimensions for rotary positional
            embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high),
                clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_value: float, max_value: float, dim: int) -> torch.Tensor:
        """
        Linear ramp function.

        Computes a linear ramp function used to smooth values between a minimum
            and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly
                interpolated between 0 and 1, clamped to the range [0, 1].
        """
        if min_value == max_value:
            max_value += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_value) / (max_value - min_value)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if factor is not None and seqlen > getattr(args, "original_seq_len", seqlen):
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, getattr(args, "original_seq_len", seqlen)
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t_index = torch.arange(seqlen)
    freqs = torch.outer(t_index, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    x_in: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True
) -> torch.Tensor:
    """Applies rotary positional embeddings to the input tensor.

    Args:
        x_in: Input tensor with positional embeddings to be applied.
        freqs_cis: Precomputed complex exponential values for positional embeddings.
        interleaved: If True (default), adjacent pairs (x0,x1),(x2,x3)... form
            complex numbers. If False, half-half layout: (x0,x_{d/2}),(x1,x_{d/2+1})...
            The DeepSeek-V3.2-Exp indexer uses interleaved=False.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x_in.dtype
    shape = x_in.shape
    if not interleaved:
        x_in = x_in.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x_in = torch.view_as_complex(x_in.float().view(*shape[:-1], -1, 2))
    # ``freqs_cis`` may be 1-D (length rope_dim), 2-D (seq_len, rope_dim), or
    # already broadcastable (1, seq_len, 1, rope_dim).  Normalize to the
    # broadcastable shape expected by the multiplication.
    if freqs_cis.dim() == 1:
        freqs_cis = freqs_cis.view(1, 1, 1, -1)
    elif freqs_cis.dim() == 2:
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    y_out = torch.view_as_real(x_in * freqs_cis).flatten(3)
    if not interleaved:
        y_out = torch.cat([y_out[..., 0::2], y_out[..., 1::2]], dim=-1)
    return y_out.to(dtype)


def _mrope_inv_freq(partial_rotary_factor: float, head_dim: int, base: float, device=None) -> torch.Tensor:
    """Compute inverse frequencies for M-RoPE.

    Matches ``Qwen3_5MoeTextRotaryEmbedding.compute_default_rope_parameters``:
    ``dim = int(head_dim * partial_rotary_factor)`` and
    ``inv_freq = 1 / base^(arange(0, dim, 2) / dim)``.
    """
    dim = int(head_dim * partial_rotary_factor)
    if dim % 2 != 0:
        dim = (dim // 2) * 2
    return 1.0 / (
        base
        ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).to(dtype=torch.float32) / dim)
    )


def _apply_interleaved_mrope(
    freqs: torch.Tensor, mrope_section: list[int]
) -> torch.Tensor:
    """Apply interleaved M-RoPE layout.

    Args:
        freqs: (3, batch, seq_len, head_dim // 2) tensor of T/H/W frequencies.
        mrope_section: [n_t, n_h, n_w] sections summing to head_dim // 2.

    Returns:
        (batch, seq_len, head_dim // 2) tensor with interleaved T/H/W segments.
    """
    # Matches ``Qwen3_5MoeTextRotaryEmbedding.apply_interleaved_mrope``.
    freqs_t = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):  # H, W
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


def precompute_mrope_embed(
    args,
    max_seq_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute M-RoPE cosine/sine embeddings for Qwen3.5-MoE/Qwen3.6.

    Args:
        args: Model arguments. Must contain ``rope_theta``, ``qk_head_dim`` or
            ``rope_dim``, ``partial_rotary_factor``, ``mrope_section``, and
            optionally ``use_mrope``.
        max_seq_len: Sequence length to precompute. Defaults to ``args.max_seq_len``.

    Returns:
        Tuple of (cos, sin) tensors with shape ``(max_seq_len, rope_dim)``
        where ``rope_dim = int(head_dim * partial_rotary_factor)``.
    """
    head_dim = getattr(args, "qk_head_dim", getattr(args, "head_dim", None))
    if head_dim is None:
        raise AttributeError("args must contain qk_head_dim or head_dim")
    partial_rotary_factor = getattr(args, "partial_rotary_factor", 1.0)
    mrope_section = getattr(args, "mrope_section", None)
    use_mrope = getattr(args, "use_mrope", mrope_section is not None)
    base = getattr(args, "rope_theta", 10000.0)
    seqlen = max_seq_len if max_seq_len is not None else args.max_seq_len

    if not use_mrope or mrope_section is None:
        # Fallback to standard 1D RoPE (keeps existing callers working).
        dim = getattr(args, "rope_dim", int(head_dim * partial_rotary_factor))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t_index = torch.arange(seqlen, dtype=torch.float32)
        freqs = torch.outer(t_index, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    # Compute M-RoPE tables entirely in numpy to avoid any DCU/HIP tensor
    # operations that can segfault inside a cuda device context.
    import numpy as np

    dim = int(head_dim * partial_rotary_factor)
    if dim % 2 != 0:
        dim = (dim // 2) * 2
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    t_index = np.arange(seqlen, dtype=np.float32)
    freqs_1d = np.outer(t_index, inv_freq).astype(np.float32)  # (seq_len, dim//2)

    # Replicate to 3 T/H/W grids and apply interleaved M-RoPE in numpy.
    freqs_3d = np.tile(freqs_1d[np.newaxis, np.newaxis, :, :], (3, 1, 1, 1))
    freqs_t = freqs_3d[0].copy()
    section = list(mrope_section)
    for dim_idx, offset in enumerate((1, 2), start=1):  # H, W
        length = section[dim_idx] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs_3d[dim_idx, ..., idx]
    freqs_mrope = freqs_t.squeeze(0)  # (seq_len, dim//2)

    emb = np.concatenate([freqs_mrope, freqs_mrope], axis=-1).astype(np.float32)
    cos = torch.from_numpy(np.cos(emb).copy())
    sin = torch.from_numpy(np.sin(emb).copy())
    return cos.contiguous(), sin.contiguous()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input (non-interleaved)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_mrope_embed(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply M-RoPE cosine/sine embeddings to query and key tensors.

    Matches the official ``apply_rotary_pos_emb`` (non-interleaved) used by
    Qwen3.5-MoE/Qwen3.6.

    Args:
        q: Query tensor, shape ``(..., seq_len, head_dim)`` or
            ``(..., n_heads, seq_len, head_dim)``.
        k: Key tensor with same rank as ``q``.
        cos: Cosine embedding, shape ``(seq_len, rotary_dim)`` or broadcastable.
        sin: Sine embedding, shape ``(seq_len, rotary_dim)`` or broadcastable.
        unsqueeze_dim: Dimension along which to unsqueeze cos/sin so they
            broadcast to q/k. Defaults to 1 (heads dim for
            ``(batch, heads, seq, head_dim)``).

    Returns:
        Rotated ``(q_embed, k_embed)`` tensors.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = q_rot * cos + _rotate_half(q_rot) * sin
    k_embed = k_rot * cos + _rotate_half(k_rot) * sin

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class SwizzleMode(IntEnum):
    """Swizzle mode."""

    SWIZZLE_NONE = 0
    SWIZZLE_32B = 32 // 16
    SWIZZLE_64B = 64 // 16
    SWIZZLE_128B = 128 // 16


def gen_tensor_swizzle_map_1d(
    rows: int, cols_in_16bytes: int, swizzle_mode: SwizzleMode = SwizzleMode.SWIZZLE_128B
) -> torch.Tensor:
    """
    Generate flattened 1D swizzle map for given tensor dimensions.

    Args:
        rows (int): Number of rows in the tensor.
        cols_in_16bytes (int): Number of columns in the tensor, in 16-byte units.
        swizzle_mode (SwizzleMode): Swizzle mode to use. Default is SWIZZLE_128B.

    Returns:
        torch.Tensor: Flattened 1D swizzle map, in 16-byte units.
    """
    idxs = torch.arange(rows * cols_in_16bytes, dtype=torch.int32)
    if swizzle_mode == SwizzleMode.SWIZZLE_NONE:
        return idxs
    row_ids = idxs // cols_in_16bytes
    col_ids = idxs % cols_in_16bytes
    col_ids = (row_ids % swizzle_mode) ^ col_ids
    return row_ids * cols_in_16bytes + col_ids
