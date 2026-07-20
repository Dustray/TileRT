"""DeltaNet linear attention operation module for Qwen3.6.

This is a Python-side placeholder wrapper for the future CUDA kernel
``delta_net_op``.  DeltaNet is a linear attention variant used in 30 of the
40 Qwen3.6 layers.  The Python wrapper only defines the weight alias / converter
interfaces and a trivial reference forward; the actual recurrence / chunk-wise
kernel will be implemented in CUDA.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn.functional as F

from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.common import RMSNorm, linear
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.utils import get_profile_log_tensor


# ---------------------------------------------------------------------------
# Pure-PyTorch fallback for Gated DeltaNet.
# Adapted from the Hugging Face transformers Qwen3_5MoeGatedDeltaNet fallback
# implementation, used as the golden reference until the dedicated CUDA kernel
# is available.
# ---------------------------------------------------------------------------


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalization aligned with the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _torch_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str = "silu",
) -> torch.Tensor:
    """Fallback causal conv1d: x shape (B, C, L), weight shape (C, K)."""
    batch_size, channels, seq_len = x.shape
    kernel_size = weight.shape[-1]
    # causal padding
    x = F.pad(x, (kernel_size - 1, 0))
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=channels)
    out = out[:, :, :seq_len]
    if activation == "silu":
        out = F.silu(out)
    return out.to(x.dtype)


def _torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked gated delta rule (pure torch, matches HF fallback)."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    for i in range(0, (sequence_length + pad_size) // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None])
            .transpose(-1, -2)
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Recurrent gated delta rule for single-token decode (pure torch)."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim, dtype=torch.float32, device=value.device
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _rmsnorm_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Gated RMSNorm: x shape (..., head_dim), gate shape same."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    gate_f = gate.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = weight * x
    x = x * F.silu(gate_f)
    return x.to(input_dtype)

__all__ = [
    "delta_net",
    "DeltaNetAlgorithm",
    "DeltaNetRefWeightsAlias",
    "DeltaNetTilertWeightsAlias",
    "DeltaNetWeightsConverter",
    "DeltaNetOp",
]


def delta_net(
    hidden_in: torch.Tensor,
    state_in: torch.Tensor,
    hidden_out: torch.Tensor,
    state_out: torch.Tensor,
    start_pos: torch.Tensor,
    profile_logs: torch.Tensor,
    model_arch: str,
    compute_kernel_type: str = "general",
) -> None:
    """DeltaNet linear attention operation.

    Args:
        hidden_in: Input hidden states (bsz, seq, dim).
        state_in: Recurrent state from previous chunk (bsz, n_kv_heads, head_dim, head_dim).
        hidden_out: Output hidden states (bsz, seq, dim).
        state_out: Updated recurrent state.
        start_pos: Current start position (int32 scalar).
        profile_logs: Profile logs tensor.
        model_arch: Architecture string.
        compute_kernel_type: Kernel type ("general" for now).
    """
    torch.ops.tilert.delta_net_op(
        hidden_in,
        state_in,
        hidden_out,
        state_out,
        start_pos,
        profile_logs,
        model_arch,
        compute_kernel_type,
    )


@dataclass
class DeltaNetRefWeightsAlias:
    """Reference weights alias for DeltaNet (Qwen3.6 linear_attention layer)."""

    key_prefix: str = "linear_attn"

    @property
    def ref_tensor_alias(self) -> list[str]:
        return [
            f"{self.key_prefix}.in_proj_qkv.weight",
            f"{self.key_prefix}.in_proj_z.weight",
            f"{self.key_prefix}.in_proj_a.weight",
            f"{self.key_prefix}.in_proj_b.weight",
            f"{self.key_prefix}.conv1d.weight",
            f"{self.key_prefix}.A_log",
            f"{self.key_prefix}.dt_bias",
            f"{self.key_prefix}.norm.weight",
            f"{self.key_prefix}.out_proj.weight",
        ]

    def __call__(self) -> list[str]:
        return self.ref_tensor_alias


@dataclass
class DeltaNetTilertWeightsAlias:
    """TileRT weights alias for DeltaNet."""

    in_proj_qkv_weights = "in_proj_qkv_weights"
    in_proj_z_weights = "in_proj_z_weights"
    in_proj_a_weights = "in_proj_a_weights"
    in_proj_b_weights = "in_proj_b_weights"
    conv1d_weights = "conv1d_weights"
    A_log = "A_log"
    dt_bias = "dt_bias"
    norm_weights = "norm_weights"
    out_proj_weights = "out_proj_weights"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [
            self.in_proj_qkv_weights,
            self.in_proj_z_weights,
            self.in_proj_a_weights,
            self.in_proj_b_weights,
            self.conv1d_weights,
            self.A_log,
            self.dt_bias,
            self.norm_weights,
            self.out_proj_weights,
        ]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class DeltaNetAlgorithm(Enum):
    """DeltaNet algorithm."""

    GENERAL = "general"


class DeltaNetWeightsConverter(TilertWeightsConverter):
    """DeltaNet weights converter.

    The checkpoint stores the linear attention weights in plain bf16.  The
    converter preserves the original layout so the CUDA kernel can interpret the
    tensors directly; no FP8 quantization is performed for Qwen3.6.
    """

    def convert_to_general(
        self, weights_list: list[torch.Tensor]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, conv1d, A_log, dt_bias, norm, out_proj = (
            weights_list
        )
        return (
            in_proj_qkv,
            in_proj_z,
            in_proj_a,
            in_proj_b,
            conv1d,
            A_log,
            dt_bias,
            norm,
            out_proj,
        )


class DeltaNetOp(TileRTModule):
    """DeltaNet linear attention op wrapper for Qwen3.6."""

    _SUPPORTED_ALGORITHMS = {
        "qwen3_6": [DeltaNetAlgorithm.GENERAL],
    }

    def __init__(
        self,
        model_args: ModelArgsQwen36,
        device_id: int,
        num_devices: int,
        algorithm: DeltaNetAlgorithm = DeltaNetAlgorithm.GENERAL,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
        )
        self.algorithm = algorithm
        self.n_heads = model_args.delta_q_heads
        self.n_k_heads = model_args.delta_k_heads
        self.n_v_heads = model_args.delta_v_heads
        self.head_dim = model_args.delta_key_head_dim
        self.value_head_dim = model_args.delta_value_head_dim
        self.dim = model_args.dim
        self.num_local_heads = self.n_heads // num_devices
        self.num_local_k_heads = max(1, self.n_k_heads // num_devices)
        self.num_local_v_heads = max(1, self.n_v_heads // num_devices)

        self.tilert_weights_alias = DeltaNetTilertWeightsAlias()
        self.ref_weights_alias = DeltaNetRefWeightsAlias()

        self.in_proj_qkv_weights: torch.Tensor | None = None
        self.in_proj_z_weights: torch.Tensor | None = None
        self.in_proj_a_weights: torch.Tensor | None = None
        self.in_proj_b_weights: torch.Tensor | None = None
        self.conv1d_weights: torch.Tensor | None = None
        self.A_log: torch.Tensor | None = None
        self.dt_bias: torch.Tensor | None = None
        self.norm_weights: torch.Tensor | None = None
        self.out_proj_weights: torch.Tensor | None = None

        self.hidden_out: torch.Tensor | None = None
        self.state_out: torch.Tensor | None = None
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
            self.in_proj_qkv_weights,
            self.in_proj_z_weights,
            self.in_proj_a_weights,
            self.in_proj_b_weights,
            self.conv1d_weights,
            self.A_log,
            self.dt_bias,
            self.norm_weights,
            self.out_proj_weights,
        ]

    def device_sharding(
        self, weights_map: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Shard DeltaNet reference weights across devices.

        Splits the column dimensions of the projection weights so each device
        owns a contiguous slice.  The output dimensions differ per tensor:

        - in_proj_qkv: (n_q_heads + 2*n_kv_heads) * head_dim
        - in_proj_z/out_proj: value_dim (n_kv_heads * value_head_dim)
        - in_proj_a/in_proj_b: n_kv_heads * a/b dim
        - conv1d: same as in_proj_qkv
        - A_log/dt_bias/norm: small per-head/per-value vectors, split similarly
        """
        prefix = self.ref_weights_alias.key_prefix
        aliases = self.ref_weights_alias.ref_tensor_alias
        out_slices = self._get_local_out_slices()

        args = self.model_args
        sharded: dict[str, list[torch.Tensor]] = {alias: [] for alias in self.tilert_weights_alias()}
        for dev in range(self.num_devices):
            slc = out_slices[dev]
            qkv = weights_map[aliases[0]][slc[0]]
            z = weights_map[aliases[1]][slc[1]]
            a = weights_map[aliases[2]][slc[2]]
            b = weights_map[aliases[3]][slc[3]]
            # conv1d weight layout is (out_channels, 1, kernel_size); the
            # out-channel dimension matches ``in_proj_qkv`` so slice dim 0.
            conv1d = weights_map[aliases[4]][slc[0], :, :]
            A_log = weights_map[aliases[5]][slc[2]]
            dt_bias = weights_map[aliases[6]][slc[3]]
            # ``norm.weight`` has shape (delta_value_head_dim,).  It is not
            # the same size as z/gate dim, so split it evenly across devices.
            norm_slc = slc[4]
            norm = weights_map[aliases[7]][norm_slc]
            out_proj = weights_map[aliases[8]][:, slc[1]]
            sharded[self.tilert_weights_alias.in_proj_qkv_weights].append(qkv)
            sharded[self.tilert_weights_alias.in_proj_z_weights].append(z)
            sharded[self.tilert_weights_alias.in_proj_a_weights].append(a)
            sharded[self.tilert_weights_alias.in_proj_b_weights].append(b)
            sharded[self.tilert_weights_alias.conv1d_weights].append(conv1d)
            sharded[self.tilert_weights_alias.A_log].append(A_log)
            sharded[self.tilert_weights_alias.dt_bias].append(dt_bias)
            sharded[self.tilert_weights_alias.norm_weights].append(norm)
            sharded[self.tilert_weights_alias.out_proj_weights].append(out_proj)

        return {
            alias: torch.stack(tensors, dim=0).contiguous()
            for alias, tensors in sharded.items()
        }

    def _get_local_out_slices(self) -> list[list[slice]]:
        """Return per-device column slices for each DeltaNet projection."""
        args = self.model_args
        qkv_out = args.delta_conv_dim  # n_q_heads * head_dim + 2 * n_kv_heads * head_dim
        z_out = args.delta_gate_dim    # n_kv_heads * value_head_dim
        a_out = args.delta_a_dim       # n_kv_heads * a_dim
        b_out = args.delta_b_dim       # n_kv_heads * b_dim
        norm_out = args.delta_value_head_dim  # per-device split for norm.weight
        qkv_per_dev = qkv_out // self.num_devices
        z_per_dev = z_out // self.num_devices
        a_per_dev = a_out // self.num_devices
        b_per_dev = b_out // self.num_devices
        norm_per_dev = norm_out // self.num_devices
        slices = []
        for dev in range(self.num_devices):
            slices.append([
                slice(dev * qkv_per_dev, (dev + 1) * qkv_per_dev),
                slice(dev * z_per_dev, (dev + 1) * z_per_dev),
                slice(dev * a_per_dev, (dev + 1) * a_per_dev),
                slice(dev * b_per_dev, (dev + 1) * b_per_dev),
                slice(dev * norm_per_dev, (dev + 1) * norm_per_dev),
            ])
        return slices

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        sharded = self.device_sharding(state_dict)
        did = self.device_id
        self.in_proj_qkv_weights = sharded[self.tilert_weights_alias.in_proj_qkv_weights][did]
        self.in_proj_z_weights = sharded[self.tilert_weights_alias.in_proj_z_weights][did]
        self.in_proj_a_weights = sharded[self.tilert_weights_alias.in_proj_a_weights][did]
        self.in_proj_b_weights = sharded[self.tilert_weights_alias.in_proj_b_weights][did]
        self.conv1d_weights = sharded[self.tilert_weights_alias.conv1d_weights][did]
        self.A_log = sharded[self.tilert_weights_alias.A_log][did]
        self.dt_bias = sharded[self.tilert_weights_alias.dt_bias][did]
        self.norm_weights = sharded[self.tilert_weights_alias.norm_weights][did]
        self.out_proj_weights = sharded[self.tilert_weights_alias.out_proj_weights][did]

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        weights_list = [state_dict[alias] for alias in self.tilert_weights_alias()]
        converter = DeltaNetWeightsConverter(self.model_args, self.num_devices)
        (
            self.in_proj_qkv_weights,
            self.in_proj_z_weights,
            self.in_proj_a_weights,
            self.in_proj_b_weights,
            self.conv1d_weights,
            self.A_log,
            self.dt_bias,
            self.norm_weights,
            self.out_proj_weights,
        ) = converter.dispatch(self.algorithm, weights_list)

    def init_tilert_vars(
        self, batch_size: int, seq_len: int, device: str = "cuda"
    ) -> None:
        self.hidden_out = torch.zeros(
            batch_size,
            seq_len,
            self.dim,
            dtype=torch.bfloat16,
            device=device,
        )
        self.state_out = torch.zeros(
            batch_size,
            self.num_local_kv_heads,
            self.head_dim,
            self.head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        self.profile_logs = get_profile_log_tensor(device=device)
        self.is_init = True

    def init_random_weights(self, device: str = "cuda") -> None:
        args = self.model_args
        # Scale random weights by 1/sqrt(fan_in) so each layer preserves the
        # input variance.  This makes the 40-layer reference forward numerically
        # stable when running sanity tests without a real checkpoint.
        in_proj_qkv = torch.randn(
            args.delta_conv_dim, args.dim, dtype=torch.bfloat16, device=device
        ) / (args.dim ** 0.5)
        in_proj_z = torch.randn(
            args.delta_gate_dim, args.dim, dtype=torch.bfloat16, device=device
        ) / (args.dim ** 0.5)
        in_proj_a = torch.randn(
            args.delta_a_dim, args.dim, dtype=torch.bfloat16, device=device
        ) / (args.dim ** 0.5)
        in_proj_b = torch.randn(
            args.delta_b_dim, args.dim, dtype=torch.bfloat16, device=device
        ) / (args.dim ** 0.5)
        conv1d = torch.randn(
            args.delta_conv_dim,
            1,
            args.delta_conv_kernel_dim,
            dtype=torch.bfloat16,
            device=device,
        ) / (args.delta_conv_kernel_dim ** 0.5)
        A_log = torch.randn(args.delta_a_dim, dtype=torch.float32, device=device)
        dt_bias = torch.randn(args.delta_b_dim, dtype=torch.float32, device=device)
        norm = torch.randn(args.delta_value_head_dim, dtype=torch.float32, device=device)
        out_proj = torch.randn(
            args.dim, args.delta_v_dim, dtype=torch.bfloat16, device=device
        ) / (args.delta_v_dim ** 0.5)
        converter = DeltaNetWeightsConverter(self.model_args, self.num_devices)
        (
            self.in_proj_qkv_weights,
            self.in_proj_z_weights,
            self.in_proj_a_weights,
            self.in_proj_b_weights,
            self.conv1d_weights,
            self.A_log,
            self.dt_bias,
            self.norm_weights,
            self.out_proj_weights,
        ) = converter.convert_to_general(
            [in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, conv1d, A_log, dt_bias, norm, out_proj]
        )

    def _gated_delta_net_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference forward using the Hugging Face Gated DeltaNet fallback.

        This mirrors ``Qwen3_5MoeGatedDeltaNet.forward`` so that the golden path
        produces logits identical to the original model when the same weights
        are used.
        """
        batch_size, seq_len, _ = x.shape

        # Projections.
        mixed_qkv = x @ self.in_proj_qkv_weights.T
        mixed_qkv = mixed_qkv.transpose(1, 2)  # (B, conv_dim, L)

        z = x @ self.in_proj_z_weights.T
        z = z.reshape(batch_size, seq_len, self.n_v_heads, self.value_head_dim)

        b = x @ self.in_proj_b_weights.T
        a = x @ self.in_proj_a_weights.T

        # Causal conv1d fallback.
        mixed_qkv = _torch_causal_conv1d(
            mixed_qkv,
            self.conv1d_weights.squeeze(1),
            bias=None,
            activation="silu",
        )
        mixed_qkv = mixed_qkv.transpose(1, 2)  # (B, L, conv_dim)

        query, key, value = torch.split(
            mixed_qkv,
            [
                self.n_k_heads * self.head_dim,
                self.n_k_heads * self.head_dim,
                self.n_v_heads * self.value_head_dim,
            ],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, self.n_k_heads, self.head_dim)
        key = key.reshape(batch_size, seq_len, self.n_k_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.n_v_heads, self.value_head_dim)

        beta = torch.sigmoid(b)
        # A_log and dt_bias are (num_v_heads,); a/b are (B, L, num_v_heads).
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())

        if self.n_v_heads // self.n_k_heads > 1:
            query = query.repeat_interleave(self.n_v_heads // self.n_k_heads, dim=2)
            key = key.repeat_interleave(self.n_v_heads // self.n_k_heads, dim=2)

        # Transpose to (B, num_heads, L, head_dim) for the delta-rule kernels.
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        beta = beta.transpose(1, 2)
        g = g.transpose(1, 2)

        if state is None:
            core_attn_out, new_state = _torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                chunk_size=64,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            # Decode mode: state is the recurrent state matrix.
            core_attn_out, new_state = _torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        # Gated RMSNorm.
        core_attn_out = core_attn_out.reshape(-1, self.value_head_dim)
        z = z.reshape(-1, self.value_head_dim)
        core_attn_out = _rmsnorm_gated(core_attn_out, self.norm_weights, z, eps=1e-6)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        out = core_attn_out @ self.out_proj_weights.T
        return out, new_state

    def golden_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference forward using the HF Gated DeltaNet fallback."""
        return self._gated_delta_net_forward(x, start_pos, state)

    def tilert_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optimized forward placeholder."""
        assert self.is_init
        assert self.hidden_out is not None
        assert self.state_out is not None
        assert self.profile_logs is not None
        delta_net(
            x,
            state if state is not None else torch.zeros_like(self.state_out),
            self.hidden_out,
            self.state_out,
            torch.tensor([start_pos], dtype=torch.int32, device=x.device),
            self.profile_logs,
            model_arch=self.model_args.arch_name,
        )
        return self.hidden_out, self.state_out

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.flag_enable_tilert:
            return self.tilert_forward(x, start_pos, state)
        return self.golden_forward(x, start_pos, state)
