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

from tilert import logger
from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.common import RMSNorm, linear
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.utils import get_profile_log_tensor

try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )

    _TRANSFORMERS_DELTA_AVAILABLE = True
except Exception:  # pragma: no cover - transformers may be older
    _TRANSFORMERS_DELTA_AVAILABLE = False

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
        # EP8: no tensor parallelism for DeltaNet weights; replicate full
        # projection matrices on every device.
        self.num_local_heads = self.n_heads
        self.num_local_k_heads = self.n_k_heads
        self.num_local_v_heads = self.n_v_heads

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
        """Replicate full DeltaNet reference weights on every device (EP8).

        Under EP8 only the MoE experts are sharded across devices; attention
        and DeltaNet projection weights are replicated in full.  This method
        stacks the same full tensors ``num_devices`` times so the downstream
        ``init_reference_weights`` can still index ``[device_id]``.
        """
        prefix = self.ref_weights_alias.key_prefix
        aliases = self.ref_weights_alias.ref_tensor_alias
        
        logger.info(f"[device_sharding] DeltaNet, key_prefix: {prefix}, num_devices: {self.num_devices}")

        result = {
            alias: torch.stack(
                [weights_map[ref_alias] for _ in range(self.num_devices)], dim=0
            ).contiguous()
            for alias, ref_alias in zip(self.tilert_weights_alias(), aliases)
        }
        
        # Log sharding details for each key
        for alias, ref_alias in zip(self.tilert_weights_alias(), aliases):
            original_shape = weights_map[ref_alias].shape
            logger.info(f"[device_sharding] key: {ref_alias}, original shape: {original_shape}, sharded shape: {result[alias].shape}, dtype: {result[alias].dtype}")
        
        return result

    def _get_local_out_slices(self) -> list[list[slice]]:
        """Unused under EP8; kept for backward compatibility only."""
        return []

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        logger.debug(f"{self.op_name}: init_reference_weights on device {self.device_id}")
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
        self.is_ref_weights_init = True

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        logger.debug(f"{self.op_name}: init_tilert_weights on device {self.device_id}")
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
        logger.debug(f"{self.op_name}: init_random_weights on {device}")
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
        norm = torch.ones(args.delta_value_head_dim, dtype=torch.float32, device=device)
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

    def _linear_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simple DeltaNet-style linear attention reference.

        Uses a state matrix S of shape (bsz, n_kv_heads, head_dim, head_dim).
        q/k are expanded to n_heads and then collapsed back via mean over groups.
        """
        bsz, seq_len, _ = q.shape
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_k_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_v_heads, self.value_head_dim).transpose(1, 2)

        # Expand q/k to match v head count for the per-head recurrence.
        reps = self.n_v_heads // self.n_k_heads
        q = q.repeat_interleave(reps, dim=1)
        k = k.repeat_interleave(reps, dim=1)

        # Use a numerically-stable linear-attention kernel (elu+1 +
        # cumulative-sum normalization).  This reference path intentionally
        # deviates from the true DeltaNet recurrence; its only purpose is to
        # produce bounded, sensible layer outputs for the golden forward.
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        # Temperature to keep the dot-products from amplifying too much over
        # a long cumulative state.
        q = q / (self.head_dim ** 0.5)
        k = k / (self.head_dim ** 0.5)

        if state is None:
            state = torch.zeros(
                bsz,
                self.n_v_heads,
                self.value_head_dim,
                self.head_dim,
                dtype=q.dtype,
                device=q.device,
            )
            norm_state = torch.zeros(
                bsz,
                self.n_v_heads,
                self.head_dim,
                dtype=q.dtype,
                device=q.device,
            )
        else:
            # Existing state is already (S, norm_state) from a previous call.
            state, norm_state = state

        outputs = []
        for t in range(seq_len):
            qt = q[:, :, t, :]
            kt = k[:, :, t, :]
            vt = v[:, :, t, :]
            state = state + kt.unsqueeze(-1) * vt.unsqueeze(-2)
            norm_state = norm_state + kt
            out_t = (qt.unsqueeze(-1) * state).sum(dim=-2)
            out_t = out_t / ((qt * norm_state).sum(dim=-1, keepdim=True) + 1e-6)
            outputs.append(out_t)
        output = torch.stack(outputs, dim=2)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_v_heads * self.value_head_dim)
        return output, (state, norm_state)

    def _gated_delta_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        state: torch.Tensor | None = None,
        chunk_size: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference gated DeltaNet recurrence matching the official rule.

        Delegates to the official ``transformers`` helpers when available:

        * ``torch_recurrent_gated_delta_rule`` for single-token decode steps,
          avoiding the heavy chunk padding/loop overhead of the chunked path.
        * ``torch_chunk_gated_delta_rule`` for longer sequences.

        This keeps TileRT's golden path bit-level compatible with the official
        Qwen3.5-MoE/Qwen3.6 model without relying on ``fla`` / ``causal-conv1d``.
        """
        initial_dtype = q.dtype
        bsz, seq_len, _ = q.shape
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.n_k_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.n_v_heads, self.value_head_dim)

        if self.n_v_heads // self.n_k_heads > 1:
            reps = self.n_v_heads // self.n_k_heads
            q = q.repeat_interleave(reps, dim=2)
            k = k.repeat_interleave(reps, dim=2)

        # beta/g currently (bsz, seq_len, num_v_heads).
        beta = beta.view(bsz, seq_len, q.shape[2])
        g = g.view(bsz, seq_len, q.shape[2])

        if _TRANSFORMERS_DELTA_AVAILABLE:
            if seq_len == 1 and state is not None:
                # Fast recurrent decode path: O(1) per token instead of
                # padding to a full chunk and running the chunk loop.
                attn_out, recurrent_state = torch_recurrent_gated_delta_rule(
                    q, k, v, g, beta,
                    initial_state=state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
                return attn_out, recurrent_state

            attn_out, recurrent_state = torch_chunk_gated_delta_rule(
                q, k, v, g, beta,
                chunk_size=chunk_size,
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            return attn_out, recurrent_state

        # Pure-PyTorch fallback when the official helpers are unavailable.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = self._l2norm(q.float(), dim=-1, eps=1e-6).to(initial_dtype)
        k = self._l2norm(k.float(), dim=-1, eps=1e-6).to(initial_dtype)
        beta = beta.transpose(1, 2)
        g = g.transpose(1, 2)
        q, k, v, beta, g = [x.contiguous().to(torch.float32) for x in (q, k, v, beta, g)]

        batch_size, num_heads, sequence_length, k_head_dim = k.shape
        v_head_dim = v.shape[-1]
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        q = F.pad(q, (0, 0, 0, pad_size))
        k = F.pad(k, (0, 0, 0, pad_size))
        v = F.pad(v, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
        total_sequence_length = sequence_length + pad_size
        scale = 1.0 / (q.shape[-1] ** 0.5)
        q = q * scale

        v_beta = v * beta.unsqueeze(-1)
        k_beta = k * beta.unsqueeze(-1)
        q, k, v, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
            for x in (q, k, v, k_beta, v_beta)
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)

        g = g.cumsum(dim=-1)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        attn = -((k_beta @ k.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
        v = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
        last_recurrent_state = (
            torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(v)
            if state is None
            else state.to(v)
        )
        core_attn_out = torch.zeros_like(v)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)

        for i in range(0, total_sequence_length // chunk_size):
            q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
            attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
            v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
            v_new = v_i - v_prime
            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
            core_attn_out[:, :, i] = attn_inter + attn @ v_new
            last_recurrent_state = (
                last_recurrent_state * g[:, :, i, -1, None, None].exp()
                + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
            )

        core_attn_out = core_attn_out.reshape(
            core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
        )
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        core_attn_out = core_attn_out.reshape(bsz, seq_len, -1)
        return core_attn_out, last_recurrent_state

    @staticmethod
    def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

    def _rmsnorm_gated(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Gated RMSNorm matching ``Qwen3_5MoeRMSNormGated``.

        The official implementation normalises first, multiplies by the
        learnable weight (not ``1 + weight``), and then applies the SiLU gate.
        """
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        x = weight * x.to(input_dtype)
        x = x * F.silu(gate.to(torch.float32))
        return x.to(input_dtype)

    def _causal_conv1d_update(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor | None,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stateful causal depthwise convolution update matching FLA convention.

        Args:
            hidden_states: (bsz, hidden_size, seq_len).
            conv_state: (bsz, hidden_size, state_len) or None.
            weight: (hidden_size, kernel_size).

        Returns:
            (output, updated_conv_state) with output shape
            (bsz, hidden_size, seq_len).
        """
        bsz, hidden_size, seq_len = hidden_states.shape
        kernel_size = weight.shape[-1]
        if conv_state is None:
            conv_state = torch.zeros(bsz, hidden_size, kernel_size, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_new = torch.cat([conv_state, hidden_states], dim=-1)
        conv_state = hidden_new[:, :, -kernel_size:].contiguous()
        out = F.conv1d(
            hidden_new,
            weight.unsqueeze(1),
            groups=hidden_size,
        )
        out = F.silu(out[:, :, -seq_len:])
        return out, conv_state

    def golden_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Reference forward implementing the gated DeltaNet recurrence.

        Replicates the official Qwen3.5-MoE/Qwen3.6 ``Qwen3_5MoeGatedDeltaNet``
        computation: in_proj_qkv/z/b/a, causal depthwise conv1d, gated delta
        rule, and gated RMSNorm, followed by the output projection.

        ``state`` is a tuple ``(conv_state, recurrent_state)`` so that both
        the causal convolution and the linear recurrence remember history
        across decode steps.  ``conv_state`` has shape
        ``(bsz, conv_dim, kernel_size)`` and ``recurrent_state`` has shape
        ``(bsz, num_heads, k_head_dim, v_head_dim)``.
        """
        logger.info(f"[DeltaNetOp.golden_forward_{self.device_id}] ENTRY: x.shape={x.shape}, start_pos={start_pos}, has_state={state is not None}")
        
        del start_pos
        assert self.in_proj_qkv_weights is not None
        assert self.in_proj_z_weights is not None
        assert self.in_proj_a_weights is not None
        assert self.in_proj_b_weights is not None
        assert self.conv1d_weights is not None
        assert self.A_log is not None
        assert self.dt_bias is not None
        assert self.norm_weights is not None
        assert self.out_proj_weights is not None

        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
        bsz, seq_len, dim = x.shape
        logger.info(f"[DeltaNetOp.golden_forward_{self.device_id}] Input: bsz={bsz}, seq_len={seq_len}, dim={dim}")

        # Unpack persistent state.
        conv_state, recurrent_state = state if state is not None else (None, None)

        # Projections.
        logger.info(f"[DeltaNetOp.golden_forward_{self.device_id}] Step1: QKV/Z/B/A projections")
        mixed_qkv = x @ self.in_proj_qkv_weights.T
        z = x @ self.in_proj_z_weights.T
        b = x @ self.in_proj_b_weights.T
        a = x @ self.in_proj_a_weights.T
        logger.info(f"[DeltaNetOp.golden_forward_{self.device_id}]   mixed_qkv.shape={mixed_qkv.shape}, z.shape={z.shape}")

        # Causal depthwise conv1d (groups = conv_dim) with cross-step state.
        logger.info(f"[DeltaNetOp.golden_forward_{self.device_id}] Step2: Causal conv1d")
        mixed_qkv = mixed_qkv.transpose(1, 2)  # (bsz, conv_dim, seq_len)
        conv_weight = self.conv1d_weights.squeeze(1)  # (conv_dim, kernel_size)
        mixed_qkv, conv_state = self._causal_conv1d_update(mixed_qkv, conv_state, conv_weight)
        mixed_qkv = mixed_qkv.transpose(1, 2)  # (bsz, seq_len, conv_dim)
        logger.info(f"[DeltaNetOp.golden_forward_{self.device_id}]   After conv1d: mixed_qkv.shape={mixed_qkv.shape}")

        # Split into q/k/v.
        logger.info(f"[DeltaNetOp.golden_forward_{self.device_id}] Step3: Split Q/K/V")
        q, k, v = torch.split(
            mixed_qkv,
            [
                self.n_heads * self.head_dim,
                self.n_k_heads * self.head_dim,
                self.n_v_heads * self.value_head_dim,
            ],
            dim=-1,
        )
        logger.info(f"[DeltaNetOp.golden_forward_{self.device_id}]   q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}")

        # Decay gate g and beta.
        logger.info(f"[DeltaNetOp.golden_forward_{self.device_id}] Step4: Compute beta and g (gating)")
        beta = torch.sigmoid(b)
        # A is stored as log(A); official uses -exp(A_log) * softplus(a + dt_bias).
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        logger.info(f"[DeltaNetOp.golden_forward_{self.device_id}]   beta.shape={beta.shape}, g.shape={g.shape}")

        logger.info(f"[DeltaNetOp.golden_forward_{self.device_id}] Step5: Gated delta attention")
        attn_out, recurrent_state = self._gated_delta_attention(q, k, v, beta, g, recurrent_state)
        logger.info(f"[DeltaNetOp.golden_forward_{self.device_id}]   attn_out.shape={attn_out.shape}")

        # Gated RMSNorm.
        logger.info(f"[DeltaNetOp.golden_forward_{self.device_id}] Step6: Gated RMSNorm")
        attn_out = attn_out.reshape(-1, self.value_head_dim)
        z_gate = z.reshape(-1, self.value_head_dim)
        attn_out = self._rmsnorm_gated(attn_out, z_gate, self.norm_weights)
        attn_out = attn_out.reshape(bsz, seq_len, -1)
        logger.info(f"[DeltaNetOp.golden_forward_{self.device_id}]   After RMSNorm: attn_out.shape={attn_out.shape}")

        logger.info(f"[DeltaNetOp.golden_forward_{self.device_id}] Step7: Output projection")
        out = attn_out @ self.out_proj_weights.T
        out = out.view(original_shape)
        logger.info(f"[DeltaNetOp.golden_forward_{self.device_id}] EXIT: out.shape={out.shape}")
        
        return out, (conv_state, recurrent_state)

    def tilert_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optimized forward placeholder."""
        logger.info(f"[DeltaNetOp.tilert_forward_{self.device_id}] ENTRY: x.shape={x.shape}, start_pos={start_pos}, has_state={state is not None}")
        
        assert self.is_init
        assert self.hidden_out is not None
        assert self.state_out is not None
        assert self.profile_logs is not None
        
        logger.info(f"[DeltaNetOp.tilert_forward_{self.device_id}] Calling CUDA kernel delta_net")
        delta_net(
            x,
            state if state is not None else torch.zeros_like(self.state_out),
            self.hidden_out,
            self.state_out,
            torch.tensor([start_pos], dtype=torch.int32, device=x.device),
            self.profile_logs,
            model_arch=self.model_args.arch_name,
        )
        logger.info(f"[DeltaNetOp.tilert_forward_{self.device_id}] CUDA kernel returned, hidden_out.shape={self.hidden_out.shape}")
        
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
