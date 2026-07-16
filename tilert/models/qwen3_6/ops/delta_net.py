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
        self.n_kv_heads = model_args.delta_kv_heads
        self.head_dim = model_args.delta_head_dim
        self.dim = model_args.dim
        self.num_local_heads = self.n_heads // num_devices
        self.num_local_kv_heads = max(1, self.n_kv_heads // num_devices)

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
        del weights_map
        return {}

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        del state_dict
        pass

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
        in_proj_qkv = torch.randn(
            args.delta_conv_dim, args.dim, dtype=torch.bfloat16, device=device
        )
        in_proj_z = torch.randn(
            args.delta_gate_dim, args.dim, dtype=torch.bfloat16, device=device
        )
        in_proj_a = torch.randn(
            args.delta_a_dim, args.dim, dtype=torch.bfloat16, device=device
        )
        in_proj_b = torch.randn(
            args.delta_b_dim, args.dim, dtype=torch.bfloat16, device=device
        )
        conv1d = torch.randn(
            args.delta_conv_dim,
            1,
            args.delta_conv_kernel_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        A_log = torch.randn(args.delta_a_dim, dtype=torch.float32, device=device)
        dt_bias = torch.randn(args.delta_b_dim, dtype=torch.float32, device=device)
        norm = torch.randn(args.delta_value_head_dim, dtype=torch.float32, device=device)
        out_proj = torch.randn(
            args.dim, args.delta_v_dim, dtype=torch.bfloat16, device=device
        )
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
        k = k.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Expand k/v to match q head count for the per-head recurrence.
        if self.n_heads != self.n_kv_heads:
            reps = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)

        if state is None:
            state = torch.zeros(
                bsz,
                self.n_heads,
                self.head_dim,
                self.head_dim,
                dtype=q.dtype,
                device=q.device,
            )

        outputs = []
        for t in range(seq_len):
            qt = q[:, :, t, :]
            kt = k[:, :, t, :]
            vt = v[:, :, t, :]
            state = state + kt.unsqueeze(-1) * vt.unsqueeze(-2)
            out_t = (qt.unsqueeze(-1) * state).sum(dim=-2)
            outputs.append(out_t)
        output = torch.stack(outputs, dim=2)
        # Collapse from n_heads to n_kv_heads by averaging groups.
        if self.n_heads != self.n_kv_heads:
            output = output.view(bsz, self.n_kv_heads, reps, seq_len, self.head_dim).mean(dim=2)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_kv_heads * self.head_dim)
        return output, state

    def golden_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference forward using simple linear attention.

        Uses only ``in_proj_qkv`` and ``out_proj`` for a quick sanity check.
        The full gated delta-net recurrence requires the FLA kernel.
        """
        del start_pos
        assert self.in_proj_qkv_weights is not None
        assert self.out_proj_weights is not None
        qkv = x @ self.in_proj_qkv_weights.T
        q, k, v = torch.split(
            qkv,
            [
                self.n_heads * self.head_dim,
                self.n_kv_heads * self.head_dim,
                self.n_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        attn_out, new_state = self._linear_attention(q, k, v, state)
        # Project only the KV-head portion (n_heads may differ from n_kv_heads).
        out = attn_out @ self.out_proj_weights.T
        return out, new_state

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
