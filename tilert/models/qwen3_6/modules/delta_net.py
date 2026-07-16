"""DeltaNet module for Qwen3.6."""

from typing import Any

import torch
import torch.nn.functional as F

from tilert.models.base import SerializableTileRTModule, TileRTModule
from tilert.models.common import RMSNorm, linear
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.models.qwen3_6.ops.delta_net import (
    DeltaNetOp,
    DeltaNetAlgorithm,
)
from tilert.models.qwen3_6.ops.rmsnorm_up_gate_silu import (
    RMSNormUpGateSiLU,
    RMSNormUpGateSiLUAlgorithm,
)


class QwenDeltaNetRef(TileRTModule):
    """Reference-only holder for DeltaNet weights.

    Mirrors the weight aliases in ``ops.delta_net.DeltaNetRefWeightsAlias``.
    The original checkpoint stores the linear attention weights under the
    ``linear_attn.*`` prefix; this holder keeps those tensors available for
    the golden/reference path while the optimized path consumes the TileRT
    sharded aliases produced by the weight converter.
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
        self.delta_q_heads = model_args.delta_q_heads
        self.delta_kv_heads = model_args.delta_kv_heads
        self.delta_head_dim = model_args.delta_head_dim

        self.in_proj_qkv_weight: torch.Tensor | None = None
        self.in_proj_z_weight: torch.Tensor | None = None
        self.in_proj_a_weight: torch.Tensor | None = None
        self.in_proj_b_weight: torch.Tensor | None = None
        self.conv1d_weight: torch.Tensor | None = None
        self.A_log: torch.Tensor | None = None
        self.dt_bias: torch.Tensor | None = None
        self.norm_weight: torch.Tensor | None = None
        self.out_proj_weight: torch.Tensor | None = None
        self.input_layernorm_weight: torch.Tensor | None = None
        self.post_attention_layernorm_weight: torch.Tensor | None = None

    def get_weights_list(self) -> list[torch.Tensor]:
        return []

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        del weights_map
        return {}

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        prefix = "linear_attn"
        self.in_proj_qkv_weight = state_dict[f"{prefix}.in_proj_qkv.weight"]
        self.in_proj_z_weight = state_dict[f"{prefix}.in_proj_z.weight"]
        self.in_proj_a_weight = state_dict[f"{prefix}.in_proj_a.weight"]
        self.in_proj_b_weight = state_dict[f"{prefix}.in_proj_b.weight"]
        self.conv1d_weight = state_dict[f"{prefix}.conv1d.weight"]
        self.A_log = state_dict[f"{prefix}.A_log"]
        self.dt_bias = state_dict[f"{prefix}.dt_bias"]
        self.norm_weight = state_dict[f"{prefix}.norm.weight"]
        self.out_proj_weight = state_dict[f"{prefix}.out_proj.weight"]
        self.input_layernorm_weight = state_dict["input_layernorm.weight"]
        self.post_attention_layernorm_weight = state_dict["post_attention_layernorm.weight"]

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        del state_dict

    def init_random_weights(self) -> None:
        pass

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        del batch_size, seq_len


class DeltaNet(SerializableTileRTModule):
    """DeltaNet linear attention layer for Qwen3.6.

    DeltaNet is a linear attention mechanism used in the DeltaNet layers
    (3 per block × 10 blocks = 30 layers).

    For now the Python wrapper exposes:
      - RMSNormUpGateSiLU for the FFN half (reuse the dense-MLP path because
        DeltaNet layers have an MLP-like up/gate/down projection after the
        linear attention).
      - A reference weight holder for the Q/K/V/O projections.

    The actual DeltaNet recurrence / chunk-wise kernel will live in a dedicated
    CUDA kernel; this module wires the Python-side plumbing so the layer can
    be instantiated inside ``QwenDsa``.
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

        self.delta_ref = QwenDeltaNetRef(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.delta_ref)

        self.attn = DeltaNetOp(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            algorithm=DeltaNetAlgorithm.GENERAL,
        )
        self.register_op(self.attn)

        self.ffn = RMSNormUpGateSiLU(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            algorithm=RMSNormUpGateSiLUAlgorithm.FP8MMA,
        )
        self.register_op(self.ffn)

    def golden_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Reference forward: DeltaNet linear attention + FFN."""
        prev_state = state.get("delta_state") if state is not None else None
        attn_out, new_state = self.attn.golden_forward(x, start_pos, prev_state)
        ffn_out = self.ffn.golden_forward(attn_out)
        if ffn_out.dim() == 4:
            ffn_out = ffn_out.sum(dim=2)
        next_state = {"delta_state": new_state} if state is not None else None
        return ffn_out, next_state

    def tilert_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Optimized forward using ``DeltaNetOp`` + ``RMSNormUpGateSiLU``."""
        prev_state = state.get("delta_state") if state is not None else None
        attn_out, new_state = self.attn.forward(x, start_pos, prev_state)
        ffn_out = self.ffn.forward(attn_out)
        if ffn_out.dim() == 4:
            ffn_out = ffn_out.sum(dim=2)
        next_state = {"delta_state": new_state} if state is not None else None
        return ffn_out, next_state

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if self.flag_enable_tilert:
            return self.tilert_forward(x, start_pos, state)
        return self.golden_forward(x, start_pos, state)