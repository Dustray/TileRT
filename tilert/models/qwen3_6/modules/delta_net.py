"""DeltaNet module for Qwen3.6."""

from typing import Any

import torch
import torch.nn.functional as F

from tilert.models.base import SerializableTileRTModule, TileRTModule
from tilert.models.common import RMSNorm, linear
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.models.qwen3_6.ops.rmsnorm_up_gate_silu import (
    RMSNormUpGateSiLU,
    RMSNormUpGateSiLUAlgorithm,
)


class QwenDeltaNetRef(TileRTModule):
    """Reference-only holder for DeltaNet weights.

    This module stores the per-layer q_proj/k_proj/v_proj/o_proj and the two
    RMSNorm weights needed by the DeltaNet block.  It is not used on the
    optimized path; it exists so that the golden path can be implemented as
    soon as the exact DeltaNet math is available.
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

        self.q_proj_weight: torch.Tensor | None = None
        self.k_proj_weight: torch.Tensor | None = None
        self.v_proj_weight: torch.Tensor | None = None
        self.o_proj_weight: torch.Tensor | None = None
        self.input_layernorm_weight: torch.Tensor | None = None
        self.post_attention_layernorm_weight: torch.Tensor | None = None

    def get_weights_list(self) -> list[torch.Tensor]:
        return []

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        del weights_map
        return {}

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.q_proj_weight = state_dict["self_attn.q_proj.weight"]
        self.k_proj_weight = state_dict["self_attn.k_proj.weight"]
        self.v_proj_weight = state_dict["self_attn.v_proj.weight"]
        self.o_proj_weight = state_dict["self_attn.o_proj.weight"]
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
        """Reference forward placeholder.

        A full implementation would run the DeltaNet linear attention
        recurrence and then the FFN.  Until the exact recurrence is available,
        this returns the FFN-only output so the module stack can be wired.
        """
        del start_pos
        ffn_out = self.ffn.golden_forward(x)
        # Sum over the per-SM expert dimension to collapse back to (B, S, dim).
        if ffn_out.dim() == 4:
            ffn_out = ffn_out.sum(dim=2)
        return ffn_out, state

    def tilert_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """TileRT forward placeholder.

        The optimized path will invoke a dedicated ``delta_net_op`` kernel.
        For now it falls back to the golden path.
        """
        return self.golden_forward(x, start_pos, state)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if self.flag_enable_tilert:
            return self.tilert_forward(x, start_pos, state)
        return self.golden_forward(x, start_pos, state)