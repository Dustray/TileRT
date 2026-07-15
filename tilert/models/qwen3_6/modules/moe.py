"""MoE (Mixture of Experts) module for Qwen3.6."""

import torch

from tilert.models.base import SerializableTileRTModule
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.models.qwen3_6.ops.expert_down_allreduce import (
    ExpertDownAllReduce,
    ExpertDownAllReduceAlgorithm,
)
from tilert.models.qwen3_6.ops.expert_sel_up_gate_silu import (
    ExpertSelectUpGateSiLU,
    ExpertSelectUpGateSiLUAlgorithm,
)
from tilert.models.qwen3_6.ops.rmsnorm_expert_proj import RMSNormExpertProj


class QwenMoe(SerializableTileRTModule):
    """Qwen3.6 MoE FFN operations.

    Follows the DSv3.2/GLM5 pattern but uses Qwen3.6 dimensions and keeps
    only the algorithms supported by the Qwen3.6 op wrappers:
      - RMSNormExpertProj: GENERAL
      - ExpertSelectUpGateSiLU: FP8MMA / FP16MMA
      - ExpertDownAllReduce: GENERAL
    """

    def __init__(
        self,
        model_args: ModelArgsQwen36,
        device_id: int,
        num_devices: int,
    ):
        super().__init__(model_args=model_args, device_id=device_id, num_devices=num_devices)

        self.rmsnorm_expert_proj = RMSNormExpertProj(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
        )
        self.register_op(self.rmsnorm_expert_proj)

        self.exp_sel_up_gate_silu = ExpertSelectUpGateSiLU(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            algorithm=ExpertSelectUpGateSiLUAlgorithm.FP8MMA,
        )
        self.register_op(self.exp_sel_up_gate_silu)

        self.expert_down_allreduce = ExpertDownAllReduce(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            algorithm=ExpertDownAllReduceAlgorithm.GENERAL,
        )
        self.register_op(self.expert_down_allreduce)

    def get_weights_list(self) -> list[torch.Tensor]:
        return super().get_weights_list()


class QwenMoeBlock(SerializableTileRTModule):
    """MoE block for Qwen3.6.

    Wraps the MoE FFN as a standalone block.  For Qwen3.6, the attention
    path lives in the heterogeneous layer stack managed by ``QwenDsa``;
    this block only represents the FFN half so it can be cached/shared per
    layer (the same pattern DSv3.2 uses with ``cached_ffn_ops``).
    """

    def __init__(
        self,
        model_args: ModelArgsQwen36,
        device_id: int,
        num_devices: int,
        remove_selected: bool = False,
        moe: QwenMoe | None = None,
    ):
        super().__init__(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            remove_selected=remove_selected,
        )

        self.moe = (
            moe
            if moe is not None
            else QwenMoe(
                model_args=model_args,
                device_id=device_id,
                num_devices=num_devices,
            )
        )
        self.register_op(self.moe)

    def golden_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reference forward: rmsnorm -> gate score -> up/gate/silu -> down."""
        norm_x, scores = self.moe.rmsnorm_expert_proj.golden_forward(x, residual)
        up_gate_out, weights, indices = self.moe.exp_sel_up_gate_silu.golden_forward(
            norm_x, scores
        )
        down_out = self.moe.expert_down_allreduce.golden_forward(up_gate_out, indices, weights)
        if residual is not None:
            down_out = down_out + residual
        return down_out

    def tilert_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """TileRT forward: dispatches to registered ops."""
        norm_x, scores = self.moe.rmsnorm_expert_proj.tilert_forward(x)
        up_gate_out, weights, indices = self.moe.exp_sel_up_gate_silu.tilert_forward(
            norm_x, scores
        )
        return self.moe.expert_down_allreduce.tilert_forward(up_gate_out, indices, weights, x, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.flag_enable_tilert:
            return self.tilert_forward(x)
        return self.golden_forward(x)