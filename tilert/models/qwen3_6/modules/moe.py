"""MoE (Mixture of Experts) module for Qwen3.6."""

import torch

from tilert import logger
from tilert.models.base import TileRTModule
from tilert.models.common import init_func
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


class QwenMoe:
    """Qwen3.6 MoE FFN operations.

    Follows the DSv3.2/GLM5 pattern but uses Qwen3.6 dimensions and keeps
    only the algorithms supported by the Qwen3.6 op wrappers:
      - RMSNormExpertProj: GENERAL
      - ExpertSelectUpGateSiLU: FP8MMA / FP16MMA
      - ExpertDownAllReduce: GENERAL

    This is a plain composition container, not a ``TileRTModule`` sub-class,
    because it only aggregates sub-ops that are themselves registered under
    ``QwenMoeBlock``.
    """

    def __init__(
        self,
        model_args: ModelArgsQwen36,
        device_id: int,
        num_devices: int,
    ):
        self.rmsnorm_expert_proj = RMSNormExpertProj(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
        )
        self.exp_sel_up_gate_silu = ExpertSelectUpGateSiLU(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            algorithm=ExpertSelectUpGateSiLUAlgorithm.FP8MMA,
        )
        self.expert_down_allreduce = ExpertDownAllReduce(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            algorithm=ExpertDownAllReduceAlgorithm.GENERAL,
        )

    def get_weights_list(self) -> list[torch.Tensor]:
        return [
            *self.rmsnorm_expert_proj.get_weights_list(),
            *self.exp_sel_up_gate_silu.get_weights_list(),
            *self.expert_down_allreduce.get_weights_list(),
        ]


class QwenMoeBlock(TileRTModule):
    """MoE block for Qwen3.6.

    Wraps the MoE FFN as a standalone block.  For Qwen3.6, the attention
    path lives in the heterogeneous layer stack managed by ``QwenTransformerStack``;
    this block only represents the FFN half so it can be cached/shared per
    layer (the same pattern DSv3.2 uses with ``cached_ffn_ops``).
    """

    def __init__(
        self,
        model_args: ModelArgsQwen36,
        device_id: int,
        num_devices: int,
        moe: QwenMoe | None = None,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
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

    def init_random_weights(self, device: str | None = None) -> None:
        """Lazy initialize random weights for sanity testing."""
        logger.debug(f"{self.op_name}: init_random_weights on {device}")
        if device is None:
            device = f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"
        if isinstance(device, str) and device.startswith("cuda:"):
            device_id = int(device.split(":")[-1])
        else:
            device_id = 0
        self.moe.rmsnorm_expert_proj.init_random_weights(device=device)
        self.moe.exp_sel_up_gate_silu.init_random_weights(device=device)
        self.moe.expert_down_allreduce.init_random_weights(device_id=device_id)

    def golden_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reference forward: rmsnorm -> gate score -> up/gate/silu -> down."""
        if not self.moe.rmsnorm_expert_proj.is_ref_weights_init:
            self.init_random_weights(device=str(x.device))
        norm_x, scores = self.moe.rmsnorm_expert_proj.golden_forward(x)
        up_gate_out, weights, indices = self.moe.exp_sel_up_gate_silu.golden_forward(
            norm_x, scores
        )
        return self.moe.expert_down_allreduce.golden_forward(up_gate_out, indices, weights)

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        logger.debug(f"{self.op_name}: init_tilert_weights")
        self.moe.rmsnorm_expert_proj.init_tilert_weights(state_dict)
        self.moe.exp_sel_up_gate_silu.init_tilert_weights(state_dict)
        self.moe.expert_down_allreduce.init_tilert_weights(state_dict)
        self.is_tilert_weights_init = True
        logger.debug(f"{self.op_name}: tilert weights initialized")

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        logger.debug(f"{self.op_name}: init_reference_weights")
        self.moe.rmsnorm_expert_proj.init_reference_weights(state_dict)
        self.moe.exp_sel_up_gate_silu.init_reference_weights(state_dict)
        self.moe.expert_down_allreduce.init_reference_weights(state_dict)

    def tilert_forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def get_weights_list(self) -> list[torch.Tensor]:
        return self.moe.get_weights_list()

    def get_ref_weights_alias(self) -> list[str]:
        return self.moe.rmsnorm_expert_proj.get_ref_weights_alias() + \
               self.moe.exp_sel_up_gate_silu.get_ref_weights_alias() + \
               self.moe.expert_down_allreduce.get_ref_weights_alias()

    def get_tilert_weights_alias(self) -> list[str]:
        return self.moe.rmsnorm_expert_proj.get_tilert_weights_alias() + \
               self.moe.exp_sel_up_gate_silu.get_tilert_weights_alias() + \
               self.moe.expert_down_allreduce.get_tilert_weights_alias()

    def device_sharding(self, raw_weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sharded = {}
        sharded.update(self.moe.rmsnorm_expert_proj.device_sharding(raw_weights_map))
        sharded.update(self.moe.exp_sel_up_gate_silu.device_sharding(raw_weights_map))
        sharded.update(self.moe.expert_down_allreduce.device_sharding(raw_weights_map, key_prefix="mlp"))
        return sharded