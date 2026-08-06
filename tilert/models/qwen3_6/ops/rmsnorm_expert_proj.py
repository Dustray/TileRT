"""RMSNormExpertProj operation module."""

from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn

from tilert import logger
from tilert.models.base import TileRTModule
from tilert.models.common import RMSNorm, init_func, linear
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.utils import get_profile_log_tensor

__all__ = [
    "RMSNormExpertProj",
    "RMSNormExpertProjRefWeightsAlias",
    "RMSNormExpertProjTilertWeightsAlias",
]


@dataclass
class RMSNormExpertProjRefWeightsAlias:
    """Reference weights alias for RMSNormExpertProj."""

    post_attention_layernorm_weight = "post_attention_layernorm.weight"
    mlp_gate_weight = "mlp.gate.weight"

    def __call__(self) -> list[str]:
        return [self.post_attention_layernorm_weight, self.mlp_gate_weight]


@dataclass
class RMSNormExpertProjTilertWeightsAlias:
    """TileRT weights alias for RMSNormExpertProj."""

    unproj_o_gamma = "unproj_o_gamma"
    exp_proj_weights = "exp_proj_weights"

    def __call__(self) -> list[str]:
        return [self.unproj_o_gamma, self.exp_proj_weights]


class RMSNormExpertProjAlgorithm(Enum):
    """RMSNormExpertProj algorithm."""

    GENERAL = "general"


class RMSNormExpertProj(TileRTModule):
    """RMS Norm followed by expert projection."""

    _SUPPORTED_ALGORITHMS = {
        "qwen3_6": [RMSNormExpertProjAlgorithm.GENERAL],
        "glm_5": [RMSNormExpertProjAlgorithm.GENERAL],
    }

    def __init__(
        self,
        model_args: ModelArgsQwen36,
        num_devices: int,
        device_id: int = 0,
        ref_weights_alias: RMSNormExpertProjRefWeightsAlias | None = None,
        tilert_weights_alias: RMSNormExpertProjTilertWeightsAlias | None = None,
    ):
        super().__init__(
            type(self).__name__,
            model_args=model_args,
            num_devices=num_devices,
            device_id=device_id,
        )
        self.dim = model_args.dim
        self.eps = model_args.eps

        self.ref_weights_alias = (
            ref_weights_alias
            if ref_weights_alias is not None
            else RMSNormExpertProjRefWeightsAlias()
        )
        self.tilert_weights_alias = (
            tilert_weights_alias
            if tilert_weights_alias is not None
            else RMSNormExpertProjTilertWeightsAlias()
        )

        self.n_activated_experts = self.model_args.n_activated_experts
        self.is_ref_weights_init = False
        self.is_tilert_weights_init = False

        self.ref_rmsnorm: RMSNorm | None = None
        self.ref_gate: RMSNorm | None = None
        self.ref_proj_weight: torch.Tensor | None = None
        self.proj_weight = nn.Parameter(
            init_func(torch.empty(model_args.n_routed_experts, model_args.dim))
        )
        self.n_routed_experts = model_args.n_routed_experts

        self.tilert_proj_weight: torch.Tensor | None = None
        self.tilert_rms_norm_weight: torch.Tensor | None = None

        self.profile_logs = get_profile_log_tensor()

    def get_weights_list(self) -> list[torch.Tensor]:
        return [self.tilert_rms_norm_weight, self.tilert_proj_weight]

    def device_sharding(
        self, rms_norm_weight: torch.Tensor, proj_weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logger.info(f"[device_sharding] RMSNormExpertProj, num_devices: {self.num_devices}")
        
        original_rms_shape = rms_norm_weight.shape
        original_proj_shape = proj_weight.shape
        
        result_rms = rms_norm_weight.float().contiguous()
        result_proj = proj_weight.contiguous()
        
        # Log sharding details
        logger.info(f"[device_sharding] key: post_attention_layernorm.weight, original shape: {original_rms_shape}, sharded shape: {result_rms.shape}, dtype: {result_rms.dtype}")
        logger.info(f"[device_sharding] key: mlp.gate.weight, original shape: {original_proj_shape}, sharded shape: {result_proj.shape}, dtype: {result_proj.dtype}")
        
        return result_rms, result_proj

    def init_reference_weights(
        self, state_dict: dict[str, torch.Tensor], device_id: int | None = None
    ) -> None:
        del device_id
        logger.debug(f"{self.op_name}: init_reference_weights")
        rms_w = state_dict[self.ref_weights_alias.post_attention_layernorm_weight]
        gate_w = state_dict[self.ref_weights_alias.mlp_gate_weight]
        # Move to compute device if still on CPU; RMSNorm expects a float32
        # weight tensor on the target device.
        if rms_w.device.type != "meta":
            rms_w = rms_w.to(dtype=torch.float32, device=f"cuda:{self.device_id}")
        if gate_w.device.type != "meta":
            gate_w = gate_w.to(dtype=torch.float32, device=f"cuda:{self.device_id}")
        self.ref_rmsnorm = RMSNorm(self.dim, self.eps)
        self.ref_rmsnorm.weight.data = rms_w
        self.ref_gate = RMSNorm(gate_w.shape[0]*gate_w.shape[1], self.eps)
        self.ref_gate.weight.data = gate_w
        self.ref_proj_weight = gate_w
        self.is_ref_weights_init = True

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        logger.debug(f"{self.op_name}: init_tilert_weights")
        self.tilert_proj_weight = (
            state_dict[self.tilert_weights_alias.exp_proj_weights].detach().clone()
        )
        self.tilert_rms_norm_weight = (
            state_dict[self.tilert_weights_alias.unproj_o_gamma].detach().clone()
        )
        self.is_tilert_weights_init = True

    def init_random_weights(self, device: str | None = None) -> None:
        if device is None:
            device = f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"
        proj_weight = torch.randn(self.n_routed_experts, self.dim, device=device)
        rms_norm_weight = torch.randn(self.dim, dtype=torch.float32, device=device)
        ref_state_dict = dict(
            zip(
                self.ref_weights_alias(),
                [rms_norm_weight, proj_weight],
            )
        )
        self.init_reference_weights(ref_state_dict)
        assert self.ref_rmsnorm is not None and self.ref_proj_weight is not None
        sharded_weights = self.device_sharding(self.ref_rmsnorm.weight, self.ref_proj_weight)
        self.init_tilert_weights(dict(zip(self.tilert_weights_alias(), sharded_weights)))

    def golden_forward(
        self, x_in: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logger.info(f"[RMSNormExpertProjOp.golden_forward_{self.device_id}] ENTRY: x_in.shape={x_in.shape}, residual={residual is not None}")
        import torch.nn.functional as F
        
        assert self.is_ref_weights_init, "Reference weights must be initialized before forward pass"
        assert self.ref_gate is not None and self.ref_rmsnorm is not None and self.ref_proj_weight is not None
        
        logger.info(f"[RMSNormExpertProjOp.golden_forward_{self.device_id}] Applying RMSNorm")
        B, M, D = x_in.shape  # 解包输入形状
        x_flat = x_in.view(B * M, D)  # 展平为 [total_tokens, D]
        router_logits = F.linear(x_flat.float(), self.ref_proj_weight.float())  # 计算 router logits
        routing_weights, expert_indices = torch.topk(router_logits, self.n_activated_experts, dim=-1)  # top-K 专家和未归一化权重
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(x_in.dtype)  # softmax 归一化后转回输入 dtype
        return x_flat, routing_weights, expert_indices
        # moe_out = torch.zeros_like(h_flat)  # 初始化输出缓冲区
        # norm_x = self.ref_gate(x_flat, residual)
        # logger.info(f"[RMSNormExpertProjOp.golden_forward_{self.device_id}] Computing scores via linear projection")
        # scores = linear(norm_x.view(-1, self.dim).float(), self.ref_proj_weight.float())
        
        # logger.info(f"[RMSNormExpertProjOp.golden_forward_{self.device_id}] EXIT: norm_x.shape={norm_x.shape}, scores.shape={scores.shape}")
        
        # return norm_x, scores

    def tilert_forward(self, x_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logger.info(f"[RMSNormExpertProjOp.tilert_forward_{self.device_id}] ENTRY: x_in.shape={x_in.shape}")
        
        assert self.is_tilert_weights_init, "Tilert weights must be initialized before forward pass"
        assert self.tilert_rms_norm_weight is not None and self.tilert_proj_weight is not None
        x_in = x_in.to(torch.bfloat16)
        hidden_out = torch.zeros_like(x_in)
        scores_out = torch.zeros(
            (x_in.shape[0], x_in.shape[1], self.n_routed_experts), dtype=torch.float32
        )
        logger.info(f"[RMSNormExpertProjOp.tilert_forward_{self.device_id}] Calling CUDA kernel rmsnorm_expert_proj_op")
        torch.ops.tilert.rmsnorm_expert_proj_op(
            x_in,
            self.tilert_rms_norm_weight,
            self.tilert_proj_weight,
            scores_out,
            hidden_out,
            self.model_args.arch_name,
            "bf16",
            self.profile_logs,
        )
        logger.info(f"[RMSNormExpertProjOp.tilert_forward_{self.device_id}] EXIT: hidden_out.shape={hidden_out.shape}, scores_out.shape={scores_out.shape}")
        
        return hidden_out, scores_out

    def __call__(self, x_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tilert_forward(x_in)
