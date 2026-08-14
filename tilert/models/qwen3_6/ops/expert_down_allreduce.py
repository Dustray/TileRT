"""ExpertDownAllreduce operation module."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F

from tilert import logger
from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.common import _safe_weight_dequant
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.utils import get_profile_log_tensor

__all__ = [
    "expert_down_allreduce",
    "ExpertDownAllReduceAlgorithm",
    "ExpertDownAllReduce",
    "ExpertDownAllReduceTilertWeightsAlias",
]

VALID_SEQ_LENS = (1, 2, 4)


def expert_down_allreduce(
    vec_in: torch.Tensor,
    mat_in: torch.Tensor,
    mat_scale: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    x_in: torch.Tensor,
    flag: int,
    vec_out: torch.Tensor,
    profile_logs: torch.Tensor,
    model_arch: str,
    compute_kernel_type: str = "bf16",
) -> None:
    """
    Fused expert down + allreduce (unified for DSv32 and GLM5).

    Args:
        vec_in: [1, seq_len, n_experts, 256], bfloat16.
        mat_in: [n_experts, dim, 256], float8_e4m3fn.
        mat_scale: [n_experts, 1024, 2], bfloat16 (DSv32) or float32 (GLM5).
        indices: [1, seq_len, 8], int32.
        scores: [1, seq_len, 8], float32.
        x_in: [1, seq_len, dim], bfloat16.
        flag: User flag.
        vec_out: [1, seq_len, dim], bfloat16 (output).
        profile_logs: 1D tensor for profile logs.
        compute_kernel_type: "bf16".

    """
    logger.info(f'[{__file__.split(chr(47))[-1]}] expert_down_allreduce')
    torch.ops.tilert.expert_down_allreduce_op(
        vec_in,
        mat_in,
        mat_scale,
        indices,
        scores,
        x_in,
        flag,
        vec_out,
        profile_logs,
        model_arch,
        compute_kernel_type,
    )


class ExpertDownAllReduceAlgorithm(Enum):
    """ExpertDownAllReduce algorithm."""

    GENERAL = "general"


class ExpertDownAllReduceWeightsConverter(TilertWeightsConverter):
    """ExpertDownAllReduce weights converter."""

    @staticmethod
    def _swizzle_qmma_16x32(mat_in: torch.Tensor) -> torch.Tensor:

        logger.info(f'[{__file__.split(chr(47))[-1]}] ExpertDownAllReduceWeightsConverter._swizzle_qmma_16x32')
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 32
        assert mat_in.dtype == torch.float8_e4m3fn
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 4).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 4).transpose(-3, -2)

    def convert_to_general(
        self, weights_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert weights to general (tilert) format.

        EP8: each device keeps the full intermediate dimension for its local
        experts.  The swizzling therefore processes the full ``inter_dim``
        rather than ``inter_dim // num_devices``.

        """
        logger.info(f'[{__file__.split(chr(47))[-1]}] ExpertDownAllReduceWeightsConverter.convert_to_general')
        args = self.model_args
        assert args.arch_name in ("qwen3_6", "glm_5")
        arch_name = args.arch_name
        dim = args.dim
        num_sms = 128
        dim_per_sm = dim // num_sms
        dim_scale_dim = dim // args.block_size
        # TP8: each device owns inter_dim // num_devices.
        expert_dim = args.inter_dim // self.num_devices
        k_chunks = expert_dim // 32
        scale_cols = expert_dim // args.block_size
        # TP8 may produce tiny local shards (e.g. inter_dim=512/8=64 < block_size).
        # The swizzler needs 32-wide chunks; fall back to raw weights + scalar
        # scale when this condition is not met so random-init sanity tests can run.
        if expert_dim % 32 != 0:
            return self._convert_to_general_fallback(weights_list)

        with torch.inference_mode():
            mat_in, scale_in = weights_list
            exp_num = mat_in.shape[0]
            mat_in_s = mat_in.reshape(exp_num, num_sms, dim_per_sm, expert_dim)

            if scale_cols == 0:
                scale_cols = 1
                # Scale was generated for full inter_dim and then sharded.
                # Collapse all scale columns per expert to a single scalar so
                # the downstream scale grid is valid.
                scale_in = scale_in.reshape(exp_num, -1).mean(
                    dim=-1, keepdim=True
                )
                if scale_in.dim() < mat_in.dim():
                    scale_in = scale_in.unsqueeze(-1)

            if arch_name == "qwen3_6":
                assert dim_per_sm == 16, f"Qwen3.6 expects dim_per_sm=16, got {dim_per_sm}"
                mat_in_0 = (
                    mat_in_s[:, :, :16].reshape(exp_num, num_sms, 16, k_chunks, 32).transpose(2, 3)
                )
                mat_in_0 = self._swizzle_qmma_16x32(mat_in_0).reshape(exp_num, num_sms, -1)
                mat_in_swizzled = mat_in_0.reshape(exp_num, dim, expert_dim)
            else:
                mat_in_0 = (
                    mat_in_s[:, :, :16].reshape(exp_num, num_sms, 16, k_chunks, 32).transpose(2, 3)
                )
                mat_in_0 = self._swizzle_qmma_16x32(mat_in_0).reshape(exp_num, num_sms, -1)
                mat_in_1 = (
                    mat_in_s[:, :, 16:32].reshape(exp_num, num_sms, 16, k_chunks, 32).transpose(2, 3)
                )
                mat_in_1 = self._swizzle_qmma_16x32(mat_in_1).reshape(exp_num, num_sms, -1)
                mat_in_2 = (
                    mat_in_s[:, :, 32:48].reshape(exp_num, num_sms, 16, k_chunks, 32).transpose(2, 3)
                )
                mat_in_2 = self._swizzle_qmma_16x32(mat_in_2).reshape(exp_num, num_sms, -1)
                mats_to_cat = [mat_in_0, mat_in_1, mat_in_2]
                mat_in_3 = (
                    mat_in_s[:, :, 48:56].reshape(exp_num, num_sms, 8, k_chunks, 32).transpose(2, 3)
                )
                mat_in_3 = self._swizzle_qmma_8x32(mat_in_3).reshape(exp_num, num_sms, -1)
                mats_to_cat.append(mat_in_3)
                mat_in_swizzled = torch.cat(mats_to_cat, dim=2).reshape(exp_num, dim, expert_dim)

            if scale_in.numel() == exp_num:
                # Tiny local shard: one scalar scale per expert.  Replicate
                # it across all dim blocks and scale columns.
                mat_scale_tilert = (
                    scale_in.reshape(exp_num, 1, 1)
                    .expand(exp_num, dim_scale_dim, scale_cols)
                    .unsqueeze(2)
                    .repeat(1, 1, dim_per_sm, 1)
                    .reshape(exp_num, num_sms, -1)
                )
            else:
                mat_scale_tilert = (
                    scale_in.reshape(exp_num, dim_scale_dim, scale_cols)
                    .unsqueeze(2)
                    .repeat(1, 1, dim_per_sm, 1)
                    .reshape(exp_num, num_sms, -1)
                )
            target_cols_per_sm = dim_per_sm * scale_cols
            pad_amount = target_cols_per_sm - mat_scale_tilert.shape[-1]
            if pad_amount > 0:
                padding_zeros = torch.zeros(
                    (exp_num, num_sms, pad_amount),
                    dtype=scale_in.dtype,
                    device=scale_in.device,
                )
                mat_scale_tilert = torch.cat([mat_scale_tilert, padding_zeros], dim=2)
            elif pad_amount < 0:
                mat_scale_tilert = mat_scale_tilert[:, :, :target_cols_per_sm]
            mat_scale_tilert = mat_scale_tilert.reshape(exp_num, dim, scale_cols)
            if mat_scale_tilert.dtype != torch.float32:
                print(
                    "Warning: ExpertDownAllReduceWeightsConverter: "
                    + f"mat_scale_tilert.dtype: {mat_scale_tilert.dtype} "
                    + "is not float32, convert to float32."
                )
                mat_scale_tilert = mat_scale_tilert.to(torch.float32)
            return mat_in_swizzled.contiguous(), mat_scale_tilert.contiguous()

    def _convert_to_general_fallback(
        self, weights_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fallback for tiny TP shards that the swizzler cannot tile.

        Returns the raw FP8 weights and a scalar scale.  The resulting tilert
        tensors are not layout-swizzled, but they keep ``init_tilert_weights``
        and the tilert path importable for random-init smoke tests.

        """
        logger.info(f'[{__file__.split(chr(47))[-1]}] ExpertDownAllReduceWeightsConverter._convert_to_general_fallback')
        mat_in, scale_in = weights_list
        # Preserve a 3-D scale tensor so the rest of the op can keep using
        # scale_in as if it came from ``process_down_weights``.
        exp_num = mat_in.shape[0]
        if scale_in.numel() == 1:
            scale_out = scale_in.to(torch.float32).view(exp_num, 1, 1)
        else:
            scale_out = (
                scale_in.reshape(exp_num, -1)
                .mean(dim=1, keepdim=True)
                .to(torch.float32)
                .view(exp_num, 1, 1)
            )
        return mat_in.contiguous(), scale_out.contiguous()


@dataclass
class ExpertDownAllReduceTilertWeightsAlias:
    """TileRT weights alias for ExpertDownAllReduce."""

    exp_down_weights = "exp_down_weights"
    exp_down_scales = "exp_down_scales"

    @property
    def tilert_tensor_alias(self) -> list[str]:

        logger.info(f'[{__file__.split(chr(47))[-1]}] ExpertDownAllReduceTilertWeightsAlias.tilert_tensor_alias')
        return [self.exp_down_weights, self.exp_down_scales]

    def __call__(self) -> list[str]:

        logger.info(f'[{__file__.split(chr(47))[-1]}] ExpertDownAllReduceTilertWeightsAlias.__call__')
        return self.tilert_tensor_alias


@dataclass
class ExpertDownAllReduceRefWeightsAlias:
    """Reference weights alias for ExpertDownAllReduce.

    The checkpoint stores shared and routed down-projection weights (and their
    optional scales) under dot-weight keys.  We expose the canonical list here
    so that callers such as ``QwenMoeBlock.get_ref_weights_alias`` can discover
    these keys consistently with the rest of the Qwen3.6 ops.
    """

    key_prefix: str = "mlp"

    @property
    def ref_tensor_alias(self) -> list[str]:

        logger.info(f'[{__file__.split(chr(47))[-1]}] ExpertDownAllReduceRefWeightsAlias.ref_tensor_alias')
        prefix = self.key_prefix
        return [
            f"{prefix}.shared_expert.down_proj.weight",
            f"{prefix}.experts.down_proj",
            f"{prefix}.shared_expert.down_proj.weight_scale_inv",
            f"{prefix}.experts.down_proj.weight_scale_inv",
        ]

    def __call__(self) -> list[str]:

        logger.info(f'[{__file__.split(chr(47))[-1]}] ExpertDownAllReduceRefWeightsAlias.__call__')
        return self.ref_tensor_alias


class ExpertDownAllReduce(TileRTModule):
    """ExpertDownAllReduce module."""

    _SUPPORTED_ALGORITHMS = {
        "qwen3_6": [ExpertDownAllReduceAlgorithm.GENERAL],
        "glm_5": [ExpertDownAllReduceAlgorithm.GENERAL],
    }
    # Removed BF16MMA for Qwen3.6; keep GENERAL only.

    def __init__(
        self,
        model_args: ModelArgsQwen36,
        device_id: int,
        num_devices: int,
        algorithm: ExpertDownAllReduceAlgorithm = ExpertDownAllReduceAlgorithm.GENERAL,
    ):

        logger.info(f'[{__file__.split(chr(47))[-1]}] ExpertDownAllReduce.__init__')
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
        )
        self.arch_name = self.model_args.arch_name
        self.dim = self.model_args.dim
        self.n_activated_experts: int = self.model_args.n_activated_experts
        self.n_routed_experts: int = self.model_args.n_routed_experts
        self.n_shared_experts: int = self.model_args.n_shared_experts
        # TP8: this op stores the *local* intermediate dimension.
        self.moe_inter_dim = self.model_args.inter_dim // self.num_devices
        self.block_size = self.model_args.block_size
        self.hidden_size = self.model_args.dim
        self.algorithm = algorithm

        self.ref_down: torch.Tensor | None = None
        self.ref_shared_expert_gate: torch.Tensor | None = None
        self.tilert_weights: torch.Tensor | None = None
        self.tilert_scales: torch.Tensor | None = None
        self.hidden_out: torch.Tensor | None = None
        self.profile_logs: torch.Tensor | None = None
        self.is_init = False
        self.moe_sync_callback: Callable[[torch.Tensor], torch.Tensor] | None = None

        if self.arch_name in ("qwen3_6", "glm_5"):
            self.compute_kernel_type = "bf16"
        else:
            raise ValueError(f"Unsupported architecture: {self.arch_name}")

        self.model_arch = self.arch_name

        self.tilert_weights_alias = ExpertDownAllReduceTilertWeightsAlias()
        self.ref_weights_alias = ExpertDownAllReduceRefWeightsAlias()
        # ``tensor_alias`` is kept for internal tilert init bookkeeping.
        self._tensor_alias = ["exp_down_weights", "exp_down_scales"]

    @property
    def tilert_tensor_alias(self) -> list[str]:

        logger.info(f'[{__file__.split(chr(47))[-1]}] ExpertDownAllReduce.tilert_tensor_alias')
        return self.tilert_weights_alias.tilert_tensor_alias

    @property
    def tensor_alias(self) -> list[str]:

        logger.info(f'[{__file__.split(chr(47))[-1]}] ExpertDownAllReduce.tensor_alias')
        return self._tensor_alias

    def get_ref_weights_alias(self) -> list[str]:

        logger.info(f'[{__file__.split(chr(47))[-1]}] ExpertDownAllReduce.get_ref_weights_alias')
        return list(self.ref_weights_alias())

    def get_weights_list(self) -> list[torch.Tensor | None]:

        logger.info(f'[{__file__.split(chr(47))[-1]}] ExpertDownAllReduce.get_weights_list')
        return [self.tilert_weights, self.tilert_scales]

    @staticmethod
    def process_down_weights(
        key_prefix: str,
        weights_hf: dict[str, torch.Tensor],
        num_devices: int,
        is_stacked_experts: bool = False,
        tp_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract and shard down weights for EP8 or TP8.

        EP8 (``tp_mode=False``): weights are sharded along the expert dimension.
        Each device owns ``n_routed_experts // num_devices`` routed experts plus
        one replicated shared expert, while inter_dim stays full on each device.

        TP8 (``tp_mode=True``): every device keeps all experts, but the input
        intermediate dimension is split across devices.  Each device therefore owns
        ``inter_dim // num_devices`` columns of every expert's down projection.
        The partial outputs from each device are all-reduced by the caller.

        Returns down weights/scales of shape
          EP8: (n_local_experts, num_devices, dim, inter_dim)
          TP8: (n_experts,       num_devices, dim, inter_dim // num_devices)
        where n_local_experts = 1 shared + n_routed_experts // num_devices routed.

        """
        logger.info(f'[{__file__.split(chr(47))[-1]}] ExpertDownAllReduce.process_down_weights')
        if is_stacked_experts:
            down_proj_weight = weights_hf[f"{key_prefix}.down_proj"]
            down_proj_scale = weights_hf.get(
                f"{key_prefix}.down_proj.weight_scale_inv",
                ExpertDownAllReduce._fake_ones_scale(
                    down_proj_weight,
                    is_stacked_experts=True,
                    num_devices=num_devices,
                    tp_mode=tp_mode,
                ),
            )
        else:
            down_proj_weight = weights_hf[f"{key_prefix}.down_proj.weight"]
            down_proj_scale = weights_hf.get(
                f"{key_prefix}.down_proj.weight_scale_inv",
                ExpertDownAllReduce._fake_ones_scale(
                    down_proj_weight,
                    is_stacked_experts=False,
                    num_devices=num_devices,
                    tp_mode=tp_mode,
                ),
            )

        if is_stacked_experts:
            n_experts, dim, moe_inter_dim = down_proj_weight.shape
            dim_scale_dim, in_scale_dim = down_proj_scale.shape[-2:]
        else:
            n_experts = 1
            dim, moe_inter_dim = down_proj_weight.shape
            dim_scale_dim, in_scale_dim = down_proj_scale.shape

        if tp_mode:
            local_inter_dim = moe_inter_dim // num_devices
            local_in_scale_dim = max(local_inter_dim // 128, 1)
        else:
            local_inter_dim = moe_inter_dim
            local_in_scale_dim = in_scale_dim

        def _tp_shard_scale(scale: torch.Tensor, n_experts: int) -> torch.Tensor:
            """Expand/trim full-inter_dim scale columns to per-device shards.

            Input scale layout is (n_experts, dim_scale_dim, in_scale_dim) or
            (dim_scale_dim, in_scale_dim) for shared experts.  The downstream
            ``convert_to_general`` consumes per-device scales as
            ``(n_experts, dim_scale_dim, scale_cols)``, so we return the
            stacked layout ``(n_experts, num_devices, dim_scale_dim,
            local_in_scale_dim)``.
            """
            logger.info(f'[{__file__.split(chr(47))[-1]}] _tp_shard_scale')
            if scale.dim() == 2:
                scale = scale.unsqueeze(0)
            cols_needed = num_devices * local_in_scale_dim
            cols_available = scale.shape[-1]
            if cols_available < cols_needed:
                repeat = (cols_needed + cols_available - 1) // cols_available
                scale = scale.repeat_interleave(repeat, dim=-1)
            scale = scale[:, :, :cols_needed]
            return scale.reshape(
                n_experts, num_devices, local_in_scale_dim, dim_scale_dim
            ).transpose(2, 3)

        if is_stacked_experts:
            if tp_mode:
                down_proj_weight = down_proj_weight.reshape(
                    n_experts, num_devices, dim, local_inter_dim
                )
                down_proj_scale = _tp_shard_scale(down_proj_scale, n_experts)
            else:
                assert n_experts % num_devices == 0, (
                    f"n_routed_experts {n_experts} must be divisible by num_devices {num_devices}"
                )
                n_local_experts = n_experts // num_devices
                down_proj_weight = down_proj_weight.reshape(
                    num_devices, n_local_experts, dim, local_inter_dim
                ).transpose(0, 1)
                down_proj_scale = down_proj_scale.reshape(
                    num_devices, n_local_experts, dim_scale_dim, local_in_scale_dim
                ).transpose(0, 1)
        else:
            if tp_mode:
                down_proj_weight = down_proj_weight.reshape(
                    1, num_devices, dim, local_inter_dim
                )
                down_proj_scale = _tp_shard_scale(down_proj_scale, 1)
            else:
                # Shared expert: full inter_dim, replicate across devices.
                down_proj_weight = down_proj_weight.reshape(1, dim, local_inter_dim)[
                    None, ...
                ].repeat(num_devices, 1, 1, 1).transpose(0, 1)
                down_proj_scale = down_proj_scale.reshape(1, dim_scale_dim, local_in_scale_dim)[
                    None, ...
                ].repeat(num_devices, 1, 1, 1).transpose(0, 1)

        return down_proj_weight, down_proj_scale

    @staticmethod
    def _fake_ones_scale(
        weight: torch.Tensor,
        is_stacked_experts: bool = False,
        num_devices: int = 1,
        tp_mode: bool = False,
    ) -> torch.Tensor:

        logger.info(f'[{__file__.split(chr(47))[-1]}] ExpertDownAllReduce._fake_ones_scale')
        if is_stacked_experts:
            _, dim, inter_dim = weight.shape
        else:
            *_, dim, inter_dim = weight.shape
        block_size = 128
        # Always generate the full unsharded scale grid; the TP split is done
        # later in ``process_down_weights`` so fake scales match real checkpoints.
        local_inter_dim = inter_dim
        shape = (dim // block_size, max(local_inter_dim // block_size, 1))
        if is_stacked_experts:
            shape = (weight.shape[0],) + shape
        return torch.ones(
            shape,
            dtype=torch.float32,
            device=weight.device,
        )

    def device_sharding(
        self,
        weights_dict: dict[str, torch.Tensor],
        key_prefix: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        logger.info(f'[{__file__.split(chr(47))[-1]}] ExpertDownAllReduce.device_sharding')
        if key_prefix is None:
            key_prefix = self.ref_weights_alias.key_prefix

        logger.info(f"[dev={self.device_id}] [device_sharding] key_prefix: {key_prefix}，num_devices: {self.num_devices}")

        assert self.n_shared_experts == 1, "Only one shared expert is supported"
        down_weights_list = []
        down_scales_list = []
        exp_prefix = f"{key_prefix}.shared_expert"
        down_weights, down_scales = self.process_down_weights(
            exp_prefix, weights_dict, self.num_devices, is_stacked_experts=False, tp_mode=True
        )
        down_weights_list.append(down_weights)
        down_scales_list.append(down_scales)
        exp_prefix = f"{key_prefix}.experts"
        down_weights, down_scales = self.process_down_weights(
            exp_prefix, weights_dict, self.num_devices, is_stacked_experts=True, tp_mode=True
        )
        down_weights_list.append(down_weights)
        down_scales_list.append(down_scales)

        original_down_weights_shape = down_weights_list[0].shape
        original_down_scales_shape = down_scales_list[0].shape

        # Concatenate along the expert dimension (first dim).  Under TP8 both
        # shared and routed outputs have rank 4: (n_experts, num_devices,
        # dim, inter_dim // num_devices).
        down_weights = torch.cat(down_weights_list, dim=0)
        down_scales = torch.cat(down_scales_list, dim=0)

        # Log sharding details
        logger.info(f"[dev={self.device_id}] [device_sharding] key: {key_prefix}.shared_expert.down_proj + {key_prefix}.experts.down_proj，原始形状: {original_down_weights_shape}，{down_weights_list[1].shape}，分片 down_weights 形状: {down_weights.shape}，数据类型: {down_weights.dtype}")
        logger.info(f"[dev={self.device_id}] [device_sharding] key: down_scales，原始形状: {original_down_scales_shape}，{down_scales_list[1].shape}，分片 down_scales 形状: {down_scales.shape}，数据类型: {down_scales.dtype}")

        return down_weights.contiguous(), down_scales.contiguous()

    def _dequant_expert_stack(
        self,
        weights: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:

        """Dequantize a stack of expert down weights to bf16."""
        logger.info(f'[{__file__.split(chr(47))[-1]}] ExpertDownAllReduce._dequant_expert_stack')
        return torch.stack(
            [
                _safe_weight_dequant(weights[i], scales[i]).to(torch.bfloat16)
                for i in range(weights.shape[0])
            ],
            dim=0,
        )

    def init_reference_weights(
        self,
        state_dict: dict[str, torch.Tensor],
        key_prefix: str | None = None,
        device_id: int = 0,
    ) -> None:
        logger.debug(f"[dev={device_id}] {self.op_name}: 在设备上初始化参考权重 {device_id}")
        if key_prefix is None:
            key_prefix = self.ref_weights_alias.key_prefix

        # TP8: keep the full expert count but only the local inter_dim shard
        # on each device.  Global expert indices select directly into the
        # local full-expert table.
        local_inter_dim = self.moe_inter_dim
        down_proj = state_dict[f"{key_prefix}.experts.down_proj"]
        shared_down = state_dict[f"{key_prefix}.shared_expert.down_proj.weight"]

        # ``load_hf_source_weights`` already TP-shards the reference MoE
        # weights per device.  Detect that and skip the unsharded
        # ``device_sharding`` path to avoid double-sharding.
        if down_proj.size(-1) == local_inter_dim:
            down_weights = torch.cat([shared_down.unsqueeze(0), down_proj], dim=0)
            scale_dtype = (
                torch.float32 if self.arch_name in ("glm_5", "qwen3_6") else torch.bfloat16
            )
            down_scales = torch.ones(
                down_weights.shape[0],
                down_weights.shape[1] // self.block_size,
                max(down_weights.shape[2] // self.block_size, 1),
                dtype=scale_dtype,
                device=down_weights.device,
            )
        else:
            sharded_list = self.device_sharding(state_dict, key_prefix)
            down_weights = sharded_list[0][:, device_id]
            down_scales = sharded_list[1][:, device_id]
        self.ref_down = self._dequant_expert_stack(down_weights, down_scales)

        # Load the shared-expert gate so the golden path can apply it after
        # the shared expert down-projection, matching HF Qwen3_5MoeSparseMoeBlock.
        shared_expert_gate = state_dict.get(f"{key_prefix}.shared_expert_gate.weight")
        if shared_expert_gate is None:
            shared_expert_gate = state_dict.get("shared_expert_gate")
        if shared_expert_gate is not None:
            if shared_expert_gate.dim() == 1:
                shared_expert_gate = shared_expert_gate.unsqueeze(0)
            self.ref_shared_expert_gate = shared_expert_gate.to(torch.bfloat16)

        self.is_ref_weights_init = True

    def get_tilert_weights_alias(self) -> list[str]:

        """Return the alias list keyed into ``state_dict`` for this op."""
        logger.info(f'[{__file__.split(chr(47))[-1]}] ExpertDownAllReduce.get_tilert_weights_alias')
        return list(self.tilert_weights_alias())

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        logger.debug(f"[dev={self.device_id}] {self.op_name}: 在设备上初始化TileRT权重 {self.device_id}")
        assert self.algorithm is not None, "Algorithm is not set"
        weights_list = [state_dict[alias] for alias in self.tensor_alias]

        # ``init_tilert_weights`` receives already-per-device tensors from
        # ``init_random_weights`` (or an external loader).  No further
        # slicing should be performed here, because the per-device scale tensor
        # may legitimately have a middle dimension equal to ``num_devices``
        # (e.g. scale_cols=2 under TP2) and an extra slice would corrupt it.

        # Real Qwen3.6 converted checkpoints store down weights in bf16.
        # The GENERAL swizzler expects float8_e4m3fn; cast if needed.
        if weights_list[0].dtype != torch.float8_e4m3fn:
            weights_list[0] = weights_list[0].to(torch.float8_e4m3fn)
        self.tilert_weights, self.tilert_scales = ExpertDownAllReduceWeightsConverter(
            self.model_args, self.num_devices
        ).dispatch(self.algorithm, weights_list)

    def init_tilert_vars(self, batch_size: int, seq_len: int, device_id: int = 0) -> None:

        logger.info(f'[{__file__.split(chr(47))[-1]}] ExpertDownAllReduce.init_tilert_vars')
        self.hidden_out = torch.zeros(
            (batch_size, seq_len, self.dim),
            dtype=torch.bfloat16,
            device=f"cuda:{device_id}",
        )
        self.profile_logs = get_profile_log_tensor(
            device=torch.device("cuda", device_id)
        )
        self.is_init = True

    def init_random_weights(self, device_id: int | None = None) -> None:

        logger.info(f'[{__file__.split(chr(47))[-1]}] ExpertDownAllReduce.init_random_weights')
        if device_id is None:
            device_id = self.device_id
        if device_id is None:
            device_id = 0
        logger.debug(f"[dev={device_id}] {self.op_name}: 初始化随机权重，设备为 cuda:{device_id}")
        dev = f"cuda:{device_id}"
        # TP8: generate *full* down weights; ``process_down_weights`` will
        # perform the TP8 split along the intermediate dimension.
        full_inter_dim = self.model_args.inter_dim
        shared_down = (
            torch.randn(
                self.dim, full_inter_dim, dtype=torch.bfloat16, device=dev
            )
            / (full_inter_dim ** 0.5)
        ).to(torch.float8_e4m3fn)
        # Reference/golden path needs the full set of routed experts.
        routed_down = (
            torch.randn(
                self.n_routed_experts,
                self.dim,
                full_inter_dim,
                dtype=torch.bfloat16,
                device=dev,
            )
            / (full_inter_dim ** 0.5)
        ).to(torch.float8_e4m3fn)
        dim_scale_dim = self.dim // self.block_size
        moe_inter_dim_scale_dim = max(full_inter_dim // self.block_size, 1)
        scale_dtype = torch.float32
        shared_scale = torch.randn(
            dim_scale_dim, moe_inter_dim_scale_dim, dtype=scale_dtype, device=dev
        )
        routed_scale = torch.randn(
            self.n_routed_experts,
            dim_scale_dim,
            moe_inter_dim_scale_dim,
            dtype=scale_dtype,
            device=dev,
        )
        shared_expert_gate = (
            torch.randn(1, self.dim, dtype=torch.bfloat16, device=dev)
            / (self.dim ** 0.5)
        )
        state_dict = dict(
            zip(
                self.ref_weights_alias(),
                [shared_down, routed_down, shared_scale, routed_scale],
            )
        )
        state_dict["mlp.shared_expert_gate.weight"] = shared_expert_gate
        self.init_reference_weights(state_dict, "mlp", device_id)
        # Keep reference weights in bf16 to avoid a 4x memory spike from the
        # fp32 dequantization fallback used during random-init sanity tests.
        assert self.ref_down is not None
        self.ref_down = self.ref_down.to(torch.bfloat16)
        sharded_list = self.device_sharding(state_dict, "mlp")
        # ``sharded_list`` has shape (n_experts, num_devices, ...); under TP8
        # each device gets a different inter_dim shard.
        sharded_state_dict = {
            alias: sharded_list[i][:, device_id] for i, alias in enumerate(self.tensor_alias)
        }
        self.init_tilert_weights(sharded_state_dict)

    def golden_forward(
        self,
        h_flat: torch.Tensor,
        expert_indices: torch.Tensor,
        moe_intermediate: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """TP8 local down-projection + weighted sum + all-reduce.

        Args:
            h_flat: [num_tokens, dim], RMSNorm-ed token hidden states.  Kept for
                the shared-expert gate computation.
            expert_indices: unused, kept for interface compatibility.
            moe_intermediate: output from ``ExpertSelectUpGateSiLU.golden_forward``,
                a list of ``(expert_id, token_idx, ffn_e, weight_e)`` tuples.

        Returns:
            [num_tokens, dim] fully all-reduced MoE output.

        """
        logger.info(f'[{__file__.split(chr(47))[-1]}] ExpertDownAllReduce.golden_forward')
        assert self.ref_down is not None

        num_tokens = h_flat.size(0)
        hidden_size = self.hidden_size
        device = h_flat.device
        dtype = h_flat.dtype

        # Partial output accumulator for this TP rank.
        moe_out = torch.zeros(
            (num_tokens, hidden_size), device=device, dtype=dtype
        )

        # Separate legacy per-expert tuples from the new batched routed tuple.
        routed_batched = None
        shared_tuple = None
        for item in moe_intermediate:
            expert_id = item[0]
            if expert_id == 0:
                shared_tuple = item
            elif expert_id == -1:
                routed_batched = item
            else:
                # Legacy per-expert path (kept for safety/fallback).
                token_idx, ffn_e, weight_e = item[1], item[2], item[3]
                down_weight = self.ref_down[expert_id].to(dtype=ffn_e.dtype)
                down_e = F.linear(ffn_e, down_weight)
                down_e = down_e * weight_e.to(dtype)
                moe_out.index_add_(0, token_idx, down_e)

        # Batched routed down-projection: one big bmm instead of many tiny linears.
        if routed_batched is not None:
            token_idx, ffn_e, weight_e, flat_expert_ids = (
                routed_batched[1], routed_batched[2], routed_batched[3], routed_batched[4]
            )
            # ref_down: [1 + n_routed, dim, local_inter_dim]; select down weight per assignment.
            down_weights_selected = self.ref_down[1 + flat_expert_ids].to(dtype)  # [N, dim, local_inter_dim]
            down_proj = torch.bmm(ffn_e.unsqueeze(1), down_weights_selected.transpose(1, 2)).squeeze(1)
            down_proj = down_proj * weight_e.to(dtype)
            moe_out.index_add_(0, token_idx, down_proj)

        # Shared expert contribution
        if shared_tuple is not None:
            expert_id, token_idx, ffn_e, weight_e = shared_tuple
            down_weight = self.ref_down[expert_id].to(dtype=ffn_e.dtype)
            down_e = F.linear(ffn_e, down_weight)
            if self.ref_shared_expert_gate is not None:
                shared_gate_weight = self.ref_shared_expert_gate.to(dtype=h_flat.dtype)
                shared_gate = torch.sigmoid(
                    h_flat[token_idx].float() @ shared_gate_weight.float().mT
                ).to(dtype)
                down_e = down_e * shared_gate
            down_e = down_e * weight_e.to(dtype)
            moe_out.index_add_(0, token_idx, down_e)

        # TP row parallel all-reduce.
        if dist.is_initialized():
            dist.all_reduce(moe_out, op=dist.ReduceOp.SUM)
        elif self.num_devices > 1 and self.moe_sync_callback is not None:
            seq_len = getattr(self, "_last_seq_len", num_tokens)
            if seq_len <= 0:
                seq_len = num_tokens
            partial_3d = moe_out.reshape(1, seq_len, hidden_size)
            full_3d = self.moe_sync_callback(partial_3d.contiguous())
            moe_out = full_3d.reshape(num_tokens, hidden_size)

        return moe_out

    def tilert_forward(
        self,
        vec_in: torch.Tensor,
        indices: torch.Tensor,
        scores: torch.Tensor,
        x_in: torch.Tensor,
        flag: int,
    ) -> torch.Tensor:
        logger.info(f"[dev={self.device_id}] [ExpertDownAllReduceOp.tilert_forward_{self.device_id}] 入口: vec_in.shape={vec_in.shape}，indices.shape={indices.shape}，scores.shape={scores.shape}，flag={flag}")

        assert self.hidden_out is not None
        assert self.tilert_weights is not None
        assert self.tilert_scales is not None
        assert self.profile_logs is not None
        logger.info(f"[dev={self.device_id}] [ExpertDownAllReduceOp.tilert_forward_{self.device_id}] 调用CUDA内核 expert_down_allreduce")
        expert_down_allreduce(
            vec_in,
            self.tilert_weights,
            self.tilert_scales,
            indices,
            scores,
            x_in,
            flag,
            self.hidden_out,
            self.profile_logs,
            self.model_arch,
            self.compute_kernel_type,
        )
        logger.info(f"[dev={self.device_id}] [ExpertDownAllReduceOp.tilert_forward_{self.device_id}] 出口: hidden_out.shape={self.hidden_out.shape}")

        return self.hidden_out

    def __call__(
        self,
        h_flat: torch.Tensor,
        expert_indices: torch.Tensor,
        moe_intermediate: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:

        logger.info(f'[{__file__.split(chr(47))[-1]}] ExpertDownAllReduce.__call__')
        return self.golden_forward(h_flat, expert_indices, moe_intermediate)