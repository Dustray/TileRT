"""ExpertDownAllreduce operation module."""

from dataclasses import dataclass
from enum import Enum

import torch

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
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 32
        assert mat_in.dtype == torch.float8_e4m3fn
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 4).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 4).transpose(-3, -2)

    @staticmethod
    def _swizzle_qmma_8x32(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 8 and mat_in.shape[-1] == 32
        pre_shape = mat_in.shape[:-2]
        return mat_in.reshape(*pre_shape, 8, 2, 4, 4).transpose(-2, -3).contiguous()

    @staticmethod
    def _swizzle_bf16mma_full_16x32(mat_in: torch.Tensor) -> torch.Tensor:
        """Swizzle a (16, 32) FP8 sub-block for the BF16 MMA kernel."""
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 32
        assert mat_in.dtype == torch.float8_e4m3fn
        pre = mat_in.shape[:-2]
        mat = mat_in.reshape(*pre, 2, 8, 2, 2, 4, 2)
        n = len(pre)
        mat = mat.permute(*range(n), 1 + n, 4 + n, 2 + n, 3 + n, 0 + n, 5 + n)
        return mat.reshape(*pre, 32, 16).contiguous()

    @staticmethod
    def _swizzle_bf16mma_partial_8x32(mat_in: torch.Tensor) -> torch.Tensor:
        """Swizzle a (8, 32) FP8 partial sub-block for the BF16 MMA kernel."""
        assert mat_in.shape[-2] == 8 and mat_in.shape[-1] == 32
        assert mat_in.dtype == torch.float8_e4m3fn
        pre = mat_in.shape[:-2]
        mat = mat_in.reshape(*pre, 8, 2, 2, 4, 2)
        n = len(pre)
        mat = mat.permute(*range(n), 0 + n, 3 + n, 1 + n, 2 + n, 4 + n)
        return mat.reshape(*pre, 32, 8).contiguous()

    def convert_to_general(
        self, weights_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert weights to general (tilert) format.

        EP8: each device keeps the full intermediate dimension for its local
        experts.  The swizzling therefore processes the full ``inter_dim``
        rather than ``inter_dim // num_devices``.
        """
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

    def convert_to_bf16mma(
        self, weights_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack FP8 weights for the BF16 MMA kernel (DSv32 only)."""
        args = self.model_args
        assert args.arch_name == "deepseek_v3_2", "BF16 MMA layout is only valid for DSv32."
        dim = args.dim
        num_sms = 128
        dim_per_sm = dim // num_sms
        expert_dim = args.moe_inter_dim // 8
        k_chunks = expert_dim // 32
        scale_cols = expert_dim // args.block_size
        assert dim_per_sm == 56, "BF16 MMA layout currently assumes dim_per_sm=56 (DSv32)."

        with torch.inference_mode():
            mat_in, scale_in = weights_list
            exp_num = mat_in.shape[0]
            mat_per_cta = mat_in.reshape(exp_num, num_sms, dim_per_sm, expert_dim)

            full_part = mat_per_cta[:, :, :48, :]
            partial_part = mat_per_cta[:, :, 48:, :]

            full_tiles = full_part.reshape(exp_num, num_sms, 3, 16, k_chunks, 32)
            full_tiles = full_tiles.transpose(3, 4)
            full_swizzled = self._swizzle_bf16mma_full_16x32(full_tiles)
            full_swizzled = full_swizzled.reshape(exp_num, num_sms, 3 * k_chunks * 32 * 16)

            partial_tiles = partial_part.reshape(exp_num, num_sms, 1, 8, k_chunks, 32).transpose(
                3, 4
            )
            partial_swizzled = self._swizzle_bf16mma_partial_8x32(partial_tiles)
            partial_swizzled = partial_swizzled.reshape(exp_num, num_sms, k_chunks * 32 * 8)

            mat_swizzled = torch.cat([full_swizzled, partial_swizzled], dim=2)
            mat_swizzled = mat_swizzled.reshape(exp_num, dim, expert_dim)

            mat_scale_tilert = (
                scale_in.reshape(exp_num, dim // args.block_size, 1, scale_cols)
                .repeat(1, 1, 16, 1)
                .reshape(exp_num, num_sms, -1)
            )
            target_cols_per_sm = 1024 * scale_cols // num_sms
            pad_amount = target_cols_per_sm - mat_scale_tilert.shape[-1]
            if pad_amount > 0:
                padding_zeros = torch.zeros(
                    (exp_num, num_sms, pad_amount),
                    dtype=scale_in.dtype,
                    device=scale_in.device,
                )
                mat_scale_tilert = torch.cat([mat_scale_tilert, padding_zeros], dim=2)
            mat_scale_tilert = mat_scale_tilert.reshape(exp_num, 1024, scale_cols)
            mat_scale_tilert = mat_scale_tilert.to(torch.bfloat16)

            return mat_swizzled.contiguous(), mat_scale_tilert.contiguous()


@dataclass
class ExpertDownAllReduceTilertWeightsAlias:
    """TileRT weights alias for ExpertDownAllReduce."""

    exp_down_weights = "exp_down_weights"
    exp_down_scales = "exp_down_scales"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [self.exp_down_weights, self.exp_down_scales]

    def __call__(self) -> list[str]:
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
        prefix = self.key_prefix
        return [
            f"{prefix}.shared_expert.down_proj.weight",
            f"{prefix}.experts.down_proj",
            f"{prefix}.shared_expert.down_proj.weight_scale_inv",
            f"{prefix}.experts.down_proj.weight_scale_inv",
        ]

    def __call__(self) -> list[str]:
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
        self.algorithm = algorithm

        self.ref_down: torch.Tensor | None = None
        self.ref_shared_expert_gate: torch.Tensor | None = None
        self.tilert_weights: torch.Tensor | None = None
        self.tilert_scales: torch.Tensor | None = None
        self.hidden_out: torch.Tensor | None = None
        self.profile_logs: torch.Tensor | None = None
        self.is_init = False

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
        return self.tilert_weights_alias.tilert_tensor_alias

    @property
    def tensor_alias(self) -> list[str]:
        return self._tensor_alias

    def get_ref_weights_alias(self) -> list[str]:
        return list(self.ref_weights_alias())

    def get_weights_list(self) -> list[torch.Tensor]:
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
        if key_prefix is None:
            key_prefix = self.ref_weights_alias.key_prefix
        
        logger.info(f"[device_sharding] key_prefix: {key_prefix}, num_devices: {self.num_devices}")
        
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
        logger.info(f"[device_sharding] key: {key_prefix}.shared_expert.down_proj + {key_prefix}.experts.down_proj, original shapes: {original_down_weights_shape}, {down_weights_list[1].shape}, sharded down_weights shape: {down_weights.shape}, dtype: {down_weights.dtype}")
        logger.info(f"[device_sharding] key: down_scales, original shapes: {original_down_scales_shape}, {down_scales_list[1].shape}, sharded down_scales shape: {down_scales.shape}, dtype: {down_scales.dtype}")
        
        return down_weights.contiguous(), down_scales.contiguous()

    def _dequant_expert_stack(
        self,
        weights: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize a stack of expert down weights to bf16."""
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
        logger.debug(f"{self.op_name}: init_reference_weights on device {device_id}")
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
        return list(self.tilert_weights_alias())

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        logger.debug(f"{self.op_name}: init_tilert_weights on device {self.device_id}")
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
        self.hidden_out = torch.zeros(
            (batch_size, seq_len, self.dim),
            dtype=torch.bfloat16,
            device=f"cuda:{device_id}",
        )
        self.profile_logs = get_profile_log_tensor(device=f"cuda:{device_id}")
        self.is_init = True

    def init_random_weights(self, device_id: int | None = None) -> None:
        if device_id is None:
            device_id = self.device_id
        if device_id is None:
            device_id = 0
        logger.debug(f"{self.op_name}: init_random_weights on cuda:{device_id}")
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
        vec_in: torch.Tensor,
        indices: torch.Tensor,
        scores: torch.Tensor,
        x_in: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logger.info(f"[ExpertDownAllReduceOp.golden_forward_{self.device_id}] ENTRY: vec_in.shape={vec_in.shape}, indices.shape={indices.shape}, scores.shape={scores.shape}")
        
        assert self.ref_down is not None
        assert vec_in.dim() == 4 and vec_in.size(0) == 1
        # ``rmsnorm_expert_proj`` returns scores as a 2-D tensor when the input
        # batch dimension is 1 (it calls ``view(-1, dim)``).  Promote back to
        # ``[1, seq_len, ...]`` so the token-wise indexing below is consistent.
        if indices.ndim == 2:
            indices = indices.unsqueeze(0)
            scores = scores.unsqueeze(0)
            logger.info(f"[ExpertDownAllReduceOp.golden_forward_{self.device_id}] Promoted indices/scores from 2D to 3D")
        # TP8: reference weights contain every expert but only the local
        # inter_dim shard, so global expert indices index directly into ref_down
        # (the shared expert lives at index 0).
        local_indices = indices
        seq_len = vec_in.shape[1]
        logger.info(f"[ExpertDownAllReduceOp.golden_forward_{self.device_id}] Processing {seq_len} tokens, n_activated_experts={self.n_activated_experts}")
        
        hidden_out_list = []
        for s in range(seq_len):
            hidden_out_w2_list = []
            logger.debug(f"[ExpertDownAllReduceOp.golden_forward_{self.device_id}] Token {s}: computing shared expert")
            hidden_out_w2_shared = vec_in[0, s, 0].float() @ self.ref_down[0].float().mT
            # Apply the shared-expert gate in the same place as the HF model:
            # after the shared expert down-projection and before adding the
            # routed expert outputs.
            if x_in is not None:
                shared_gate = torch.sigmoid(
                    x_in[0, s].float() @ self.ref_shared_expert_gate.float().mT
                )
                hidden_out_w2_shared = hidden_out_w2_shared * shared_gate.squeeze(-1)
                logger.debug(f"[ExpertDownAllReduceOp.golden_forward_{self.device_id}] Token {s}: applied shared gate")
            hidden_out_w2_list.append(hidden_out_w2_shared)
            ref_down_sel = self.ref_down[1:][local_indices[0, s]]
            for i in range(self.n_activated_experts):
                hidden_out_w2_sel = vec_in[0, s, i + 1].float() @ ref_down_sel[i].float().mT
                hidden_out_w2_list.append(hidden_out_w2_sel * scores[0, s, i])
            hidden_out_w2 = torch.stack(hidden_out_w2_list, dim=0).to(torch.bfloat16)
            hidden_out_w2 = torch.sum(hidden_out_w2, dim=0)

            hidden_out_list.append(hidden_out_w2)
        hidden_out = torch.stack(hidden_out_list, dim=0)
        result = hidden_out[None, ...]
        logger.info(f"[ExpertDownAllReduceOp.golden_forward_{self.device_id}] EXIT: result.shape={result.shape}")
        
        return result

    def tilert_forward(
        self,
        vec_in: torch.Tensor,
        indices: torch.Tensor,
        scores: torch.Tensor,
        x_in: torch.Tensor,
        flag: int,
    ) -> torch.Tensor:
        logger.info(f"[ExpertDownAllReduceOp.tilert_forward_{self.device_id}] ENTRY: vec_in.shape={vec_in.shape}, indices.shape={indices.shape}, scores.shape={scores.shape}, flag={flag}")
        
        assert self.hidden_out is not None
        logger.info(f"[ExpertDownAllReduceOp.tilert_forward_{self.device_id}] Calling CUDA kernel expert_down_allreduce")
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
        logger.info(f"[ExpertDownAllReduceOp.tilert_forward_{self.device_id}] EXIT: hidden_out.shape={self.hidden_out.shape}")
        
        return self.hidden_out

    def __call__(
        self,
        x_in: torch.Tensor,
        indices: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        return self.golden_forward(x_in, indices, scores)
