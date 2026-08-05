"""ExpertSelectUpGateSiLU operation module."""

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F
import torch.distributed as dist

from tilert import logger
from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.common import _safe_weight_dequant
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.utils import get_profile_log_tensor
__all__ = [
    "ExpertSelectUpGateSiLUAlgorithm",
    "ExpertSelectUpGateSiLU",
    "ExpertSelectUpGateSiLURefWeightsAlias",
    "ExpertSelectUpGateSiLUTilertWeightsAlias",
    "expert_select_up_gate_silu",
]


def expert_select_up_gate_silu(
    hidden_in: torch.Tensor,
    scores_in: torch.Tensor,
    bias_in: torch.Tensor,
    experts_weights_in: torch.Tensor,
    hidden_out: torch.Tensor,
    expert_probs_out: torch.Tensor,
    expert_indices_out: torch.Tensor,
    profile_logs: torch.Tensor,
    algorithm: str = "fp8mma",
    *,
    model_arch: str,
) -> None:
    """Expert SelectUpGateSiLU operation."""
    torch.ops.tilert.expert_select_up_gate_silu_op(
        hidden_in,
        scores_in,
        bias_in,
        experts_weights_in,
        hidden_out,
        expert_probs_out,
        expert_indices_out,
        profile_logs,
        model_arch,
        algorithm,
    )


@dataclass
class ExpertSelectUpGateSiLURefWeightsAlias:
    """Reference weights alias for ExpertSelectUpGateSiLU (Qwen3.6).

    Qwen3.6 stores the routed experts as stacked tensors:
      - ``mlp.experts.gate_up_proj`` (n_routed_experts, 2*inter_dim, dim)
      - ``mlp.experts.down_proj``    (n_routed_experts, dim, inter_dim)
    The shared expert uses conventional per-tensor gate/up/down weights:
      - ``mlp.shared_expert.gate_proj.weight`` (shared_inter_dim, dim)
      - ``mlp.shared_expert.up_proj.weight``   (shared_inter_dim, dim)
      - ``mlp.shared_expert_gate.weight``      (1, dim)
    The checkpoint is bf16 only, so fake all-ones ``weight_scale_inv`` tensors are
    synthesised by the converter to satisfy the existing FP8 converters.
    """

    key_prefix: str = "mlp"
    n_routed_experts: int = 256

    @property
    def ref_tensor_alias(self) -> list[str]:
        return (
            [f"{self.key_prefix}.shared_expert.gate_proj.weight"]
            + [f"{self.key_prefix}.shared_expert.up_proj.weight"]
            + [f"{self.key_prefix}.experts.gate_up_proj"]
            + [f"{self.key_prefix}.shared_expert.gate_proj.weight_scale_inv"]
            + [f"{self.key_prefix}.shared_expert.up_proj.weight_scale_inv"]
            + [f"{self.key_prefix}.experts.gate_up_proj.weight_scale_inv"]
            + [f"{self.key_prefix}.gate.e_score_correction_bias"]
            + [f"{self.key_prefix}.shared_expert_gate.weight"]
        )

    def __call__(self) -> list[str]:
        return self.ref_tensor_alias


@dataclass
class ExpertSelectUpGateSiLUTilertWeightsAlias:
    """TileRT weights alias for ExpertSelectUpGateSiLU."""

    exp_bias = "exp_bias"
    exp_gate_weights = "exp_gate_weights"
    exp_gate_scales = "exp_gate_scales"
    exp_up_weights = "exp_up_weights"
    exp_up_scales = "exp_up_scales"

    @property
    def tilert_tensor_alias(self) -> list[str]:
        return [
            self.exp_bias,
            self.exp_gate_weights,
            self.exp_gate_scales,
            self.exp_up_weights,
            self.exp_up_scales,
        ]

    def __call__(self) -> list[str]:
        return self.tilert_tensor_alias


class ExpertSelectUpGateSiLUAlgorithm(Enum):
    """ExpertSelectUpGateSiLU algorithm"""

    FP8MMA = "fp8mma"
    FP16MMA = "fp16mma"
    BF16MMA = "bf16mma"


class ExpertSelectUpGateSiLUWeightsConverter(TilertWeightsConverter):
    """ExpertSelectUpGateSiLU weights converter"""

    @staticmethod
    def _swizzle_qmma_16x32(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 32
        assert mat_in.dtype == torch.float8_e4m3fn
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 4).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 4).transpose(-3, -2)

    @staticmethod
    def _swizzle_mma_16x32(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 32
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 4).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 4).transpose(-3, -2)

    @staticmethod
    def _swizzle_mma_16x16(mat_in: torch.Tensor) -> torch.Tensor:
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 16
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 2).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 2).transpose(-3, -2)

    @staticmethod
    def tilert_to_tilert_144sm(
        mat_in: torch.Tensor, mat_scale_in: torch.Tensor, mma_type: str | None = None
    ) -> torch.Tensor:
        """
        Convert tilert weights and scales to tilert_144sm input format.

        Args:
            mat_in: tilert weights
            mat_scale_in: tilert scales
            mma_type: MMA type, None,"16x32" or "16x16"
        Returns:
            tilert_144sm weights and scales
        """
        exp_num = mat_in.shape[0]
        assert mat_in.shape == (exp_num, 512, 7168)
        assert mat_scale_in.shape == (exp_num, 4, 64)
        weights_trt = mat_in.reshape(exp_num, 128, 4, 7168)
        weights_w1 = weights_trt[:, :, :2].reshape(exp_num, 256, 7168)
        weights_w3 = weights_trt[:, :, 2:].reshape(exp_num, 256, 7168)
        weights_w1 = weights_w1.reshape(exp_num, 16, 16, 7, 1024).transpose(2, 3)
        weights_w3 = weights_w3.reshape(exp_num, 16, 16, 7, 1024).transpose(2, 3)
        if mma_type == "16x32":
            weights_w1 = weights_w1.reshape(exp_num, 16, 7, 16, 32, 32).transpose(3, 4)
            weights_w1 = ExpertSelectUpGateSiLUWeightsConverter._swizzle_mma_16x32(weights_w1)
            weights_w1 = weights_w1.reshape(exp_num, 16, 7, 16, 1024)
            weights_w3 = weights_w3.reshape(exp_num, 16, 7, 16, 32, 32).transpose(3, 4)
            weights_w3 = ExpertSelectUpGateSiLUWeightsConverter._swizzle_mma_16x32(weights_w3)
            weights_w3 = weights_w3.reshape(exp_num, 16, 7, 16, 1024)
        elif mma_type == "16x16":
            weights_w1 = weights_w1.reshape(exp_num, 16, 7, 16, 64, 16).transpose(3, 4)
            weights_w1 = ExpertSelectUpGateSiLUWeightsConverter._swizzle_mma_16x16(weights_w1)
            weights_w1 = weights_w1.reshape(exp_num, 16, 7, 16, 1024)
            weights_w3 = weights_w3.reshape(exp_num, 16, 7, 16, 64, 16).transpose(3, 4)
            weights_w3 = ExpertSelectUpGateSiLUWeightsConverter._swizzle_mma_16x16(weights_w3)
            weights_w3 = weights_w3.reshape(exp_num, 16, 7, 16, 1024)

        weights = torch.cat([weights_w1, weights_w3], dim=3)
        assert weights.shape == (exp_num, 16, 7, 32, 1024)
        weights = weights.reshape(exp_num, 16, 7, 32 * 1024)

        scales_unswizzled = torch.zeros(exp_num, 4, 56)
        for i in range(64):
            if ((i % 8) * 8 + i // 8) < 56:
                scales_unswizzled[..., ((i % 8) * 8 + i // 8)] = mat_scale_in[..., i]
        scales_unswizzled = scales_unswizzled.reshape(exp_num, 2, 2, 56)

        scales_w1 = scales_unswizzled[:, :, :1].repeat(1, 1, 8, 1).reshape(exp_num, 16, 1, 7, 8)
        scales_w1 = scales_w1.transpose(2, 3)
        scales_w3 = scales_unswizzled[:, :, 1:].repeat(1, 1, 8, 1).reshape(exp_num, 16, 1, 7, 8)
        scales_w3 = scales_w3.transpose(2, 3)
        scales = torch.cat([scales_w1, scales_w3], dim=3)
        assert scales.shape == (exp_num, 16, 7, 2, 8)
        scales = (
            scales.reshape(exp_num, 16, 7, 2 * 8).to(torch.bfloat16).view(dtype=torch.float8_e4m3fn)
        )
        weights_and_scales = torch.zeros(
            exp_num, 16, 7, 32 * 1024 + 128, dtype=torch.float8_e4m3fn, device=mat_in.device
        )
        weights_and_scales[:, :, :, : 32 * 1024].copy_(weights)
        weights_and_scales[:, :, :, 32 * 1024 : 32 * 1024 + 32].copy_(scales)
        return weights_and_scales

    @staticmethod
    def tilert_to_tilert_144sm_mma(
        mat_in: torch.Tensor, mat_scale_in: torch.Tensor, mma_type: str = "16x32"
    ) -> torch.Tensor:
        """
        Convert tilert weights and scales to tilert_144sm_mma input format.

        Args:
            mat_in: tilert weights
            mat_scale_in: tilert scales
        Returns:
            tilert_144sm weights and scales
        """
        return ExpertSelectUpGateSiLUWeightsConverter.tilert_to_tilert_144sm(
            mat_in, mat_scale_in, mma_type
        )

    def convert_to_mma(
        self, weights_list: list[torch.Tensor], algorithm: str = "fp8mma"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert the weights to mma format."""
        args = self.model_args
        dim = args.dim
        pages = dim // 1024
        dim_scale_dim = dim // args.block_size
        with torch.inference_mode():
            bias_or_gamma, weights_w1, scales_w1, weights_w3, scales_w3 = weights_list
            exp_num = weights_w1.shape[0]
            moe_rows = weights_w1.shape[1]
            n_row_groups = moe_rows // 16
            # Use the actual number of scale rows instead of
            # ``moe_rows // block_size``.  When the per-device intermediate dim
            # is smaller than ``block_size`` (e.g. Qwen3.6 with 8 devices:
            # 512//8 = 64 < 128) the latter becomes zero, and the scale rows
            # are already provided by ``process_gate_up_weights`` based on the
            # original unsharded intermediate dimension.
            scale_m_dim = scales_w1.shape[1]
            weights_w1 = weights_w1.reshape(exp_num, n_row_groups, 16, pages, 1024).transpose(2, 3)
            weights_w3 = weights_w3.reshape(exp_num, n_row_groups, 16, pages, 1024).transpose(2, 3)
            if algorithm == "fp8mma":
                weights_w1 = weights_w1.reshape(exp_num, n_row_groups, pages, 16, 32, 32).transpose(
                    3, 4
                )
                weights_w1 = self._swizzle_qmma_16x32(weights_w1)
                weights_w1 = weights_w1.reshape(exp_num, n_row_groups, pages, 16, 1024)
                weights_w3 = weights_w3.reshape(exp_num, n_row_groups, pages, 16, 32, 32).transpose(
                    3, 4
                )
                weights_w3 = self._swizzle_qmma_16x32(weights_w3)
                weights_w3 = weights_w3.reshape(exp_num, n_row_groups, pages, 16, 1024)
            elif algorithm == "fp16mma":
                weights_w1 = weights_w1.reshape(exp_num, n_row_groups, pages, 16, 64, 16).transpose(
                    3, 4
                )
                weights_w1 = self._swizzle_mma_16x16(weights_w1)
                weights_w1 = weights_w1.reshape(exp_num, n_row_groups, pages, 16, 1024)
                weights_w3 = weights_w3.reshape(exp_num, n_row_groups, pages, 16, 64, 16).transpose(
                    3, 4
                )
                weights_w3 = self._swizzle_mma_16x16(weights_w3)
                weights_w3 = weights_w3.reshape(exp_num, n_row_groups, pages, 16, 1024)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            weights: torch.Tensor = torch.cat([weights_w1, weights_w3], dim=3)
            assert weights.shape == (exp_num, n_row_groups, pages, 32, 1024)
            weights = weights.reshape(exp_num, n_row_groups, pages, 32 * 1024)

            scales_per_page = 1024 // args.block_size
            repeat_factor = n_row_groups // scale_m_dim
            scales_w1 = (
                scales_w1.reshape(exp_num, scale_m_dim, 1, dim_scale_dim)
                .repeat(1, 1, repeat_factor, 1)
                .reshape(exp_num, n_row_groups, 1, pages, scales_per_page)
            )
            scales_w1 = scales_w1.transpose(2, 3)
            scales_w3 = (
                scales_w3.reshape(exp_num, scale_m_dim, 1, dim_scale_dim)
                .repeat(1, 1, repeat_factor, 1)
                .reshape(exp_num, n_row_groups, 1, pages, scales_per_page)
            )
            scales_w3 = scales_w3.transpose(2, 3)
            scales = torch.cat([scales_w1, scales_w3], dim=3)
            assert scales.shape == (exp_num, n_row_groups, pages, 2, scales_per_page)

            if self.model_args.arch_name in ("glm_5", "qwen3_6"):
                if scales.dtype != torch.float32:
                    print(
                        "Warning: ExpertSelectUpGateSiLUWeightsConverter: "
                        + f"scales.dtype: {scales.dtype} "
                        + "is not float32, convert to float32."
                    )
                scales = scales.to(torch.float32)
            else:
                scales = scales.to(torch.bfloat16)

            scales = scales.reshape(exp_num, n_row_groups, pages, 2 * scales_per_page).view(
                dtype=torch.float8_e4m3fn
            )

            weights_and_scales = torch.zeros(
                exp_num,
                n_row_groups,
                pages,
                32 * 1024 + 128,
                dtype=torch.float8_e4m3fn,
                device=weights_w1.device,
            )
            weights_and_scales[:, :, :, : 32 * 1024].copy_(weights)
            weights_and_scales[:, :, :, 32 * 1024 : 32 * 1024 + scales.shape[-1]].copy_(scales)

            return bias_or_gamma.float(), weights_and_scales.contiguous()

    def convert_to_fp8mma(
        self, weights_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the weights to fp8mma format.

        Args:
            weights: List of weights.

        Returns:
            Tuple of weights.
        """
        return self.convert_to_mma(weights_list, "fp8mma")

    def convert_to_fp16mma(
        self, weights_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the weights to fp16mma format.

        Args:
            weights: List of weights.

        Returns:
            Tuple of weights.
        """
        return self.convert_to_mma(weights_list, "fp16mma")

    def convert_to_bf16mma(
        self, weights_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert the weights to bf16mma format."""
        return self.convert_to_mma(weights_list, "fp16mma")


class ExpertSelectUpGateSiLU(TileRTModule):
    """ExpertSelectUpGateSiLU module"""

    _SUPPORTED_ALGORITHMS = {
        "qwen3_6": [
            ExpertSelectUpGateSiLUAlgorithm.FP8MMA,
            ExpertSelectUpGateSiLUAlgorithm.FP16MMA,
        ],
        "glm_5": [
            ExpertSelectUpGateSiLUAlgorithm.FP8MMA,
            ExpertSelectUpGateSiLUAlgorithm.FP16MMA,
        ],
    }

    def __init__(
        self,
        model_args: ModelArgsQwen36,
        num_devices: int,
        device_id: int = 0,
        ref_weights_alias: ExpertSelectUpGateSiLURefWeightsAlias | None = None,
        tilert_weights_alias: ExpertSelectUpGateSiLUTilertWeightsAlias | None = None,
        algorithm: ExpertSelectUpGateSiLUAlgorithm = ExpertSelectUpGateSiLUAlgorithm.FP8MMA,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            num_devices=num_devices,
            device_id=device_id,
        )

        self.arch_name = self.model_args.arch_name
        self.dim = self.model_args.dim

        self.n_activated_experts = self.model_args.n_activated_experts
        self.n_routed_experts = self.model_args.n_routed_experts
        self.n_shared_experts = self.model_args.n_shared_experts
        self.moe_inter_dim = self.model_args.inter_dim
        # Qwen3.6 model_args does not define n_expert_groups / n_limited_groups.
        self.n_expert_groups = getattr(self.model_args, "n_expert_groups", 1)
        self.n_limited_groups = getattr(self.model_args, "n_limited_groups", 1)
        self.route_scale = self.model_args.route_scale
        self.block_size = self.model_args.block_size
        self.algorithm = algorithm
        self.gate_up_proj_weight : torch.Tensor | None = None

        self.tilert_weights_alias = (
            tilert_weights_alias
            if tilert_weights_alias is not None
            else ExpertSelectUpGateSiLUTilertWeightsAlias()
        )
        if not isinstance(self.tilert_weights_alias, ExpertSelectUpGateSiLUTilertWeightsAlias):
            self.tilert_weights_alias = ExpertSelectUpGateSiLUTilertWeightsAlias()
        self.ref_weights_alias = (
            ref_weights_alias
            if ref_weights_alias is not None
            else ExpertSelectUpGateSiLURefWeightsAlias(
                key_prefix="mlp", n_routed_experts=self.n_routed_experts
            )
        )
        if not isinstance(self.ref_weights_alias, ExpertSelectUpGateSiLURefWeightsAlias):
            self.ref_weights_alias = ExpertSelectUpGateSiLURefWeightsAlias(
                key_prefix="mlp", n_routed_experts=self.n_routed_experts
            )

        self.ref_bias: torch.Tensor | None = None
        self.ref_gate: torch.Tensor | None = None
        self.ref_up: torch.Tensor | None = None
        self.ref_shared_expert_gate: torch.Tensor | None = None

        self.tilert_bias: torch.Tensor | None = None
        self.tilert_weights: torch.Tensor | None = None
        self.tilert_scales = (
            torch.zeros(1, dtype=torch.bfloat16, device=torch.device("cuda"))
            if torch.cuda.is_available()
            else None
        )

        self.hidden_out: torch.Tensor | None = None
        self.expert_probs: torch.Tensor | None = None
        self.expert_indices: torch.Tensor | None = None

        self.profile_logs: torch.Tensor | None = None
        self.is_init = False

        self._tensor_alias = self.tilert_weights_alias()
        self._tilert_tensor_alias = [
            self.tilert_weights_alias.exp_bias,
            "exp_upgate_weights",
            "exp_upgate_scales",
        ]

    @property
    def tensor_alias(self) -> list[str]:
        return self._tensor_alias

    @property
    def tilert_tensor_alias(self) -> list[str]:
        """Output weight names for get_weights_list (backward compat)."""
        return self._tilert_tensor_alias

    def get_weights_list(self) -> list[torch.Tensor]:
        """
        Get the weights list.

        Returns:
            List of weights.
        """
        return [self.tilert_bias, self.tilert_weights, self.tilert_scales]

    @staticmethod
    def process_gate_up_weights(
        key_prefix: str,
        weights_hf: dict[str, torch.Tensor],
        num_devices: int,
        inter_dim: int,
        is_stacked_experts: bool = False,
        tp_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract and shard gate/up weights for EP8 or TP8.

        EP8 (``tp_mode=False``): attention weights are replicated; only MoE
        routed experts are sharded along the expert dimension.  Each device
        owns ``n_routed_experts // num_devices`` routed experts plus one
        replicated shared expert.

        TP8 (``tp_mode=True``): all experts keep their full expert count on
        every device, but the intermediate dimension is split across devices.
        Each device owns ``inter_dim // num_devices`` columns of every expert's
        gate/up projection.  This matches the Qwen3.6 MoE TP plan where only the
        FFN intermediate dimension is tensor-parallel sharded and an all-reduce is
        required after the down projection.

        For Qwen3.6 routed experts the weight is a stacked ``gate_up_proj`` tensor
        of shape ``(n_experts, 2*inter_dim, dim)``.  It is split into gate and up
        along the output dimension, then either sharded along the expert dimension
        (EP8) or along ``inter_dim`` (TP8).  Shared experts use conventional
        ``gate_proj`` / ``up_proj`` tensors of shape ``(inter_dim, dim)``.

        Returns gate/up weights/scales of shape
          EP8: (n_local_experts, num_devices, inter_dim, dim)
          TP8: (n_experts,      num_devices, inter_dim // num_devices, dim)
        where n_local_experts = 1 shared + n_routed_experts // num_devices routed.
        The ``num_devices`` dim is kept for compatibility with the existing
        ``device_sharding`` stacking convention.
        """
        if is_stacked_experts:
            gate_up_proj = weights_hf[f"{key_prefix}.gate_up_proj"]
            gate_proj_weight = gate_up_proj[:, :inter_dim, :]
            up_proj_weight = gate_up_proj[:, inter_dim:, :]
            gate_up_scale = weights_hf.get(
                f"{key_prefix}.gate_up_proj.weight_scale_inv",
                ExpertSelectUpGateSiLU._fake_ones_scale(
                    gate_up_proj,
                    is_stacked_experts=True,
                    num_devices=num_devices,
                    tp_mode=tp_mode,
                ),
            )
            # The scale tensor returned by ``_fake_ones_scale`` already
            # follows the local TP-sharded layout (one row per local inter_dim
            # block).  Real checkpoints use a fused scale over the full
            # unsharded ``2 * inter_dim`` rows; detect this by checking whether
            # the row count matches the full fused layout.
            full_fused_scale_rows = max(2 * inter_dim // 128, 1)
            if tp_mode and gate_up_scale.shape[1] == full_fused_scale_rows:
                half_scale_rows = gate_up_scale.shape[1] // 2
            else:
                # Already sharded: keep all rows for both gate and up.
                half_scale_rows = gate_up_scale.shape[1]
            # Guard against the degenerate split when ``half_scale_rows`` is 0
            # (can happen for tiny local shards where ``_fake_ones_scale``
            # collapses to a single row).
            if half_scale_rows == 0:
                half_scale_rows = gate_up_scale.shape[1]
            # For fake unit scales without a real fused scale, gate and up share
            # the same scale row (the single all-ones row).  Copy it so both
            # projections receive a valid scale tensor.
            if gate_up_scale.shape[1] == 1:
                gate_proj_scale = gate_up_scale
                up_proj_scale = gate_up_scale
            else:
                gate_proj_scale = gate_up_scale[:, :half_scale_rows, :]
                up_proj_scale = gate_up_scale[:, half_scale_rows:, :]
        else:
            gate_proj_weight_key = f"{key_prefix}.gate_proj.weight"
            gate_proj_scale_key = f"{key_prefix}.gate_proj.weight_scale_inv"
            up_proj_weight_key = f"{key_prefix}.up_proj.weight"
            up_proj_scale_key = f"{key_prefix}.up_proj.weight_scale_inv"

            gate_proj_weight = weights_hf[gate_proj_weight_key]
            up_proj_weight = weights_hf[up_proj_weight_key]
            gate_proj_scale = weights_hf.get(
                gate_proj_scale_key,
                ExpertSelectUpGateSiLU._fake_ones_scale(
                    gate_proj_weight,
                    is_stacked_experts=False,
                    num_devices=num_devices,
                    tp_mode=tp_mode,
                ),
            )
            up_proj_scale = weights_hf.get(
                up_proj_scale_key,
                ExpertSelectUpGateSiLU._fake_ones_scale(
                    up_proj_weight,
                    is_stacked_experts=False,
                    num_devices=num_devices,
                    tp_mode=tp_mode,
                ),
            )

        dim = gate_proj_weight.shape[-1]
        scale_dim = gate_proj_scale.shape[-1]
        # ``in_scale_dim`` describes the *original* (unsharded) number of scale
        # rows.  After ``process_gate_up_weights`` returns, scale tensors are
        # already sharded along the intermediate dimension, so do not divide
        # by ``num_devices`` again.
        in_scale_dim = gate_proj_scale.shape[-2]

        if is_stacked_experts:
            n_experts = gate_proj_weight.shape[0]
            if tp_mode:
                local_inter_dim = inter_dim // num_devices
                # The routed scale has already been split to the gate or up
                # half, so its rows describe the unsharded ``inter_dim``.
                # After sharding, each device owns ``local_inter_dim`` rows.
                if local_inter_dim >= 128:
                    local_in_scale_dim = local_inter_dim // 128
                else:
                    local_in_scale_dim = 1
                # Under TP each scale row covers ``block_size`` rows of the
                # *unsharded* weight.  When ``local_inter_dim < block_size`` a
                # single scale row spans multiple consecutive devices; expand
                # to ``n_experts * num_devices * local_in_scale_dim`` rows.
                rows_needed = n_experts * num_devices * local_in_scale_dim
                rows_available = gate_proj_scale.shape[0] * gate_proj_scale.shape[-2]
                if rows_available < rows_needed:
                    repeats = (rows_needed + rows_available - 1) // rows_available
                    gate_proj_scale = gate_proj_scale.repeat_interleave(repeats, dim=-2)
                    up_proj_scale = up_proj_scale.repeat_interleave(repeats, dim=-2)
                # Reshape as (n_experts, num_devices * local_in_scale_dim, scale_dim)
                # then trim to exactly that many rows per expert.
                gate_proj_scale = gate_proj_scale.view(n_experts, -1, scale_dim)
                up_proj_scale = up_proj_scale.view(n_experts, -1, scale_dim)
                gate_proj_scale = gate_proj_scale[:, : num_devices * local_in_scale_dim, :]
                up_proj_scale = up_proj_scale[:, : num_devices * local_in_scale_dim, :]
                gate_proj_weight = gate_proj_weight.reshape(
                    n_experts, num_devices, local_inter_dim, dim
                )
                gate_proj_scale = gate_proj_scale.reshape(
                    n_experts, num_devices, local_in_scale_dim, scale_dim
                )
                up_proj_weight = up_proj_weight.reshape(
                    n_experts, num_devices, local_inter_dim, dim
                )
                up_proj_scale = up_proj_scale.reshape(
                    n_experts, num_devices, local_in_scale_dim, scale_dim
                )
            else:
                n_local_experts = n_experts // num_devices
                gate_proj_weight = gate_proj_weight.reshape(
                    num_devices, n_local_experts, inter_dim, dim
                ).transpose(0, 1)
                gate_proj_scale = gate_proj_scale.reshape(
                    num_devices, n_local_experts, in_scale_dim, scale_dim
                ).transpose(0, 1)
                up_proj_weight = up_proj_weight.reshape(
                    num_devices, n_local_experts, inter_dim, dim
                ).transpose(0, 1)
                up_proj_scale = up_proj_scale.reshape(
                    num_devices, n_local_experts, in_scale_dim, scale_dim
                ).transpose(0, 1)
        else:
            if tp_mode:
                local_inter_dim = inter_dim // num_devices
                # Under TP each scale row covers ``block_size`` rows of the
                # *unsharded* weight.  When ``local_inter_dim < block_size`` a
                # single scale row spans multiple consecutive devices, so we
                # replicate the unsharded rows to cover ``num_devices``.
                if local_inter_dim >= 128:
                    local_in_scale_dim = local_inter_dim // 128
                else:
                    local_in_scale_dim = 1
                rows_needed = num_devices * local_in_scale_dim
                rows_available = gate_proj_scale.shape[-2]
                if rows_available < rows_needed:
                    repeats = (rows_needed + rows_available - 1) // rows_available
                    gate_proj_scale = gate_proj_scale.repeat_interleave(repeats, dim=-2)
                    up_proj_scale = up_proj_scale.repeat_interleave(repeats, dim=-2)
                gate_proj_scale = gate_proj_scale[:rows_needed]
                up_proj_scale = up_proj_scale[:rows_needed]
                gate_proj_weight = gate_proj_weight.reshape(
                    1, num_devices, local_inter_dim, dim
                )
                gate_proj_scale = gate_proj_scale.reshape(
                    1, num_devices, local_in_scale_dim, scale_dim
                )
                up_proj_weight = up_proj_weight.reshape(
                    1, num_devices, local_inter_dim, dim
                )
                up_proj_scale = up_proj_scale.reshape(
                    1, num_devices, local_in_scale_dim, scale_dim
                )
            else:
                # Shared expert: keep full inter_dim and replicate across devices.
                gate_proj_weight = gate_proj_weight.reshape(1, inter_dim, dim)[None, ...].repeat(
                    num_devices, 1, 1, 1
                ).transpose(0, 1)
                gate_proj_scale = gate_proj_scale.reshape(1, in_scale_dim, scale_dim)[
                    None, ...
                ].repeat(num_devices, 1, 1, 1).transpose(0, 1)
                up_proj_weight = up_proj_weight.reshape(1, inter_dim, dim)[None, ...].repeat(
                    num_devices, 1, 1, 1
                ).transpose(0, 1)
                up_proj_scale = up_proj_scale.reshape(1, in_scale_dim, scale_dim)[
                    None, ...
                ].repeat(num_devices, 1, 1, 1).transpose(0, 1)

        return gate_proj_weight, gate_proj_scale, up_proj_weight, up_proj_scale

    @staticmethod
    def _fake_ones_scale(
        weight: torch.Tensor,
        is_stacked_experts: bool = True,
        num_devices: int = 1,
        tp_mode: bool = False,
    ) -> torch.Tensor:
        """Return a float32 all-ones scale tensor matching the expected layout.

        Under TP8 the scale grid follows the local (sharded) intermediate
        dimension.  If ``inter_dim // num_devices`` is smaller than
        ``block_size``, we keep one scale column per row block.
        """
        block_size = 128
        if is_stacked_experts:
            n_experts, in_dim, dim = weight.shape
            # Always generate the full unsharded scale rows; the TP split is
            # performed later in ``process_gate_up_weights`` so that fake
            # scales match the layout of real checkpoint scales.
            local_in_dim = in_dim
            in_scale_dim = max(local_in_dim // block_size, 1)
            scale_dim = dim // block_size
            shape = (n_experts, in_scale_dim, scale_dim)
        else:
            *_, in_dim, dim = weight.shape
            local_in_dim = in_dim
            in_scale_dim = max(local_in_dim // block_size, 1)
            scale_dim = dim // block_size
            shape = (in_scale_dim, scale_dim)
        return torch.ones(shape, dtype=torch.float32, device=weight.device)

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Device sharding: ref state dict -> tilert sharded tensors (num_devices, ...).

        Args:
            weights_map: State dict keyed by ref_weights_alias().

        Returns:
            Dict keyed by tilert_weights_alias() with (num_devices, ...) tensors.
        """
        ref_alias = self.ref_weights_alias
        key_prefix = ref_alias.key_prefix
        
        # Log input weights
        logger.info(f"[device_sharding] key_prefix: {key_prefix}, input weights keys: {list(weights_map.keys())}")

        bias_key = f"{key_prefix}.gate.e_score_correction_bias"
        bias = weights_map.get(
            bias_key,
            torch.zeros(self.n_routed_experts, dtype=torch.float32),
        )
        bias = bias[None, :].repeat(self.num_devices, 1)

        gate_weights_list = []
        gate_scales_list = []
        up_weights_list = []
        up_scales_list = []
        assert self.n_shared_experts == 1, "Only one shared expert is supported"

        # Shared expert uses conventional gate/up tensors (no expert dim).
        exp_prefix = f"{key_prefix}.shared_expert"
        shared_gate, shared_gate_s, shared_up, shared_up_s = self.process_gate_up_weights(
            exp_prefix,
            weights_map,
            self.num_devices,
            inter_dim=self.model_args.inter_dim,
            is_stacked_experts=False,
            tp_mode=True,
        )

        # Routed experts use Qwen3.6 stacked ``gate_up_proj`` (expert dim first).
        exp_prefix = f"{key_prefix}.experts"
        routed_gate, routed_gate_s, routed_up, routed_up_s = self.process_gate_up_weights(
            exp_prefix,
            weights_map,
            self.num_devices,
            inter_dim=self.model_args.inter_dim,
            is_stacked_experts=True,
            tp_mode=True,
        )

        # The TileRT layout expects a single tensor for both routed and shared
        # experts.  Under TP8 both have rank 4: (n_experts, num_devices,
        # inter_dim // num_devices, dim).  Concatenate along the expert dimension.
        gate_weights = torch.cat([shared_gate, routed_gate], dim=0)
        gate_scales = torch.cat([shared_gate_s, routed_gate_s], dim=0)
        up_weights = torch.cat([shared_up, routed_up], dim=0)
        up_scales = torch.cat([shared_up_s, routed_up_s], dim=0)

        # Shared-expert gate is a (1, dim) matrix replicated on all devices.
        shared_expert_gate = weights_map.get(
            f"{key_prefix}.shared_expert_gate.weight",
            torch.zeros(1, self.dim, dtype=torch.bfloat16),
        )
        if shared_expert_gate.dim() == 1:
            shared_expert_gate = shared_expert_gate.unsqueeze(0)
        shared_expert_gate = shared_expert_gate[None, :, :].repeat(self.num_devices, 1, 1)

        tilert_alias = self.tilert_weights_alias
        
        # Log sharding details for each key
        logger.info(f"[device_sharding] key: {tilert_alias.exp_bias}, shape: {bias.shape}, dtype: {bias.dtype}")
        logger.info(f"[device_sharding] key: {tilert_alias.exp_gate_weights}, shape: {gate_weights.shape}, dtype: {gate_weights.dtype}")
        logger.info(f"[device_sharding] key: {tilert_alias.exp_gate_scales}, shape: {gate_scales.shape}, dtype: {gate_scales.dtype}")
        logger.info(f"[device_sharding] key: {tilert_alias.exp_up_weights}, shape: {up_weights.shape}, dtype: {up_weights.dtype}")
        logger.info(f"[device_sharding] key: {tilert_alias.exp_up_scales}, shape: {up_scales.shape}, dtype: {up_scales.dtype}")
        logger.info(f"[device_sharding] key: shared_expert_gate, shape: {shared_expert_gate.shape}, dtype: {shared_expert_gate.dtype}")
        
        return {
            tilert_alias.exp_bias: bias,
            tilert_alias.exp_gate_weights: gate_weights,
            tilert_alias.exp_gate_scales: gate_scales,
            tilert_alias.exp_up_weights: up_weights,
            tilert_alias.exp_up_scales: up_scales,
            "shared_expert_gate": shared_expert_gate,
        }

    def _dequant_expert_stack(
        self,
        weights: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize a stack of expert weights to bf16.

        ``weights`` has shape ``(n_experts, inter_dim, dim)`` or
        ``(n_experts, in_scale_dim, scale_dim)`` and ``scales`` has the matching
        scale shape.  The result is stacked along the first dimension.
        """
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
        device_id: int | None = None,
    ) -> None:
        """
        Initialize the reference weights.

        Args:
            state_dict: State dict keyed by ref_weights_alias().
            device_id: Device ID; defaults to self.device_id.
        """
        did = self.device_id if device_id is None else device_id
        logger.debug(f"{self.op_name}: init_reference_weights on device {did}")
        ref_alias = self.ref_weights_alias
        key_prefix = ref_alias.key_prefix

        # bias = state_dict.get(
        #     f"{key_prefix}.gate.e_score_correction_bias",
        #     torch.zeros(self.n_routed_experts, dtype=torch.float32),
        # )
        # self.ref_bias = bias.to(torch.float32).to(f"cuda:{did}")

        # TP8: keep the full expert count but only the local inter_dim shard on
        # each device.  Global expert indices select directly into the local
        # full-expert table, so no modulo remapping is needed.
        local_inter_dim = self.moe_inter_dim // self.num_devices
        gate_up_proj = state_dict[f"{key_prefix}.experts.gate_up_proj"] # 已经是切分完成的
        self.gate_up_proj_weight = gate_up_proj
        shared_gate = state_dict[f"{key_prefix}.shared_expert.gate_proj.weight"]
        shared_up = state_dict[f"{key_prefix}.shared_expert.up_proj.weight"]

        # ``load_hf_source_weights`` already TP-shards the reference MoE
        # weights per device.  Detect that and skip the unsharded
        # ``device_sharding`` path to avoid double-sharding.
        if gate_up_proj.size(1) == 2 * local_inter_dim:
            half = gate_up_proj.size(1) // 2
            routed_gate = gate_up_proj[:, :half, :]
            routed_up = gate_up_proj[:, half:, :]
            gate_weights = torch.cat([shared_gate.unsqueeze(0), routed_gate], dim=0)
            up_weights = torch.cat([shared_up.unsqueeze(0), routed_up], dim=0)
            scale_dtype = (
                torch.float32 if self.arch_name in ("glm_5", "qwen3_6") else torch.bfloat16
            )
            gate_scales = torch.ones(
                gate_weights.shape[0],
                max(gate_weights.shape[1] // self.block_size, 1),
                gate_weights.shape[2] // self.block_size,
                dtype=scale_dtype,
                device=gate_weights.device,
            )
            up_scales = gate_scales.clone()
        else:
            sharded = self.device_sharding(state_dict)
            tilert_alias = self.tilert_weights_alias
            gate_weights = sharded[tilert_alias.exp_gate_weights][:, did]
            gate_scales = sharded[tilert_alias.exp_gate_scales][:, did]
            up_weights = sharded[tilert_alias.exp_up_weights][:, did]
            up_scales = sharded[tilert_alias.exp_up_scales][:, did]

        self.ref_gate = self._dequant_expert_stack(gate_weights, gate_scales)
        self.ref_up = self._dequant_expert_stack(up_weights, up_scales)

        shared_expert_gate = state_dict.get(f"{key_prefix}.shared_expert_gate.weight")
        if shared_expert_gate is not None:
            if shared_expert_gate.dim() == 1:
                shared_expert_gate = shared_expert_gate.unsqueeze(0)
            self.ref_shared_expert_gate = shared_expert_gate.to(torch.bfloat16).to(f"cuda:{did}")

        self.is_ref_weights_init = True

    def get_tilert_weights_alias(self) -> list[str]:
        """Return the alias list keyed into ``state_dict`` for this op."""
        return list(self.tilert_weights_alias())

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Initialize the tilert weights."""
        logger.debug(f"{self.op_name}: init_tilert_weights on device {self.device_id}")
        assert self.algorithm is not None, "Algorithm is not set"
        weights_list = [state_dict[alias] for alias in self.tilert_weights_alias()]

        # Real Qwen3.6 converted checkpoints store gate/up weights in bf16.
        # The FP8MMA swizzler requires float8_e4m3fn, so cast them before conversion.
        if self.algorithm == ExpertSelectUpGateSiLUAlgorithm.FP8MMA:
            for i in (1, 3):  # exp_gate_weights, exp_up_weights
                if weights_list[i].dtype != torch.float8_e4m3fn:
                    weights_list[i] = weights_list[i].to(torch.float8_e4m3fn)
        converter = ExpertSelectUpGateSiLUWeightsConverter(self.model_args, self.num_devices)
        self.tilert_bias, self.tilert_weights = converter.dispatch(self.algorithm, weights_list)

    def init_tilert_vars(self, batch_size: int, seq_len: int, device: str = "cuda") -> None:
        """
        Initialize the tilert variables.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
        """
        # TP8: each device holds inter_dim // num_devices for every expert.
        local_inter_dim = max(self.moe_inter_dim // self.num_devices, 1)
        self.hidden_out = torch.zeros(
            (
                batch_size,
                seq_len,
                self.n_activated_experts + self.n_shared_experts,
                local_inter_dim,
            ),
            dtype=torch.bfloat16,
            device=device,
        )
        self.expert_probs = torch.zeros(
            (batch_size, seq_len, self.n_activated_experts),
            dtype=torch.float32,
            device=device,
        )
        self.expert_indices = torch.zeros(
            (batch_size, seq_len, self.n_activated_experts),
            dtype=torch.int32,
            device=device,
        )

        self.profile_logs = get_profile_log_tensor(device=device)
        self.is_init = True

    def init_random_weights(self, device: str | int | None = None) -> None:
        """
        Initialize the random weights.

        Args:
            device: Device to place weights on. May be a ``torch.device`` string,
                an integer device id, or ``None`` to default to
                ``f"cuda:{self.device_id}"``.
        """
        if device is None:
            device = f"cuda:{self.device_id}"
        elif isinstance(device, int):
            device = f"cuda:{device}"
        logger.debug(f"{self.op_name}: init_random_weights on {device}")

        bias = torch.randn(self.n_routed_experts, dtype=torch.float32, device=device) * 0.01
        # Shared expert first.  Scale by 1/sqrt(fan_in) for stable layer outputs.
        shared_gate = (
            torch.randn(
                self.model_args.inter_dim, self.dim, dtype=torch.bfloat16, device=device
            )
            / (self.dim ** 0.5)
        ).to(torch.float8_e4m3fn)
        shared_up = (
            torch.randn(
                self.model_args.inter_dim, self.dim, dtype=torch.bfloat16, device=device
            )
            / (self.dim ** 0.5)
        ).to(torch.float8_e4m3fn)
        routed_gate_up = (
            torch.randn(
                self.n_routed_experts,
                2 * self.model_args.inter_dim,
                self.dim,
                dtype=torch.bfloat16,
                device=device,
            )
            / (self.dim ** 0.5)
        ).to(torch.float8_e4m3fn)
        # The scale layout must be compatible with ``process_gate_up_weights``.
        # For Qwen3.6 the routed scale is the fused gate_up_proj scale of shape
        # (n_routed_experts, 2 * inter_dim // block_size, dim // block_size), while
        # the shared scale is (inter_dim // block_size, dim // block_size).
        # Use ``model_args.inter_dim`` (not ``self.moe_inter_dim``) because the
        # MoE FFN intermediate size is defined by ``inter_dim``.
        inter_dim = self.model_args.inter_dim
        moe_inter_dim_scale_dim = max(inter_dim // self.block_size, 1)
        # Under TP8 the fused routed scale still describes the full unsharded
        # gate_up_proj; ``process_gate_up_weights`` will split it in half.
        routed_moe_inter_dim_scale_dim = max(2 * inter_dim // self.block_size, 1)
        dim_scale_dim = self.dim // self.block_size
        scale_dtype = torch.float32 if self.arch_name in ("glm_5", "qwen3_6") else torch.bfloat16
        shared_gate_scale = torch.randn(
            moe_inter_dim_scale_dim, dim_scale_dim, dtype=scale_dtype, device=device
        )
        shared_up_scale = torch.randn(
            moe_inter_dim_scale_dim, dim_scale_dim, dtype=scale_dtype, device=device
        )
        routed_scale = torch.randn(
            self.n_routed_experts,
            routed_moe_inter_dim_scale_dim,
            dim_scale_dim,
            dtype=scale_dtype,
            device=device,
        )
        key_prefix = self.ref_weights_alias.key_prefix
        shared_expert_gate = torch.randn(1, self.dim, dtype=torch.bfloat16, device=device) / (
            self.dim ** 0.5
        )
        ref_state_dict = {
            f"{key_prefix}.gate.e_score_correction_bias": bias,
            f"{key_prefix}.shared_expert.gate_proj.weight": shared_gate,
            f"{key_prefix}.shared_expert.up_proj.weight": shared_up,
            f"{key_prefix}.experts.gate_up_proj": routed_gate_up,
            f"{key_prefix}.shared_expert.gate_proj.weight_scale_inv": shared_gate_scale,
            f"{key_prefix}.shared_expert.up_proj.weight_scale_inv": shared_up_scale,
            f"{key_prefix}.experts.gate_up_proj.weight_scale_inv": routed_scale,
            f"{key_prefix}.shared_expert_gate.weight": shared_expert_gate,
        }
        self.init_reference_weights(ref_state_dict)
        sharded = self.device_sharding(ref_state_dict)
        tilert_alias = self.tilert_weights_alias
        # ``device_sharding`` returns TP-stacked tensors.  Build a per-device
        # state dict matching the convention expected by ``init_tilert_weights``.
        per_device_state = {
            tilert_alias.exp_bias: sharded[tilert_alias.exp_bias][self.device_id],
            tilert_alias.exp_gate_weights: sharded[tilert_alias.exp_gate_weights][:, self.device_id],
            tilert_alias.exp_gate_scales: sharded[tilert_alias.exp_gate_scales][:, self.device_id],
            tilert_alias.exp_up_weights: sharded[tilert_alias.exp_up_weights][:, self.device_id],
            tilert_alias.exp_up_scales: sharded[tilert_alias.exp_up_scales][:, self.device_id],
            "shared_expert_gate": sharded["shared_expert_gate"][self.device_id],
        }
        self.init_tilert_weights(per_device_state)

    def _ref_expert_select_glm5(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = scores.sigmoid()
        original_scores = scores
        if self.ref_bias is not None:
            scores = scores + self.ref_bias
        indices = torch.topk(scores, self.n_activated_experts, dim=-1)[1]
        indices = indices.view(*original_scores.shape[:-1], self.n_activated_experts)
        weights = original_scores.gather(-1, indices)
        weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices

    def _ref_expert_select_qwen36(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference routing for Qwen3.6 matching HF Qwen3_5MoeTopKRouter.

        The HF implementation does softmax -> topk -> normalize; there is
        no additional route_scale.  We keep the multiplication for backwards
        compatibility with older checkpoints/conversions, but the default
        `route_scale` for qwen3_6 is now 1.0.
        """
        original_scores = scores
        if self.ref_bias is not None:
            scores = scores + self.ref_bias
        scores = F.softmax(scores, dim=-1)
        indices = torch.topk(scores, self.n_activated_experts, dim=-1)[1]
        indices = indices.view(*original_scores.shape[:-1], self.n_activated_experts)
        weights = scores.gather(-1, indices)
        # HF Qwen3.5/3.6-MoE normalizes the top-k weights to sum to 1.
        weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices

    def golden_forward(
        self,
        h_flat: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ):
        logger.info(f"[ExpertSelUpGateSiluOp.golden_forward_{self.device_id}] ENTRY: x_in.shape={x_in.shape}, scores.shape={scores.shape}")
        moe_out = torch.zeros_like(h_flat)  # 初始化输出缓冲区

        assert self.gate_up_proj_weight is not None
        assert self.ref_gate is not None
        assert self.ref_up is not None

        # 用于收集所有激活专家的中间结果
        moe_intermediate = []

        logger.info(f"[ExpertSelUpGateSiluOp.golden_forward_{self.device_id}] Running expert selection for arch={self.arch_name}")
        fused_experts = True # "mlp.experts.gate_up_proj" in w and "mlp.experts.down_proj" in w 是否存在堆叠的 BF16 gate_up/down 权重
        for e in range(256):
            mask = (expert_indices == e)  # 选中专家 e 的掩码
            if not mask.any():  # 无则跳过
                continue
            token_idx, slot_idx = mask.nonzero(as_tuple=True)  # 获取 token 和 slot 索引
            x_e = h_flat[token_idx]  # 专家 e 的输入
            gate_up = F.linear(x_e, self.gate_up_proj_weight[e])  # gate+up 合并投影，gate_up_proj_weight是TP8切分后的权重
            gate_e, up_e = gate_up.chunk(2, dim=-1)  # 切成 gate 和 up
            ffn_e = F.silu(gate_e) * up_e
            weight_e = routing_weights[ token_idx, slot_idx ].unsqueeze(-1)
            moe_intermediate.append( ( e, token_idx, ffn_e, weight_e ) )
        return moe_intermediate


   
        
        weights= routing_weights
        indices = expert_indices
        logger.info(f"[ExpertSelUpGateSiluOp.golden_forward_{self.device_id}] Expert selection done: indices.shape={indices.shape}, weights.shape={weights.shape}")
        
        # ``rmsnorm_expert_proj`` flattens the batch dimension, so scores can be
        # 2-D here.  Restore the batch dimension for token-wise indexing.
        if indices.ndim == 2:
            indices = indices.unsqueeze(0)
            weights = weights.unsqueeze(0)
            logger.info(f"[ExpertSelUpGateSiluOp.golden_forward_{self.device_id}] Promoted indices/weights from 2D to 3D")
        # TP8: reference weights contain every expert but only the local
        # inter_dim shard, so global expert IDs index directly into ref_gate/ref_up
        # (the shared expert lives at index 0).
        local_indices = indices
        logger.info(f"[ExpertSelUpGateSiluOp.golden_forward_{self.device_id}] Processing {seq_len} tokens, n_activated_experts={self.n_activated_experts}")
        
        hidden_out_list = []
        for s in range(seq_len):
            hidden_out_w1_list = []
            hidden_out_w3_list = []
            logger.debug(f"[ExpertSelUpGateSiluOp.golden_forward_{self.device_id}] Token {s}: computing shared expert gate/up")
            hidden_out_w1_shared = x_in[0, s].float() @ self.ref_gate[0].float().mT
            hidden_out_w3_shared = x_in[0, s].float() @ self.ref_up[0].float().mT
            hidden_out_w1_list.append(hidden_out_w1_shared)
            hidden_out_w3_list.append(hidden_out_w3_shared)
            ref_gate_sel = self.ref_gate[1:][local_indices[0, s]]
            ref_up_sel = self.ref_up[1:][local_indices[0, s]]
            for i in range(self.n_activated_experts):
                hidden_out_w1_sel = x_in[0, s].float() @ ref_gate_sel[i].float().mT
                hidden_out_w3_sel = x_in[0, s].float() @ ref_up_sel[i].float().mT
                hidden_out_w1_list.append(hidden_out_w1_sel)
                hidden_out_w3_list.append(hidden_out_w3_sel)
            hidden_out_w1 = torch.stack(hidden_out_w1_list, dim=0)
            hidden_out_w3 = torch.stack(hidden_out_w3_list, dim=0)
            hidden_out = F.silu(hidden_out_w1.float()) * hidden_out_w3.float()
            hidden_out = hidden_out.to(torch.bfloat16)
            hidden_out_list.append(hidden_out)
        hidden_out = torch.stack(hidden_out_list, dim=0)
        hidden_out = hidden_out[None, ...]
        
        logger.info(f"[ExpertSelUpGateSiluOp.golden_forward_{self.device_id}] EXIT: hidden_out.shape={hidden_out.shape}, weights.shape={weights.shape}, indices.shape={indices.shape}")
        
        return hidden_out, weights, indices

    def tilert_forward(
        self,
        x_in: torch.Tensor,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the kernel."""
        logger.info(f"[ExpertSelUpGateSiluOp.tilert_forward_{self.device_id}] ENTRY: x_in.shape={x_in.shape}, scores.shape={scores.shape}")
        
        assert self.algorithm is not None, "Algorithm is not set"
        logger.info(f"[ExpertSelUpGateSiluOp.tilert_forward_{self.device_id}] Calling CUDA kernel expert_select_up_gate_silu")
        expert_select_up_gate_silu(
            x_in,
            scores,
            self.tilert_bias,
            self.tilert_weights,
            self.hidden_out,
            self.expert_probs,
            self.expert_indices,
            self.profile_logs,
            self.algorithm.value,
            model_arch=self.model_args.arch_name,
        )
        logger.info(f"[ExpertSelUpGateSiluOp.tilert_forward_{self.device_id}] EXIT: hidden_out.shape={self.hidden_out.shape}, expert_probs.shape={self.expert_probs.shape}, expert_indices.shape={self.expert_indices.shape}")
        
        return self.hidden_out, self.expert_probs, self.expert_indices
