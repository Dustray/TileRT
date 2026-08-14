"""UnprojOAllreduce operation module."""

import math
from dataclasses import dataclass
from enum import Enum

import torch

from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.common import _safe_weight_dequant
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.utils import get_profile_log_tensor
from tilert import logger

__all__ = [
    "unproj_o_allreduce",
    "UnProjOAllReduce",
    "UnProjOAllReduceAlgorithm",
    "UnProjOAllReduceRefWeightsAlias",
    "UnProjOAllReduceTilertWeightsAlias",
]


def unproj_o_allreduce(
    vec_in: torch.Tensor,
    mat_in: torch.Tensor,
    mat_scale: torch.Tensor,
    x_in: torch.Tensor,
    flag: int,
    vec_out: torch.Tensor,
    profile_logs: torch.Tensor,
    model_arch: str,
    compute_kernel_type: str = "bf16",
) -> None:
    """
    Fused operation of unprojection and allreduce.

    Args:
        vec_in: Input tensor.
        mat_in: Input tensor.
        mat_scale: Input tensor.
        x_in: Input tensor.
        flag: Input flag.
        vec_out: Output tensor.
        profile_logs: Profile logs tensor.
        model_arch: Model architecture ("qwen3_6" or "glm_5").
        compute_kernel_type: Compute kernel type ("bf16", "fp16mma").

    """
    logger.info(f'[{__file__.split(chr(47))[-1]}] unproj_o_allreduce')
    torch.ops.tilert.unproj_o_allreduce_op(
        vec_in,
        mat_in,
        mat_scale,
        x_in,
        flag,
        vec_out,
        profile_logs,
        model_arch,
        compute_kernel_type,
    )


class UnProjOAllReduceAlgorithm(Enum):
    """UnprojOAllReduce algorithm"""

    FP16MMA = "fp16mma"
    BF16MMA = "bf16mma"


@dataclass
class UnProjOAllReduceRefWeightsAlias:
    """Reference weights alias for UnProjOAllReduce."""

    o_proj_weight = "self_attn.o_proj.weight"
    o_proj_scale_inv = "self_attn.o_proj.weight_scale_inv"

    @property
    def ref_tensor_alias(self) -> list[str]:

        logger.info(f'[{__file__.split(chr(47))[-1]}] UnProjOAllReduceRefWeightsAlias.ref_tensor_alias')
        return [self.o_proj_weight, self.o_proj_scale_inv]

    def __call__(self) -> list[str]:

        logger.info(f'[{__file__.split(chr(47))[-1]}] UnProjOAllReduceRefWeightsAlias.__call__')
        return self.ref_tensor_alias


@dataclass
class UnProjOAllReduceTilertWeightsAlias:
    """TileRT weights alias for UnProjOAllReduce."""

    unproj_weights = "unproj_weights"
    unproj_scales = "unproj_scales"

    @property
    def tilert_tensor_alias(self) -> list[str]:

        logger.info(f'[{__file__.split(chr(47))[-1]}] UnProjOAllReduceTilertWeightsAlias.tilert_tensor_alias')
        return [self.unproj_weights, self.unproj_scales]

    def __call__(self) -> list[str]:

        logger.info(f'[{__file__.split(chr(47))[-1]}] UnProjOAllReduceTilertWeightsAlias.__call__')
        return self.tilert_tensor_alias


class UnProjOAllReduceWeightsConverter(TilertWeightsConverter):
    """UnProjOAllReduce weights converter"""

    @staticmethod
    def _swizzle_mma_16x16(mat_in: torch.Tensor) -> torch.Tensor:

        logger.info(f'[{__file__.split(chr(47))[-1]}] UnProjOAllReduceWeightsConverter._swizzle_mma_16x16')
        assert mat_in.shape[-2] == 16 and mat_in.shape[-1] == 16
        pre_shape = mat_in.shape[:-2]
        mat_in = mat_in.reshape(*pre_shape, 2, 8, 2, 4, 2).transpose(-4, -3).transpose(-5, -4)
        return mat_in.reshape(*pre_shape, 2 * 2, 8 * 4, 2).transpose(-3, -2)

    def convert_to_fp16mma_128cta(
        self,
        weights_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:

        """Convert weights to the FP16 MMA layout for the 128-CTA config."""
        logger.info(f'[{__file__.split(chr(47))[-1]}] UnProjOAllReduceWeightsConverter.convert_to_fp16mma_128cta')
        with torch.inference_mode():
            mat, scales = weights_list
            if scales.dtype != torch.float32:
                scales = scales.to(torch.float32)

            dim = self.model_args.dim
            block_size = self.model_args.block_size
            sms = 128
            vec_dim = mat.shape[-1]
            dim_per_sm = dim // sms
            full_tiles = dim_per_sm // 16
            remainder_rows = dim_per_sm % 16
            stages = vec_dim // 512
            vec_scale_dim = vec_dim // block_size
            scale_per_stage = vec_scale_dim // stages

            dim_scale_dim = dim // block_size
            scales_per_full_tile = 2 if remainder_rows > 0 else 1
            rem_scales = 1 if remainder_rows > 0 else 0
            total_scale_slots = (full_tiles * scales_per_full_tile + rem_scales) * scale_per_stage
            repeat_factor = 8 if remainder_rows == 0 else 16

            sc = scales.reshape(dim_scale_dim, 1, vec_scale_dim)
            sc = sc.repeat(1, repeat_factor, 1)
            scales_per_cta = full_tiles * scales_per_full_tile + rem_scales
            sc = (
                sc.reshape(sms, scales_per_cta, stages, scale_per_stage)
                .transpose(1, 2)
                .reshape(sms, stages, total_scale_slots)
                .view(torch.float8_e4m3fn)
            )
            sc_packed = sc

            mat_per_sm = mat.reshape(sms, dim_per_sm, vec_dim)

            full_rows = full_tiles * 16
            mat_full = (
                mat_per_sm[:, :full_rows, :]
                .reshape(sms, full_tiles, 16, stages, 512)
                .transpose(2, 3)
                .reshape(sms, full_tiles, stages, 16, 32, 16)
                .transpose(3, 4)
                .reshape(sms, full_tiles, stages, 32, 16, 16)
            )
            mat_full = UnProjOAllReduceWeightsConverter._swizzle_mma_16x16(mat_full)
            mat_full = mat_full.transpose(1, 2).reshape(sms, stages, -1)

            if remainder_rows > 0:
                mat_rem_raw = mat_per_sm[:, full_rows:, :]
                mat_rem_padded = torch.zeros(
                    sms, 16, vec_dim, dtype=mat_rem_raw.dtype, device=mat_rem_raw.device
                )
                mat_rem_padded[:, :remainder_rows, :] = mat_rem_raw
                mat_rem = (
                    mat_rem_padded.reshape(sms, 1, 16, stages, 512)
                    .transpose(2, 3)
                    .reshape(sms, 1, stages, 16, 32, 16)
                    .transpose(3, 4)
                    .reshape(sms, 1, stages, 32, 16, 16)
                )
                mat_rem = UnProjOAllReduceWeightsConverter._swizzle_mma_16x16(mat_rem)
                mat_rem = mat_rem.transpose(1, 2).reshape(sms, stages, -1)
                mat_combined = torch.cat([mat_full, mat_rem], dim=-1)
            else:
                mat_combined = mat_full

            scales_padding = torch.zeros(
                sms,
                stages,
                128 - sc_packed.shape[-1],
                dtype=torch.float8_e4m3fn,
                device=mat.device,
            )
            mat_all = torch.cat([mat_combined, sc_packed, scales_padding], dim=-1).contiguous()
            dummy_scales = torch.zeros(1, dtype=torch.float32, device=mat.device)
            return mat_all, dummy_scales

    def convert_to_bf16mma(
        self,
        weights_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:

        """Convert common weights to the BF16 MMA layout."""
        logger.info(f'[{__file__.split(chr(47))[-1]}] UnProjOAllReduceWeightsConverter.convert_to_bf16mma')
        assert (
            self.model_args.arch_name == "qwen3_6"
        ), "BF16 MMA dispatch is wired only for DeepSeek-V3.2 DevGroupB."
        return self.convert_to_fp16mma_128cta(weights_list)

    def convert_to_fp16mma(
        self,
        weights_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:

        """Convert common weights to TileRT FP16 MMA layout."""
        logger.info(f'[{__file__.split(chr(47))[-1]}] UnProjOAllReduceWeightsConverter.convert_to_fp16mma')
        if self.model_args.arch_name == "qwen3_6":
            return self.convert_to_fp16mma_128cta(weights_list)
        assert self.model_args.arch_name == "glm_5", "Only GLM-5 and DSV3.2 support FP16 MMA"

        with torch.inference_mode():
            mat, scales = weights_list
            if scales.dtype != torch.float32:
                print(
                    "Warning: UnProjOAllReduceWeightsConverter: "
                    + f"scales.dtype: {scales.dtype} "
                    + "is not float32, convert to float32."
                )
                scales = scales.to(torch.float32)

            dim = self.model_args.dim
            block_size = self.model_args.block_size
            sms = 128
            vec_dim = mat.shape[-1]
            dim_per_sm = dim // sms
            tiles_per_stage = dim_per_sm // 16
            stages = vec_dim // 512
            dim_scale_dim = dim // block_size
            vec_scale_dim = vec_dim // block_size
            scale_per_stage = vec_scale_dim // stages

            scales = scales.reshape(dim_scale_dim, 1, vec_scale_dim)
            scales = scales.repeat(1, 8, 1)
            scales = (
                scales.reshape(sms, tiles_per_stage, stages, scale_per_stage)
                .transpose(1, 2)
                .reshape(sms, stages, tiles_per_stage * scale_per_stage)
                .view(torch.float8_e4m3fn)
            )

            mat = (
                mat.reshape(sms, dim_per_sm, vec_dim)
                .reshape(sms, tiles_per_stage, 16, stages, 512)
                .transpose(2, 3)
                .reshape(sms, tiles_per_stage, stages, 16, 32, 16)
                .transpose(3, 4)
                .reshape(sms, tiles_per_stage, stages, 32, 16, 16)
            )
            mat = UnProjOAllReduceWeightsConverter._swizzle_mma_16x16(mat)
            mat = mat.transpose(1, 2).reshape(sms, stages, -1)

            scales_padding = torch.zeros(
                sms,
                stages,
                128 - scales.shape[-1],
                dtype=torch.float8_e4m3fn,
                device=mat.device,
            )
            mat_full = torch.cat([mat, scales, scales_padding], dim=-1).contiguous()
            dummy_scales = torch.zeros(1, dtype=torch.float32, device=mat.device)
            return mat_full, dummy_scales


class UnProjOAllReduce(TileRTModule):
    """UnProjOAllReduce module"""

    _SUPPORTED_ALGORITHMS = {
        "qwen3_6": [UnProjOAllReduceAlgorithm.FP16MMA],
        "glm_5": [UnProjOAllReduceAlgorithm.FP16MMA],
    }

    def __init__(
        self,
        model_args: ModelArgsQwen36,
        num_devices: int,
        device_id: int = 0,
        ref_weights_alias: UnProjOAllReduceRefWeightsAlias | None = None,
        tilert_weights_alias: UnProjOAllReduceTilertWeightsAlias | None = None,
        algorithm: UnProjOAllReduceAlgorithm = UnProjOAllReduceAlgorithm.FP16MMA,
    ):

        logger.info(f'[{__file__.split(chr(47))[-1]}] UnProjOAllReduce.__init__')
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            num_devices=num_devices,
            device_id=device_id,
        )

        self.tilert_weights_alias = (
            tilert_weights_alias
            if tilert_weights_alias is not None
            else UnProjOAllReduceTilertWeightsAlias()
        )
        if not isinstance(self.tilert_weights_alias, UnProjOAllReduceTilertWeightsAlias):
            self.tilert_weights_alias = UnProjOAllReduceTilertWeightsAlias()
        self.ref_weights_alias = (
            ref_weights_alias
            if ref_weights_alias is not None
            else UnProjOAllReduceRefWeightsAlias()
        )
        if not isinstance(self.ref_weights_alias, UnProjOAllReduceRefWeightsAlias):
            self.ref_weights_alias = UnProjOAllReduceRefWeightsAlias()

        self.arch_name = self.model_args.arch_name
        self.dim = self.model_args.dim
        self.n_heads = self.model_args.n_heads
        self.head_dim = self.model_args.v_head_dim
        # EP8: no head-dim tensor parallelism; replicate the full o_proj on every
        # device.  num_local_heads is only used for sharding layout below.
        self.num_local_heads = self.n_heads

        self.block_size = self.model_args.block_size
        self.algorithm: UnProjOAllReduceAlgorithm = algorithm

        self.ref_unproj_o: torch.Tensor | None = None

        self.tilert_weights: torch.Tensor | None = None
        self.tilert_scales: torch.Tensor | None = None

        self.hidden_out: torch.Tensor | None = None

        self.profile_logs: torch.Tensor | None = None
        self.is_var_init = False

    def get_weights_list(self) -> list[torch.Tensor]:
        """
        Get the weights list.

        Returns:
            List of weights.

        """
        logger.info(f'[{__file__.split(chr(47))[-1]}] UnProjOAllReduce.get_weights_list')
        return [self.tilert_weights, self.tilert_scales]

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Device sharding.

        Args:
            weights_map: Map from ref weight alias to tensor (full model).

        Returns:
            Map from tilert weight alias to (num_devices, ...) tensors.

        """
        logger.info(f"[device_sharding] UnprojOAllreduce, num_devices: {self.num_devices}")
        
        unproj_o_weight = weights_map[self.ref_weights_alias.o_proj_weight]
        # Real Qwen3.6 checkpoints do not include FP8 scales for o_proj.
        # Synthesize an all-ones scale so the existing dequant/converter paths
        # can be reused without changing the checkpoint format.
        if self.ref_weights_alias.o_proj_scale_inv in weights_map:
            unproj_o_scale = weights_map[self.ref_weights_alias.o_proj_scale_inv]
        else:
            head_scale_dim = self.head_dim // self.block_size
            dim_scale_dim = self.dim // self.block_size
            unproj_o_scale = torch.ones(
                dim_scale_dim,
                self.n_heads * head_scale_dim,
                dtype=torch.float32,
                device=unproj_o_weight.device,
            )

        # EP8: replicate the full o_proj weight/scale on every device.
        result = {
            self.tilert_weights_alias.unproj_weights: torch.stack(
                [unproj_o_weight for _ in range(self.num_devices)], dim=0
            ).contiguous(),
            self.tilert_weights_alias.unproj_scales: torch.stack(
                [unproj_o_scale for _ in range(self.num_devices)], dim=0
            ).contiguous(),
        }
        
        # Log sharding details
        logger.info(f"[device_sharding] key: {self.ref_weights_alias.o_proj_weight}, original shape: {unproj_o_weight.shape}, sharded shape: {result[self.tilert_weights_alias.unproj_weights].shape}, dtype: {result[self.tilert_weights_alias.unproj_weights].dtype}")
        logger.info(f"[device_sharding] key: o_proj_scale_inv, original shape: {unproj_o_scale.shape}, sharded shape: {result[self.tilert_weights_alias.unproj_scales].shape}, dtype: {result[self.tilert_weights_alias.unproj_scales].dtype}")
        
        return result

    def init_reference_weights(
        self,
        state_dict: dict[str, torch.Tensor],
        device_id: int | None = None,
    ) -> None:
        """
        Initialize the reference weights.

        Args:
            state_dict: State dictionary keyed by ref weight alias (full model).
            device_id: Device ID for this shard; defaults to self.device_id.

        """
        logger.info(f'[{__file__.split(chr(47))[-1]}] UnProjOAllReduce.init_reference_weights')
        did = self.device_id if device_id is None else device_id
        logger.debug(f"{self.op_name}: init_reference_weights on device {did}")
        sharded = self.device_sharding(state_dict)
        weights = sharded[self.tilert_weights_alias.unproj_weights][did]
        scales = sharded[self.tilert_weights_alias.unproj_scales][did]
        self.ref_unproj_o = _safe_weight_dequant(weights, scales)

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Initialize the tilert weights.

        Args:
            state_dict: State dictionary keyed by tilert weight alias (per-device).

        """
        logger.debug(f"{self.op_name}: init_tilert_weights on device {self.device_id}")
        assert self.algorithm is not None, "Algorithm is not set"
        tilert_alias = list(self.tilert_weights_alias())
        ref_alias = self.ref_weights_alias

        # Map either the fused tilert alias or the dot-weight checkpoint keys.
        if tilert_alias[0] in state_dict:
            weight = state_dict[tilert_alias[0]]
        elif ref_alias.o_proj_weight in state_dict:
            weight = state_dict[ref_alias.o_proj_weight]
        else:
            # Qwen3.6 converted checkpoints store the attention output projection
            # as ``layer_N_o_proj.weight_dev_M`` rather than an ``unproj`` alias.
            # Search the per-device state dict for the matching o_proj key.
            o_proj_keys = [
                k for k in state_dict if "o_proj.weight" in k or k.endswith("o_proj.weight")
            ]
            if o_proj_keys:
                weight = state_dict[o_proj_keys[0]]
            else:
                raise KeyError(
                    f"Neither {tilert_alias[0]} nor {ref_alias.o_proj_weight} found in state_dict"
                )

        if tilert_alias[1] in state_dict:
            scale = state_dict[tilert_alias[1]]
        elif ref_alias.o_proj_scale_inv in state_dict:
            scale = state_dict[ref_alias.o_proj_scale_inv]
        else:
            # Real Qwen3.6 checkpoints are bf16-only; synthesize an all-ones scale.
            head_scale_dim = self.head_dim // self.block_size
            dim_scale_dim = self.dim // self.block_size
            scale = torch.ones(
                dim_scale_dim,
                self.n_heads * head_scale_dim,
                dtype=torch.float32,
                device=weight.device,
            )

        weights_list = [weight, scale]
        # The FP16 MMA converter expects float8_e4m3fn weights; cast bf16
        # checkpoints if needed.
        if weights_list[0].dtype != torch.float8_e4m3fn:
            weights_list[0] = weights_list[0].to(torch.float8_e4m3fn)
        self.tilert_weights, self.tilert_scales = UnProjOAllReduceWeightsConverter(
            self.model_args, self.num_devices
        ).dispatch(
            self.algorithm,
            weights_list,
        )

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        """
        Initialize the tilert variables.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.

        """
        logger.info(f'[{__file__.split(chr(47))[-1]}] UnProjOAllReduce.init_tilert_vars')
        self.hidden_out = torch.zeros(
            (batch_size, seq_len, self.dim),
            dtype=torch.bfloat16,
            device=f"cuda:{self.device_id}",
        )
        self.profile_logs = get_profile_log_tensor(device=f"cuda:{self.device_id}")
        self.is_var_init = True

    def init_random_weights(self) -> None:

        """Initialize the random weights."""
        logger.debug(f"{self.op_name}: init_random_weights on device {self.device_id}")
        unproj_o_weights = torch.randn(
            self.dim,
            self.n_heads * self.head_dim,
            dtype=torch.bfloat16,
            device=f"cuda:{self.device_id}",
        ).to(torch.float8_e4m3fn)

        head_scale_dim = self.head_dim // self.block_size
        dim_scale_dim = self.dim // self.block_size
        scale_dtype = torch.float32 if self.arch_name == "glm_5" else torch.bfloat16
        unproj_o_scales = torch.randn(
            dim_scale_dim,
            self.n_heads * head_scale_dim,
            dtype=scale_dtype,
            device=f"cuda:{self.device_id}",
        )
        ref_state_dict = {
            self.ref_weights_alias.o_proj_weight: unproj_o_weights,
            self.ref_weights_alias.o_proj_scale_inv: unproj_o_scales,
        }

        self.init_reference_weights(ref_state_dict)
        sharded = self.device_sharding(ref_state_dict)
        per_device_state = {k: v[self.device_id] for k, v in sharded.items()}
        self.init_tilert_weights(per_device_state)

    def golden_forward(
        self,
        vec_in: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the down-project module.

        Args:
            vec_in: Input vector.

        Returns:
            Output tensor.

        """
        logger.info(f"[UnprojOAllReduceOp.golden_forward_{self.device_id}] ENTRY: vec_in.shape={vec_in.shape}")
        
        assert self.ref_unproj_o is not None
        bsz = vec_in.shape[0]
        seq_len = vec_in.shape[1]
        assert bsz == 1
        
        logger.info(f"[UnprojOAllReduceOp.golden_forward_{self.device_id}] Computing output projection: vec_in.float() @ ref_unproj_o.T.float()")
        res = vec_in.reshape(bsz, seq_len, -1).float() @ self.ref_unproj_o.T.float()
        result = res.to(torch.bfloat16)
        
        logger.info(f"[UnprojOAllReduceOp.golden_forward_{self.device_id}] EXIT: result.shape={result.shape}")
        
        return result

    def tilert_forward(
        self,
        vec_in: torch.Tensor,
        x_in: torch.Tensor,
        flag: int,
    ) -> torch.Tensor:
        logger.info(f"[UnprojOAllReduceOp.tilert_forward_{self.device_id}] ENTRY: vec_in.shape={vec_in.shape}, x_in.shape={x_in.shape}, flag={flag}")
        
        assert self.hidden_out is not None
        assert self.profile_logs is not None
        assert self.algorithm is not None
        logger.info(f"[UnprojOAllReduceOp.tilert_forward_{self.device_id}] Calling CUDA kernel unproj_o_allreduce")
        unproj_o_allreduce(
            vec_in,
            self.tilert_weights,
            self.tilert_scales,
            x_in,
            flag,
            self.hidden_out,
            self.profile_logs,
            model_arch=self.model_args.arch_name,
            compute_kernel_type=self.algorithm.value,
        )
        logger.info(f"[UnprojOAllReduceOp.tilert_forward_{self.device_id}] EXIT: hidden_out.shape={self.hidden_out.shape}")
        
        return self.hidden_out

    def __call__(
        self,
        vec_in: torch.Tensor,
    ) -> torch.Tensor:

        logger.info(f'[{__file__.split(chr(47))[-1]}] UnProjOAllReduce.__call__')
        return self.golden_forward(vec_in)