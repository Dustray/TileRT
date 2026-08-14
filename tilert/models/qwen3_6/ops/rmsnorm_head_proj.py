"""RMSNormHeadProj operation module."""
from tilert import logger

from dataclasses import dataclass
from enum import Enum

import torch

from tilert.models.base import TileRTModule, TilertWeightsConverter
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.utils import get_profile_log_tensor

__all__ = [
    "rmsnorm_head_proj",
    "RMSNormHeadProj",
    "RMSNormHeadProjTilertWeightsAlias",
]


def rmsnorm_head_proj(
    hidden_in: torch.Tensor,
    gamma_in: torch.Tensor,
    weight_in: torch.Tensor,
    hidden_rmsnorm_out: torch.Tensor,
    logits_out: torch.Tensor,
    profile_logs: torch.Tensor,
    model_arch: str,
    compute_kernel_type: str = "general",
) -> None:

    """RMS Norm Head Projection operation."""
    logger.info(f'[{__file__.split(chr(47))[-1]}] rmsnorm_head_proj')
    torch.ops.tilert.rmsnorm_head_proj_op(
        hidden_in,
        gamma_in,
        weight_in,
        hidden_rmsnorm_out,
        logits_out,
        model_arch,
        compute_kernel_type,
        profile_logs,
    )


class RMSNormHeadProjAlgorithm(Enum):
    """RMSNormHeadProj algorithm"""

    GENERAL = "general"


class RMSNormHeadProjWeightsConverter(TilertWeightsConverter):
    """RMSNormHeadProj weights converter"""

    @staticmethod
    def tilert_to_tilert_native_bf16_warp_gemv(
        tilert_weight_in: torch.Tensor,
    ) -> torch.Tensor:

        """Convert TILERT weights to TILERT native bf16 warp gemv weights."""
        logger.info(f'[{__file__.split(chr(47))[-1]}] RMSNormHeadProjWeightsConverter.tilert_to_tilert_native_bf16_warp_gemv')
        weights = tilert_weight_in.reshape(1010, 16, 7, 1024)
        weights = weights.transpose(1, 2).reshape(7070, 16, 1024)
        return weights.contiguous()

    def convert_to_general(
        self, weights_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the weights to general format.

        Args:
            weights_list: List of weights.

        Returns:
            Tuple of weights.

        """
        logger.info(f'[{__file__.split(chr(47))[-1]}] RMSNormHeadProjWeightsConverter.convert_to_general')
        args = self.model_args
        assert args.arch_name == "qwen3_6" or args.arch_name == "glm_5"

        with torch.inference_mode():
            rmsnorm_gamma, mat_in = weights_list
            logits_dim = mat_in.shape[-2]
            dim = mat_in.shape[-1]
            num_steps = dim // 1024
            assert dim % 1024 == 0
            weights = mat_in.reshape(logits_dim // 16, 16, num_steps, 1024)
            weights = weights.transpose(1, 2).reshape(logits_dim // 16 * num_steps, 16, 1024)
            return rmsnorm_gamma.float(), weights


@dataclass
class RMSNormHeadProjTilertWeightsAlias:
    """TileRT weights alias for RMSNormHeadProj."""

    model_norm_weight = "model.norm.weight"
    lm_head_weight = "lm_head.weight"

    @property
    def tilert_tensor_alias(self) -> list[str]:

        logger.info(f'[{__file__.split(chr(47))[-1]}] RMSNormHeadProjTilertWeightsAlias.tilert_tensor_alias')
        return [self.model_norm_weight, self.lm_head_weight]

    def __call__(self) -> list[str]:

        logger.info(f'[{__file__.split(chr(47))[-1]}] RMSNormHeadProjTilertWeightsAlias.__call__')
        return self.tilert_tensor_alias


class RMSNormHeadProj(TileRTModule):
    """RMSNormHeadProj module"""

    _SUPPORTED_ALGORITHMS = {
        "qwen3_6": [RMSNormHeadProjAlgorithm.GENERAL],
        "glm_5": [RMSNormHeadProjAlgorithm.GENERAL],
    }

    def __init__(
        self,
        model_args: ModelArgsQwen36,
        device_id: int,
        num_devices: int,
        algorithm: RMSNormHeadProjAlgorithm = RMSNormHeadProjAlgorithm.GENERAL,
    ):

        logger.info(f'[{__file__.split(chr(47))[-1]}] RMSNormHeadProj.__init__')
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
        )

        self.arch_name = self.model_args.arch_name
        self.dim = self.model_args.dim
        self.logits_dim = self.model_args.vocab_size
        self.algorithm = algorithm
        self.eps = self.model_args.eps

        self.ref_rmsnorm_gamma: torch.Tensor | None = None
        self.ref_head_proj: torch.Tensor | None = None

        self.tilert_rmsnorm_gamma: torch.Tensor | None = None
        self.tilert_head_proj: torch.Tensor | None = None

        self.hidden_rmsnorm_out: torch.Tensor | None = None
        self.hidden_out: torch.Tensor | None = None

        self.profile_logs: torch.Tensor | None = None
        self.is_init = False

        self.tilert_weights_alias = RMSNormHeadProjTilertWeightsAlias()

        self.ref_tensor_alias: list[str] = [
            "model.norm.weight",
            "lm_head.weight",
        ]

    @property
    def tilert_tensor_alias(self) -> list[str]:

        logger.info(f'[{__file__.split(chr(47))[-1]}] RMSNormHeadProj.tilert_tensor_alias')
        return self.tilert_weights_alias()

    def get_weights_list(self) -> list[torch.Tensor]:
        """
        Get the weights list.

        Returns:
            List of weights.

        """
        logger.info(f'[{__file__.split(chr(47))[-1]}] RMSNormHeadProj.get_weights_list')
        return [self.tilert_rmsnorm_gamma, self.tilert_head_proj]

    def device_sharding(
        self,
        weights_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Device sharding.

        Args:
            weights_dict: Dictionary of weights.
            key_prefix: Key prefix.
        Returns:
            Tuple of weights.

        """
        logger.info(f'[{__file__.split(chr(47))[-1]}] RMSNormHeadProj.device_sharding')
        rmsnorm_gamma_key = "model.norm.weight"
        head_proj_key = "lm_head.weight"
        # Qwen3.6 checkpoint stores final norm under model.language_model.norm.weight.
        qwen36_norm_key = "model.language_model.norm.weight"
        if qwen36_norm_key in weights_dict:
            rmsnorm_gamma = weights_dict[qwen36_norm_key][None, ...]
        else:
            rmsnorm_gamma = weights_dict[rmsnorm_gamma_key][None, ...]
        rmsnorm_gamma = rmsnorm_gamma.repeat(self.num_devices, 1)
        head_proj = weights_dict[head_proj_key]

        # Detect already-sharded TileRT checkpoint: each device already owns a
        # vocab shard of shape (vocab_shard, dim).  Stack them; otherwise
        # replicate the full head projection.
        if head_proj.dim() == 2 and head_proj.size(0) * self.num_devices == self.logits_dim:
            head_proj = head_proj[None, ...].repeat(self.num_devices, 1, 1)
        elif head_proj.dim() == 3 and head_proj.size(0) == self.num_devices:
            # Already stacked (e.g. from init_random_weights/device_sharding).
            pass
        else:
            # EP8 / replicated full vocab layout.
            head_proj = head_proj[None, ...].repeat(self.num_devices, 1, 1)

        return rmsnorm_gamma.contiguous(), head_proj.contiguous()

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Initialize the reference weights.

        Args:
            state_dict: State dictionary.
            device_id: Device ID.

        """
        logger.info(f'[{__file__.split(chr(47))[-1]}] RMSNormHeadProj.init_reference_weights')
        sharded_list = self.device_sharding(state_dict)

        gamma, head_proj = sharded_list[0][self.device_id], sharded_list[1][self.device_id]
        self.ref_rmsnorm_gamma = gamma
        self.ref_head_proj = head_proj

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Initialize the tilert weights.

        Args:
            state_dict: State dictionary.

        """
        logger.info(f'[{__file__.split(chr(47))[-1]}] RMSNormHeadProj.init_tilert_weights')
        assert self.algorithm is not None
        self.tilert_rmsnorm_gamma, self.tilert_head_proj = RMSNormHeadProjWeightsConverter(
            self.model_args, self.num_devices
        ).dispatch(self.algorithm, [state_dict[alias] for alias in self.tilert_weights_alias()])

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        """
        Initialize the tilert variables.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.

        """
        logger.info(f'[{__file__.split(chr(47))[-1]}] RMSNormHeadProj.init_tilert_vars')
        self.hidden_rmsnorm_out = torch.zeros(
            (batch_size, seq_len, self.dim),
            dtype=torch.bfloat16,
            device=f"cuda:{self.device_id}",
        )
        # EP8: full vocab logits on every device.
        self.hidden_out = torch.zeros(
            (batch_size, seq_len, self.logits_dim),
            dtype=torch.float32,
            device=f"cuda:{self.device_id}",
        )
        self.profile_logs = get_profile_log_tensor(device=f"cuda:{self.device_id}")
        self.is_init = True

    def init_random_weights(self, device_id: int | None = None) -> None:

        """Initialize the random weights."""
        logger.info(f'[{__file__.split(chr(47))[-1]}] RMSNormHeadProj.init_random_weights')
        if device_id is None:
            device_id = self.device_id
        rmsnorm_gamma = torch.randn(self.dim, dtype=torch.float32, device=f"cuda:{device_id}")
        head_proj = torch.randn(
            self.logits_dim, self.dim, dtype=torch.bfloat16, device=f"cuda:{device_id}"
        )

        tensor_list = [
            rmsnorm_gamma,
            head_proj,
        ]
        state_dict = dict(zip(self.ref_tensor_alias, tensor_list))

        self.init_reference_weights(state_dict)
        sharded_list = self.device_sharding(state_dict)
        sharded_state_dict = {
            alias: sharded_list[i][self.device_id]
            for i, alias in enumerate(self.tilert_weights_alias())
        }
        self.init_tilert_weights(sharded_state_dict)

    def golden_forward(
        self,
        hidden_in: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the down-project module.

        Args:
            hidden_in: Input hidden.

        Returns:
            Output tensor.

        """
        logger.info(f'[{__file__.split(chr(47))[-1]}] RMSNormHeadProj.golden_forward')
        assert self.ref_rmsnorm_gamma is not None
        assert self.ref_head_proj is not None
        bsz = hidden_in.shape[0]
        assert bsz == 1
        hidden_in_float = hidden_in.float().detach()
        gamma = self.ref_rmsnorm_gamma.float().detach()
        # Qwen3.5-MoE/Qwen3.6 use the (1 + weight) RMSNorm convention.
        hidden_rmsnorm = hidden_in_float * torch.rsqrt(
            hidden_in_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        hidden_rmsnorm = hidden_rmsnorm * (1.0 + gamma)
        # Golden path expects a full 2-D head projection; end2end ensures this.
        head_proj = self.ref_head_proj
        if head_proj.dim() != 2:
            raise ValueError(f"Unexpected head projection layout: {head_proj.shape}")

        # Compute logits in chunks to keep peak memory low: each chunk does a
        # bf16 matmul (hidden_rmsnorm @ head_chunk.T) and immediately converts
        # the small result to float32, avoiding a full (1, vocab) float32 temp.
        head_proj_bf16 = head_proj.to(torch.bfloat16)
        chunk_size = 65536
        vocab_size = head_proj.shape[0]
        chunks = []
        for start in range(0, vocab_size, chunk_size):
            end = min(start + chunk_size, vocab_size)
            chunk_logits = hidden_rmsnorm.to(torch.bfloat16) @ head_proj_bf16[start:end, :].T
            chunks.append(chunk_logits.float())
        return torch.cat(chunks, dim=-1)

    def tilert_forward(
        self,
        hidden_in: torch.Tensor,
    ) -> torch.Tensor:

        logger.info(f'[{__file__.split(chr(47))[-1]}] RMSNormHeadProj.tilert_forward')
        assert self.hidden_out is not None
        rmsnorm_head_proj(
            hidden_in,
            self.tilert_rmsnorm_gamma,
            self.tilert_head_proj,
            self.hidden_rmsnorm_out,
            self.hidden_out,
            self.profile_logs,
            model_arch=self.model_args.arch_name,
        )
        return self.hidden_out

    def __call__(
        self,
        hidden_in: torch.Tensor,
    ) -> torch.Tensor:

        logger.info(f'[{__file__.split(chr(47))[-1]}] RMSNormHeadProj.__call__')
        return self.golden_forward(hidden_in)