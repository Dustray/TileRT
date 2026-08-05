"""Base classes for deepseek v3."""

import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar

import torch
import torch.nn as nn

from tilert import logger
from tilert.models.deepseek_config import get_rank, get_world_size
from tilert.models.deepseek_v3_2.model_args import ModelArgs
from tilert.utils import get_profile_log_tensor

__all__ = [
    "TileRTModule",
]

ModelArgsLike = Any


class TilertWeightsConverter:
    """Tilert weights converter"""

    def __init__(self, model_args: ModelArgsLike, num_devices: int):
        self.model_args = model_args
        self.num_devices = num_devices

    def dispatch(self, algorithm: Enum, weights: list[torch.Tensor]) -> Any:
        dispatch_method = getattr(self, f"convert_to_{algorithm.value}")
        return dispatch_method(weights)


class TileRTModule(nn.Module, ABC):
    """Base class for all TileRT modules.

    This class serves as an abstract base for implementing TileRT modules.
    All module classes should inherit from this class and implement their
    own forward method.
    """

    _SUPPORTED_ALGORITHMS: ClassVar[dict[str, list[Enum]]] = {}
    _VALID_COMPUTE_KERNEL_TYPES: ClassVar[frozenset[str]] = frozenset(
        {
            "bf16",
            "fp8",
            "fp8mma",
            "general",
            "bf16mma",
            "fp16mma",
            "fp8mma_68cta",
        }
    )

    @classmethod
    def get_supported_algorithms(cls, arch_name: str) -> list[Enum]:
        """Return supported algorithms for the given architecture."""
        if arch_name not in cls._SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"{cls.__name__} does not support arch '{arch_name}'. "
                f"Supported: {list(cls._SUPPORTED_ALGORITHMS.keys())}"
            )
        return cls._SUPPORTED_ALGORITHMS[arch_name]

    def __init__(
        self,
        op_name: str = "",
        golden_weights_dir: str = "",
        tilert_weights_dir: str = "",
        layer_idx: int = 0,
        compute_kernel_type: str = "bf16",
        model_args: ModelArgsLike | None = None,
        num_devices: int = 8,
        device_id: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the operation.

        Args:
            op_name: Optional operation name. Defaults to class name.
            golden_weights_dir: Optional path to golden weights directory.
            tilert_weights_path: Optional path to tilert weights directory.
            layer_idx: Layer index.
            compute_kernel_type: Compute kernel type, bf16 by default; fp8 is also supported.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)

        self.model_args: ModelArgsLike = model_args if model_args is not None else ModelArgs()
        self.num_devices = num_devices
        self.device_id = device_id
        self.algorithm: Enum | None = None

        self.is_var_init = False
        self.is_tilert_weights_init = False
        self.is_ref_weights_init = False

        self.profile_logs: torch.Tensor | None = None

        self.layer_idx = layer_idx

        self.flag_enable_tilert = False

        if compute_kernel_type not in self._VALID_COMPUTE_KERNEL_TYPES:
            raise ValueError(
                f"Invalid compute kernel type: {compute_kernel_type}, "
                f"must be one of {sorted(self._VALID_COMPUTE_KERNEL_TYPES)}."
            )
        self.compute_kernel_type = compute_kernel_type

        self.flag_enable_profiling_log = False
        self.flag_enable_external_profiling_log = False

        self.op_name = type(self).__name__ if op_name == "" else op_name
        self.profile_log_dir = "profile_logs"

        self.golden_weights_dir = golden_weights_dir
        self.tilert_weights_dir = tilert_weights_dir

        self.profile_logs = get_profile_log_tensor()

    def get_cache_vars(self) -> list[torch.Tensor]:
        return []

    def get_tilert_weights_alias(self) -> list[str]:
        return list(self.tilert_weights_alias())

    def get_ref_weights_alias(self) -> list[str]:
        return list(self.ref_weights_alias())

    def set_algorithm(self, algorithm: Enum) -> None:
        """Set the algorithm for the module.

        Args:
            algorithm: Algorithm.
        """
        if self._SUPPORTED_ALGORITHMS:
            arch = self.model_args.arch_name
            supported = self.get_supported_algorithms(arch)
            if algorithm not in supported:
                raise ValueError(
                    f"{type(self).__name__}: algorithm {algorithm} not supported "
                    f"for arch '{arch}'. Supported: {supported}"
                )
        self.algorithm = algorithm

    def register_weights(self, weights_config: dict[str, dict[str, Any]]) -> None:
        """Register weights configuration.

        Args:
            weights_config: Dictionary mapping weight names to their configurations.
        """
        self.weight_loader.register_weights(weights_config)

    def get_profile_log_path(self) -> str:
        """Get the path to the profile log file.

        Returns:
            Path to the profile log file.
        """
        return os.path.join(self.profile_log_dir, f"{self.op_name}.xlsx")

    def get_external_profile_log_path(self) -> str:
        """Get the path to the external profile log file.

        Returns:
            Path to the external profile log file.
        """
        return os.path.join(self.profile_log_dir, f"{self.op_name}.json")

    def world_size(self) -> int:
        """Get the world size.

        Returns:
            World size.
        """
        return int(get_world_size())

    def rank(self) -> int:
        """Get the rank.

        Returns:
            Rank.
        """
        return int(get_rank())

    @abstractmethod
    def golden_forward(self, *args: Any, **kwargs: Any) -> Any:
        """Golden forward pass.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        del args, kwargs
        raise NotImplementedError("Golden forward not implemented")

    @abstractmethod
    def tilert_forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: U100
        """Tilert forward method to be implemented by subclasses.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        del args, kwargs
        raise NotImplementedError("Tilert forward not implemented")

    def enable_profiling_log(self, enable: bool = True) -> None:
        """Enable profiling log for this module and all submodules.

        Args:
            enable: Whether to enable profiling.
        """
        for module in self.modules():
            if isinstance(module, TileRTModule):
                logger.info(f"Enable profiling for {module.__class__.__name__}")
                module.flag_enable_profiling_log = enable

    def enable_external_profiling_log(self, enable: bool = True) -> None:
        """Enable external profiling log for this module and all submodules.

        Args:
            enable: Whether to enable external profiling.
        """
        for module in self.modules():
            if isinstance(module, TileRTModule):
                logger.info(f"Enable external profiling for {module.__class__.__name__}")
                module.flag_enable_external_profiling_log = enable

    def enable_tilert(self, enable: bool = True) -> None:  # type: ignore
        for module in self.modules():
            if isinstance(module, TileRTModule):
                logger.info(f"Enable tilert for {module.__class__.__name__}")
                module.flag_enable_tilert = enable
                if enable:
                    module.to_tilert_weights()


class SerializableTileRTModule(TileRTModule):
    """Serializable TileRT module."""

    def __init__(
        self,
        model_args: ModelArgsLike,
        device_id: int,
        num_devices: int,
        remove_selected: bool = False,
    ):
        super().__init__(
            type(self).__name__, model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.remove_selected = remove_selected

        self.exec_seq: list[TileRTModule] = []
        self.prefix_seq: list[str] = []
        self.suffix_seq: list[str] = []
        self.retain_weights_seq: list[bool] = []

    def get_cache_vars(self) -> list[torch.Tensor]:
        cache_vars = []
        for op in self.exec_seq:
            cache_vars.extend(op.get_cache_vars())
        return cache_vars

    def register_op(
        self, op: TileRTModule, prefix: str = "", suffix: str = "", retain_weights: bool = False
    ) -> None:
        self.exec_seq.append(op)
        self.prefix_seq.append(prefix)
        self.suffix_seq.append(suffix)
        self.retain_weights_seq.append(retain_weights)

    def get_tilert_weights_alias(self) -> list[str]:
        weights_alias: list[str] = []
        for op in self.exec_seq:
            weights_alias.extend(op.get_tilert_weights_alias())
        return weights_alias

    def get_ref_weights_alias(self) -> list[str]:
        weights_alias: list[str] = []
        for op in self.exec_seq:
            weights_alias.extend(op.get_ref_weights_alias())
        return weights_alias

    def get_weights_list(self) -> list[torch.Tensor]:
        weights = []
        for op in self.exec_seq:
            weights.extend(op.get_weights_list())
        return weights

    def device_sharding(self, raw_weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sharded_weights_map: dict[str, torch.Tensor] = {}
        for op in self.exec_seq:
            sharded_weights_map.update(op.device_sharding(raw_weights_map))
        return sharded_weights_map

    @property
    def tilert_tensor_alias(self) -> list[str]:
        """Return tilert tensor alias of the first sub-op (RMSNormProjxWqkvia)."""
        tensor_alias: list[str] = []
        for op in self.exec_seq:
            tensor_alias.extend(op.tilert_weights_alias())
        return tensor_alias

    @property
    def ref_tensor_alias(self) -> list[str]:
        """Return reference tensor alias of the first sub-op (RMSNormProjxWqkvia)."""
        tensor_alias: list[str] = []
        for op in self.exec_seq:
            tensor_alias.extend(op.ref_weights_alias())
        return tensor_alias

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        logger.info(
            f"{type(self).__name__}.init_tilert_weights: "
            f"{len(self.exec_seq)} sub-op(s), state_dict has {len(state_dict)} entries"
        )
        for op, prefix, suffix, retain_weights in zip(
            self.exec_seq, self.prefix_seq, self.suffix_seq, self.retain_weights_seq
        ):
            if op.is_tilert_weights_init:
                logger.debug(f"Skipping init_tilert_weights for {op.op_name} (already initialized)")
                continue

            logger.debug(
                f"Initializing tilert weights for {op.op_name} "
                f"(prefix={prefix!r}, suffix={suffix!r})"
            )
            keys_to_remove = set()
            op_state_dict = {}
            for op_key in op.get_tilert_weights_alias():
                original_key = f"{prefix}{op_key}{suffix}"
                if original_key in state_dict:
                    op_state_dict[op_key] = state_dict[original_key]
                    if self.remove_selected:
                        keys_to_remove.add(original_key)
                    continue

                ref_original_key = f"ref_{prefix}{op_key}{suffix}"
                if ref_original_key in state_dict:
                    op_state_dict[op_key] = state_dict[ref_original_key]
                    continue

                # Special case: the GQA attention op fuses q/k/v into a single
                # ``qkv_proj_weights`` tensor, but the converted checkpoint
                # stores them as separate ``q_proj``, ``k_proj`` and ``v_proj``
                # weights.  Build the fused tensor on the fly if the pieces
                # are present, taking precedence over the single-key fallback.
                if op_key == "qkv_proj_weights":
                    pieces = []
                    piece_keys = []
                    for piece in ("q_proj", "k_proj", "v_proj"):
                        piece_key = f"{prefix}{piece}.weight{suffix}"
                        if piece_key in state_dict:
                            pieces.append(state_dict[piece_key])
                            piece_keys.append(piece_key)
                        else:
                            break
                    if len(pieces) == 3:
                        op_state_dict[op_key] = torch.cat(pieces, dim=0)
                        if self.remove_selected:
                            keys_to_remove.update(piece_keys)
                        continue
                # Fallback: some Qwen3.6 converted checkpoints use dot-weight
                # aliases (e.g. ``linear_attn.in_proj_qkv.weight``) while the
                # op's tilert alias is underscore-style (``in_proj_qkv_weights``).
                # Try the dot-weight form derived from the underscore alias.
                dot_weight_key = self._alias_to_dot_weight_key(prefix, op_key, suffix, state_dict)
                if dot_weight_key in state_dict:
                    op_state_dict[op_key] = state_dict[dot_weight_key]
                    if self.remove_selected:
                        keys_to_remove.add(dot_weight_key)
                    continue

            op.init_tilert_weights(op_state_dict)
            logger.debug(
                f"Initialized tilert weights for {op.op_name}: "
                f"{len(op_state_dict)} tensor(s)"
            )

            if self.remove_selected and not retain_weights:
                for k in keys_to_remove:
                    del state_dict[k]
        logger.info(f"{type(self).__name__}.init_tilert_weights completed")

    @staticmethod
    def _alias_to_dot_weight_key(
        prefix: str, op_key: str, suffix: str, state_dict: dict[str, torch.Tensor] | None = None
    ) -> str:
        """Map an underscore-style tilert alias to a dot-weight checkpoint key.

        The converted Qwen3.6 checkpoint uses a flat naming convention that drops the
        ``linear_attn`` / ``self_attn`` module prefixes from the original PyTorch
        state dict.  This helper translates op aliases into those flat keys.

        Examples:
            - ``layer_0_in_proj_qkv_weights_dev_0`` ->
              ``layer_0_in_proj_qkv.weight_dev_0``
            - ``layer_3_qkv_proj_weights_dev_0`` ->
              ``layer_3_q_proj.weight_dev_0``
        """
        full = f"{prefix}{op_key}{suffix}"
        # Only rewrite aliases that end with ``_weights`` or ``_weight``.
        for trailing in ("_weights", "_weight"):
            if op_key.endswith(trailing):
                stem = op_key[: -len(trailing)]
                # DeltaNet weights are stored with a flat dot-weight name.
                if stem in {
                    "in_proj_qkv",
                    "in_proj_z",
                    "in_proj_a",
                    "in_proj_b",
                    "conv1d",
                    "norm",
                    "out_proj",
                }:
                    return f"{prefix}{stem}.weight{suffix}"
                # Per-device tensors that already contain a dot in the alias.
                if stem == "A_log":
                    return f"{prefix}A_log{suffix}"
                if stem == "dt_bias":
                    return f"{prefix}dt_bias{suffix}"
                # GQA weights: the converter shards q/k/v/o separately but the
                # op currently expects a fused ``qkv_proj``.  Use the q_proj.
                if stem == "qkv_proj":
                    return f"{prefix}q_proj.weight{suffix}"
                if stem == "o_proj":
                    return f"{prefix}o_proj.weight{suffix}"
                if stem == "q_norm":
                    return f"{prefix}q_norm.weight{suffix}"
                if stem == "k_norm":
                    return f"{prefix}k_norm.weight{suffix}"
                # Qwen3.6 GatedAttention stores the attention output projection as
                # ``o_proj.weight`` but exposes it through the UnProjOAllReduce op
                # under the ``unproj_weights`` alias.  Map it here without breaking
                # DeepSeek/GLM checkpoints that use a real ``unproj.weight`` key.
                if stem == "unproj":
                    o_proj_key = f"{prefix}o_proj.weight{suffix}"
                    if o_proj_key in state_dict:
                        return o_proj_key
                    return full
                break
        return full

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        logger.info(
            f"{type(self).__name__}.init_reference_weights: "
            f"{len(self.exec_seq)} sub-op(s), state_dict has {len(state_dict)} entries"
        )
        for op, prefix, suffix in zip(
            self.exec_seq, self.prefix_seq, self.suffix_seq
        ):
            logger.debug(f"Initializing reference weights for {op.op_name}")
            op_state_dict: dict[str, torch.Tensor] = {}
            for ref_key in op.get_ref_weights_alias():
                original_key = f"{prefix}{ref_key}{suffix}"
                if original_key in state_dict:
                    op_state_dict[ref_key] = state_dict[original_key]
                    continue
                # Fallback: ``load_hf_source_weights`` stores reference tensors
                # under a dedicated ``ref_layer_{idx}_`` prefix to avoid
                # colliding with tilert keys that use the same alias names.
                ref_prefixed_key = f"ref_{prefix}{ref_key}{suffix}"
                if ref_prefixed_key in state_dict:
                    op_state_dict[ref_key] = state_dict[ref_prefixed_key]
            if False and op_state_dict: # 暂时不需要
                op.init_reference_weights(op_state_dict)
        logger.info(f"{type(self).__name__}.init_reference_weights completed")

    def init_random_weights(self) -> None:
        logger.info(
            f"{type(self).__name__}.init_random_weights: "
            f"{len(self.exec_seq)} sub-op(s)"
        )
        self._random_init_marker = True
        for op in self.exec_seq:
            logger.debug(f"Random-initializing {op.op_name}")
            op.init_random_weights()
        logger.info(f"{type(self).__name__}.init_random_weights completed")

    def _is_random_init(self) -> bool:
        return getattr(self, "_random_init_marker", False)

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        logger.debug(
            f"{type(self).__name__}.init_tilert_vars: batch_size={batch_size}, seq_len={seq_len}"
        )
        for op in self.exec_seq:
            op.init_tilert_vars(batch_size, seq_len)

    def golden_forward(
        self,
        x: torch.Tensor,
        pe_cache: torch.Tensor,
        start_pos: int,
    ) -> Any:
        del x, pe_cache, start_pos
        raise NotImplementedError("Golden forward is not implemented")

    def tilert_forward(
        self,
        x: torch.Tensor,
        pe_cache: torch.Tensor,
        start_pos: int,
    ) -> Any:
        del x, pe_cache, start_pos
        raise NotImplementedError("Tilert forward is not implemented")
