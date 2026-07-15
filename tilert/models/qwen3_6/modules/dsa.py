"""Qwen3.6-35B-A3B DSA (Deep Show Attention) module."""

from typing import Any

import torch

from tilert.models.base import SerializableTileRTModule
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.models.qwen3_6.modules.gated_attention import GatedAttention
from tilert.models.qwen3_6.modules.delta_net import DeltaNet
from tilert.models.qwen3_6.modules.moe import QwenMoeBlock
from tilert.models.qwen3_6.modules.mlp import QwenMlpBlock

__all__ = ["QwenDsa"]


class QwenDsa(SerializableTileRTModule):
    """DSA module for Qwen3.6.

    Qwen3.6 has a heterogeneous layer structure:
    - 30 DeltaNet layers (3 per block × 10 blocks)
    - 10 Gated Attention layers (1 per block × 10 blocks)

    This module handles both layer types with appropriate routing.
    
    Forward pass priority:
        1. golden_forward: PyTorch reference implementation (default)
        2. tilert_forward: Optimized CUDA kernel (when flag_enable_tilert=True)
    """

    def __init__(
        self,
        model_args: ModelArgsQwen36,
        device_id: int,
        num_devices: int,
        cached_ffn_ops: list | None = None,
    ):
        """Initialize QwenDsa module.

        Args:
            model_args: Model configuration.
            device_id: Current GPU device ID.
            num_devices: Total number of devices.
            cached_ffn_ops: Optional pre-cached FFN ops for weight reuse.
        """
        super().__init__(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            remove_selected=True,
        )

        self.model_args = model_args
        self.device_id = device_id
        self.num_devices = num_devices

        dev = f"cuda:{device_id}"

        # Layer configuration
        self.n_delta_layers = model_args.n_delta_layers  # 30
        self.n_gated_layers = model_args.n_gated_layers   # 10
        self.n_blocks = model_args.n_blocks                # 10

        # Layer type mapping: 0 = DeltaNet, 1 = Gated Attention
        # Pattern: [DeltaNet, DeltaNet, DeltaNet, GatedAttention] × 10
        self.layer_types = []
        for block_idx in range(self.n_blocks):
            # 3 DeltaNet layers per block
            for _ in range(3):
                self.layer_types.append(0)  # DeltaNet
            # 1 Gated Attention layer per block
            self.layer_types.append(1)  # Gated Attention

        # TODO: Initialize DeltaNet and Gated Attention layers
        # This requires the corresponding CUDA kernels to be built first

    def golden_forward(self, *args: Any, **kwargs: Any) -> Any:
        """Golden forward pass (PyTorch reference implementation).

        This is the default forward method, providing a correct but
        unoptimized reference implementation for testing and validation.
        """
        raise NotImplementedError(
            "QwenDsa golden_forward: PyTorch reference implementation not yet implemented. "
            "This requires implementing the reference forward pass for Qwen3.6 architecture."
        )

    def tilert_forward(self, *args: Any, **kwargs: Any) -> Any:
        """Tilert forward pass (optimized CUDA kernel).

        This is the optimized forward method using TileRT CUDA kernels.
        Requires building the libtilert_qwen36.so library first.
        """
        raise NotImplementedError(
            "QwenDsa tilert_forward: CUDA kernels not yet built. "
            "Build libtilert_qwen36.so to enable this path."
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass with automatic path selection.

        Automatically selects between golden_forward and tilert_forward
        based on the flag_enable_tilert flag.
        """
        if self.flag_enable_tilert:
            return self.tilert_forward(*args, **kwargs)
        return self.golden_forward(*args, **kwargs)

    def get_tilert_weights_alias(self) -> list[str]:
        """Get weight aliases for serialization."""
        # TODO: Implement based on actual layer structure
        return []

    def from_pretrained(self, model_path: str) -> None:
        """Load pretrained weights."""
        raise NotImplementedError(
            "QwenDsa weight loading not yet implemented."
        )

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass