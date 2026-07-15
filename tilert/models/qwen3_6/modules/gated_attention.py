"""Gated Attention module for Qwen3.6."""

from typing import Any

import torch

from tilert.models.base import TileRTModule
from tilert.models.qwen3_6.model_args import ModelArgsQwen36


class GatedAttention(TileRTModule):
    """Gated Attention layer for Qwen3.6.

    This implements GQA (Grouped Query Attention) with a gating mechanism.
    Used in the Gated Attention layers (1 per block × 10 blocks = 10 layers).
    
    Forward pass priority:
        1. golden_forward: PyTorch reference implementation (default)
        2. tilert_forward: Optimized CUDA kernel (when flag_enable_tilert=True)
    """

    def __init__(
        self,
        model_args: ModelArgsQwen36,
        device_id: int,
        num_devices: int,
    ):
        super().__init__()
        self.model_args = model_args
        self.device_id = device_id
        self.num_devices = num_devices

    def golden_forward(self, *args: Any, **kwargs: Any) -> Any:
        """Golden forward: PyTorch reference implementation."""
        raise NotImplementedError(
            "GatedAttention golden_forward: PyTorch reference not yet implemented"
        )

    def tilert_forward(self, *args: Any, **kwargs: Any) -> Any:
        """Tilert forward: CUDA kernel (requires libtilert_qwen36.so)."""
        raise NotImplementedError(
            "GatedAttention tilert_forward: CUDA kernels not yet built"
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.flag_enable_tilert:
            return self.tilert_forward(*args, **kwargs)
        return self.golden_forward(*args, **kwargs)