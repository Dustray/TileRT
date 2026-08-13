"""MTP (Multi-Token Prediction) module for Qwen3.6."""
from tilert import logger

from typing import Any

import torch

from tilert.models.base import TileRTModule
from tilert.models.qwen3_6.model_args import ModelArgsQwen36


class QwenMTP(TileRTModule):
    """MTP (Multi-Token Prediction) module for Qwen3.6.

    MTP enables speculative decoding by predicting multiple tokens ahead.
    This is a second priority feature - implement after basic inference works.
    
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

        logger.info(f'[{__file__.split(chr(47))[-1]}] QwenMTP.__init__')
        super().__init__()
        self.model_args = model_args
        self.device_id = device_id
        self.num_devices = num_devices

    def golden_forward(self, *args: Any, **kwargs: Any) -> Any:

        """Golden forward: PyTorch reference implementation."""
        logger.info(f'[{__file__.split(chr(47))[-1]}] QwenMTP.golden_forward')
        raise NotImplementedError(
            "QwenMTP golden_forward: PyTorch reference not yet implemented"
        )

    def tilert_forward(self, *args: Any, **kwargs: Any) -> Any:

        """Tilert forward: CUDA kernel (requires libtilert_qwen36.so)."""
        logger.info(f'[{__file__.split(chr(47))[-1]}] QwenMTP.tilert_forward')
        raise NotImplementedError(
            "QwenMTP tilert_forward: CUDA kernels not yet built"
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:

        logger.info(f'[{__file__.split(chr(47))[-1]}] QwenMTP.forward')
        if self.flag_enable_tilert:
            return self.tilert_forward(*args, **kwargs)
        return self.golden_forward(*args, **kwargs)
