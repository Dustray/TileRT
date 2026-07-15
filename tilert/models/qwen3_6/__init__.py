"""Qwen3.6-35B-A3B model integration for TileRT."""

from tilert.models.qwen3_6.generator import Qwen36Generator, stats_time
from tilert.models.qwen3_6.model_args import ModelArgsQwen36

__all__ = [
    "Qwen36Generator",
    "ModelArgsQwen36",
    "stats_time",
]