"""Qwen3.6-35B-A3B generator module."""

import math
import time

import torch
from transformers import AutoTokenizer

from tilert import logger
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.tilert_init import tilert_init

__all__ = [
    "Qwen36Generator",
    "stats_time",
]


def stats_time(time_list: list[float], title: str) -> None:
    """Print timing statistics."""
    if len(time_list) > 0:
        avg_time = sum(time_list) / len(time_list)
        std_dev = math.sqrt(sum((x - avg_time) ** 2 for x in time_list) / len(time_list))
        logger.info(title)
        logger.info(f"--Average time taken to generate token: {avg_time * 1000:.4f} ms")
        logger.info(f"--Standard deviation of time: {std_dev * 1000:.4f} ms")
        logger.info(f"--Effective tokens per second: {1 / avg_time:.4f}")


class Qwen36Generator:
    """Generator for Qwen3.6-35B-A3B.

    This is a placeholder implementation. The actual implementation requires
    the corresponding CUDA kernels to be built first.
    """

    def __init__(
        self,
        model_args: ModelArgsQwen36,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        model_weights_dir: str = "",
        with_mtp: bool = False,
        use_topp: bool = False,
        top_p: float = 0.9,
        top_k: int = 256,
        sampling_seed: int = 42,
        enable_thinking: bool = False,
    ):
        """Initialize the Qwen36Generator.

        Args:
            model_args: Model configuration parameters.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Temperature for sampling.
            model_weights_dir: Path to the model weights directory.
            with_mtp: Whether to use MTP (Multi-Token Prediction).
            use_topp: Whether to use top-p (nucleus) sampling.
            top_p: Top-p threshold for nucleus sampling.
            top_k: Number of top-k candidates for sampling.
            sampling_seed: Sampling seed for reproducibility.
            enable_thinking: Whether to enable thinking mode.
        """
        torch.set_num_threads(64)
        self.model_weights_dir = model_weights_dir

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.with_mtp = with_mtp
        self.use_topp = use_topp
        self.top_p = top_p
        self.top_k = top_k
        self.sampling_seed = sampling_seed
        self.enable_thinking = enable_thinking

        self.config = model_args
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_weights_dir, trust_remote_code=True
        )
        self.eos_id = self.tokenizer.eos_token_id
        self.batch_size = 1

        self.default_device = torch.device("cuda:0")

        # Placeholder for the actual decode layer
        # Will be implemented after CUDA kernels are ready
        self.decode_layer = None

        self.mtp_seq_len = 4 if with_mtp else 1

        logger.info(
            f"Qwen36Generator initialized: max_new_tokens={max_new_tokens}, "
            f"temperature={temperature}, with_mtp={with_mtp}"
        )

    def init(self) -> None:
        """Initialize the TileRT backend."""
        tilert_init()
        logger.info("TileRT backend initialized for Qwen3.6")

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.decode_layer is not None:
            self.decode_layer.cleanup()
        logger.info("Qwen36Generator cleanup completed")

    def init_random_weights(self) -> None:
        """Initialize weights randomly (for testing)."""
        logger.warning("init_random_weights: Not yet implemented for Qwen3.6")
        # TODO: Implement random weight initialization

    def from_pretrained(self) -> None:
        """Load the model weights from the given path."""
        logger.info(f"Loading weights from: {self.model_weights_dir}")
        # TODO: Implement weight loading
        # This requires the CUDA kernels to be built first

    def update_sampling_params(
        self,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 256,
        use_topp: bool = True,
    ) -> None:
        """Update sampling parameters for the next generation."""
        self.temperature = temperature
        self.use_topp = use_topp
        self.top_p = top_p
        self.top_k = top_k
        logger.debug(
            f"Updated sampling params: temp={temperature}, top_p={top_p}, "
            f"top_k={top_k}, use_topp={use_topp}"
        )

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt.

        Note: This is a placeholder. Actual generation requires CUDA kernels.
        """
        logger.warning("generate: Qwen3.6 CUDA kernels not yet implemented")

        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.default_device)

        # Placeholder: return the input as-is (for testing the interface)
        output = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return output

    def generate_streaming(self, prompt: str):
        """Generate text with streaming output.

        Note: This is a placeholder. Actual generation requires CUDA kernels.
        """
        logger.warning("generate_streaming: Qwen3.6 CUDA kernels not yet implemented")
        result = self.generate(prompt)
        yield result