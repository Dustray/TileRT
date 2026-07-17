"""Qwen3.6-35B-A3B generator module."""

import math
import time

import torch
from transformers import AutoTokenizer

from tilert import logger
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.models.qwen3_6.modules.end2end import (
    QwenShowHandsLayer,
    _extract_ffn_ops,
    _get_moe_weight_keys,
)
from tilert.models.qwen3_6.temp_var_indices import Idx
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

    The generator owns a ``QwenShowHandsLayer`` decode layer.  When the
    Qwen3.6 CUDA kernels are built, ``forward()`` will automatically use them;
    until then the generator runs a Python reference path for end-to-end smoke
    tests and architecture alignment.
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

        self.decode_layer = QwenShowHandsLayer(
            model_args=self.config,
            model_path=self.model_weights_dir,
            with_mtp=with_mtp,
            use_topp=use_topp,
            top_p=top_p,
            top_k=top_k,
        )

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
        self.decode_layer.init_random_weights()

    def from_pretrained(self) -> None:
        """Load the model weights from the given path."""
        logger.info(f"Loading weights from: {self.model_weights_dir}")
        self.decode_layer.from_pretrained(self.model_weights_dir)

    def extract_ffn_cache(self) -> tuple[dict[int, list], dict[int, set[str]]]:
        """Extract MOE/MLP op objects and skip keys from current loaded weights.

        Returns:
            Tuple of (cached_ffn_ops_per_device, skip_keys_per_device).
        """
        cached_ffn_ops: dict[int, list] = {}
        skip_keys: dict[int, set[str]] = {}
        for device_id in range(self.decode_layer.num_devices):
            stack = self.decode_layer._stack_objects[device_id]
            if stack is None:
                raise RuntimeError(
                    f"Device {device_id} QwenTransformerStack not available for cache extraction"
                )
            cached_ffn_ops[device_id] = _extract_ffn_ops(stack)
            skip_keys[device_id] = _get_moe_weight_keys(stack)
        return cached_ffn_ops, skip_keys

    def from_pretrained_with_cache(
        self,
        cached_ffn_ops_per_device: dict[int, list],
        skip_keys_per_device: dict[int, set[str]],
    ) -> None:
        """Load weights reusing cached MOE/MLP ops."""
        self.decode_layer.from_pretrained_with_cache(
            self.model_weights_dir, cached_ffn_ops_per_device, skip_keys_per_device
        )

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
        self.decode_layer.update_sampling_config(
            temperature=temperature, top_p=top_p, top_k=top_k, use_topp=use_topp
        )

    def set_cur_pos(self, cur_pos: int, with_mtp: bool | None = None) -> None:
        """Set the current decode position for RoPE."""
        active_mtp = with_mtp if with_mtp is not None else self.with_mtp
        self.decode_layer.set_cur_pos(cur_pos, with_mtp=active_mtp)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        print_log: bool = True,
        with_mtp: bool | None = None,
        prompt_tokens: list[int] | None = None,
    ) -> tuple[str, list[float], list[int], int]:
        """Main function to perform single sequence generation.

        Args:
            prompt: The input prompt string.
            print_log: Whether to print generation logs.
            with_mtp: Override MTP mode for this call. None uses self.with_mtp.
                Requires MTP weights to have been loaded (self.with_mtp=True).
            prompt_tokens: Pre-tokenized prompt tokens. If provided, skip tokenization
                and use these tokens directly (useful for exact-length benchmarking).

        Returns:
            Tuple of (result_text, time_list, accepted_counts, prompt_len).
            accepted_counts is empty for non-MTP mode.
        """
        active_mtp = with_mtp if with_mtp is not None else self.with_mtp
        if active_mtp and not self.with_mtp:
            raise ValueError("Cannot use MTP mode: MTP weights were not loaded")
        self.decode_layer.set_sampling_seed(self.sampling_seed, with_mtp=active_mtp)
        if active_mtp:
            return self._generate_with_mtp(prompt, print_log, prompt_tokens=prompt_tokens)
        result, time_list, prompt_len = self._generate_without_mtp(
            prompt, print_log, with_mtp=active_mtp, prompt_tokens=prompt_tokens
        )
        return result, time_list, [], prompt_len

    def _generate_without_mtp(
        self,
        prompt: str,
        print_log: bool = True,
        with_mtp: bool = False,
        prompt_tokens: list[int] | None = None,
    ) -> tuple[str, list[float], int]:
        """Standard generation without MTP."""
        if prompt_tokens is None:
            prompt_tokens = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                thinking=self.enable_thinking,
            )

        max_seq_len = self.config.max_seq_len
        prompt_len = len(prompt_tokens)
        total_len = min(max_seq_len, self.max_new_tokens + prompt_len)

        tokens = torch.full(
            (self.batch_size, total_len), -1, dtype=torch.long, device=self.default_device
        )
        tokens[0, :prompt_len] = torch.tensor(
            prompt_tokens, dtype=torch.long, device=self.default_device
        )
        prompt_mask = tokens != -1

        prev_pos = 0
        finished = torch.tensor(
            [False] * self.batch_size, dtype=torch.bool, device=self.default_device
        )

        time_list = []
        for cur_pos_val in range(1, total_len):
            start_time = time.time()
            multi_devices_results = self.decode_layer.forward(
                tokens[0, prev_pos], with_mtp=with_mtp, cur_pos=prev_pos
            )
            end_time = time.time()
            time_list.append(end_time - start_time)

            intermediates, *_ = multi_devices_results[0]
            next_token = intermediates[Idx.TOKEN_OUT][0, 0, 0]

            next_token = torch.where(
                prompt_mask[0, cur_pos_val], tokens[0, cur_pos_val], next_token
            )
            tokens[0, cur_pos_val] = next_token
            finished |= torch.logical_and(~prompt_mask[0, cur_pos_val], next_token == self.eos_id)
            prev_pos = cur_pos_val
            if cur_pos_val >= prompt_len:
                decoded_tokens = self.tokenizer.decode(
                    [next_token.item()], skip_special_tokens=True
                )
                if print_log:
                    print(decoded_tokens, end="", flush=True)

            if finished.all():
                break

        if print_log:
            print("\n")
            logger.info(f"--Number of tokens generated: {len(time_list)}")

            stats_time(time_list, "==== Performance ====")
            print("\n")

        self.decode_layer.reset_sequence()

        completion_tokens = []
        for _, toks in enumerate(tokens.tolist()):
            toks = toks[prompt_len : prompt_len + self.max_new_tokens]
            if self.eos_id in toks:
                toks = toks[: toks.index(self.eos_id)]
            completion_tokens.append(toks)

        decoded_tokens = self.tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)

        return f"{decoded_tokens[0]}\n" if decoded_tokens else "", time_list, prompt_len

    def _generate_with_mtp(
        self,
        prompt: str,
        print_log: bool = True,
        prompt_tokens: list[int] | None = None,
    ) -> tuple[str, list[float], list[int], int]:
        """Generation with MTP (Multi-Token Prediction) speculative decoding."""
        if prompt_tokens is None:
            prompt_tokens = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                thinking=self.enable_thinking,
            )

        max_seq_len = self.config.max_seq_len
        prompt_len = len(prompt_tokens)
        total_len = min(max_seq_len, self.max_new_tokens + prompt_len)

        tokens = torch.full(
            (self.batch_size, total_len), -1, dtype=torch.long, device=self.default_device
        )
        tokens[0, :prompt_len] = torch.tensor(
            prompt_tokens, dtype=torch.long, device=self.default_device
        )

        prefill_time_list = []
        decode_time_list = []
        decode_accepted_counts = []
        cur_pos = 0

        while cur_pos < prompt_len - 1:
            draft_end = min(cur_pos + self.mtp_seq_len, prompt_len)
            draft_tokens = tokens[0, cur_pos:draft_end].clone()
            actual_token_count = draft_tokens.shape[0]

            if actual_token_count < self.mtp_seq_len:
                pad_token = draft_tokens[-1].item()
                padding = torch.full(
                    (self.mtp_seq_len - actual_token_count,),
                    pad_token,
                    dtype=torch.long,
                    device=self.default_device,
                )
                draft_tokens = torch.cat([draft_tokens, padding])

            draft_tokens = draft_tokens.reshape(1, self.mtp_seq_len).to(torch.int32)

            mtp_extra_pos = cur_pos + self.mtp_seq_len
            if mtp_extra_pos < prompt_len:
                mtp_extra_token = int(tokens[0, mtp_extra_pos].item())
            else:
                mtp_extra_token = int(tokens[0, draft_end - 1].item())
            self.decode_layer.set_prefill_mtp_extra_token(mtp_extra_token)

            self.decode_layer.set_prefill_valid_tokens(actual_token_count)

            start_time = time.time()
            self.decode_layer.forward(draft_tokens, with_mtp=True, cur_pos=cur_pos)
            end_time = time.time()
            prefill_time_list.append(end_time - start_time)

            cur_pos += actual_token_count

        cur_pos = prompt_len - 1
        self.set_cur_pos(prompt_len - 1)

        self.decode_layer.set_prefill_valid_tokens(0)

        finished = False
        while cur_pos < total_len - 1 and not finished:
            if cur_pos == prompt_len - 1:
                last_token = tokens[0, prompt_len - 1].item()
                draft_tokens = torch.full(
                    (self.mtp_seq_len,),
                    last_token,
                    dtype=torch.long,
                    device=self.default_device,
                )
                draft_tokens = draft_tokens.reshape(1, self.mtp_seq_len).to(torch.int32)
            else:
                draft_tokens = self.decode_layer.get_next_draft_tokens(0).reshape(
                    1, self.mtp_seq_len
                )

            start_time = time.time()
            self.decode_layer.forward(draft_tokens, with_mtp=True, cur_pos=cur_pos)
            end_time = time.time()
            decode_time_list.append(end_time - start_time)

            num_accepted = self.decode_layer.get_num_accepted(0)
            predicted_tokens = self.decode_layer.get_predicted_tokens(0).flatten()
            decode_accepted_counts.append(num_accepted)

            num_output_tokens = num_accepted
            for i in range(num_output_tokens):
                if cur_pos + 1 + i >= total_len:
                    break
                new_token = int(predicted_tokens[i].item())
                tokens[0, cur_pos + 1 + i] = new_token

                if cur_pos + 1 + i >= prompt_len and print_log:
                    decoded_text = self.tokenizer.decode([new_token], skip_special_tokens=True)
                    print(decoded_text, end="", flush=True)

                if new_token == self.eos_id:
                    finished = True
                    break

            cur_pos += num_accepted

        if print_log:
            print("\n")
            total_tokens = sum(decode_accepted_counts)
            logger.info(f"--Number of forward calls (decode): {len(decode_accepted_counts)}")
            logger.info(f"--Total tokens generated: {total_tokens}")
            if len(decode_accepted_counts) > 0:
                avg_accepted = sum(decode_accepted_counts) / len(decode_accepted_counts)
                min_accepted = min(decode_accepted_counts)
                max_accepted = max(decode_accepted_counts)
                logger.info(
                    f"--Accepted tokens per call: mean={avg_accepted:.2f}, "
                    f"min={min_accepted}, max={max_accepted}"
                )

            if decode_time_list:
                total_decode_time = sum(decode_time_list)
                effective_tps = total_tokens / total_decode_time if total_decode_time > 0 else 0
                avg_time_ms = total_decode_time / len(decode_time_list) * 1000
                logger.info(f"--Avg forward time: {avg_time_ms:.2f}ms")
                logger.info(f"--Effective TPS (with MTP): {effective_tps:.2f} tokens/s")

            print("\n")

        self.decode_layer.reset_sequence()

        completion_tokens = []
        for _, toks in enumerate(tokens.tolist()):
            toks = toks[prompt_len : prompt_len + self.max_new_tokens]
            toks = [t for t in toks if t != -1]
            if self.eos_id in toks:
                toks = toks[: toks.index(self.eos_id)]
            completion_tokens.append(toks)

        decoded_tokens = self.tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)

        return (
            f"{decoded_tokens[0]}\n" if decoded_tokens else "",
            decode_time_list,
            decode_accepted_counts,
            prompt_len,
        )

    def generate_streaming(self, prompt: str):
        """Generate text with streaming output.

        Note: Streaming generation is not yet supported.  This method yields
        the final output once generation completes.
        """
        logger.warning("generate_streaming: Qwen3.6 streaming not yet implemented")
        result, *_ = self.generate(prompt)
        yield result