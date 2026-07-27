"""Diagnose Qwen3.6 generator parallelism and per-device timing.

This script runs a very short generation (a few tokens) and prints:
- per-step wall time
- per-device compute time (cuda event elapsed)
- time spent in all-reduce/all-gather (if any)
- whether torch.distributed is initialized
- the first few layer latencies on each device
"""
import logging
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s:%(lineno)d [%(levelname)s]: %(message)s",
)
from tilert import logger


def main():
    torch.set_num_threads(64)
    use_random_weights = True  # Random is faster to load and shows the same path.
    weights_dir = "/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"

    from tilert.models.qwen3_6.model_args import ModelArgsQwen36
    from tilert.models.qwen3_6.generator import Qwen36Generator

    model_args = ModelArgsQwen36()
    model_args.max_seq_len = 512
    model_args.max_batch_size = 1

    generator = Qwen36Generator(
        model_args=model_args,
        max_new_tokens=5,
        temperature=1.0,
        model_weights_dir=weights_dir,
        with_mtp=False,
        use_topp=False,
        top_p=0.9,
        top_k=256,
        sampling_seed=42,
    )
    if use_random_weights:
        generator.init_random_weights()
    else:
        generator.from_pretrained()

    decode_layer = generator.decode_layer
    num_devices = decode_layer.num_devices
    print(f"\n=== Environment ===")
    print(f"num_devices (torch.cuda.device_count): {num_devices}")
    print(f"torch.distributed.is_initialized: {torch.distributed.is_initialized()}")
    if torch.distributed.is_initialized():
        print(f"world_size: {torch.distributed.get_world_size()}")
        print(f"rank: {torch.distributed.get_rank()}")

    prompt = "Hello"
    print(f"\n=== Running 5-token generation ===")
    result = generator.generate(prompt, print_log=True)
    completion, time_list, _, prompt_len = result
    print(f"\n=== Summary ===")
    print(f"prompt_len={prompt_len}, generated_tokens={len(time_list)}")
    if time_list:
        print(f"first_token_ms={time_list[0]*1000:.2f}, last_token_ms={time_list[-1]*1000:.2f}")
        print(f"avg_ms={sum(time_list)/len(time_list)*1000:.2f}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        logger.exception("FAILED: %s", exc)
        raise
