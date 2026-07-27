"""Short-prompt generator smoke test to measure load + single decode latency."""
import logging
import os
import sys
import time

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s:%(lineno)d [%(levelname)s]: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    torch.set_num_threads(64)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    weights_dir = "/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"

    logger.info("[1/4] Importing Qwen36Generator...")
    from tilert.models.qwen3_6.model_args import ModelArgsQwen36
    from tilert.models.qwen3_6.generator import Qwen36Generator
    logger.info("[1/4] Import OK")

    logger.info("[2/4] Initializing generator from pretrained weights...")
    model_args = ModelArgsQwen36()
    model_args.max_seq_len = 512
    model_args.max_batch_size = 1

    generator = Qwen36Generator(
        model_args=model_args,
        max_new_tokens=1,
        temperature=1.0,
        model_weights_dir=weights_dir,
        with_mtp=False,
        use_topp=False,
        top_p=0.9,
        top_k=256,
        sampling_seed=42,
    )
    load_start = time.time()
    generator.from_pretrained()
    load_elapsed = time.time() - load_start
    logger.info("[2/4] Generator loaded in %.2f seconds", load_elapsed)

    # Use a tiny fixed prompt to exercise only a few prefill steps plus one decode.
    prompt_tokens = [151644, 872, 198, 26056, 311, 6075, 30, 151645, 198, 151644, 77091, 198]
    prompt = "<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\n"
    logger.info("[3/4] Running generate() with %d prompt tokens...", len(prompt_tokens))
    gen_start = time.time()
    result = generator.generate(prompt, print_log=True, prompt_tokens=prompt_tokens)
    gen_elapsed = time.time() - gen_start
    completion = result[0] if isinstance(result, tuple) else result
    logger.info("[3/4] Generated completion: %r", completion)

    time_list = result[1] if isinstance(result, tuple) and len(result) > 1 else []
    if time_list:
        avg_ms = sum(time_list) / len(time_list) * 1000
        logger.info("[4/4] Per-step average: %.2f ms, total steps: %d", avg_ms, len(time_list))
    logger.info("[4/4] Total generate() time: %.2f seconds", gen_elapsed)
    logger.info("[4/4] Load time: %.2f seconds, generate time: %.2f seconds", load_elapsed, gen_elapsed)

    logger.info("\n=== Short-prompt generator smoke test PASSED ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        logger.exception("\n=== FAILED: %s ===", exc)
        raise
