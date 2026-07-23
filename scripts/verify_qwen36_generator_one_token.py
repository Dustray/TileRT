"""验证 Qwen36Generator 使用真实 HF 权重能完成一次 forward 并产出合法 token。

与完整官方 prompt 测试不同，这里只跑 1 个新 token（prefill 后接一次 decode），
用于在合理时间内确认 generate() 路径、logits 分布和采样逻辑正确。
"""
import logging
import os
import sys
import time

import torch

logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s:%(lineno)d [%(levelname)s]: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    torch.set_num_threads(64)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    weights_dir = "/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"

    logger.info("[1/3] Importing Qwen36Generator...")
    from tilert.models.qwen3_6.model_args import ModelArgsQwen36
    from tilert.models.qwen3_6.generator import Qwen36Generator
    logger.info("[1/3] Import OK")

    logger.info("[2/3] Initializing generator from pretrained weights...")
    model_args = ModelArgsQwen36()
    model_args.max_seq_len = 512
    model_args.max_batch_size = 1

    generator = Qwen36Generator(
        model_args=model_args,
        max_new_tokens=1,  # 只生成 1 个新 token
        temperature=1.0,
        model_weights_dir=weights_dir,
        with_mtp=False,
        use_topp=False,
        top_p=0.9,
        top_k=256,
        sampling_seed=42,
    )
    generator.from_pretrained()
    logger.info("[2/3] Generator initialized from pretrained weights OK")

    prompt = (
        "Tell me three jokes:\n\n"
        "1. A dad joke,\n"
        "2. A programmer joke,\n"
        "3. A joke that only makes sense if you've ever tried "
        "to train a large language model.\n"
        "Keep each joke under 15 words."
    )

    logger.info("[3/3] Running generate() for exactly 1 new token...")
    logger.info("Prompt: %s", prompt)
    start = time.time()
    result = generator.generate(prompt, print_log=True)
    elapsed = time.time() - start
    completion = result[0] if isinstance(result, tuple) else result
    logger.info("Generated 1 token in %.2f seconds", elapsed)
    logger.info("Completion so far: %r", completion)

    assert completion is not None
    logger.info("\n=== Generator one-token smoke test PASSED ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        logger.exception("\n=== FAILED: %s ===", exc)
        raise
