"""使用 HF fallback 快速验证 Qwen36Generator 能加载并生成文本。

TILERT_QWEN36_HF_FALLBACK=1 会走 transformers AutoModelForCausalLM 路径，
绕过 TileRT golden forward 的 8-DCU all-reduce 性能问题，先确认端到端
生成能力。
"""
import logging
import os
import sys

import faulthandler
import torch

faulthandler.enable()

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s:%(lineno)d [%(levelname)s]: %(message)s",
)

from tilert import logger  # noqa: E402

# 强制走 HF fallback（见 end2end.py 中的 _hf_fallback_enabled）
os.environ["TILERT_QWEN36_HF_FALLBACK"] = "1"
# golden 路径的 forward_max_seq_len 对 fallback 不影响，但保留安全值
os.environ["TILERT_QWEN36_FORWARD_MAX_SEQ_LEN"] = "1"


def main():
    torch.set_num_threads(64)

    weights_dir = "/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"

    logger.info("[1/3] Importing Qwen36Generator...")
    from tilert.models.qwen3_6.model_args import ModelArgsQwen36
    from tilert.models.qwen3_6.generator import Qwen36Generator

    logger.info("[1/3] Import OK")

    logger.info("[2/3] Initializing generator with HF fallback...")
    model_args = ModelArgsQwen36()
    model_args.max_seq_len = 512
    model_args.max_batch_size = 1

    generator = Qwen36Generator(
        model_args=model_args,
        max_new_tokens=20,
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

    logger.info("[3/3] Running generate() with HF fallback...")
    logger.info("Prompt: %s", prompt)
    logger.info("Completion:")
    result = generator.generate(prompt, print_log=True)
    completion = result[0] if isinstance(result, tuple) else result

    assert completion and len(completion.strip()) > 0, "Empty completion"
    logger.info("\n=== Generated completion ===")
    logger.info(completion)
    logger.info("\n=== Generator HF-fallback smoke test PASSED ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        logger.exception("\n=== FAILED: %s ===", exc)
        raise
