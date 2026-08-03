"""按照 README 官方示例风格验证 Qwen36Generator.generate。

验证项：
1. Qwen36Generator 可正常构造（需要 tokenizer 文件存在于 model_weights_dir）。
2. 使用真实转换后的 TileRT 权重（或随机权重）跑通 generate(prompt)。
3. 输出与 README 中 DeepSeek 示例类似的“three jokes”提示，验证生成结果不为空。

前置条件：
- model_weights_dir 下需要存在 tokenizer 相关文件（转换脚本已自动复制）。
- 若使用随机权重，请把 ``use_random_weights`` 设为 True；若使用真实权重，请确保
  ``weights_dir`` 指向转换后的 TileRT checkpoint 目录。

执行：
    cd /public/home/panyq/yiny/projects/TileRT
    PYTHONPATH=/public/home/panyq/yiny/projects/TileRT python3 scripts/verify_qwen36_generator_official_prompt.py
"""
import logging
import os
import sys

import torch
import faulthandler
faulthandler.enable()
from tilert import logger
import torch.autograd.profiler as profiler

logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s:%(lineno)d [%(levelname)s]: %(message)s",
)


def main():
    torch.set_num_threads(64)

    use_random_weights = False
    weights_dir = "/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"

    # Sanity check for tokenizer files (copied by weight_converter now).
    required_tok_files = ("tokenizer_config.json", "tokenizer.json", "vocab.json")
    found = [f for f in required_tok_files if os.path.isfile(os.path.join(weights_dir, f))]
    if found:
        logger.info("  Found tokenizer files: %s", found)
    else:
        logger.warning("  Warning: no tokenizer json files found; generator init will likely fail.")

    logger.info("[1/3] Importing Qwen36Generator...")
    from tilert.models.qwen3_6.model_args import ModelArgsQwen36
    from tilert.models.qwen3_6.generator import Qwen36Generator
    logger.info("[1/3] Import OK")

    logger.info("[2/3] Initializing generator...")
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

    if use_random_weights:
        generator.init_random_weights()
        logger.info("[2/3] Generator initialized with random weights OK")
    else:
        generator.from_pretrained()
        logger.info("[2/3] Generator initialized from pretrained weights OK")

    prompt = (
        "hello"
    )

    logger.info("[3/3] Running generate() with official README prompt...")
    logger.info("Prompt: %s", prompt)
    logger.info("Completion:")
    result = generator.generate(prompt, print_log=True)
    # Qwen36Generator.generate returns a tuple: (completion_text, time_list, [], prompt_len)
    completion = result[0] if isinstance(result, tuple) else result

    assert completion and len(completion.strip()) > 0, "Empty completion"
    logger.info("\n=== Generator official-prompt smoke test PASSED ===")
    return 0


if __name__ == "__main__":
    try:
        with torch.autograd.profiler.profile(enabled=False, use_device="cuda", record_shapes=False, profile_memory=False) as prof:
            main()
        if prof is not None:
            print(prof.table())
            prof.export_chrome_trace('./official_test.json')
    except Exception as exc:
        logger.exception("\n=== FAILED: %s ===", exc)
        raise
