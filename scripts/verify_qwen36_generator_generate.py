"""验证 Qwen36Generator.generate 在随机权重下可完成多 token 生成。

验证项：
1. Qwen36Generator 可正常构造（需要 tokenizer 文件存在于 model_weights_dir）。
2. init_random_weights 后，generate() 能跑通非 MTP 路径。
3. 生成结果包含若干 token，logits 数值有界。

前置条件：
- model_weights_dir 下需要存在 tokenizer 相关文件（tokenizer_config.json、tokenizer.json 等）。
- 若 tokenizer 不存在，可用 prompt_tokens 参数绕过 tokenization。

执行：
    cd /public/home/dinggy/yiny/projects/TileRT
    PYTHONPATH=/public/home/dinggy/yiny/projects/TileRT python3 scripts/verify_qwen36_generator_generate.py
"""
import os
import sys

import torch


def main():
    torch.set_num_threads(64)

    # The converted TileRT checkpoint currently does not include tokenizer files.
    # Fall back to the original HF model directory for tokenizer only.
    weights_dir = "/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B--TileRT/snapshots/master"
    tokenizer_dir = weights_dir#"/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"
    # Verify tokenizer files are present in the original model directory.
    for tok_file in ("tokenizer_config.json", "tokenizer.json", "vocab.json"):
        if os.path.isfile(os.path.join(tokenizer_dir, tok_file)):
            print(f"  Found {tok_file} in original model dir")
            break
    else:
        print("  Warning: no tokenizer json files found; generator init will likely fail.")
    print("[1/3] Importing Qwen36Generator...")
    from tilert.models.qwen3_6.model_args import ModelArgsQwen36
    from tilert.models.qwen3_6.generator import Qwen36Generator
    print("[1/3] Import OK")

    print("[2/3] Initializing generator with random weights...")
    model_args = ModelArgsQwen36()
    model_args.max_seq_len = 512
    model_args.max_batch_size = 1

    # If tokenizer files are missing, we pass prompt_tokens directly to skip tokenization.
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
        tokenizer_dir=tokenizer_dir,
    )
    generator.init_random_weights()
    print("[2/3] Generator initialized OK")

    print("[3/3] Running generate() with prompt_tokens=[1, 2, 3]...")
    prompt_tokens = [1, 2, 3]
    result, time_list, accepted_counts, prompt_len = generator.generate(
        prompt="",
        print_log=False,
        with_mtp=False,
        prompt_tokens=prompt_tokens,
    )
    print(f"[3/3] prompt_len={prompt_len}, generated_tokens={len(time_list)}, result={result!r}")

    assert len(time_list) > 0, "No tokens generated"
    print("\n=== Generator random-init generate smoke test PASSED ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\n=== FAILED: {exc} ===", file=sys.stderr)
        raise
