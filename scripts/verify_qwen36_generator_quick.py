"""快速验证 Qwen36Generator 在真实权重下输出是否合理。"""
import os
import sys

import torch


def main():
    torch.set_num_threads(64)
    weights_dir = "/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B--TileRT/snapshots/master"

    from tilert.models.qwen3_6.model_args import ModelArgsQwen36
    from tilert.models.qwen3_6.generator import Qwen36Generator

    model_args = ModelArgsQwen36()
    model_args.max_seq_len = 128
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

    prompt = "Hello, how are you?"
    print("Prompt:", prompt)
    result = generator.generate(prompt, print_log=True)
    completion = result[0] if isinstance(result, tuple) else result
    print("\nCompletion:", repr(completion))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\n=== FAILED: {exc} ===", file=sys.stderr)
        raise
