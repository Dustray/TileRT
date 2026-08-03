"""验证 Qwen3.6-35B-A3B 在 8 张 DCU 上的 golden 文本生成。

本脚本使用 QwenShowHandsLayer（真正的 8 卡 TP8 MoE all-reduce），
从原始 HF checkpoint 加载权重，逐 token greedy decode，输出可读文本。
"""
import logging
import os
import sys
import time
import warnings

import torch
from transformers import AutoTokenizer

from tilert import logger

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s:%(lineno)d [%(levelname)s]: %(message)s",
)

MODEL_DIR = "/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"
PROMPT = "How many r's are in the word strawberry? Think step by step."
MAX_NEW_TOKENS = 40


def main():
    torch.set_num_threads(64)
    torch.manual_seed(42)

    logger.info("[1/4] Loading tokenizer from %s", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    chat_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_tokens = tokenizer.encode(chat_input, add_special_tokens=False)
    logger.info("Prompt tokens: %d -> %s...", len(prompt_tokens), prompt_tokens[:10])

    logger.info("[2/4] Building QwenShowHandsLayer on %d devices...", torch.cuda.device_count())
    from tilert.models.qwen3_6.model_args import ModelArgsQwen36
    from tilert.models.qwen3_6.modules.end2end import QwenShowHandsLayer

    model_args = ModelArgsQwen36()
    model_args.max_seq_len = 512
    model_args.max_batch_size = 1

    layer = QwenShowHandsLayer(
        model_args=model_args,
        model_path=MODEL_DIR,
        with_mtp=False,
        temperature=1.0,
        top_p=0.9,
        top_k=256,
        use_topp=False,
    )

    logger.info("[3/4] Loading HF pretrained weights on 8 devices...")
    layer.from_pretrained(MODEL_DIR)
    logger.info("[3/4] Weights loaded")

    logger.info("[4/4] Running greedy generation with golden_forward...")
    generated: list[int] = []
    eos_id = tokenizer.eos_token_id

    t0 = time.time()
    cur_pos = 0
    # QwenShowHandsLayer.forward 一次只接受一个 token；prompt 逐 token feed。
    for token_id in prompt_tokens:
        tid = torch.tensor(token_id, dtype=torch.int32)
        layer.forward(tid, with_mtp=False, cur_pos=cur_pos)
        cur_pos += 1

    for _ in range(MAX_NEW_TOKENS):
        tid = torch.tensor(layer.get_next_token(device_id=0), dtype=torch.int32)
        layer.forward(tid, with_mtp=False, cur_pos=cur_pos)
        next_id = layer.get_next_token(device_id=0)
        generated.append(next_id)
        if next_id == eos_id:
            break
        cur_pos += 1

    dt = time.time() - t0
    n_decode = len(generated)
    ms_per_tok = 1000 * dt / max(1, n_decode + len(prompt_tokens))
    logger.info(
        "Generated %d tokens in %.2fs (%.1f ms/tok, prompt %d tokens)",
        n_decode, dt, ms_per_tok, len(prompt_tokens),
    )

    text = tokenizer.decode(generated, skip_special_tokens=False)
    print("\n===== GENERATED TEXT =====\n", text)
    print("===== TOKEN IDS =====\n", generated)

    layer.cleanup()
    logger.info("\n=== 8-GPU text-generation test PASSED ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        logger.exception("\n=== FAILED: %s ===", exc)
        raise
