"""纯 transformers greedy baseline，用于与 TileRT HF fallback 对比输出。"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"
MAX_NEW_TOKENS = 20
PROMPT = (
    "Tell me three jokes:\\n\\n"
    "1. A dad joke,\\n"
    "2. A programmer joke,\\n"
    "3. A joke that only makes sense if you've ever tried "
    "to train a large language model.\\n"
    "Keep each joke under 15 words."
)
SEED = 42


def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print("[1/2] Loading tokenizer and model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="balanced",
        max_memory={0: "48GiB", 1: "48GiB", "cpu": "300GiB"},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    load_time = time.time() - t0
    print(f"[1/2] Model loaded in {load_time:.1f}s on {model.device}", flush=True)

    chat_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(chat_input, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    print(f"[2/2] Prompt length: {prompt_len}", flush=True)

    t0 = time.time()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )
    gen_time = time.time() - t0

    output_text = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)
    total_new = generated.shape[1] - prompt_len
    print(f"\n=== Generated ({total_new} new tokens, {gen_time:.1f}s, {total_new/gen_time:.2f} tok/s) ===")
    print(output_text)
    print("\n=== Token IDs ===")
    print(generated[0, prompt_len:].tolist())


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
