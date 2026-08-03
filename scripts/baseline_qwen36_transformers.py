"""Baseline: use transformers AutoModelForCausalLM to generate coherent text."""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"

def main():
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    print("Loading model on 2 GPUs...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        max_memory={0: "60GiB", 1: "60GiB", "cpu": "200GiB"},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()

    prompt = (
        "Tell me three jokes:\n\n"
        "1. A dad joke,\n"
        "2. A programmer joke,\n"
        "3. A joke that only makes sense if you've ever tried "
        "to train a large language model.\n"
        "Keep each joke under 15 words."
    )

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    input_ids = inputs["input_ids"].to(model.device)

    print("Generating...", flush=True)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            max_new_tokens=80,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][input_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print("\n=== Generated text ===")
    print(text)
    print("\n=== Token IDs ===")
    print(generated.tolist())
    return 0

if __name__ == "__main__":
    sys.exit(main())
