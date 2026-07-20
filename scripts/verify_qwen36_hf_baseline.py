"""用原始 Hugging Face Qwen3.6 模型跑一个 token，作为 golden forward 参考基准。

由于原始模型约 68GB，使用 ``device_map="auto"`` 分布到多张 GPU 上。
输出：
- prompt tokens
- 第一个生成 token id
- 前 5 个候选 token id 及其 logit

执行：
    cd /public/home/dinggy/yiny/projects/TileRT
    PYTHONPATH=/public/home/dinggy/yiny/projects/TileRT python3 scripts/verify_qwen36_hf_baseline.py
"""
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main():
    model_dir = "/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"
    prompt = (
        "Tell me three jokes:\n\n"
        "1. A dad joke,\n"
        "2. A programmer joke,\n"
        "3. A joke that only makes sense if you've ever tried "
        "to train a large language model.\n"
        "Keep each joke under 15 words."
    )

    print("[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    chat_out = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        thinking=False,
    )
    input_ids = list(chat_out["input_ids"]) if hasattr(chat_out, "input_ids") else list(chat_out)
    print(f"Prompt tokens ({len(input_ids)}): {input_ids[:20]}...")

    print("[2/3] Loading original HF model with device_map='auto'...")
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    print(f"Architecture: {cfg.architectures}")
    print(f"Model type: {cfg.model_type}")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=cfg,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("Model loaded.")

    print("[3/3] Running forward on the last prompt token...")
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        outputs = model(input_tensor, use_cache=True)
    logits = outputs.logits[0, -1, :]  # (vocab_size,)
    next_token_id = int(logits.argmax().item())
    top5 = torch.topk(logits, 5)

    print(f"\nNext token id: {next_token_id}")
    print(f"Next token text: {tokenizer.decode([next_token_id], skip_special_tokens=True)!r}")
    print("\nTop-5 candidates:")
    for i in range(5):
        tok = int(top5.indices[i].item())
        score = float(top5.values[i].item())
        text = tokenizer.decode([tok], skip_special_tokens=True)
        print(f"  {i + 1}. id={tok}, logit={score:.4f}, text={text!r}")

    print("\n=== HF baseline captured ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\n=== FAILED: {exc} ===", file=sys.stderr)
        raise
