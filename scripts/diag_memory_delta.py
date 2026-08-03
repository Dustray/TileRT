"""Capture per-token CUDA memory snapshot and compare block counts."""
import gc
import os
import pickle
import warnings

import torch

warnings.filterwarnings("ignore")

MODEL_DIR = "/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"
PROMPT = "How many r's are in the word strawberry? Think step by step."


def summarize_snapshot(path):
    with open(path, "rb") as f:
        snap = pickle.load(f)
    dev0 = snap["device_traces"][0]
    allocs = [b for b in dev0 if b["action"] == "alloc"]
    active_addrs = {b["addr"] for b in dev0 if b["action"] == "alloc"}
    freed = {b["addr"] for b in dev0 if b["action"] == "free"}
    active = active_addrs - freed
    from collections import Counter
    sizes = Counter(b["size"] for b in allocs)
    return active, sizes


def main():
    torch.set_num_threads(64)
    from transformers import AutoTokenizer
    from tilert.models.qwen3_6.model_args import ModelArgsQwen36
    from tilert.models.qwen3_6.modules.end2end import QwenShowHandsLayer

    model_args = ModelArgsQwen36()
    model_args.max_seq_len = 512
    model_args.max_batch_size = 1
    layer = QwenShowHandsLayer(model_args=model_args, model_path=MODEL_DIR, with_mtp=False)
    layer.from_pretrained(MODEL_DIR)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    chat = tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False)
    tokens = tokenizer.encode(chat, add_special_tokens=False)

    def snapshot_path(i):
        return f"/tmp/token_{i}_snap.pickle"

    torch.cuda.memory._record_memory_history(max_entries=100000)
    torch.cuda.synchronize()

    for i, tok in enumerate(tokens):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"\n=== token {i}: allocated={torch.cuda.memory_allocated(0)/1e9:.2f} GB reserved={torch.cuda.memory_reserved(0)/1e9:.2f} GB")
        with torch.no_grad():
            layer.forward(torch.tensor(tok, dtype=torch.int32), with_mtp=False, cur_pos=i)
        torch.cuda.synchronize()
        snap = torch.cuda.memory._snapshot()
        with open(snapshot_path(i), "wb") as f:
            pickle.dump(snap, f)
        torch.cuda.memory._record_memory_history()

    for i in range(len(tokens)):
        active, sizes = summarize_snapshot(snapshot_path(i))
        print(f"\ntoken {i}: active blocks={len(active)} top sizes={sizes.most_common(10)}")


if __name__ == "__main__":
    main()
