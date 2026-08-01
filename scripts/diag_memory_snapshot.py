"""Capture CUDA memory snapshots before/after forward to find growth."""
import gc
import warnings

import torch
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

MODEL_DIR = "/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"
PROMPT = "How many r's are in the word strawberry? Think step by step."


def main():
    torch.set_num_threads(64)
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

    # Device 0 only for snapshot.
    torch.cuda.synchronize(0)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    # Record memory history.
    torch.cuda.memory._record_memory_history(max_entries=100000)
    with torch.no_grad():
        for i, tid in enumerate(tokens[:3]):
            torch.cuda.synchronize(0)
            before = torch.cuda.memory_allocated(0)
            layer.forward(torch.tensor(tid, dtype=torch.int32), with_mtp=False, cur_pos=i)
            torch.cuda.synchronize(0)
            after = torch.cuda.memory_allocated(0)
            print(f"token {i}: allocated {before/1e9:.2f} -> {after/1e9:.2f} GB (+{(after-before)/1e9:.2f} GB)")
            if i == 0:
                # Dump snapshot after first token to see what survived.
                snapshot_path = "/tmp/after_token0.pickle"
                torch.cuda.memory._dump_snapshot(snapshot_path)
                print(f"dumped snapshot to {snapshot_path}")
    torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()
