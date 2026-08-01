"""Check whether per-token memory growth persists under no_grad."""
import gc
import warnings

import torch

warnings.filterwarnings("ignore")

MODEL_DIR = "/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"
PROMPT = "How many r's are in the word strawberry? Think step by step."


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

    print(f"After load: allocated={torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    for i, tok in enumerate(tokens[:25]):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated(0)
        with torch.no_grad():
            layer.forward(torch.tensor(tok, dtype=torch.int32), with_mtp=False, cur_pos=i)
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated(0)
        print(f"token {i}: allocated_before={before/1e9:.3f} GB allocated_after={after/1e9:.3f} GB delta={(after-before)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
