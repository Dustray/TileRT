"""Find which tensor sizes grow per forward."""
import gc
import logging
import warnings

import torch
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(name)s:%(lineno)d [%(levelname)s]: %(message)s")

MODEL_DIR = "/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"
PROMPT = "How many r's are in the word strawberry? Think step by step."


def tensor_sizes(did):
    gc.collect()
    torch.cuda.empty_cache()
    sizes = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.device.type == "cuda" and obj.device.index == did:
                key = f"{tuple(obj.shape)}_{obj.dtype}_{obj.storage().data_ptr()}"
                sizes[key] = obj.numel() * obj.element_size()
        except Exception:
            pass
    return sizes


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
    prompt_tokens = tokenizer.encode(chat, add_special_tokens=False)

    base = tensor_sizes(0)
    print(f"After load unique tensors: {len(base)} total bytes: {sum(base.values())/1e9:.2f} GB")
    big_after_load = {k: v for k, v in base.items() if v > 50e6}
    print("big tensors after load (>50MB):", sorted(big_after_load.values(), reverse=True)[:10])

    with torch.no_grad():
        layer.forward(torch.tensor(prompt_tokens[0], dtype=torch.int32), with_mtp=False, cur_pos=0)
    after1 = tensor_sizes(0)
    diff = {k: v for k, v in after1.items() if k not in base}
    print(f"After 1 token unique tensors: {len(after1)} total bytes: {sum(after1.values())/1e9:.2f} GB")
    print("new tensors after 1 token (>50MB):", sorted([v for v in diff.values() if v > 50e6], reverse=True)[:10])
    for k, v in sorted(diff.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {v/1e9:.3f} GB")

    with torch.no_grad():
        layer.forward(torch.tensor(prompt_tokens[1], dtype=torch.int32), with_mtp=False, cur_pos=1)
    after2 = tensor_sizes(0)
    diff2 = {k: v for k, v in after2.items() if k not in after1}
    print(f"After 2 tokens unique tensors: {len(after2)} total bytes: {sum(after2.values())/1e9:.2f} GB")
    print("new tensors after 2nd token (>50MB):", sorted([v for v in diff2.values() if v > 50e6], reverse=True)[:10])
    for k, v in sorted(diff2.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {v/1e9:.3f} GB")

    layer.cleanup()


if __name__ == "__main__":
    main()
