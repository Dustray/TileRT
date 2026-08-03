"""Trace referrers of tensors that survive across forward calls."""
import gc
import warnings

import torch
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

MODEL_DIR = "/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"
PROMPT = "How many r's are in the word strawberry? Think step by step."


def live_tensors(did):
    out = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.device.type == "cuda" and obj.device.index == did:
                out[id(obj)] = obj
        except Exception:
            pass
    return out


def trace_referrers(t, depth=0, max_depth=4, seen=None):
    if seen is None:
        seen = set()
    if depth > max_depth or id(t) in seen:
        return []
    seen.add(id(t))
    refs = gc.get_referrers(t)
    # Remove the frame object of this function to avoid recursion noise.
    refs = [r for r in refs if r is not seen and not isinstance(r, type(None))]
    lines = []
    for r in refs:
        name = getattr(r, "__name__", None) or type(r).__name__
        detail = ""
        if isinstance(r, torch.nn.Module):
            detail = f" class={r.__class__.__name__}"
        elif isinstance(r, dict):
            keys = [k for k, v in r.items() if v is t]
            detail = f" dict keys={keys[:5]}"
        elif isinstance(r, list):
            idxs = [i for i, v in enumerate(r) if v is t]
            detail = f" list idxs={idxs[:5]}"
        elif isinstance(r, tuple):
            idxs = [i for i, v in enumerate(r) if v is t]
            detail = f" tuple idxs={idxs[:5]}"
        lines.append((depth, f"{'  ' * depth}{name}{detail}"))
        if isinstance(r, (dict, list, tuple, torch.nn.Module)):
            lines.extend(trace_referrers(r, depth + 1, max_depth, seen))
    return lines


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

    base = live_tensors(0)
    print(f"After load: {len(base)} live tensors on dev0, allocated={torch.cuda.memory_allocated(0)/1e9:.2f} GB")

    with torch.no_grad():
        layer.forward(torch.tensor(tokens[0], dtype=torch.int32), with_mtp=False, cur_pos=0)
    after = live_tensors(0)
    new = [t for tid, t in after.items() if tid not in base]
    new.sort(key=lambda x: -x.numel() * x.element_size())
    print(f"\nAfter token 0: {len(after)} live tensors, allocated={torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    print(f"New tensors count: {len(new)}")
    for t in new[:20]:
        size = t.numel() * t.element_size()
        print(f"\nNEW tensor shape={tuple(t.shape)} dtype={t.dtype} device={t.device} size={size/1e6:.2f} MB data_ptr={t.data_ptr()}")
        for depth, line in trace_referrers(t, max_depth=3):
            print(line)


if __name__ == "__main__":
    main()
