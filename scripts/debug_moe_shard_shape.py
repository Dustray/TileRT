"""Debug: print MoE sharded shapes for one HF layer."""
import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, "/public/home/dinggy/yiny/projects/TileRT")

from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.models.qwen3_6.modules.transformer_stack import QwenTransformerStack
from tilert.models.qwen3_6.modules.moe import QwenMoeBlock

MODEL_DIR = "/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"
LAYER_IDX = 0


def load_layer_state(model_dir: str, layer_idx: int) -> dict[str, torch.Tensor]:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as f:
        weight_index = json.load(f)["weight_map"]
    prefix = f"model.language_model.layers.{layer_idx}."
    needed_files = sorted({weight_index[k] for k in weight_index if k.startswith(prefix)})
    out: dict[str, torch.Tensor] = {}
    for fname in needed_files:
        with safe_open(os.path.join(model_dir, fname), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    out[key[len(prefix):]] = f.get_tensor(key)
    return out


def main():
    args = ModelArgsQwen36()
    args.max_seq_len = 512
    args.max_batch_size = 1

    stack = QwenTransformerStack(args, device_id=0, num_devices=8)
    block = stack.exec_seq[LAYER_IDX]
    raw = load_layer_state(MODEL_DIR, LAYER_IDX)
    print("Raw keys:", sorted(raw.keys())[:20])
    print("mlp.experts.down_proj shape:", raw.get("mlp.experts.down_proj").shape if "mlp.experts.down_proj" in raw else None)
    print("mlp.experts.gate_up_proj shape:", raw.get("mlp.experts.gate_up_proj").shape if "mlp.experts.gate_up_proj" in raw else None)

    sharded = block.device_sharding(raw)
    print("\nSharded aliases:")
    for alias, tensor in sorted(sharded.items()):
        print(f"  {alias}: {tuple(tensor.shape)}  size={tensor.numel()}")

    from tilert.models.qwen3_6.modules.hf_source_loader import _unshard_to_per_device
    for did in [0, 1]:
        per_dev = _unshard_to_per_device(sharded, did, 8)
        print(f"\nDevice {did}:")
        for alias, tensor in sorted(per_dev.items()):
            print(f"  {alias}: {tuple(tensor.shape)}  size={tensor.numel()}")


if __name__ == "__main__":
    main()
