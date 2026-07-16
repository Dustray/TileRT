import json
from safetensors.torch import load_file
import sys
idx_path = '/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master/model.safetensors.index.json'
with open(idx_path) as f:
    wm = json.load(f)['weight_map']
base = '/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master/'
keys = ['model.language_model.layers.0.mlp.experts.gate_up_proj']
files = set(wm[k] for k in keys)
weights = {}
for fn in files:
    weights.update(load_file(base + fn, device='cpu'))
for k in keys:
    print(k, tuple(weights[k].shape), weights[k].dtype, file=sys.stderr)
