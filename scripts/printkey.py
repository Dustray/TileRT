import json
from pathlib import Path
root = Path('/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master')
idx = json.load(open(root / 'model.safetensors.index.json'))
keys = sorted(idx['weight_map'].keys())
for k in keys:
    print(k)