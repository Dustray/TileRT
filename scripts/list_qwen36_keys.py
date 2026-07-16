import json
with open('/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master/model.safetensors.index.json') as f:
    wm = json.load(f)['weight_map']
keys = [k for k in wm if 'layers.0.mlp.experts' in k]
print('\n'.join(keys))
print("--- all .gate_proj .up_proj .down_proj for layer 0 ---")
for k in sorted(wm):
    if 'layers.0.mlp' in k:
        print(k)
