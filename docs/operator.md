
## 权重转换 

For **Qwen3.6-35B-A3B**

```toml
python -m tilert.models.preprocess.weight_converter \
  --model_type qwen3_6 \
  --model_dir "/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master/" \
  --save_dir "/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B--TileRT/snapshots/master/"
```

```toml
[weights]
qwen3_6_35b_a3b = "/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B--TileRT/snapshots/master/"
```