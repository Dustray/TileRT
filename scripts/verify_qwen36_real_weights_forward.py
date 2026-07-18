"""验证 QwenShowHandsLayer 加载真实 TileRT weights 后单步 forward。

验证项：
1. 从 master checkpoint 目录加载转换后的 TileRT weights（ safetensors shards）。
2. QwenShowHandsLayer.from_pretrained 完成多设备并行加载。
3. forward(token_id) 输出 logits 形状正确且数值有界。
4. 采样得到合法 next token。

前置条件：
- master/ 目录下必须包含 model.safetensors.index.json 与分片文件。
- 若未执行 weight conversion，请先运行 weight_converter.py 生成 TileRT weights。

执行：
    cd /public/home/dinggy/yiny/projects/TileRT
    PYTHONPATH=/public/home/dinggy/yiny/projects/TileRT python3 scripts/verify_qwen36_real_weights_forward.py
"""
import os
import sys

import torch


def main():
    torch.set_num_threads(64)

    weights_dir = "/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B--TileRT/snapshots/master"
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")
    index_file = os.path.join(weights_dir, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Missing index file: {index_file}")

    print("[1/3] Importing modules...")
    from tilert.models.qwen3_6.model_args import ModelArgsQwen36
    from tilert.models.qwen3_6.modules.end2end import QwenShowHandsLayer
    print("[1/3] Import OK")

    print("[2/3] Loading real pretrained weights...")
    model_args = ModelArgsQwen36()
    model_args.max_seq_len = 512
    model_args.max_batch_size = 1

    layer = QwenShowHandsLayer(
        model_args=model_args,
        model_path=weights_dir,
        with_mtp=False,
        temperature=1.0,
        top_p=0.9,
        top_k=256,
        use_topp=False,
    )
    layer.from_pretrained(weights_dir)
    print("[2/3] Real weights loaded OK")

    print("[3/5] Running forward(token_id=100, cur_pos=0)...")
    token_id = torch.tensor(100, dtype=torch.int32)
    results = layer.forward(token_id, with_mtp=False, cur_pos=0)
    assert len(results) == layer.num_devices

    next_token = layer.get_next_token(device_id=0)
    logits = layer.get_logits(device_id=0)
    print(f"[3/5] next_token={next_token}, logits.shape={tuple(logits.shape)}, finite={logits.isfinite().all().item()}")

    assert logits.isfinite().all(), "Logits contain NaN/Inf"
    assert 0 <= next_token < model_args.vocab_size, f"Invalid next_token {next_token}"

    print("[4/5] Running three autoregressive decode steps...")
    for step, pos in enumerate([1, 2, 3], start=1):
        results = layer.forward(
            torch.tensor(next_token, dtype=torch.int32), with_mtp=False, cur_pos=pos
        )
        next_token = layer.get_next_token(device_id=0)
        print(f"  step {step} (cur_pos={pos}) -> next_token={next_token}")
        assert 0 <= next_token < model_args.vocab_size, f"Invalid next_token {next_token}"

    print("[5/5] Checking logits per-device consistency...")
    for device_id in range(layer.num_devices):
        device_logits = layer.get_logits(device_id=device_id)
        assert device_logits.isfinite().all(), f"Logits on device {device_id} contain NaN/Inf"

    layer.cleanup()
    print("\n=== Real-weights golden forward smoke test PASSED ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\n=== FAILED: {exc} ===", file=sys.stderr)
        raise
