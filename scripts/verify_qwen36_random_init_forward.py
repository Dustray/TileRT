"""验证 QwenShowHandsLayer 在随机初始化下的 golden 路径。

验证项：
1. ModelArgsQwen36 可正常实例化。
2. QwenShowHandsLayer 随机初始化不触发 import/runtime 错误。
3. forward(token_id) 在 golden 路径下返回结果，logits 形状正确且数值有界。
4. 采样得到的 next token 为合法 int。

执行：
    cd /public/home/dinggy/yiny/projects/TileRT
    PYTHONPATH=/public/home/dinggy/yiny/projects/TileRT python3 scripts/verify_qwen36_random_init_forward.py
"""
import sys

import torch


def main():
    torch.set_num_threads(64)

    print("[1/4] Importing Qwen3.6 modules...")
    from tilert.models.qwen3_6.model_args import ModelArgsQwen36
    from tilert.models.qwen3_6.modules.end2end import QwenShowHandsLayer
    print("[1/4] Import OK")

    print("[2/4] Building ModelArgsQwen36 (max_seq_len=512 for smoke test)...")
    model_args = ModelArgsQwen36()
    model_args.max_seq_len = 512
    model_args.max_batch_size = 1
    print(f"[2/4] {model_args}")

    print("[3/4] Initializing QwenShowHandsLayer with random weights on 8 devices...")
    layer = QwenShowHandsLayer(
        model_args=model_args,
        model_path="",
        with_mtp=False,
        temperature=1.0,
        top_p=0.9,
        top_k=256,
        use_topp=False,
    )
    layer.init_random_weights()
    print("[3/4] Random weights initialized OK")

    print("[4/4] Running forward(token_id=100, cur_pos=0) on golden path...")
    token_id = torch.tensor(100, dtype=torch.int32)
    results = layer.forward(token_id, with_mtp=False, cur_pos=0)
    assert len(results) == layer.num_devices, f"Expected {layer.num_devices} results, got {len(results)}"

    next_token = layer.get_next_token(device_id=0)
    logits = layer.get_logits(device_id=0)
    print(f"[4/4] next_token={next_token}, logits.shape={tuple(logits.shape)}, logits.finite={logits.isfinite().all().item()}")

    # LOGITS_OUT is (max_batch_size, max_seq_len, vocab_size // num_devices).
    # The first two dims are padded to the max layout sizes.
    assert logits.shape[2] * layer.num_devices == model_args.vocab_size, (
        f"Logits last dim {logits.shape[2]} not vocab_size/num_devices"
    )
    assert logits.isfinite().all(), "Logits contain NaN/Inf"
    assert 0 <= next_token < model_args.vocab_size, f"Invalid next_token {next_token}"

    layer.cleanup()
    print("\n=== Random-init golden forward smoke test PASSED ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\n=== FAILED: {exc} ===", file=sys.stderr)
        raise
