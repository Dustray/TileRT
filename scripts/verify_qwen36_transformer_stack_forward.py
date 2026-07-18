"""验证 QwenTransformerStack 的 golden 路径。

验证项：
1. 40 层异构栈（30 DeltaNet + 10 GatedAttention）可正常构造。
2. 共享 cached_ffn_ops 模式下仍可进行随机初始化。
3. golden_forward 对 seq_len=1/2 输出形状正确且数值有界。

执行：
    cd /public/home/dinggy/yiny/projects/TileRT
    PYTHONPATH=/public/home/dinggy/yiny/projects/TileRT python3 scripts/verify_qwen36_transformer_stack_forward.py
"""
import sys

import torch


def main():
    torch.set_num_threads(64)

    print("[1/3] Importing modules...")
    from tilert.models.qwen3_6.model_args import ModelArgsQwen36
    from tilert.models.qwen3_6.modules.transformer_stack import QwenTransformerStack
    from tilert.models.qwen3_6.modules.moe import QwenMoeBlock
    from tilert.models.utils import precompute_freqs_cis
    print("[1/3] Import OK")

    print("[2/3] Building QwenTransformerStack with shared cached_ffn_ops...")
    model_args = ModelArgsQwen36()
    model_args.max_seq_len = 512
    model_args.max_batch_size = 1

    device_id = 0
    num_devices = 8

    # Build a single shared QwenMoeBlock to mimic QwenShowHandsLayer's injection.
    shared_ffn = QwenMoeBlock(model_args=model_args, device_id=device_id, num_devices=num_devices)
    shared_ffn.init_random_weights(device=f"cuda:{device_id}")
    cached_ffn_ops = [shared_ffn] * model_args.n_layers

    stack = QwenTransformerStack(
        model_args=model_args,
        device_id=device_id,
        num_devices=num_devices,
        cached_ffn_ops=cached_ffn_ops,
    )
    stack.init_random_weights()
    print("[2/3] Stack initialized OK")

    print("[3/3] Running golden_forward for seq_len=1 and seq_len=2...")
    freqs_cis_real = torch.view_as_real(precompute_freqs_cis(model_args))
    # precompute_freqs_cis returns complex [max_seq_len, rope_dim/2];
    # view_as_real yields [max_seq_len, rope_dim/2, 2]. Collapse to [max_seq_len, rope_dim].
    freqs_cis = freqs_cis_real.reshape(model_args.max_seq_len, -1).to(device_id)

    for seq_len in (1, 2):
        x = torch.zeros(1, seq_len, model_args.dim, dtype=torch.bfloat16, device=f"cuda:{device_id}")
        out, caches = stack.golden_forward(x, start_pos=0, freqs_cis=freqs_cis)
        assert out.shape == (1, seq_len, model_args.dim), f"seq_len={seq_len}: unexpected shape {out.shape}"
        assert out.isfinite().all(), f"seq_len={seq_len}: output contains NaN/Inf"
        print(f"[3/3] seq_len={seq_len}: output.shape={tuple(out.shape)}, finite={out.isfinite().all().item()}")

    print("\n=== Transformer stack golden forward smoke test PASSED ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\n=== FAILED: {exc} ===", file=sys.stderr)
        raise
