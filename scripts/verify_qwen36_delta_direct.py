"""直接验证 TileRT 的 Gated DeltaNet forward 实现是否与 HF 一致。

使用 HF layer 0 的原始权重直接赋值给 DeltaNetOp，绕过 weight converter 的切分逻辑。
"""
import sys

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.models.qwen3_6.ops.delta_net import DeltaNetOp


def main():
    model_dir = "/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"
    prompt = "Tell me three jokes:"

    print("[1/3] Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    chat_out = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        thinking=False,
    )
    input_ids = list(chat_out["input_ids"]) if hasattr(chat_out, "input_ids") else list(chat_out)
    print(f"Input ids: {input_ids}")

    print("[2/3] Loading HF model and capturing layer 0 attention output...")
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=cfg,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    hf_model.eval()
    text_model = getattr(hf_model.model, "text_model", hf_model.model)
    layer0 = text_model.layers[0]
    attn = layer0.linear_attn

    with torch.no_grad():
        outs = text_model(torch.tensor([input_ids], device=hf_model.device), output_hidden_states=True)
    h0 = outs.hidden_states[0]
    hf_attn_out = outs.hidden_states[1] - outs.hidden_states[0]

    print(f"HF layer0 input shape: {h0.shape}")
    print(f"HF layer0 attn_out shape: {hf_attn_out.shape}")

    print("[3/3] Running TileRT DeltaNetOp with HF weights directly...")
    model_args = ModelArgsQwen36(layer_types=list(cfg.text_config.layer_types))
    tile_op = DeltaNetOp(model_args=model_args, device_id=0, num_devices=1)

    # Assign HF weights directly, no sharding.
    tile_op.in_proj_qkv_weights = attn.in_proj_qkv.weight.data.to("cuda:0")
    tile_op.in_proj_z_weights = attn.in_proj_z.weight.data.to("cuda:0")
    tile_op.in_proj_a_weights = attn.in_proj_a.weight.data.to("cuda:0")
    tile_op.in_proj_b_weights = attn.in_proj_b.weight.data.to("cuda:0")
    tile_op.conv1d_weights = attn.conv1d.weight.data.to("cuda:0")
    tile_op.A_log = attn.A_log.data.to("cuda:0")
    tile_op.dt_bias = attn.dt_bias.data.to("cuda:0")
    tile_op.norm_weights = attn.norm.weight.data.to("cuda:0")
    tile_op.out_proj_weights = attn.out_proj.weight.data.to("cuda:0")

    x = h0.to("cuda:0").to(torch.bfloat16)
    with torch.no_grad():
        tile_out, _ = tile_op.golden_forward(x, start_pos=0, state=None)

    hf_out = hf_attn_out.to("cuda:0").to(torch.bfloat16)
    diff = (tile_out - hf_out).abs()
    print(f"\nTileRT DeltaNetOp (HF weights) vs HF layer 0 attn_out:")
    print(f"  max abs diff: {diff.max().item():.6f}")
    print(f"  mean abs diff: {diff.mean().item():.6f}")
    print(f"  HF max: {hf_out.abs().max().item():.6f}")
    print(f"  rel diff: {diff.max().item() / (hf_out.abs().max().item() + 1e-6):.6f}")

    if diff.max().item() < 0.05:
        print("  --> Forward implementation matches HF.")
    else:
        print("  --> Forward implementation still diverges from HF.")

    print("\n=== Direct DeltaNet comparison complete ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\n=== FAILED: {exc} ===", file=sys.stderr)
        raise
