"""对比 HF Qwen3.6 layer 0 (DeltaNet) 与 TileRT DeltaNet golden forward。

只加载 HF 模型并取 layer 0 的权重/输入/输出，避免完整 TileRT 栈导致 OOM。

执行：
    cd /public/home/dinggy/yiny/projects/TileRT
    PYTHONPATH=/public/home/dinggy/yiny/projects/TileRT python3 scripts/verify_qwen36_hf_layer_comparison.py
"""
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main():
    model_dir = "/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"
    prompt = (
        "Tell me three jokes:\n\n"
        "1. A dad joke,\n"
        "2. A programmer joke,\n"
        "3. A joke that only makes sense if you've ever tried "
        "to train a large language model.\n"
        "Keep each joke under 15 words."
    )

    print("[1/3] Loading tokenizer and tokenizing prompt...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    chat_out = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        thinking=False,
    )
    input_ids = list(chat_out["input_ids"]) if hasattr(chat_out, "input_ids") else list(chat_out)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    print(f"Prompt length: {len(input_ids)}")

    print("[2/3] Loading original HF model...")
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
    print(f"HF text model type: {type(text_model)}")
    print(f"HF text model layers: {len(text_model.layers)}")

    print("[3/3] Running HF layer 0 attention directly and comparing with TileRT DeltaNet golden...")
    layer0 = text_model.layers[0]
    attn = layer0.linear_attn

    with torch.no_grad():
        hf_outputs = text_model(input_tensor.to(hf_model.device), output_hidden_states=True)
    hf_layer0_in = hf_outputs.hidden_states[0]  # post-embedding input to layer 0

    # HF attention-only output (without residual and without MoE FFN).
    with torch.no_grad():
        hf_attn_out = attn(hf_layer0_in)

    print(f"HF layer 0 input shape: {hf_layer0_in.shape}, device: {hf_layer0_in.device}")
    print(f"HF attention-only output shape: {hf_attn_out.shape}, device: {hf_attn_out.device}")

    # Extract HF layer 0 weights and run TileRT DeltaNetOp on cuda:0.
    from tilert.models.qwen3_6.model_args import ModelArgsQwen36
    from tilert.models.qwen3_6.ops.delta_net import DeltaNetOp

    model_args = ModelArgsQwen36(layer_types=list(cfg.text_config.layer_types))

    state_dict = {
        "linear_attn.in_proj_qkv.weight": attn.in_proj_qkv.weight.data,
        "linear_attn.in_proj_z.weight": attn.in_proj_z.weight.data,
        "linear_attn.in_proj_a.weight": attn.in_proj_a.weight.data,
        "linear_attn.in_proj_b.weight": attn.in_proj_b.weight.data,
        "linear_attn.conv1d.weight": attn.conv1d.weight.data,
        "linear_attn.A_log": attn.A_log.data,
        "linear_attn.dt_bias": attn.dt_bias.data,
        "linear_attn.norm.weight": attn.norm.weight.data,
        "linear_attn.out_proj.weight": attn.out_proj.weight.data,
        "input_layernorm.weight": layer0.input_layernorm.weight.data,
        "post_attention_layernorm.weight": layer0.post_attention_layernorm.weight.data,
    }

    # Move to cuda:0 for TileRT comparison.
    state_dict_cuda = {k: v.to("cuda:0") for k, v in state_dict.items()}
    x0 = hf_layer0_in.to("cuda:0").to(torch.bfloat16)

    tile_op = DeltaNetOp(
        model_args=model_args,
        device_id=0,
        num_devices=1,
    )
    tile_op.init_reference_weights(state_dict_cuda)

    with torch.no_grad():
        tile_out, _ = tile_op.golden_forward(x0, start_pos=0, state=None)

    hf_attn_out = hf_attn_out.to("cuda:0").to(torch.bfloat16)

    diff = (tile_out - hf_attn_out).abs()
    print(f"\nDeltaNetOp golden vs HF layer 0 attention-only output:")
    print(f"  TileRT out shape: {tile_out.shape}")
    print(f"  HF attn_out shape: {hf_attn_out.shape}")
    print(f"  max abs diff: {diff.max().item():.6f}")
    print(f"  mean abs diff: {diff.mean().item():.6f}")
    print(f"  HF attn_out max: {hf_attn_out.abs().max().item():.6f}")
    print(f"  rel diff (max/max): {diff.max().item() / (hf_attn_out.abs().max().item() + 1e-6):.6f}")

    if diff.max().item() > 0.1:
        print("  --> Significant divergence: TileRT DeltaNet golden_forward does not match HF.")
    else:
        print("  --> Outputs are close.")

    print("\n=== Layer 0 comparison complete ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\n=== FAILED: {exc} ===", file=sys.stderr)
        raise
