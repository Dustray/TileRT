"""对比 HF Qwen3.6 全注意力层 (GatedAttention) 与 TileRT golden forward。

加载 HF 模型，取第一个 Gated Attention 层（默认 layer 3）的权重/输入/输出，
分别比较：
  1. attention-only 输出
  2. 完整 layer 输出（含 residual + MoE）

执行：
    cd /public/home/dinggy/yiny/projects/TileRT
    PYTHONPATH=/public/home/dinggy/yiny/projects/TileRT python3 scripts/verify_qwen36_gated_attention_hf.py
"""
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def compare_tensors(name, a, b, tol=0.1):
    diff = (a - b).abs()
    print(f"\n{name}:")
    print(f"  max abs diff: {diff.max().item():.6f}")
    print(f"  mean abs diff: {diff.mean().item():.6f}")
    print(f"  ref max: {b.abs().max().item():.6f}")
    print(f"  rel diff: {diff.max().item() / (b.abs().max().item() + 1e-6):.6f}")
    ok = diff.max().item() < tol
    print(f"  --> {'PASS' if ok else 'FAIL'}")
    return ok


def main(layer_idx=3):
    model_dir = "/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"
    prompt = (
        "Tell me three jokes:\n\n"
        "1. A dad joke,\n"
        "2. A programmer joke,\n"
        "3. A joke that only makes sense if you've ever tried "
        "to train a large language model.\n"
        "Keep each joke under 15 words."
    )

    print(f"[1/4] Loading tokenizer and tokenizing prompt...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    chat_out = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        thinking=False,
    )
    input_ids = list(chat_out["input_ids"]) if hasattr(chat_out, "input_ids") else list(chat_out)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    print(f"Prompt length: {len(input_ids)}")

    print("[2/4] Loading original HF model...")
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
    print(f"HF text model layers: {len(text_model.layers)}")

    layer = text_model.layers[layer_idx]
    print(f"[3/4] Running HF layer {layer_idx} ({type(layer).__name__})...")

    with torch.no_grad():
        hf_outputs = text_model(input_tensor.to(hf_model.device), output_hidden_states=True)
    hf_layer_in = hf_outputs.hidden_states[layer_idx]
    hf_layer_out = hf_outputs.hidden_states[layer_idx + 1]

    # HF full attention needs position_embeddings (cos, sin) and causal mask.
    position_ids = torch.arange(hf_layer_in.size(1), device=hf_layer_in.device).view(1, -1)
    cos, sin = text_model.rotary_emb(hf_layer_in, position_ids)
    position_embeddings = (cos, sin)
    attention_mask = None  # single sequence, no padding; HF eager handles causal internally when None

    with torch.no_grad():
        hf_attn_out, _ = layer.self_attn(
            hf_layer_in,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )

    print(f"HF layer {layer_idx} input shape: {hf_layer_in.shape}")
    print(f"HF attention-only output shape: {hf_attn_out.shape}")
    print(f"HF full layer output shape: {hf_layer_out.shape}")

    # Build TileRT GatedAttention on cuda:0 with HF weights.
    from tilert.models.qwen3_6.model_args import ModelArgsQwen36
    from tilert.models.qwen3_6.modules.gated_attention import GatedAttention

    model_args = ModelArgsQwen36(layer_types=list(cfg.text_config.layer_types))

    state_dict = {
        "self_attn.q_proj.weight": layer.self_attn.q_proj.weight.data,
        "self_attn.k_proj.weight": layer.self_attn.k_proj.weight.data,
        "self_attn.v_proj.weight": layer.self_attn.v_proj.weight.data,
        "self_attn.o_proj.weight": layer.self_attn.o_proj.weight.data,
        "self_attn.q_norm.weight": layer.self_attn.q_norm.weight.data,
        "self_attn.k_norm.weight": layer.self_attn.k_norm.weight.data,
        "input_layernorm.weight": layer.input_layernorm.weight.data,
        "post_attention_layernorm.weight": layer.post_attention_layernorm.weight.data,
    }

    # MoE weights for the same layer.
    mlp_prefix = "mlp"
    state_dict.update({
        f"{mlp_prefix}.gate.weight": layer.mlp.gate.weight.data,
        f"{mlp_prefix}.experts.gate_up_proj": layer.mlp.experts.gate_up_proj.data,
        f"{mlp_prefix}.experts.down_proj": layer.mlp.experts.down_proj.data,
        f"{mlp_prefix}.shared_expert.gate_proj.weight": layer.mlp.shared_expert.gate_proj.weight.data,
        f"{mlp_prefix}.shared_expert.up_proj.weight": layer.mlp.shared_expert.up_proj.weight.data,
        f"{mlp_prefix}.shared_expert.down_proj.weight": layer.mlp.shared_expert.down_proj.weight.data,
        f"{mlp_prefix}.shared_expert_gate.weight": layer.mlp.shared_expert_gate.weight.data,
    })

    state_dict_cuda = {k: v.to("cuda:0") for k, v in state_dict.items()}
    # Ensure the fallback zero bias is also on cuda:0.
    bias_key = "mlp.gate.e_score_correction_bias"
    if bias_key not in state_dict_cuda:
        state_dict_cuda[bias_key] = torch.zeros(
            model_args.n_routed_experts, dtype=torch.float32, device="cuda:0"
        )
    x0 = hf_layer_in.to("cuda:0").to(torch.bfloat16)

    tile_layer = GatedAttention(
        model_args=model_args,
        device_id=0,
        num_devices=1,
    )
    tile_layer.init_reference_weights(state_dict_cuda)
    # Make sure the RMSNorm modules inside GatedAttention also get the checkpoint weights.
    tile_layer.input_layernorm.weight.data.copy_(
        state_dict_cuda["input_layernorm.weight"]
    )
    tile_layer.post_attention_layernorm.weight.data.copy_(
        state_dict_cuda["post_attention_layernorm.weight"]
    )

    # Build RoPE freqs_cis for the sequence length.
    from tilert.models.utils import precompute_freqs_cis
    freqs_cis = precompute_freqs_cis(model_args).to("cuda:0")
    freqs_cis_real = torch.view_as_real(freqs_cis).reshape(freqs_cis.shape[0], -1)

    max_seq_len = model_args.max_seq_len + model_args.kv_cache_pad
    k_cache = torch.zeros(
        1, max_seq_len, model_args.n_kv_heads, model_args.qk_head_dim,
        dtype=torch.bfloat16, device="cuda:0",
    )
    v_cache = torch.zeros(
        1, max_seq_len, model_args.n_kv_heads, model_args.v_head_dim,
        dtype=torch.bfloat16, device="cuda:0",
    )

    with torch.no_grad():
        tile_full_out, _, _ = tile_layer.golden_forward(
            x0, start_pos=0, freqs_cis=freqs_cis_real, k_cache=k_cache, v_cache=v_cache
        )

    # Attention-only: feed the *pre-norm* input through the attention op directly.
    with torch.no_grad():
        norm_x = tile_layer.input_layernorm(x0)
        tile_attn_out, _, _ = tile_layer.attn.golden_forward(
            norm_x, start_pos=0, freqs_cis=freqs_cis_real, k_cache=k_cache.clone(), v_cache=v_cache.clone()
        )

    hf_attn_out = hf_attn_out.to("cuda:0").to(torch.bfloat16)
    hf_layer_out = hf_layer_out.to("cuda:0").to(torch.bfloat16)

    ok1 = compare_tensors(
        f"TileRT GQA attention-only vs HF layer {layer_idx} self_attn",
        tile_attn_out, hf_attn_out, tol=0.1
    )
    ok2 = compare_tensors(
        f"TileRT GatedAttention full layer vs HF layer {layer_idx} output",
        tile_full_out, hf_layer_out, tol=0.1
    )

    print("\n=== GatedAttention comparison complete ===")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\n=== FAILED: {exc} ===", file=sys.stderr)
        raise
