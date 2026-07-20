"""Validate Qwen3.6 GatedAttention golden path vs Hugging Face.

This script compares TileRT's reference (golden) attention-only output against
Hugging Face's Qwen3.5MoeAttention.forward for a full_attention layer.  It
passes proper position_embeddings=(cos, sin) obtained from the HF model's
rotary embedding, avoiding the None-tuple error seen in earlier attempts.
"""

import os
import sys
import math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.environ["TILERT_ROOT"])

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.models.qwen3_6.modules.gated_attention import QwenAttentionRef
from tilert.models.utils import precompute_freqs_cis, apply_rotary_emb

HF_DIR = os.environ.get("HF_MODEL_DIR", "/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master")
TILERT_WEIGHTS = os.environ.get("TILERT_WEIGHTS_DIR", "/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B--TileRT/snapshots/master")
PROMPT = "Tell me three jokes:"
LAYER_IDX = 3

def main():
    tok = AutoTokenizer.from_pretrained(HF_DIR, trust_remote_code=True)
    chat_out = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, thinking=False)
    input_ids = chat_out["input_ids"]
    seq_len = len(input_ids)

    cfg = AutoConfig.from_pretrained(HF_DIR, trust_remote_code=True)
    print("HF config loaded, model_type =", cfg.model_type)

    # Build TileRT model args from HF config.
    ma = ModelArgsQwen36(layer_types=list(cfg.text_config.layer_types))
    print("TileRT ModelArgs:", ma)

    # Load HF model for rotary_emb and baseline attention.
    print("Loading HF model...")
    m = AutoModelForCausalLM.from_pretrained(
        HF_DIR,
        config=cfg,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    mtext = getattr(m.model, "text_model", m.model)
    layer = mtext.layers[LAYER_IDX]
    assert layer.layer_type == "full_attention", f"Layer {LAYER_IDX} is {layer.layer_type}, not full_attention"

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=m.device)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=m.device).unsqueeze(0)
    # Pass a float tensor so rotary_emb returns float cos/sin instead of int64.
    cos, sin = mtext.rotary_emb(input_tensor.float(), position_ids)
    print("cos/sin shapes/dtypes", cos.shape, cos.dtype, sin.shape, sin.dtype)

    # Get hidden states at layer input by running up to previous layer.
    with torch.no_grad():
        hidden_states_all = mtext(input_tensor, output_hidden_states=True).hidden_states
    hidden_input = hidden_states_all[LAYER_IDX]
    print("hidden_input shape", hidden_input.shape)

    # HF attention-only output (no residual/norm/FFN).
    norm_hidden = layer.input_layernorm(hidden_input)
    with torch.no_grad():
        hf_attn_out, _ = layer.self_attn(
            norm_hidden.to(torch.bfloat16),
            position_embeddings=(cos.to(torch.bfloat16), sin.to(torch.bfloat16)),
            attention_mask=None,
        )
    print("HF attn out shape", hf_attn_out.shape)

    # TileRT attention-only output.
    # Load self_attn weights from safetensors.
    safetensor_files = sorted([
        os.path.join(TILERT_WEIGHTS, f)
        for f in os.listdir(TILERT_WEIGHTS)
        if f.endswith(".safetensors") and "model.safetensors-" in f
    ])
    print("Loading", len(safetensor_files), "safetensor files")
    state_dict = {}
    for f in safetensor_files:
        state_dict.update(load_file(f, device="cpu"))

    # Build reference state dict for the layer.  The converted TileRT
    # checkpoint uses layer_N_<name>_dev_0 keys; use device 0 replica.
    prefix = f"layer_{LAYER_IDX}_"
    attn_state = {
        "self_attn.q_proj.weight": state_dict[f"{prefix}q_proj.weight_dev_0"].to(torch.bfloat16),
        "self_attn.k_proj.weight": state_dict[f"{prefix}k_proj.weight_dev_0"].to(torch.bfloat16),
        "self_attn.v_proj.weight": state_dict[f"{prefix}v_proj.weight_dev_0"].to(torch.bfloat16),
        "self_attn.o_proj.weight": state_dict[f"{prefix}o_proj.weight_dev_0"].to(torch.bfloat16),
        "self_attn.q_norm.weight": state_dict[f"{prefix}q_norm.weight_dev_0"].to(torch.bfloat16),
        "self_attn.k_norm.weight": state_dict[f"{prefix}k_norm.weight_dev_0"].to(torch.bfloat16),
        "input_layernorm.weight": torch.ones(ma.dim, dtype=torch.bfloat16),  # not used for attn-only
        "post_attention_layernorm.weight": torch.ones(ma.dim, dtype=torch.bfloat16),
    }

    # TileRT QwenAttentionRef is single-device reference.
    ref = QwenAttentionRef(ma, device_id=0, num_devices=1)
    ref.init_reference_weights(attn_state)
    # Move weights to same device as hidden_input.
    device = hidden_input.device
    for name, w in [
        ("q_proj_weight", attn_state["self_attn.q_proj.weight"]),
        ("k_proj_weight", attn_state["self_attn.k_proj.weight"]),
        ("v_proj_weight", attn_state["self_attn.v_proj.weight"]),
        ("o_proj_weight", attn_state["self_attn.o_proj.weight"]),
        ("q_norm_weight", attn_state["self_attn.q_norm.weight"]),
        ("k_norm_weight", attn_state["self_attn.k_norm.weight"]),
    ]:
        setattr(ref, name, w.to(device))

    # Precompute freqs_cis for TileRT.
    ma.max_seq_len = seq_len
    freqs_cis = precompute_freqs_cis(
        ma,
        theta_override=ma.rope_theta,
        factor_override=ma.rope_factor,
    ).to(device).unsqueeze(0).unsqueeze(2)

    k_cache = torch.zeros((1, seq_len, ma.n_kv_heads, ma.qk_head_dim), dtype=torch.bfloat16, device=device)
    v_cache = torch.zeros((1, seq_len, ma.n_kv_heads, ma.qk_head_dim), dtype=torch.bfloat16, device=device)

    with torch.no_grad():
        trt_attn_out, _, _ = ref.golden_forward(
            norm_hidden.to(device), 0, freqs_cis, k_cache, v_cache, mask=None
        )
    print("TileRT attn out shape", trt_attn_out.shape)

    # Compare.
    a = hf_attn_out.float()
    b = trt_attn_out.float()
    diff = (a - b).abs()
    print("Mean abs diff:", diff.mean().item())
    print("Max abs diff:", diff.max().item())
    print("HF mean abs:", a.abs().mean().item())
    print("TRT mean abs:", b.abs().mean().item())
    rel = diff / (a.abs() + 1e-6)
    print("Mean relative diff:", rel.mean().item())
    print("Max relative diff:", rel.max().item())

    # Tolerance for bfloat16.
    if diff.max().item() < 1e-2 or rel.max().item() < 2e-2:
        print("PASS: GatedAttention golden matches HF.")
    else:
        print("FAIL: GatedAttention golden diverges from HF.")

if __name__ == "__main__":
    main()
