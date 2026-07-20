"""逐子模块对比 TileRT 与 HF 的 DeltaNet 实现。

从 in_proj_qkv 开始，逐步对比 mixed_qkv、conv1d 输出、split q/k/v、
g/beta、chunk_gated_delta_rule 输出、rmsnorm gated、out_proj。
"""
import sys

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    torch_chunk_gated_delta_rule,
)

from tilert.models.qwen3_6.ops.delta_net import (
    _torch_causal_conv1d,
    _torch_chunk_gated_delta_rule,
    _rmsnorm_gated,
)


def report(name, a, b):
    diff = (a - b).abs()
    print(
        f"{name:40s} max_diff={diff.max().item():.6f} "
        f"mean_diff={diff.mean().item():.6f} "
        f"rel={diff.max().item() / (b.abs().max().item() + 1e-6):.6f}"
    )


def main():
    model_dir = "/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"
    prompt = "Tell me three jokes:"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    chat_out = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        thinking=False,
    )
    input_ids = list(chat_out["input_ids"]) if hasattr(chat_out, "input_ids") else list(chat_out)

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
    attn = text_model.layers[0].linear_attn

    with torch.no_grad():
        outs = text_model(torch.tensor([input_ids], device=hf_model.device), output_hidden_states=True)
    x = outs.hidden_states[0].to("cuda:0").to(torch.bfloat16)
    hf_attn_out = (outs.hidden_states[1] - outs.hidden_states[0]).to("cuda:0").to(torch.bfloat16)
    batch_size, seq_len, dim = x.shape

    # 1. in_proj_qkv
    mixed_qkv = x @ attn.in_proj_qkv.weight.T
    report("in_proj_qkv", mixed_qkv, mixed_qkv)  # self-check

    # 2. conv1d
    mixed_qkv_t = mixed_qkv.transpose(1, 2)
    conv_hf = attn.conv1d(mixed_qkv_t)
    conv_hf = F.silu(conv_hf[:, :, :seq_len])
    conv_tile = _torch_causal_conv1d(
        mixed_qkv_t,
        attn.conv1d.weight.squeeze(1),
        bias=None,
        activation="silu",
    )
    report("conv1d output", conv_tile, conv_hf)

    # 3. split q/k/v
    conv_hf = conv_hf.transpose(1, 2)
    query_hf, key_hf, value_hf = torch.split(
        conv_hf,
        [attn.key_dim, attn.key_dim, attn.value_dim],
        dim=-1,
    )
    query_tile, key_tile, value_tile = torch.split(
        conv_tile.transpose(1, 2),
        [attn.key_dim, attn.key_dim, attn.value_dim],
        dim=-1,
    )
    report("query", query_tile, query_hf)
    report("key", key_tile, key_hf)
    report("value", value_tile, value_hf)

    # 4. z/a/b
    z_hf = (x @ attn.in_proj_z.weight.T).reshape(batch_size, seq_len, attn.num_v_heads, attn.head_v_dim)
    b_hf = x @ attn.in_proj_b.weight.T
    a_hf = x @ attn.in_proj_a.weight.T
    report("z", z_hf, z_hf)
    report("b", b_hf, b_hf)
    report("a", a_hf, a_hf)

    beta_hf = torch.sigmoid(b_hf)
    g_hf = -attn.A_log.float().exp() * F.softplus(a_hf.float() + attn.dt_bias.float())
    report("beta", beta_hf, beta_hf)
    report("g", g_hf, g_hf)

    # 5. reshape and repeat_interleave
    query_hf = query_hf.reshape(batch_size, seq_len, attn.num_k_heads, attn.head_k_dim)
    key_hf = key_hf.reshape(batch_size, seq_len, attn.num_k_heads, attn.head_k_dim)
    value_hf = value_hf.reshape(batch_size, seq_len, attn.num_v_heads, attn.head_v_dim)
    query_hf = query_hf.repeat_interleave(attn.num_v_heads // attn.num_k_heads, dim=2)
    key_hf = key_hf.repeat_interleave(attn.num_v_heads // attn.num_k_heads, dim=2)

    q = query_hf.transpose(1, 2)
    k = key_hf.transpose(1, 2)
    v = value_hf.transpose(1, 2)
    beta_t = beta_hf.transpose(1, 2)
    g_t = g_hf.transpose(1, 2)

    # 6. chunk_gated_delta_rule: HF vs TileRT fallback
    core_hf, _ = torch_chunk_gated_delta_rule(
        q, k, v, g_t, beta_t, chunk_size=64, initial_state=None,
        output_final_state=False, use_qk_l2norm_in_kernel=True,
    )
    core_tile, _ = _torch_chunk_gated_delta_rule(
        q, k, v, g_t, beta_t, chunk_size=64, initial_state=None,
        output_final_state=False, use_qk_l2norm_in_kernel=True,
    )
    report("chunk_gated_delta_rule", core_tile, core_hf)

    # 7. gated rmsnorm
    core_flat = core_hf.reshape(-1, attn.head_v_dim)
    z_flat = z_hf.reshape(-1, attn.head_v_dim)
    norm_hf = attn.norm(core_flat, z_flat)
    norm_tile = _rmsnorm_gated(core_flat, attn.norm.weight, z_flat, eps=1e-6)
    report("gated rmsnorm", norm_tile, norm_hf)

    # 8. out_proj
    out_hf = norm_hf.reshape(batch_size, seq_len, -1) @ attn.out_proj.weight.T
    out_tile = norm_tile.reshape(batch_size, seq_len, -1) @ attn.out_proj.weight.T
    report("out_proj", out_tile, out_hf)

    print(f"\nFinal vs HF attn_out: max_diff={((out_tile - hf_attn_out).abs().max().item()):.6f}")

    print("\n=== Submodule comparison complete ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\n=== FAILED: {exc} ===", file=sys.stderr)
        raise
