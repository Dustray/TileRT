import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tilert.models.utils import precompute_freqs_cis
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.models.qwen3_6.modules.gated_attention import GatedAttention

model_dir = '/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master'
cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, config=cfg, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto', low_cpu_mem_usage=True, attn_implementation='eager')
text_model = model.model
layer_idx = 3
layer = text_model.layers[layer_idx]
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
input_tensor = tok.apply_chat_template([{'role':'user','content':'hello'}], add_generation_prompt=True, thinking=False, return_tensors='pt')['input_ids']
with torch.no_grad():
    hf_outputs = text_model(input_tensor.to(model.device), output_hidden_states=True)
hf_layer_in = hf_outputs.hidden_states[layer_idx]
hf_layer_out = hf_outputs.hidden_states[layer_idx+1] - hf_layer_in
args = ModelArgsQwen36(layer_types=list(cfg.text_config.layer_types))
args.top_k = 8
freqs_cis = precompute_freqs_cis(args, theta_override=args.rope_theta, factor_override=args.rope_factor).to('cuda:0')[:input_tensor.size(1)]
state_dict = {
    'input_layernorm.weight': layer.input_layernorm.weight.data,
    'post_attention_layernorm.weight': layer.post_attention_layernorm.weight.data,
    'self_attn.q_proj.weight': layer.self_attn.q_proj.weight.data,
    'self_attn.k_proj.weight': layer.self_attn.k_proj.weight.data,
    'self_attn.v_proj.weight': layer.self_attn.v_proj.weight.data,
    'self_attn.o_proj.weight': layer.self_attn.o_proj.weight.data,
    'self_attn.q_norm.weight': layer.self_attn.q_norm.weight.data,
    'self_attn.k_norm.weight': layer.self_attn.k_norm.weight.data,
    'mlp.gate.weight': layer.mlp.gate.weight.data,
    'mlp.experts.gate_up_proj': layer.mlp.experts.gate_up_proj,
    'mlp.experts.down_proj': layer.mlp.experts.down_proj,
    'mlp.shared_expert.gate_proj.weight': layer.mlp.shared_expert.gate_proj.weight.data,
    'mlp.shared_expert.up_proj.weight': layer.mlp.shared_expert.up_proj.weight.data,
    'mlp.shared_expert.down_proj.weight': layer.mlp.shared_expert.down_proj.weight.data,
    'mlp.shared_expert_gate.weight': layer.mlp.shared_expert_gate.weight.data,
    'mlp.experts.gate_up_proj.weight_scale_inv': torch.ones(256, 8, 2048//128, device='cuda'),
    'mlp.experts.down_proj.weight_scale_inv': torch.ones(256, 2048//128, 512//128, device='cuda'),
    'mlp.shared_expert.gate_proj.weight_scale_inv': torch.ones(512//128, 2048//128, device='cuda'),
    'mlp.shared_expert.up_proj.weight_scale_inv': torch.ones(512//128, 2048//128, device='cuda'),
    'mlp.shared_expert.down_proj.weight_scale_inv': torch.ones(2048//128, 512//128, device='cuda'),
    'mlp.gate.e_score_correction_bias': torch.zeros(256, device='cuda'),
}
attn = GatedAttention(args, device_id=0, num_devices=1).to('cuda:0')
attn.init_reference_weights(state_dict)
max_seq_len = input_tensor.size(1)
k_cache = torch.zeros(1, max_seq_len, args.n_kv_heads, args.qk_head_dim, device='cuda:0', dtype=torch.bfloat16)
v_cache = torch.zeros(1, max_seq_len, args.n_kv_heads, args.qk_head_dim, device='cuda:0', dtype=torch.bfloat16)

x_in = hf_layer_in.to('cuda:0')
norm_x = attn.input_layernorm(x_in)
attn_out, _, _ = attn.attn_ref.golden_forward(norm_x, 0, freqs_cis, k_cache, v_cache)
h = x_in + attn_out
norm_h = attn.post_attention_layernorm(h)
ffn_out = attn.ffn.golden_forward(norm_h)

trt_layer_out = h + ffn_out

pos_ids = torch.arange(hf_layer_in.size(1), device='cuda:0').view(1,-1)
rot_emb = text_model.rotary_emb(x_in, pos_ids)
causal_mask = torch.triu(torch.full((input_tensor.size(1), input_tensor.size(1)), float('-inf')), 1).to('cuda:0')
hf_attn_out = layer.self_attn(norm_x, position_embeddings=rot_emb, attention_mask=causal_mask.unsqueeze(0).unsqueeze(0))[0]
hf_h = x_in + hf_attn_out
hf_norm_h = layer.post_attention_layernorm(hf_h)
hf_ffn_out = layer.mlp(hf_norm_h)
hf_layer_out_direct = hf_h + hf_ffn_out

print('attn diff', (attn_out - hf_attn_out).abs().max().item())
print('post norm diff', (norm_h - hf_norm_h).abs().max().item())
print('ffn diff', (ffn_out - hf_ffn_out).abs().max().item())
print('layer_out diff (hf ref direct)', (trt_layer_out - hf_layer_out_direct).abs().max().item())
print('layer_out diff (hf hidden states residual)', (trt_layer_out - hf_layer_out.to('cuda:0')).abs().max().item())
