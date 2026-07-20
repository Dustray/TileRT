#!/usr/bin/env python3
"""MoE block parity check for Qwen3.6-35B-A3B.

Compares the TileRT golden (reference) MoE block against HuggingFace's
``Qwen3_5MoeMLP`` for layer 3 using a single prompt.
"""

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.models.qwen3_6.ops.rmsnorm_expert_proj import RMSNormExpertProj
from tilert.models.qwen3_6.ops.expert_sel_up_gate_silu import ExpertSelectUpGateSiLU
from tilert.models.qwen3_6.ops.expert_down_allreduce import ExpertDownAllReduce


def main() -> None:
    model_dir = "/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=cfg,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    layer = model.model.layers[3]
    moe = layer.mlp

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    input_tensor = tok.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        add_generation_prompt=True,
        thinking=False,
        return_tensors="pt",
    )["input_ids"]
    position_ids = torch.arange(
        input_tensor.shape[1], dtype=torch.long, device=model.device
    ).unsqueeze(0)

    with torch.no_grad():
        outputs = model.model(
            input_tensor.to(model.device),
            output_hidden_states=True,
            return_dict=True,
        )
    hf_layer_in = outputs.hidden_states[3]
    position_embeddings = model.model.rotary_emb(
        input_tensor.to(model.device), position_ids
    )
    hf_layer_out = layer(
        hf_layer_in, position_embeddings=position_embeddings
    )[0]
    post_att = hf_layer_in + hf_layer_out
    norm_x = layer.post_attention_layernorm(post_att)
    hf_moe = moe(norm_x)

    args = ModelArgsQwen36(layer_types=list(cfg.text_config.layer_types))
    args.top_k = 8

    moe_state = {
        "post_attention_layernorm.weight": layer.post_attention_layernorm.weight.data,
        "mlp.gate.weight": moe.gate.weight.data,
        "mlp.experts.gate_up_proj": moe.experts.gate_up_proj,
        "mlp.experts.down_proj": moe.experts.down_proj,
        "mlp.shared_expert.gate_proj.weight": moe.shared_expert.gate_proj.weight.data,
        "mlp.shared_expert.up_proj.weight": moe.shared_expert.up_proj.weight.data,
        "mlp.shared_expert.down_proj.weight": moe.shared_expert.down_proj.weight.data,
        "mlp.shared_expert_gate.weight": moe.shared_expert_gate.weight.data,
        "mlp.experts.gate_up_proj.weight_scale_inv": torch.ones(
            256, 8, 2048 // 128, device="cuda"
        ),
        "mlp.experts.down_proj.weight_scale_inv": torch.ones(
            256, 2048 // 128, 512 // 128, device="cuda"
        ),
        "mlp.shared_expert.gate_proj.weight_scale_inv": torch.ones(
            512 // 128, 2048 // 128, device="cuda"
        ),
        "mlp.shared_expert.up_proj.weight_scale_inv": torch.ones(
            512 // 128, 2048 // 128, device="cuda"
        ),
        "mlp.shared_expert.down_proj.weight_scale_inv": torch.ones(
            2048 // 128, 512 // 128, device="cuda"
        ),
        "mlp.gate.e_score_correction_bias": torch.zeros(256, device="cuda"),
    }

    rms = RMSNormExpertProj(args, device_id=0, num_devices=1)
    rms.init_reference_weights(moe_state)
    tr_norm_x, scores = rms.golden_forward(post_att.to("cuda:0"))
    print("norm diff", (tr_norm_x - norm_x.to("cuda:0")).abs().max().item())

    sel = ExpertSelectUpGateSiLU(args, device_id=0, num_devices=1)
    sel.init_reference_weights(moe_state)
    sel.ref_bias = None
    sel_up_gate, trt_weights, trt_indices = sel.golden_forward(
        tr_norm_x.to("cuda:0"), scores
    )

    allred = ExpertDownAllReduce(args, device_id=0, num_devices=1)
    allred.init_reference_weights(moe_state)
    tr_moe = allred.golden_forward(
        sel_up_gate.to("cuda:0"),
        trt_indices,
        trt_weights,
        x_in=tr_norm_x.to("cuda:0"),
    )
    print("tr_moe vs hf_moe diff", (tr_moe - hf_moe.to("cuda:0")).abs().max().item())


if __name__ == "__main__":
    main()
