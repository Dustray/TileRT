import os, sys
sys.path.insert(0, os.environ['TILERT_ROOT'])
import torch
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from transformers import AutoConfig
from tilert.models.utils import precompute_freqs_cis
from tilert.models.qwen3_6.modules.gated_attention import QwenAttentionRef

cfg = AutoConfig.from_pretrained(os.environ['HF_MODEL_DIR'], trust_remote_code=True)
ma = ModelArgsQwen36(layer_types=list(cfg.text_config.layer_types))
ma.max_seq_len = 15
freqs = precompute_freqs_cis(ma).cuda().unsqueeze(0).unsqueeze(2)
print('freqs', freqs.shape)
ref = QwenAttentionRef(ma, 0, 1)
ref.q_proj_weight = torch.randn(8192, 2048, dtype=torch.bfloat16, device='cuda')
ref.k_proj_weight = torch.randn(512, 2048, dtype=torch.bfloat16, device='cuda')
ref.v_proj_weight = torch.randn(512, 2048, dtype=torch.bfloat16, device='cuda')
ref.o_proj_weight = torch.randn(2048, 4096, dtype=torch.bfloat16, device='cuda')
ref.q_norm_weight = torch.ones(256, dtype=torch.bfloat16, device='cuda')
ref.k_norm_weight = torch.ones(256, dtype=torch.bfloat16, device='cuda')
x = torch.randn(1, 15, 2048, dtype=torch.bfloat16, device='cuda')
k = torch.zeros(1, 15, 2, 256, dtype=torch.bfloat16, device='cuda')
v = torch.zeros(1, 15, 2, 256, dtype=torch.bfloat16, device='cuda')
try:
    out, _, _ = ref.golden_forward(x, 0, freqs, k, v)
    print('ok', out.shape)
except Exception:
    import traceback
    traceback.print_exc()
