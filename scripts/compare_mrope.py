import torch
import sys

sys.path.insert(0, '/public/home/dinggy/yiny/projects/TileRT')
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.models.utils import precompute_mrope_embed
from transformers import Qwen3_5MoeConfig

MODEL_DIR = '/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master'

cfg = Qwen3_5MoeConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
args = ModelArgsQwen36()
head_dim = args.qk_head_dim
print('args qk_head_dim', head_dim, 'partial', args.partial_rotary_factor,
      'section', args.mrope_section, 'theta', args.rope_theta)

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
emb_official = Qwen3_5MoeTextRotaryEmbedding(config=cfg.get_text_config(), device='cpu')

# Bypass dynamic_rope_update decorator which may reset cache and produce zeros.
def official_forward(emb, x, position_ids):
    if position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    inv_freq_expanded = emb.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
    freqs = emb.apply_interleaved_mrope(freqs, emb.mrope_section)
    emb_tensor = torch.cat((freqs, freqs), dim=-1)
    return emb_tensor.cos(), emb_tensor.sin()

cos_o, sin_o = official_forward(
    emb_official,
    torch.zeros(1, 512, dtype=torch.float32),
    position_ids=torch.arange(512).unsqueeze(0),
)
print('official cos', cos_o.shape, cos_o.device)

cos_t, sin_t = precompute_mrope_embed(args, max_seq_len=512)
print('tilert cos', cos_t.shape)
diff_cos = (cos_t - cos_o[0].float()).abs()
diff_sin = (sin_t - sin_o[0].float()).abs()
print('max diff cos', diff_cos.max().item())
print('max diff sin', diff_sin.max().item())
max_pos = int(diff_cos.argmax().item())
row = max_pos // cos_t.shape[1]
col = max_pos % cos_t.shape[1]
print('max diff pos row', row, 'col', col)
print('tilert cos at pos', cos_t[row, col].item(), 'sin', sin_t[row, col].item())
print('official cos at pos', cos_o[0, row, col].item(), 'sin', sin_o[0, row, col].item())
print('tilert row', row, 'cos:', cos_t[row, :16].tolist())
print('official row', row, 'cos:', cos_o[0, row, :16].tolist())
