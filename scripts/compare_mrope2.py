import torch
import sys

sys.path.insert(0, '/public/home/dinggy/yiny/projects/TileRT')
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.models.utils import precompute_mrope_embed

MODEL_DIR = '/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master'

args = ModelArgsQwen36()

# Use local docs copy to avoid container transformers version mismatch
sys.path.insert(0, '/public/home/dinggy/yiny/projects/TileRT/docs')
from modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
from transformers import Qwen3_5MoeConfig

cfg = Qwen3_5MoeConfig.from_pretrained(MODEL_DIR)
emb_official = Qwen3_5MoeTextRotaryEmbedding(config=cfg.get_text_config(), device='cpu')

x = torch.zeros(1, 16, dtype=torch.int64)
position_ids = torch.arange(16).unsqueeze(0)
cos_o, sin_o = emb_official(x, position_ids=position_ids)
print('official cos shape', cos_o.shape)
print('official cos row 0', cos_o[0, 0].tolist())
print('official cos row 1', cos_o[0, 1].tolist())
print('official cos row 2', cos_o[0, 2].tolist())
print('official sin row 1', sin_o[0, 1].tolist())
print('official inv_freq', emb_official.inv_freq[:8].tolist())

cos_t, sin_t = precompute_mrope_embed(args, max_seq_len=16)
print('tilert cos row 0', cos_t[0].tolist())
print('tilert cos row 1', cos_t[1].tolist())
print('tilert cos row 2', cos_t[2].tolist())
print('tilert sin row 1', sin_t[1].tolist())

print('max diff cos', (cos_t[:16] - cos_o[0].float()).abs().max().item())
print('max diff sin', (sin_t[:16] - sin_o[0].float()).abs().max().item())
