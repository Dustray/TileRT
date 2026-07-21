import sys
sys.path.insert(0, '/public/home/dinggy/yiny/projects/TileRT')
from transformers import Qwen3_5MoeConfig

MODEL_DIR = '/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master'
cfg = Qwen3_5MoeConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
text_cfg = cfg.get_text_config()
print('text config:', text_cfg)
print('hidden_size', text_cfg.hidden_size)
print('num_attention_heads', text_cfg.num_attention_heads)
print('num_key_value_heads', text_cfg.num_key_value_heads)
print('head_dim attr', getattr(text_cfg, 'head_dim', None))
print('rope_parameters', text_cfg.rope_parameters)
print('partial_rotary_factor', text_cfg.rope_parameters.get('partial_rotary_factor'))
print('mrope_section', text_cfg.rope_parameters.get('mrope_section'))
print('rope_theta', text_cfg.rope_parameters.get('rope_theta'))
print('rope_type', text_cfg.rope_parameters.get('rope_type'))
print('max_position_embeddings', text_cfg.max_position_embeddings)
