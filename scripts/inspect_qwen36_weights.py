import logging

from safetensors.torch import load_file

from tilert import logger

logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s:%(lineno)d [%(levelname)s]: %(message)s",
)

path = '/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master/model-00002-of-00026.safetensors'
weights = load_file(path, device='cpu')
keys = [
    'model.language_model.layers.0.mlp.experts.down_proj',
    'model.language_model.layers.0.mlp.experts.gate_up_proj',
    'model.language_model.layers.0.mlp.gate.weight',
    'model.language_model.layers.0.mlp.shared_expert.down_proj.weight',
    'model.language_model.layers.0.mlp.shared_expert.gate_proj.weight',
    'model.language_model.layers.0.mlp.shared_expert.up_proj.weight',
    'model.language_model.layers.0.mlp.shared_expert_gate.weight',
]
for k in keys:
    if k in weights:
        logger.info("%s %s %s", k, tuple(weights[k].shape), weights[k].dtype)
    else:
        logger.warning("%s MISSING", k)
