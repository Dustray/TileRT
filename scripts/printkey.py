import json
import logging
from pathlib import Path

from tilert import logger

logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s:%(lineno)d [%(levelname)s]: %(message)s",
)

root = Path('/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master')
idx = json.load(open(root / 'model.safetensors.index.json'))
keys = sorted(idx['weight_map'].keys())
for k in keys:
    logger.info(k)
