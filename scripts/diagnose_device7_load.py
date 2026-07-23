"""Diagnose why HF-source weight loading is slow on device 7."""
import logging
import os
import sys
import time

import torch

logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s:%(lineno)d [%(levelname)s]: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(64)

    model_path = "/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"
    sys.path.insert(0, "/public/home/dinggy/yiny/projects/TileRT")

    from tilert.models.qwen3_6.model_args import ModelArgsQwen36
    from tilert.models.qwen3_6.modules.transformer_stack import QwenTransformerStack
    from tilert.models.qwen3_6.modules.hf_source_loader import load_hf_source_weights

    model_args = ModelArgsQwen36()
    num_devices = 8
    device_id = 7

    logger.info("Building stack on cuda:%d", device_id)
    stack = QwenTransformerStack(model_args, device_id, num_devices)

    logger.info("Loading HF source weights for device %d...", device_id)
    start = time.time()
    state_dict = load_hf_source_weights(model_path, model_args, num_devices, device_id, stack)
    elapsed = time.time() - start
    logger.info("Loaded %d tensors in %.1f seconds", len(state_dict), elapsed)

    # Print a few key tensor shapes/devices.
    for key in list(state_dict.keys())[:10]:
        t = state_dict[key]
        logger.info("  %s: %s on %s", key, tuple(t.shape), t.device)


if __name__ == "__main__":
    main()
