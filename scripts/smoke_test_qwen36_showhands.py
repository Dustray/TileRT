"""Smoke test QwenShowHandsLayer init_random_weights + forward in container."""
import logging

import torch

from tilert import logger

logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s:%(lineno)d [%(levelname)s]: %(message)s",
)
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.models.qwen3_6.modules.end2end import QwenShowHandsLayer


def main():
    torch.set_num_threads(64)
    model_args = ModelArgsQwen36()
    # Limit max seq len to reduce KV cache allocation on random-init smoke test.
    model_args.max_seq_len = 512
    model_args.max_batch_size = 1

    layer = QwenShowHandsLayer(
        model_args=model_args,
        model_path="",
        with_mtp=False,
        temperature=1.0,
        top_p=0.9,
        top_k=256,
        use_topp=False,
    )

    logger.info("Initializing random weights on 8 devices...")
    layer.init_random_weights()
    logger.info("Random weights initialized.")

    token_id = torch.tensor(100, dtype=torch.int32)
    logger.info("Running forward(token_id=%d) on golden path...", token_id.item())
    results = layer.forward(token_id, with_mtp=False, cur_pos=0)
    logger.info("Forward returned %d device results.", len(results))

    next_token = layer.get_next_token(device_id=0)
    logger.info("Sampled next token: %d", next_token)

    logits = layer.get_logits(device_id=0)
    logger.info(
        "Logits shape: %s, finite=%s",
        logits.shape,
        logits.isfinite().all().item(),
    )

    layer.cleanup()
    logger.info("Smoke test passed.")


if __name__ == "__main__":
    main()
