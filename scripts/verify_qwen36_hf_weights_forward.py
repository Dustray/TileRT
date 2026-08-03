"""验证 QwenShowHandsLayer 从原始 HF checkpoint 加载并单步 forward。"""
import logging
import os
import sys

import torch

from tilert import logger

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s:%(lineno)d [%(levelname)s]: %(message)s",
)


def main():
    torch.set_num_threads(64)

    weights_dir = "/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master"
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

    logger.info("[1/3] Importing modules...")
    from tilert.models.qwen3_6.model_args import ModelArgsQwen36
    from tilert.models.qwen3_6.modules.end2end import QwenShowHandsLayer
    logger.info("[1/3] Import OK")

    logger.info("[2/3] Loading HF pretrained weights on 8 devices...")
    model_args = ModelArgsQwen36()
    model_args.max_seq_len = 512
    model_args.max_batch_size = 1

    layer = QwenShowHandsLayer(
        model_args=model_args,
        model_path=weights_dir,
        with_mtp=False,
        temperature=1.0,
        top_p=0.9,
        top_k=256,
        use_topp=False,
    )
    layer.from_pretrained(weights_dir)
    logger.info("[2/3] HF weights loaded OK")

    logger.info("[3/3] Running forward(token_id=100, cur_pos=0)...")
    token_id = torch.tensor(100, dtype=torch.int32)
    results = layer.forward(token_id, with_mtp=False, cur_pos=0)
    assert len(results) == layer.num_devices

    next_token = layer.get_next_token(device_id=0)
    logits = layer.get_logits(device_id=0)
    logger.info(
        "[3/3] next_token=%d, logits.shape=%s, finite=%s",
        next_token,
        tuple(logits.shape),
        logits.isfinite().all().item(),
    )

    assert logits.isfinite().all(), "Logits contain NaN/Inf"
    assert 0 <= next_token < model_args.vocab_size, f"Invalid next_token {next_token}"

    layer.cleanup()
    logger.info("\n=== HF-weights golden forward smoke test PASSED ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        logger.exception("\n=== FAILED: %s ===", exc)
        raise
