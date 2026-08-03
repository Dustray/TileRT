"""Diagnose per-device GPU memory bloat in QwenShowHandsLayer HF-weights forward."""
import gc
import logging
import os
import sys

import torch

from tilert import logger

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s:%(lineno)d [%(levelname)s]: %(message)s",
)


def mem_summary(device: int | str = None, title: str = "") -> str:
    if isinstance(device, int):
        device = f"cuda:{device}"
    allocated = torch.cuda.memory_allocated(device) / 2**30
    reserved = torch.cuda.memory_reserved(device) / 2**30
    return f"[{title}] cuda:{device.split(':')[-1]} allocated={allocated:.2f} GiB reserved={reserved:.2f} GiB"


def _tensor_size(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def estimate_module_memory(module, device_id: int | None = None, prefix: str = "") -> int:
    """Sum nbytes of all tensor attributes in a module/op."""
    total = 0
    seen_ids = set()
    for name, attr in module.named_parameters(recurse=True):
        if id(attr) in seen_ids:
            continue
        if device_id is None or attr.device.index == device_id:
            total += _tensor_size(attr)
            seen_ids.add(id(attr))
    for name, attr in module.named_buffers(recurse=True):
        if id(attr) in seen_ids:
            continue
        if device_id is None or attr.device.index == device_id:
            total += _tensor_size(attr)
            seen_ids.add(id(attr))
    # Also catch plain tensor attributes that are not registered as params/buffers
    for name, val in vars(module).items():
        if isinstance(val, torch.Tensor):
            if id(val) in seen_ids:
                continue
            if device_id is None or val.device.index == device_id:
                total += _tensor_size(val)
                seen_ids.add(id(val))
        elif isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, torch.Tensor):
                    if id(item) in seen_ids:
                        continue
                    if device_id is None or item.device.index == device_id:
                        total += _tensor_size(item)
                        seen_ids.add(id(item))
    return total


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

    for did in range(layer.num_devices):
        logger.info(mem_summary(did, "after load"))

    # Try to estimate tensor memory on device 0
    stack0 = layer._stack_objects[0]
    head0 = layer._head_proj_objects[0]
    stack_mem = estimate_module_memory(stack0, device_id=0) / 2**30
    head_mem = estimate_module_memory(head0, device_id=0) / 2**30
    logger.info(f"Estimated tensor memory on device 0: stack={stack_mem:.2f} GiB head={head_mem:.2f} GiB total={stack_mem+head_mem:.2f} GiB")

    logger.info("[3/3] Running forward(token_id=100, cur_pos=0)...")
    token_id = torch.tensor(100, dtype=torch.int32)
    try:
        results = layer.forward(token_id, with_mtp=False, cur_pos=0)
    except torch.cuda.OutOfMemoryError as e:
        logger.exception("OOM during forward")
        for did in range(layer.num_devices):
            logger.info("\n" + torch.cuda.memory_summary(device=did, abbreviated=True))
        return 1

    assert len(results) == layer.num_devices
    next_token = layer.get_next_token(device_id=0)
    logits = layer.get_logits(device_id=0)
    logger.info(
        "[3/3] next_token=%d, logits.shape=%s, finite=%s",
        next_token,
        tuple(logits.shape),
        logits.isfinite().all().item(),
    )

    for did in range(layer.num_devices):
        logger.info(mem_summary(did, "after forward"))

    layer.cleanup()
    logger.info("\n=== Memory diagnosis completed ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        logger.exception("\n=== FAILED: %s ===", exc)
        raise
