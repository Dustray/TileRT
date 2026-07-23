"""HF-source weight loader for QwenShowHandsLayer.

This module converts an original Hugging Face Qwen3.6 checkpoint into the
per-device TileRT weight layout expected by ``QwenShowHandsLayer``.  It does
not modify the source files; all sharding is performed in memory and the
resulting state dicts are stored on the target devices.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import torch
from safetensors import safe_open

from tilert import logger
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.models.qwen3_6.modules.transformer_stack import QwenTransformerStack
from tilert.models.qwen3_6.ops.rmsnorm_head_proj import RMSNormHeadProj

# Module-level cache so the full HF checkpoint is loaded into CPU memory only once
# when serving multiple devices.  The cache is keyed by the model directory path.
_HF_CHECKPOINT_CPU_CACHE: dict[str, dict[str, torch.Tensor]] = {}


def _is_hf_checkpoint(model_path: str) -> bool:
    """Return True if ``model_path`` points to an HF Qwen3.6 checkpoint."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return False
    try:
        with open(index_path, encoding="utf-8") as f:
            idx = json.load(f)
    except Exception:
        return False
    # HF text-only keys always contain ``model.language_model``.
    for key in idx.get("weight_map", {}):
        if "model.language_model" in key:
            return True
    return False


def _strip_language_model_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip the ``model.language_model`` prefix from HF text-only keys."""
    out: dict[str, torch.Tensor] = {}
    prefix = "model.language_model."
    for key, tensor in state_dict.items():
        if key.startswith(prefix):
            out[key[len(prefix) :]] = tensor
        else:
            out[key] = tensor
    return out


def _layer_state_from_hf(
    state_dict: dict[str, torch.Tensor], layer_idx: int
) -> dict[str, torch.Tensor]:
    """Extract a single layer's weights from the (stripped) HF state dict."""
    prefix = f"layers.{layer_idx}."
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.startswith(prefix):
            out[key[len(prefix) :]] = tensor
    return out


def _unshard_to_per_device(
    sharded: dict[str, torch.Tensor],
    device_id: int,
    num_devices: int,
) -> dict[str, torch.Tensor]:
    """Convert a sharded state dict with a device dimension to per-device keys.

    TileRT ``device_sharding`` methods stack per-device slices either at
    dimension 0 (the default, e.g. QKV/O projections) or at dimension 1 for
    MoE weights whose first dimension is the expert count.  This helper
    extracts the slice belonging to ``device_id`` and avoids keeping full
    replicated tensors on every device.

    Heuristic:
      * 0-D tensors: returned as-is.
      * Dim 0 == ``num_devices``: device shards are along dim 0 (e.g.
        ``qkv_proj_weights``, ``o_proj_weights``, ``shared_expert_gate``).
      * Dim 1 == ``num_devices``: device shards are along dim 1 for stacked
        expert weights whose first dimension is the expert count (e.g.
        ``exp_gate_weights``, ``exp_up_weights``, ``exp_down_weights``).
      * Otherwise: keep the full tensor (small replicated tensors such as
        ``unproj_o_gamma`` and ``exp_proj_weights``).
    """
    per_device: dict[str, torch.Tensor] = {}
    for key, tensor in sharded.items():
        if tensor.dim() == 0:
            per_device[key] = tensor
            continue
        # Device shards stacked along dimension 0.
        if tensor.size(0) == num_devices:
            per_device[key] = tensor[device_id]
            continue
        # Device shards stacked along dimension 1 for MoE expert-stacked
        # weights; the first dimension is the expert count (257 routed + 1
        # shared = 258 for gate/up, 257 for down after concatenation).
        if tensor.dim() >= 2 and tensor.size(1) == num_devices:
            per_device[key] = tensor[:, device_id]
            continue
        per_device[key] = tensor
    return per_device


def _head_state_from_hf(
    state_dict: dict[str, torch.Tensor],
    head_proj: RMSNormHeadProj,
    num_devices: int,
    device_id: int,
) -> dict[str, torch.Tensor]:
    """Build per-device TileRT head/norm state from the HF state dict."""
    head_input: dict[str, torch.Tensor | None] = {
        "model.language_model.norm.weight": state_dict.get("model.language_model.norm.weight"),
        "lm_head.weight": state_dict.get("lm_head.weight"),
    }
    # Also accept the stripped key name if present.
    if "norm.weight" in state_dict:
        head_input["model.language_model.norm.weight"] = state_dict["norm.weight"]
    head_input = {k: v for k, v in head_input.items() if v is not None}
    gamma, head = head_proj.device_sharding(head_input)
    # TileRT keys expect the layer_{n_layers}_ prefix in the end2end loader.
    n_layers = head_proj.model_args.n_layers
    return {
        f"layer_{n_layers}_model.norm.weight_dev_{device_id}": gamma[device_id],
        f"layer_{n_layers}_lm_head.weight_dev_{device_id}": head[device_id],
    }


def _load_hf_checkpoint_into_cpu(
    model_path: str,
    weight_index: dict[str, str],
) -> dict[str, torch.Tensor]:
    """Load all text-only HF weights into CPU memory once and cache it.

    This is a one-time cost; the full checkpoint is ~35B bf16 (~70 GiB) and
    should fit into the container's CPU RAM (several hundred GiB).  The cache is
    shared across per-device loader calls so the 26 safetensors shards are only
    read from disk once.
    """
    global _HF_CHECKPOINT_CPU_CACHE
    if model_path in _HF_CHECKPOINT_CPU_CACHE:
        logger.info("HF-source loader: reusing cached CPU checkpoint")
        return _HF_CHECKPOINT_CPU_CACHE[model_path]

    logger.info("HF-source loader: loading full HF checkpoint into CPU memory")
    target_files = sorted(set(weight_index.values()))
    state_dict: dict[str, torch.Tensor] = {}
    for weight_file in target_files:
        filepath = os.path.join(model_path, weight_file)
        logger.info(f"HF-source loader: loading {weight_file}")
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    logger.info(f"HF-source loader: loaded {len(state_dict)} tensors")
    _HF_CHECKPOINT_CPU_CACHE[model_path] = state_dict
    return state_dict


def _split_state_dict_by_layer(
    state_dict: dict[str, torch.Tensor],
    n_layers: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Split full HF state dict into embeddings, per-layer, and head/norm."""
    embeddings: dict[str, torch.Tensor] = {}
    head_norm: dict[str, torch.Tensor] = {}
    layers: list[dict[str, torch.Tensor]] = [dict() for _ in range(n_layers)]

    for key, tensor in state_dict.items():
        if key.startswith("model.language_model.embed_tokens.") or key == "model.embed_tokens.weight":
            embeddings[key] = tensor
            continue
        if key == "model.language_model.norm.weight" or key == "norm.weight":
            head_norm[key] = tensor
            continue
        if key == "lm_head.weight":
            head_norm[key] = tensor
            continue
        # text-only layers live under ``model.language_model.layers.{idx}.``
        if "model.language_model.layers." in key:
            parts = key.split(".")
            # model . language_model . layers . idx . rest
            layer_idx = int(parts[3])
            layers[layer_idx][key] = tensor
    return embeddings, layers, head_norm


def load_hf_source_weights(
    model_path: str,
    model_args: ModelArgsQwen36,
    num_devices: int,
    device_id: int,
    stack: QwenTransformerStack,
) -> dict[str, torch.Tensor]:
    """Build the per-device TileRT state dict from an HF checkpoint.

    This function is intended to be called once per device.  It shards the
    loaded CPU tensors according to each op's ``device_sharding`` method and
    moves only the selected device shard to ``cuda:{device_id}``.

    The returned dict contains keys that match the existing
    ``init_tilert_weights`` conventions:
      * ``model.embed_tokens.weight`` (replicated)
      * ``freqs_cos`` / ``freqs_sin``
      * ``layer_{idx}_{alias}_dev_{device_id}`` for every layer tensor
      * ``layer_{n_layers}_model.norm.weight_dev_{device_id}``
      * ``layer_{n_layers}_lm_head.weight_dev_{device_id}``
    """
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as f:
        weight_index = json.load(f)["weight_map"]

    # Load once.  In a multi-thread/multi-process setting the caller may want
    # to cache this; for now we load per-call to keep the loader self-contained.
    full_state = _load_hf_checkpoint_into_cpu(model_path, weight_index)
    embeddings, per_layer, head_norm = _split_state_dict_by_layer(
        full_state, model_args.n_layers
    )

    # Build head projection on the target device (small, cheap).
    dev = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    head_proj = RMSNormHeadProj(
        model_args=model_args,
        device_id=device_id,
        num_devices=num_devices,
    )
    head_state = _head_state_from_hf(
        {**head_norm, **embeddings},
        head_proj,
        num_devices,
        device_id,
    )

    result: dict[str, torch.Tensor] = {}

    # Replicate embedding table on every device.
    if "model.embed_tokens.weight" in embeddings:
        embed_key = "model.embed_tokens.weight"
    else:
        # Original HF checkpoint key.
        embed_key = "model.language_model.embed_tokens.weight"
    result["model.embed_tokens.weight"] = embeddings[embed_key].to(dev)

    # M-RoPE tables.  Keep these in CPU-compatible float32; they are tiny.
    from tilert.models.utils import precompute_mrope_embed

    cos, sin = precompute_mrope_embed(model_args)
    result["freqs_cos"] = cos
    result["freqs_sin"] = sin
    result["freqs_cis"] = cos  # backward compat

    # Shard each layer and rename to TileRT per-device keys.
    for layer_idx, layer_state in enumerate(per_layer):
        if not layer_state:
            continue
        stripped = _strip_language_model_prefix(layer_state)
        # Strip the leading ``layers.{idx}.`` so the block sees the local
        # aliases that match its op-level ``ref_weights_alias``.
        local_state = {}
        lead = f"layers.{layer_idx}."
        for key, tensor in stripped.items():
            if key.startswith(lead):
                local_state[key[len(lead):]] = tensor
            else:
                local_state[key] = tensor
        block = stack.exec_seq[layer_idx]
        sharded = block.device_sharding(local_state)
        per_device = _unshard_to_per_device(sharded, device_id, num_devices)
        for alias, tensor in per_device.items():
            result[f"layer_{layer_idx}_{alias}_dev_{device_id}"] = tensor.to(dev)

    result.update(head_state)
    # Move head tensors to target device.
    for key in list(head_state.keys()):
        result[key] = result[key].to(dev)

    logger.info(
        f"HF-source loader: device {device_id} received {len(result)} tensors"
    )
    return result
