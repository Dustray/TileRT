"""HF-source weight loader for QwenShowHandsLayer.

This module converts an original Hugging Face Qwen3.6 checkpoint into the
per-device TileRT weight layout expected by ``QwenShowHandsLayer``.  It does
not modify the source files; all sharding is performed in memory and the
resulting state dicts are stored on the target devices.
"""
from __future__ import annotations

from tilert import logger
import json
import os
from typing import Any
import torch
from safetensors import safe_open
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.models.qwen3_6.modules.transformer_stack import QwenTransformerStack
from tilert.models.qwen3_6.ops.rmsnorm_head_proj import RMSNormHeadProj
_HF_CHECKPOINT_CPU_CACHE: dict[str, dict[str, torch.Tensor]] = {}

def _is_hf_checkpoint(model_path: str) -> bool:

    """Return True if ``model_path`` points to an HF Qwen3.6 checkpoint."""
    logger.info(f'[{__file__.split(chr(47))[-1]}] _is_hf_checkpoint')
    index_path = os.path.join(model_path, 'model.safetensors.index.json')
    if not os.path.exists(index_path):
        return False
    try:
        with open(index_path, encoding='utf-8') as f:
            idx = json.load(f)
    except Exception:
        return False
    for key in idx.get('weight_map', {}):
        if 'model.language_model' in key:
            return True
    return False

def _strip_language_model_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

    """Strip the ``model.language_model`` prefix from HF text-only keys."""
    logger.info(f'[{__file__.split(chr(47))[-1]}] _strip_language_model_prefix')
    out: dict[str, torch.Tensor] = {}
    prefix = 'model.language_model.'
    for (key, tensor) in state_dict.items():
        if key.startswith(prefix):
            out[key[len(prefix):]] = tensor
        else:
            out[key] = tensor
    return out

def _layer_state_from_hf(state_dict: dict[str, torch.Tensor], layer_idx: int) -> dict[str, torch.Tensor]:

    """Extract a single layer's weights from the (stripped) HF state dict."""
    logger.info(f'[{__file__.split(chr(47))[-1]}] _layer_state_from_hf')
    prefix = f'layers.{layer_idx}.'
    out: dict[str, torch.Tensor] = {}
    for (key, tensor) in state_dict.items():
        if key.startswith(prefix):
            out[key[len(prefix):]] = tensor
    return out

def _unshard_to_per_device(sharded: dict[str, torch.Tensor], device_id: int, num_devices: int) -> dict[str, torch.Tensor]:
    """Convert a sharded state dict with a device dimension to per-device keys.

    Layout conventions:
      * Attention / DeltaNet / O-projection / lm_head weights are replicated on
        every device; ``device_sharding`` stacks them along dim 0.
      * MoE gate/up/down weights are tensor-parallel sharded along the
        *intermediate* dimension, while the second dimension is the
        ``num_devices`` TP shard slot.  They have shape
        ``(n_experts, num_devices, ...)`` and we extract ``[:, device_id]`` to
        keep the local inter_dim shard for every expert (shared + all routed).
      * Small replicated tensors such as ``unproj_o_gamma`` and
        ``exp_proj_weights`` are returned as-is.

    """
    logger.info(f'[{__file__.split(chr(47))[-1]}] _unshard_to_per_device')
    per_device: dict[str, torch.Tensor] = {}
    for (key, tensor) in sharded.items():
        if tensor.dim() == 0:
            per_device[key] = tensor
            continue
        if tensor.dim() >= 3 and tensor.size(1) == num_devices:
            per_device[key] = tensor[:, device_id]
            continue
        if tensor.size(0) == num_devices:
            per_device[key] = tensor[device_id]
            continue
        per_device[key] = tensor
    return per_device

def _load_hf_checkpoint_into_cpu(model_path: str, weight_index: dict[str, str]) -> dict[str, torch.Tensor]:

    """Load all text-only HF weights into CPU memory once and cache it."""
    logger.info(f'[{__file__.split(chr(47))[-1]}] _load_hf_checkpoint_into_cpu')
    global _HF_CHECKPOINT_CPU_CACHE
    if model_path in _HF_CHECKPOINT_CPU_CACHE:
        return _HF_CHECKPOINT_CPU_CACHE[model_path]
    target_files = sorted(set(weight_index.values()))
    state_dict: dict[str, torch.Tensor] = {}
    for weight_file in target_files:
        filepath = os.path.join(model_path, weight_file)
        with safe_open(filepath, framework='pt', device='cpu') as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    _HF_CHECKPOINT_CPU_CACHE[model_path] = state_dict
    return state_dict

def _split_state_dict_by_layer(state_dict: dict[str, torch.Tensor], n_layers: int) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:

    """Split full HF state dict into embeddings, per-layer, and head/norm."""
    logger.info(f'[{__file__.split(chr(47))[-1]}] _split_state_dict_by_layer')
    embeddings: dict[str, torch.Tensor] = {}
    head_norm: dict[str, torch.Tensor] = {}
    layers: list[dict[str, torch.Tensor]] = [dict() for _ in range(n_layers)]
    for (key, tensor) in state_dict.items():
        if key.startswith('model.language_model.embed_tokens.') or key == 'model.embed_tokens.weight':
            embeddings[key] = tensor
            continue
        if key == 'model.language_model.norm.weight' or key == 'norm.weight':
            head_norm[key] = tensor
            continue
        if key == 'lm_head.weight':
            head_norm[key] = tensor
            continue
        if 'model.language_model.layers.' in key:
            parts = key.split('.')
            layer_idx = int(parts[3])
            layers[layer_idx][key] = tensor
    return (embeddings, layers, head_norm)

def _precompute_all_device_states(model_path: str, model_args: ModelArgsQwen36, num_devices: int, stack: QwenTransformerStack) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[dict[str, dict[str, torch.Tensor]]], list[dict[str, dict[str, torch.Tensor]]]]:
    """Load the HF checkpoint once and shard every layer for all devices.

    Returns:
      * ``embeddings``: embedding tensors (replicated across devices later).
      * ``head_state_per_device``: dict mapping device_id -> head state dict.
      * ``per_device_layer_states``: list of length ``n_layers``; each element is
        a dict mapping device_id -> per-device TileRT state dict for that layer.
      * ``per_device_ref_layer_states``: same layout, but containing raw HF-style
        reference keys per device suitable for ``init_reference_weights``.

    This centralizes the CPU-heavy ``device_sharding`` work so that it runs
    exactly once, regardless of how many devices are being loaded.

    """
    logger.info(f'[{__file__.split(chr(47))[-1]}] _precompute_all_device_states')
    index_path = os.path.join(model_path, 'model.safetensors.index.json')
    with open(index_path, encoding='utf-8') as f:
        weight_index = json.load(f)['weight_map']
    full_state = _load_hf_checkpoint_into_cpu(model_path, weight_index)
    (embeddings, per_layer, head_norm) = _split_state_dict_by_layer(full_state, model_args.n_layers)
    head_proj = RMSNormHeadProj(model_args=model_args, device_id=0, num_devices=num_devices)
    (head_sharded_gamma, head_sharded_head) = head_proj.device_sharding({**head_norm, **embeddings})
    n_layers = model_args.n_layers
    head_state_per_device: dict[str, torch.Tensor] = {}
    for did in range(num_devices):
        head_state_per_device[str(did)] = {f'layer_{n_layers}_model.norm.weight_dev_{did}': head_sharded_gamma[did], f'layer_{n_layers}_lm_head.weight_dev_{did}': head_sharded_head[did]}
    per_device_layer_states: list[dict[str, dict[str, torch.Tensor]]] = []
    per_device_ref_layer_states: list[dict[str, dict[str, torch.Tensor]]] = []
    for (layer_idx, layer_state) in enumerate(per_layer):
        if not layer_state:
            per_device_layer_states.append({str(did): {} for did in range(num_devices)})
            per_device_ref_layer_states.append({str(did): {} for did in range(num_devices)})
            continue
        stripped = _strip_language_model_prefix(layer_state)
        local_state: dict[str, torch.Tensor] = {}
        lead = f'layers.{layer_idx}.'
        for (key, tensor) in stripped.items():
            if key.startswith(lead):
                local_state[key[len(lead):]] = tensor
            else:
                local_state[key] = tensor
        block = stack.exec_seq[layer_idx]
        sharded = block.device_sharding(local_state)
        layer_states: dict[str, dict[str, torch.Tensor]] = {}
        for did in range(num_devices):
            per_device = _unshard_to_per_device(sharded, did, num_devices)
            device_state = {f'layer_{layer_idx}_{alias}_dev_{did}': tensor for (alias, tensor) in per_device.items()}
            layer_states[str(did)] = device_state
        per_device_layer_states.append(layer_states)
        ref_aliases = block.get_ref_weights_alias()
        ref_layer_states: dict[str, dict[str, torch.Tensor]] = {}
        for did in range(num_devices):
            ref_device_state: dict[str, torch.Tensor] = {}
            for ref_key in ref_aliases:
                if ref_key not in local_state:
                    continue
                tensor = local_state[ref_key]
                if tensor.dim() == 0:
                    ref_device_state[ref_key] = tensor
                elif 'mlp.experts.gate_up_proj' in ref_key:
                    full_inter_dim = tensor.size(1) // 2
                    local_inter_dim = full_inter_dim // num_devices
                    start = did * local_inter_dim
                    end = start + local_inter_dim
                    ref_device_state[ref_key] = torch.cat([tensor[:, start:end, :], tensor[:, full_inter_dim + start:full_inter_dim + end, :]], dim=1)
                elif 'mlp.experts.down_proj' in ref_key:
                    local_inter_dim = tensor.size(2) // num_devices
                    start = did * local_inter_dim
                    end = start + local_inter_dim
                    ref_device_state[ref_key] = tensor[:, :, start:end]
                elif tensor.dim() >= 3 and tensor.size(1) == num_devices:
                    ref_device_state[ref_key] = tensor[:, did]
                elif tensor.size(0) == num_devices and ref_key.endswith('.weight'):
                    ref_device_state[ref_key] = tensor[did]
                elif 'mlp.shared_expert.gate_proj.weight' == ref_key or 'mlp.shared_expert.up_proj.weight' == ref_key:
                    local_inter_dim = tensor.size(0) // num_devices
                    start = did * local_inter_dim
                    end = start + local_inter_dim
                    ref_device_state[ref_key] = tensor[start:end, :]
                elif 'mlp.shared_expert.down_proj.weight' == ref_key:
                    local_inter_dim = tensor.size(1) // num_devices
                    start = did * local_inter_dim
                    end = start + local_inter_dim
                    ref_device_state[ref_key] = tensor[:, start:end]
                else:
                    ref_device_state[ref_key] = tensor
            ref_layer_states[str(did)] = ref_device_state
        per_device_ref_layer_states.append(ref_layer_states)
    return (embeddings, head_state_per_device, per_device_layer_states, per_device_ref_layer_states)

def load_hf_source_weights(model_path: str, model_args: ModelArgsQwen36, num_devices: int, device_id: int, stack: QwenTransformerStack, precomputed: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[dict[str, dict[str, torch.Tensor]]], list[dict[str, dict[str, torch.Tensor]]]] | None=None) -> dict[str, torch.Tensor]:
    """Build the per-device TileRT state dict from an HF checkpoint.

    When ``precomputed`` is provided (the common case for multi-device loading),
    this function only moves the already-sharded CPU tensors to
    ``cuda:{device_id}`` and assembles the final state dict.  The expensive
    ``device_sharding`` calls happen once in ``_precompute_all_device_states``.

    The returned dict contains keys that match the existing
    ``init_tilert_weights`` conventions:
      * ``model.embed_tokens.weight`` (replicated)
      * ``freqs_cos`` / ``freqs_sin``
      * ``layer_{idx}_{alias}_dev_{device_id}`` for every layer tensor
      * ``layer_{n_layers}_model.norm.weight_dev_{device_id}``
      * ``layer_{n_layers}_lm_head.weight_dev_{device_id}``

    Additionally, for each layer the raw reference tensors (HF keys) for this
    device are stored under ``ref_layer_{idx}_{hf_key}_dev_{device_id}`` so that
    ``init_reference_weights`` can be called once without triggering lazy random
    initialization on every forward step.

    """
    logger.info(f'[{__file__.split(chr(47))[-1]}] load_hf_source_weights')
    dev = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
    if precomputed is None:
        (embeddings, head_state_per_device, per_device_layer_states, per_device_ref_layer_states) = _precompute_all_device_states(model_path, model_args, num_devices, stack)
    else:
        (embeddings, head_state_per_device, per_device_layer_states, per_device_ref_layer_states) = precomputed
    result: dict[str, torch.Tensor] = {}
    if 'model.embed_tokens.weight' in embeddings:
        embed_key = 'model.embed_tokens.weight'
    else:
        embed_key = 'model.language_model.embed_tokens.weight'
    result['model.embed_tokens.weight'] = embeddings[embed_key].to(dev)
    from tilert.models.utils import precompute_mrope_embed
    (cos, sin) = precompute_mrope_embed(model_args)
    result['freqs_cos'] = cos
    result['freqs_sin'] = sin
    result['freqs_cis'] = cos
    for layer_states in per_device_layer_states:
        device_state = layer_states.get(str(device_id), {})
        for (alias, tensor) in device_state.items():
            result[alias] = tensor.to(dev)
    for (layer_idx, ref_layer_states) in enumerate(per_device_ref_layer_states):
        device_state = ref_layer_states.get(str(device_id), {})
        for (ref_key, tensor) in device_state.items():
            result[f'ref_layer_{layer_idx}_{ref_key}_dev_{device_id}'] = tensor.to(dev)
    for (alias, tensor) in head_state_per_device[str(device_id)].items():
        result[alias] = tensor.to(dev)
    return result
