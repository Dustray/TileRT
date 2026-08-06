"""Qwen3.6-35B-A3B Transformer layer stack module.

This module is not related to DeepSeek's DSA (DeepSeek Sparse Attention).
Qwen3.6 uses GQA + DeltaNet; this file simply stacks the 40 heterogeneous
Transformer layers and provides the golden / TileRT forward dispatchers.
"""
from typing import Any
import torch
from tilert import logger
from tilert.models.base import SerializableTileRTModule
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.models.qwen3_6.modules.gated_attention import GatedAttention
from tilert.models.qwen3_6.modules.delta_net import DeltaNet
from tilert.models.qwen3_6.modules.moe import QwenMoeBlock
__all__ = ['QwenTransformerStack']

class QwenTransformerStack(SerializableTileRTModule):
    """Transformer layer stack for Qwen3.6.

    Qwen3.6 has a heterogeneous layer structure:
    - 30 DeltaNet layers (3 per block × 10 blocks)
    - 10 Gated Attention layers (1 per block × 10 blocks)

    This module instantiates those layers in order and provides the high-level
    ``forward`` dispatcher.  The optimized TileRT path will be implemented once
    the dedicated kernels are available; until then the golden path serves as a
    reference and sanity check.

    ``cached_ffn_ops`` is an optional layer-level FFN/MoE cache shared across
    layers (similar to the mechanism in DSv3.2's DSA).  It is used here purely
    to reduce memory during random-init reference sanity tests, not because
    Qwen3.6 itself uses DeepSeek Sparse Attention.
    """

    def __init__(self, model_args: ModelArgsQwen36, device_id: int, num_devices: int, cached_ffn_ops: list | None=None, moe_sync_callback: Any | None=None):
        super().__init__(model_args=model_args, device_id=device_id, num_devices=num_devices, remove_selected=True)
        self.model_args = model_args
        self.device_id = device_id
        self.num_devices = num_devices
        self.moe_sync_callback = moe_sync_callback
        if cached_ffn_ops is not None:
            assert len(cached_ffn_ops) == model_args.n_layers, f'Expected {model_args.n_layers} cached FFN ops, got {len(cached_ffn_ops)}'
        self.layer_types: list[int] = []
        for _ in range(model_args.n_blocks):
            self.layer_types.extend([0, 0, 0, 1])
        self.layer_types = self.layer_types[:model_args.n_layers]
        logger.info(f'QwenTransformerStack: building {len(self.layer_types)} layers on cuda:{device_id} (num_devices={num_devices}), cached_ffn_ops={cached_ffn_ops is not None}')
        delta_count = sum((1 for t in self.layer_types if t == 0))
        gqa_count = len(self.layer_types) - delta_count
        logger.info(f'QwenTransformerStack: {delta_count} DeltaNet + {gqa_count} GatedAttention layers')
        for (layer_idx, layer_type) in enumerate(self.layer_types):
            ffn_op = cached_ffn_ops[layer_idx] if cached_ffn_ops else None
            if layer_type == 0:
                block = DeltaNet(model_args=model_args, device_id=device_id, num_devices=num_devices, ffn_op=ffn_op)
            else:
                block = GatedAttention(model_args=model_args, device_id=device_id, num_devices=num_devices)
            self.register_op(block, prefix=f'layer_{layer_idx}_', suffix=f'_dev_{device_id}')
            block.moe_sync_callback = self.moe_sync_callback
            block.ffn.moe.expert_down_allreduce.moe_sync_callback = self.moe_sync_callback
            logger.debug(f"Registered layer {layer_idx}: {('DeltaNet' if layer_type == 0 else 'GatedAttention')}")
        logger.info('[dev={device_id}]QwenTransformerStack 构建完成')

    def _prepare_mrope_embed(self, mrope_embed: tuple[torch.Tensor, torch.Tensor] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize RoPE tables to a (cos, sin) tuple."""
        if isinstance(mrope_embed, tuple):
            return mrope_embed
        freqs_cis = mrope_embed
        if torch.is_complex(freqs_cis):
            freqs_cis = torch.view_as_real(freqs_cis)
        else:
            freqs_cis = freqs_cis.view(freqs_cis.size(0), -1, 2)
        return (freqs_cis[..., 0], freqs_cis[..., 1])

    def golden_forward(self, x: torch.Tensor, start_pos: int, mrope_embed: tuple[torch.Tensor, torch.Tensor] | torch.Tensor, caches: dict[str, Any] | None=None) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reference forward through all 40 heterogeneous layers.

        Dispatches each block via ``block.forward()`` so that individual ops
        can decide between golden/tilert based on ``flag_enable_tilert``.
        """
        logger.info(f'[QwenTransformerStack.golden_forward_{self.device_id}] ENTRY: x.shape={x.shape}, start_pos={start_pos}, exec_seq_len={len(self.exec_seq)}')
        mrope_embed = self._prepare_mrope_embed(mrope_embed)
        if caches is None:
            logger.info(f'[QwenTransformerStack.golden_forward_{self.device_id}] Initializing layer caches')
            caches = self._init_layer_caches(mrope_embed)
        h = x
        shared_k_cache = caches['k_cache']
        shared_v_cache = caches['v_cache']
        seq_len = x.size(1)
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float('-inf'), dtype=torch.float32, device=x.device)
            mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)
            logger.info(f'[QwenTransformerStack.golden_forward_{self.device_id}] Created attention mask for seq_len={seq_len}')
        logger.info(f'[QwenTransformerStack.golden_forward_{self.device_id}] Starting layer loop: {len(self.exec_seq)} layers')
        for (layer_idx, block) in enumerate(self.exec_seq):
            block_type = type(block).__name__
            logger.info(f'[QwenTransformerStack.golden_forward_{self.device_id}] === Layer {layer_idx}/{len(self.exec_seq) - 1}: {block_type} ===')
            logger.info(f'[QwenTransformerStack.golden_forward_{self.device_id}]   Input h: shape={h.shape}, dtype={h.dtype}, device={h.device}')
            if isinstance(block, DeltaNet):
                logger.info(f'[QwenTransformerStack.golden_forward_{self.device_id}]   Calling DeltaNet.forward')
                (out, layer_state) = block.forward(h, start_pos, caches.get('delta_state', {}).get(layer_idx))
                if layer_state is not None:
                    caches.setdefault('delta_state', {})[layer_idx] = layer_state['delta_state']
                logger.info(f'[QwenTransformerStack.golden_forward_{self.device_id}]   DeltaNet.forward output: shape={out.shape}')
            elif isinstance(block, GatedAttention):
                logger.info(f'[QwenTransformerStack.golden_forward_{self.device_id}]   Calling GatedAttention.forward')
                (out, shared_k_cache, shared_v_cache) = block.forward(h, start_pos, mrope_embed, shared_k_cache, shared_v_cache, mask)
                logger.info(f'[QwenTransformerStack.golden_forward_{self.device_id}]   GatedAttention.forward output: shape={out.shape}')
                logger.info(f'[QwenTransformerStack.golden_forward_{self.device_id}]   Updated caches: k_cache={shared_k_cache.shape}, v_cache={shared_v_cache.shape}')
            else:
                raise TypeError(f'Unsupported block type: {type(block)}')
            if torch.isnan(out).any() or torch.isinf(out).any():
                logger.warning(f'QwenTransformerStack layer {layer_idx} produced NaN/Inf; mean={out.float().mean().item():.4f}, std={out.float().std().item():.4f}')
            h = out
            if layer_idx == 9:
                break
            if layer_idx % 10 == 0 or layer_idx == len(self.exec_seq) - 1:
                logger.info(f'[QwenTransformerStack.golden_forward_{self.device_id}]   Layer {layer_idx} output stats: mean={h.float().mean().item():.4f}, std={h.float().std().item():.4f}, min={h.float().min().item():.4f}, max={h.float().max().item():.4f}')
        caches['k_cache'] = shared_k_cache
        caches['v_cache'] = shared_v_cache
        logger.info(f'[QwenTransformerStack.golden_forward_{self.device_id}] EXIT: h.shape={h.shape}, dtype={h.dtype}')
        return (h, caches)

    def tilert_forward(self, x: torch.Tensor, start_pos: int, mrope_embed: tuple[torch.Tensor, torch.Tensor], caches: dict[str, Any] | None=None) -> tuple[torch.Tensor, dict[str, Any]]:
        """Optimized forward placeholder.

        Currently routes through ``golden_forward`` because the dedicated
        Qwen3.6 CUDA-graph wrappers are not yet implemented.
        """
        return self.golden_forward(x, start_pos, mrope_embed, caches)

    def forward(self, x: torch.Tensor, start_pos: int, mrope_embed: tuple[torch.Tensor, torch.Tensor], caches: dict[str, Any] | None=None) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.flag_enable_tilert:
            return self.tilert_forward(x, start_pos, mrope_embed, caches)
        return self.golden_forward(x, start_pos, mrope_embed, caches)

    def _init_layer_caches(self, mrope_embed: tuple[torch.Tensor, torch.Tensor]) -> dict[str, Any]:
        """Allocate KV caches for Gated Attention layers.

        DeltaNet layers carry their own recurrent state in ``caches["delta_state"]``.
        """
        dev = f'cuda:{self.device_id}'
        cache_seq_len = self.model_args.max_seq_len + self.model_args.kv_cache_pad
        return {'k_cache': torch.zeros(self.model_args.max_batch_size, cache_seq_len, self.model_args.n_kv_heads, self.model_args.qk_head_dim, dtype=torch.bfloat16, device=dev), 'v_cache': torch.zeros(self.model_args.max_batch_size, cache_seq_len, self.model_args.n_kv_heads, self.model_args.v_head_dim, dtype=torch.bfloat16, device=dev), 'mrope_embed': mrope_embed, 'delta_state': {}}

    def get_tilert_weights_alias(self) -> list[str]:
        """Aggregate aliases from all registered ops."""
        return super().get_tilert_weights_alias()

    def from_pretrained(self, model_path: str) -> None:
        """Load pretrained weights."""
        raise NotImplementedError('QwenTransformerStack weight loading not yet implemented.')

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass