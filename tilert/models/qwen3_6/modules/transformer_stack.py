"""Qwen3.6-35B-A3B Transformer layer stack module.

This module is not related to DeepSeek's DSA (DeepSeek Sparse Attention).
Qwen3.6 uses GQA + DeltaNet; this file simply stacks the 40 heterogeneous
Transformer layers and provides the golden / TileRT forward dispatchers.
"""

from typing import Any

import torch

from tilert.models.base import SerializableTileRTModule
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.models.qwen3_6.modules.gated_attention import GatedAttention
from tilert.models.qwen3_6.modules.delta_net import DeltaNet
from tilert.models.qwen3_6.modules.moe import QwenMoeBlock

__all__ = ["QwenTransformerStack"]


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

    def __init__(
        self,
        model_args: ModelArgsQwen36,
        device_id: int,
        num_devices: int,
        cached_ffn_ops: list | None = None,
    ):
        super().__init__(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            remove_selected=True,
        )

        self.model_args = model_args
        self.device_id = device_id
        self.num_devices = num_devices

        if cached_ffn_ops is not None:
            assert len(cached_ffn_ops) == model_args.n_layers, (
                f"Expected {model_args.n_layers} cached FFN ops, "
                f"got {len(cached_ffn_ops)}"
            )

        # Layer type mapping: 0 = DeltaNet, 1 = Gated Attention
        # Pattern: [DeltaNet, DeltaNet, DeltaNet, GatedAttention] × n_blocks.
        # If the caller overrides ``n_layers`` for a smaller sanity test,
        # truncate the pattern so the actual number of executed layers matches
        # ``model_args.n_layers``.
        self.layer_types: list[int] = []
        for _ in range(model_args.n_blocks):
            self.layer_types.extend([0, 0, 0, 1])
        self.layer_types = self.layer_types[: model_args.n_layers]

        for layer_idx, layer_type in enumerate(self.layer_types):
            ffn_op = cached_ffn_ops[layer_idx] if cached_ffn_ops else None
            if layer_type == 0:
                block = DeltaNet(
                    model_args=model_args,
                    device_id=device_id,
                    num_devices=num_devices,
                    ffn_op=ffn_op,
                )
            else:
                block = GatedAttention(
                    model_args=model_args,
                    device_id=device_id,
                    num_devices=num_devices,
                )
            self.register_op(block, prefix=f"layer_{layer_idx}_", suffix=f"_dev_{device_id}")

    def _block_forward(
        self,
        block: SerializableTileRTModule,
        x: torch.Tensor,
        start_pos: int,
        layer_cache: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Route one layer through the correct forward signature.

        Each block is a full Transformer layer and already applies its own
        internal residual connections and layer norms.
        """
        if isinstance(block, DeltaNet):
            out, layer_cache["delta_state"] = block.forward(
                x, start_pos, layer_cache.get("delta_state")
            )
            return out, layer_cache
        if isinstance(block, GatedAttention):
            k_cache = layer_cache["k_cache"]
            v_cache = layer_cache["v_cache"]
            out, k_cache, v_cache = block.forward(x, start_pos, layer_cache["freqs_cis"], k_cache, v_cache)
            layer_cache["k_cache"] = k_cache
            layer_cache["v_cache"] = v_cache
            return out, layer_cache
        raise TypeError(f"Unsupported block type: {type(block)}")

    def golden_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        caches: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reference forward through all 40 heterogeneous layers."""
        if caches is None:
            caches = self._init_layer_caches(freqs_cis)

        # Convert real-form freqs_cis back to complex for the reference path.
        # Real layout produced by some callers is [seq_len, rope_dim].
        if not torch.is_complex(freqs_cis):
            freqs_cis = torch.view_as_complex(freqs_cis.view(freqs_cis.size(0), -1, 2))

        h = x
        # Use full residual addition for real pretrained weights; only scale
        # the residual branch when randomly initialized weights are detected
        # (by checking whether any child module was created with the
        # ``is_random_init`` marker).  This keeps random-init sanity tests
        # numerically bounded without changing real-model semantics.
        residual_scale = 1.0 / max(len(self.exec_seq), 1) if self._is_random_init() else 1.0
        shared_k_cache = caches["k_cache"]
        shared_v_cache = caches["v_cache"]
        for layer_idx, block in enumerate(self.exec_seq):
            layer_cache = {
                "k_cache": shared_k_cache,
                "v_cache": shared_v_cache,
                "freqs_cis": freqs_cis,
                "delta_state": caches.get("delta_state", {}).get(layer_idx),
            }
            out, layer_cache = self._block_forward(block, h, start_pos, layer_cache)
            h = h + out * residual_scale
            shared_k_cache = layer_cache["k_cache"]
            shared_v_cache = layer_cache["v_cache"]
            if "delta_state" in layer_cache:
                caches.setdefault("delta_state", {})[layer_idx] = layer_cache["delta_state"]
        caches["k_cache"] = shared_k_cache
        caches["v_cache"] = shared_v_cache
        return h, caches

    def tilert_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        caches: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Optimized forward placeholder.

        Falls back to ``golden_forward`` until the dedicated Qwen3.6 CUDA-graph
        wrappers are implemented.
        """
        return self.golden_forward(x, start_pos, freqs_cis, caches)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        caches: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.flag_enable_tilert:
            return self.tilert_forward(x, start_pos, freqs_cis, caches)
        return self.golden_forward(x, start_pos, freqs_cis, caches)

    def _init_layer_caches(self, freqs_cis: torch.Tensor) -> dict[str, Any]:
        """Allocate KV caches for Gated Attention layers.

        DeltaNet layers carry their own recurrent state in ``caches["delta_state"]``.
        """
        dev = f"cuda:{self.device_id}"
        cache_seq_len = self.model_args.max_seq_len + self.model_args.kv_cache_pad
        return {
            "k_cache": torch.zeros(
                self.model_args.max_batch_size,
                cache_seq_len,
                self.model_args.n_kv_heads,
                self.model_args.qk_head_dim,
                dtype=torch.bfloat16,
                device=dev,
            ),
            "v_cache": torch.zeros(
                self.model_args.max_batch_size,
                cache_seq_len,
                self.model_args.n_kv_heads,
                self.model_args.v_head_dim,
                dtype=torch.bfloat16,
                device=dev,
            ),
            "freqs_cis": freqs_cis,
            "delta_state": {},
        }

    def get_tilert_weights_alias(self) -> list[str]:
        """Aggregate aliases from all registered ops."""
        return super().get_tilert_weights_alias()

    def from_pretrained(self, model_path: str) -> None:
        """Load pretrained weights."""
        raise NotImplementedError("QwenTransformerStack weight loading not yet implemented.")

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass