"""DeltaNet module for Qwen3.6."""

from typing import Any

import torch
import torch.nn.functional as F

from tilert.models.base import SerializableTileRTModule, TileRTModule
from tilert.models.common import RMSNorm, init_func, linear
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.models.qwen3_6.ops.delta_net import (
    DeltaNetOp,
    DeltaNetAlgorithm,
)
from tilert.models.qwen3_6.modules.moe import QwenMoeBlock


class QwenDeltaNetRef(TileRTModule):
    """Reference-only holder for DeltaNet weights.

    Mirrors the weight aliases in ``ops.delta_net.DeltaNetRefWeightsAlias``.
    The original checkpoint stores the linear attention weights under the
    ``linear_attn.*`` prefix; this holder keeps those tensors available for
    the golden/reference path while the optimized path consumes the TileRT
    sharded aliases produced by the weight converter.
    """

    def __init__(
        self,
        model_args: ModelArgsQwen36,
        device_id: int,
        num_devices: int,
    ):
        super().__init__(
            self.__class__.__name__,
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
        )
        self.delta_q_heads = model_args.delta_q_heads
        self.delta_k_heads = model_args.delta_k_heads
        self.delta_v_heads = model_args.delta_v_heads
        self.delta_key_head_dim = model_args.delta_key_head_dim
        self.delta_value_head_dim = model_args.delta_value_head_dim

        self.in_proj_qkv_weight: torch.Tensor | None = None
        self.in_proj_z_weight: torch.Tensor | None = None
        self.in_proj_a_weight: torch.Tensor | None = None
        self.in_proj_b_weight: torch.Tensor | None = None
        self.conv1d_weight: torch.Tensor | None = None
        self.A_log: torch.Tensor | None = None
        self.dt_bias: torch.Tensor | None = None
        self.norm_weight: torch.Tensor | None = None
        self.out_proj_weight: torch.Tensor | None = None
        self.input_layernorm_weight: torch.Tensor | None = None
        self.post_attention_layernorm_weight: torch.Tensor | None = None

    def get_weights_list(self) -> list[torch.Tensor]:
        return []

    def get_tilert_weights_alias(self) -> list[str]:
        return []

    def get_ref_weights_alias(self) -> list[str]:
        return []

    def device_sharding(self, weights_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        del weights_map
        return {}

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        prefix = "linear_attn"
        self.in_proj_qkv_weight = state_dict[f"{prefix}.in_proj_qkv.weight"]
        self.in_proj_z_weight = state_dict[f"{prefix}.in_proj_z.weight"]
        self.in_proj_a_weight = state_dict[f"{prefix}.in_proj_a.weight"]
        self.in_proj_b_weight = state_dict[f"{prefix}.in_proj_b.weight"]
        self.conv1d_weight = state_dict[f"{prefix}.conv1d.weight"]
        self.A_log = state_dict[f"{prefix}.A_log"]
        self.dt_bias = state_dict[f"{prefix}.dt_bias"]
        self.norm_weight = state_dict[f"{prefix}.norm.weight"]
        self.out_proj_weight = state_dict[f"{prefix}.out_proj.weight"]
        self.input_layernorm_weight = state_dict["input_layernorm.weight"]
        self.post_attention_layernorm_weight = state_dict["post_attention_layernorm.weight"]

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        del state_dict

    def init_random_weights(self) -> None:
        pass

    def init_tilert_vars(self, batch_size: int, seq_len: int) -> None:
        del batch_size, seq_len

    def golden_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference forward: simple DeltaNet wrapper.

        Delegates to ``DeltaNetOp.golden_forward`` to keep the reference
        computation in one place.  Weights are lazily initialized with random
        values on first call so the module can be sanity-tested without a
        checkpoint.
        """
        if self.attn.in_proj_qkv_weights is None:
            self.attn.init_random_weights(device=str(x.device))
        return self.attn.golden_forward(x, start_pos, state)

    def tilert_forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("QwenDeltaNetRef is reference-only")


class DeltaNet(SerializableTileRTModule):
    """DeltaNet linear attention layer for Qwen3.6.

    Implements a full Transformer layer:
      input_layernorm  -> DeltaNet attention -> residual ->
      post_attention_layernorm -> MoE FFN -> residual

    The actual DeltaNet recurrence / chunk-wise kernel will live in a dedicated
    CUDA kernel; this module wires the Python-side plumbing so the layer can
    be instantiated inside ``QwenTransformerStack``.
    """

    def __init__(
        self,
        model_args: ModelArgsQwen36,
        device_id: int,
        num_devices: int,
        remove_selected: bool = False,
        ffn_op: QwenMoeBlock | None = None,
    ):
        super().__init__(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            remove_selected=remove_selected,
        )

        self.delta_ref = QwenDeltaNetRef(
            model_args=model_args, device_id=device_id, num_devices=num_devices
        )
        self.register_op(self.delta_ref)

        self.attn = DeltaNetOp(
            model_args=model_args,
            device_id=device_id,
            num_devices=num_devices,
            algorithm=DeltaNetAlgorithm.GENERAL,
        )
        self.register_op(self.attn)

        self.ffn = (
            ffn_op
            if ffn_op is not None
            else QwenMoeBlock(
                model_args=model_args,
                device_id=device_id,
                num_devices=num_devices,
            )
        )
        self.register_op(self.ffn)

        self.input_layernorm = RMSNorm(model_args.dim, eps=model_args.eps)
        self.post_attention_layernorm = RMSNorm(model_args.dim, eps=model_args.eps)

    def _ensure_weights(self, x: torch.Tensor) -> None:
        """Lazy initialize weights for sanity testing without a checkpoint."""
        if self.attn.in_proj_qkv_weights is None:
            self.attn.init_random_weights(device=str(x.device))
        if not self.ffn.moe.rmsnorm_expert_proj.is_ref_weights_init:
            self.ffn.init_random_weights(device=str(x.device))
        if self.input_layernorm.weight.device != x.device:
            self.input_layernorm.to(x.device)
        if self.post_attention_layernorm.weight.device != x.device:
            self.post_attention_layernorm.to(x.device)

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load weights and also set the RMSNorm module weights from the checkpoint."""
        super().init_tilert_weights(state_dict)
        self._load_layernorm_weights(state_dict)

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load reference weights and also set the RMSNorm module weights."""
        super().init_reference_weights(state_dict)
        self._load_layernorm_weights(state_dict)

    def _load_layernorm_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Copy checkpoint layernorm weights into the RMSNorm modules.

        The checkpoint stores ``input_layernorm.weight`` and
        ``post_attention_layernorm.weight`` for each layer.  These must be loaded
        into the ``RMSNorm`` modules so the golden/reference path uses the real
        values instead of random initialization.
        """
        if "input_layernorm.weight" in state_dict:
            self.input_layernorm.weight.data.copy_(state_dict["input_layernorm.weight"])
        if "post_attention_layernorm.weight" in state_dict:
            self.post_attention_layernorm.weight.data.copy_(state_dict["post_attention_layernorm.weight"])

    def golden_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Reference forward: full DeltaNet layer with residuals and layer norms."""
        self._ensure_weights(x)
        prev_state = state.get("delta_state") if state is not None else None

        # Pre-attention norm + attention + residual.
        norm_x = self.input_layernorm(x)
        attn_out, new_state = self.attn.golden_forward(norm_x, start_pos, prev_state)
        h = x + attn_out

        # Post-attention norm + MoE FFN + residual.
        norm_h = self.post_attention_layernorm(h)
        ffn_out = self.ffn.golden_forward(norm_h)
        out = h + ffn_out

        next_state = {"delta_state": new_state} if state is not None else None
        return out, next_state

    def tilert_forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Optimized forward using ``DeltaNetOp`` + ``QwenMoeBlock``."""
        prev_state = state.get("delta_state") if state is not None else None
        attn_out, new_state = self.attn.forward(x, start_pos, prev_state)
        ffn_out = self.ffn.forward(attn_out)
        next_state = {"delta_state": new_state} if state is not None else None
        return ffn_out, next_state

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if self.flag_enable_tilert:
            return self.tilert_forward(x, start_pos, state)
        return self.golden_forward(x, start_pos, state)