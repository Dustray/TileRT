"""DeltaNet module for Qwen3.6."""

from typing import Any

import torch
import torch.nn.functional as F

from tilert import logger
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
        return self.get_ref_weights_alias()

    def get_ref_weights_alias(self) -> list[str]:
        return ["input_layernorm.weight", 
                "post_attention_layernorm.weight", 
                "mlp.gate.weight",
                "mlp.experts.gate_up_proj",
                "mlp.shared_expert.gate_proj.weight",
                "mlp.shared_expert.up_proj.weight",
                "mlp.shared_expert_gate.weight",
                "mlp.experts.down_proj",
                "mlp.shared_expert.down_proj.weight"
                ]

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
        self.recurrent_state = None
        self.conv_state = None

    def _ensure_weights(self, x: torch.Tensor) -> None:
        """Lazy initialize weights for sanity testing without a checkpoint."""
        # if self.attn.in_proj_qkv_weights is None:
        #     self.attn.init_tilert_weights(device=str(x.device))
        # if not self.ffn.moe.rmsnorm_expert_proj.is_ref_weights_init:
        #     self.ffn.init_tilert_weights(device=str(x.device))
        if self.input_layernorm.weight.device != x.device:
            self.input_layernorm.to(x.device)
        if self.post_attention_layernorm.weight.device != x.device:
            self.post_attention_layernorm.to(x.device)

    def init_tilert_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load weights and also set the RMSNorm module weights from the checkpoint."""
        logger.debug(f"{self.op_name}: loading tilert weights + layernorms")
        super().init_tilert_weights(state_dict)
        self._load_layernorm_weights(state_dict)
        if self.attn.in_proj_qkv_weights is None:
            self.attn.init_reference_weights(state_dict)
        if not self.ffn.moe.rmsnorm_expert_proj.is_ref_weights_init:
            self.ffn.init_reference_weights(state_dict)

    def init_reference_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load reference weights and also set the RMSNorm module weights."""
        logger.debug(f"{self.op_name}: loading reference weights + layernorms")
        super().init_reference_weights(state_dict)
        # ``DeltaNetOp`` currently only loads tilert-layout weights.  The
        # reference holder already received the HF reference tensors; copy
        # them into the op so golden_forward does not need lazy random init.
        if self.delta_ref.in_proj_qkv_weight is not None:
            self.attn.in_proj_qkv_weights = self.delta_ref.in_proj_qkv_weight
            self.attn.in_proj_z_weights = self.delta_ref.in_proj_z_weight
            self.attn.in_proj_a_weights = self.delta_ref.in_proj_a_weight
            self.attn.in_proj_b_weights = self.delta_ref.in_proj_b_weight
            self.attn.conv1d_weights = self.delta_ref.conv1d_weight
            self.attn.A_log = self.delta_ref.A_log
            self.attn.dt_bias = self.delta_ref.dt_bias
            self.attn.norm_weights = self.delta_ref.norm_weight
            self.attn.out_proj_weights = self.delta_ref.out_proj_weight
            self.attn.is_ref_weights_init = True
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
        logger.info(f"[DeltaNet.golden_forward_{self.device_id}] ENTRY: x.shape={x.shape}, start_pos={start_pos}, has_state={state is not None}")
        
        self._ensure_weights(x)
        prev_state = state.get("delta_state") if state is not None else None
        logger.info(f"[DeltaNet.golden_forward_{self.device_id}] prev_state: {type(prev_state)}")

        # Pre-attention norm + attention + residual.
        logger.info(f"[DeltaNet.golden_forward_{self.device_id}] Step1: input_layernorm, input shape={x.shape}")
        norm_x = self.input_layernorm(x)
        logger.info(f"[DeltaNet.golden_forward_{self.device_id}] Step2: attention (attn.golden_forward)")
        attn_out, new_state = self.attn.golden_forward(norm_x, start_pos, prev_state)
        logger.info(f"[DeltaNet.golden_forward_{self.device_id}] attn_out: shape={attn_out.shape}, new_state type={type(new_state)}")
        self.recurrent_state = new_state[0] # 保存一下 recurrent_state
        self.conv_state = new_state[1] # 保存一下conv_state
        h = x + attn_out
        logger.info(f"[DeltaNet.golden_forward_{self.device_id}] After residual: h.shape={h.shape}")

        # Post-attention norm + MoE FFN (partial TP8 sum) + all-reduce.
        logger.info(f"[DeltaNet.golden_forward_{self.device_id}] Step3: post_attention_layernorm")
        norm_h = self.post_attention_layernorm(h)
        logger.info(f"[DeltaNet.golden_forward_{self.device_id}] Step4: MoE FFN (ffn.golden_forward)")
        
        ffn_partial = self.ffn.golden_forward(norm_h)
        logger.info(
            f"[DeltaNet.golden_forward_{self.device_id}] ffn_partial: "
            f"shape={ffn_partial.shape} mean={ffn_partial.float().mean().item():.6f} "
            f"std={ffn_partial.float().std().item():.6f}"
        )

        # The TP8 MoE down-op already performs the all-reduce internally.
        # Applying the callback again here would double-aggregate the same
        # tensor and corrupt the result on multi-GPU runs.
        ffn_full = ffn_partial

        # Final residual uses the all-reduced (full) FFN output.
        out = h + ffn_full
        logger.info(f"[DeltaNet.golden_forward_{self.device_id}] Final output: shape={out.shape}, mean={out.float().mean().item():.4f}")

        # ``new_state`` is a tuple ``(conv_state, recurrent_state)`` produced by
        # ``DeltaNetOp.golden_forward`` to keep both the causal convolution and
        # the gated delta recurrence alive across decode steps.  It must always
        # be returned so that ``QwenTransformerStack`` can persist it, even on
        # the very first call when no prior state was supplied.
        next_state = {"delta_state": new_state}
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
        next_state = {"delta_state": new_state}
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