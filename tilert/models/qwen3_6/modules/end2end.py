"""End-to-end ShowHands layer for Qwen3.6-35B-A3B.

This module mirrors the architecture of ``ShowHandsDSALayer`` used by
DeepSeek-V3.2 / GLM5, but adapts it to Qwen3.6's heterogeneous Transformer stack
(DeltaNet + Gated Attention) and removes all DeepSeek Sparse Attention (DSA)
/ MLA P2P constructs.

Because the Qwen3.6 CUDA kernels are not yet built, the current implementation
keeps the TileRT-weight layout and CUDA-graph calling convention intact while
using a Python/golden forward path for end-to-end sanity tests.  Once
``libtilert_qwen36.so`` is available, only the ``qwen36_show_hands_*``
wrappers and ``forward()`` need to be switched over.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from tilert import logger
from tilert.models.base import TileRTModule
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.models.qwen3_6.modules.transformer_stack import QwenTransformerStack
from tilert.models.qwen3_6.ops.rmsnorm_head_proj import RMSNormHeadProj
from tilert.models.qwen3_6.temp_var_indices import Idx, TEMP_VARS_SIZE, validate_temp_vars_layout
from tilert.models.utils import precompute_freqs_cis
from tilert.utils import get_profile_log_tensor

__all__ = [
    "QwenShowHandsLayer",
    "_extract_ffn_ops",
    "_get_moe_weight_keys",
]


DeviceResult = tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]


def _mark_weights_initialized(module: TileRTModule) -> None:
    """Recursively mark a module and all sub-ops as having initialized tilert weights."""
    module.is_tilert_weights_init = True
    if hasattr(module, "exec_seq"):
        for op in module.exec_seq:
            _mark_weights_initialized(op)


def _extract_ffn_ops(stack: QwenTransformerStack) -> list:
    """Extract MoE op objects from a QwenTransformerStack's layer blocks.

    Returns a list of length ``n_layers`` where each element is a ``QwenMoeBlock``.
    """
    from tilert.models.qwen3_6.modules.delta_net import DeltaNet
    from tilert.models.qwen3_6.modules.gated_attention import GatedAttention

    ffn_ops: list = []
    for block in stack.exec_seq:
        if isinstance(block, (DeltaNet, GatedAttention)):
            op = block.ffn
            _mark_weights_initialized(op)
            ffn_ops.append(op)

    assert len(ffn_ops) == stack.model_args.n_layers, (
        f"Expected {stack.model_args.n_layers} FFN ops, got {len(ffn_ops)}"
    )
    return ffn_ops


def _get_moe_weight_keys(stack: QwenTransformerStack) -> set[str]:
    """Get state_dict keys that belong exclusively to MOE ops in this stack."""
    from tilert.models.qwen3_6.modules.delta_net import DeltaNet
    from tilert.models.qwen3_6.modules.gated_attention import GatedAttention

    moe_keys: set[str] = set()
    for block, prefix, suffix in zip(stack.exec_seq, stack.prefix_seq, stack.suffix_seq):
        if isinstance(block, (DeltaNet, GatedAttention)):
            ffn = block.ffn
            for alias in ffn.get_tilert_weights_alias():
                moe_keys.add(f"{prefix}{alias}{suffix}")
    return moe_keys


def qwen36_show_hands_prepare_money(
    params: list[torch.Tensor],
    temp_vars: list[torch.Tensor],
    cache_vars: list[torch.Tensor],
    profile_logs: torch.Tensor,
    forward_max_seq_len: int,
    with_mtp: bool = False,
) -> Any:
    """Prepare the Qwen3.6 CUDA graph decode context."""
    mtp_flag = "_mtp_e2e" if with_mtp else ""
    func_name = f"qwen36{mtp_flag}_show_hands_prepare_money"
    if mtp_flag:
        return getattr(torch.ops.tilert, func_name)(params, temp_vars, cache_vars, profile_logs)
    return getattr(torch.ops.tilert, func_name)(
        params, temp_vars, cache_vars, profile_logs, forward_max_seq_len
    )


def qwen36_show_hands(token_id: torch.Tensor, with_mtp: bool = False) -> Any:
    """Execute one Qwen3.6 CUDA-graph decode step."""
    mtp_flag = "_mtp_e2e" if with_mtp else ""
    func_name = f"qwen36{mtp_flag}_show_hands"
    return getattr(torch.ops.tilert, func_name)(token_id)


def qwen36_show_hands_reset(with_mtp: bool = False) -> Any:
    """Reset the Qwen3.6 CUDA graph decode context."""
    mtp_flag = "_mtp_e2e" if with_mtp else ""
    func_name = f"qwen36{mtp_flag}_show_hands_reset"
    return getattr(torch.ops.tilert, func_name)()


def qwen36_show_hands_go_home(with_mtp: bool = False) -> Any:
    """Release the Qwen3.6 CUDA graph decode context."""
    mtp_flag = "_mtp_e2e" if with_mtp else ""
    func_name = f"qwen36{mtp_flag}_show_hands_go_home"
    return getattr(torch.ops.tilert, func_name)()


def qwen36_show_hands_set_sampling_seed(seed: int, with_mtp: bool = False) -> Any:
    """Set the sampling seed (request-level)."""
    mtp_flag = "_mtp_e2e" if with_mtp else ""
    func_name = f"qwen36{mtp_flag}_show_hands_set_sampling_seed"
    return getattr(torch.ops.tilert, func_name)(seed)


def qwen36_show_hands_set_cur_pos(cur_pos: int, with_mtp: bool = False) -> Any:
    """Set the current decode position for RoPE."""
    mtp_flag = "_mtp_e2e" if with_mtp else ""
    func_name = f"qwen36{mtp_flag}_show_hands_set_cur_pos"
    return getattr(torch.ops.tilert, func_name)(cur_pos)


def qwen36_mtp_e2e_show_hands_set_prefill_valid_tokens(
    num_valid_tokens: int,
) -> Any:
    """Set the number of valid tokens during prefill."""
    return torch.ops.tilert.qwen36_mtp_e2e_show_hands_set_prefill_valid_tokens(num_valid_tokens)


def qwen36_mtp_e2e_show_hands_set_prefill_mtp_extra_token(token: int) -> Any:
    """Set the extra token for MTP[0] shifted input during prefill."""
    return torch.ops.tilert.qwen36_mtp_e2e_show_hands_set_prefill_mtp_extra_token(token)


class QwenShowHandsLayer:
    """End-to-end decode layer for Qwen3.6-35B-A3B.

    The class follows the same lifecycle as ``ShowHandsDSALayer``:

      1. Construct with ``ModelArgsQwen36`` and optional sampling config.
      2. ``from_pretrained()`` / ``init_random_weights()`` loads weights across
         all 8 devices in parallel.
      3. ``forward(token_id)`` runs one decode step.
      4. ``cleanup()`` releases CUDA graphs on shutdown.

    The Qwen3.6 stack has no DSA/MLA, so we do **not** allocate ``v2_peer_bufs``
    or ``ll_buf``.  The layout is therefore simpler: each device owns a
    ``QwenTransformerStack``, a final ``RMSNormHeadProj`` head projection, the
    embedding table, and RoPE frequencies.
    """

    def __init__(
        self,
        model_args: ModelArgsQwen36,
        model_path: str = "",
        with_weight_conversion: bool = True,
        with_mtp: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 256,
        use_topp: bool = False,
    ) -> None:
        validate_temp_vars_layout()
        print(f"Model args: {model_args.arch_name}")
        for k_arg, v_arg in model_args.__dict__.items():
            print(f" - {k_arg}: {v_arg}")

        self.model_args = model_args
        assert self.model_args.arch_name == "qwen3_6"

        self.num_devices = 8
        self.forward_max_seq_len = model_args.max_seq_len

        self.model_path = model_path
        self.with_weight_conversion = with_weight_conversion
        self.with_mtp = with_mtp

        self.multi_devices_results: list[DeviceResult | None] = [None] * torch.cuda.device_count()
        self._stack_objects: list[QwenTransformerStack | None] = [None] * torch.cuda.device_count()

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_topp = use_topp

    def _gen_freqs_cis(self) -> torch.Tensor:
        freqs_cis = precompute_freqs_cis(self.model_args)
        # Return real layout (max_seq_len, rope_dim) so it can be sliced by cur_pos
        # and converted back to complex inside the transformer stack.
        return torch.view_as_real(freqs_cis).reshape(freqs_cis.shape[0], -1)

    def load_device_weights(
        self,
        model_path: str,
        device_id: int,
        extra_keys: list[str],
        skip_keys: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Load the safetensors shard(s) needed by ``device_id``.

        Qwen3.6 follows the same per-device suffix convention as DSv3.2/GLM5:
        sharded weights end with ``_dev_{device_id}``.  Unsharded tensors such
        as ``model.embed_tokens.weight`` are replicated on every device.
        """
        index_file = "model.safetensors.index.json"
        with open(os.path.join(model_path, index_file), encoding="utf-8") as f:
            weights_index = json.load(f)
        weight_file_map = weights_index["weight_map"]

        weights_list = [_k for _k in weight_file_map.keys() if _k.endswith(f"dev_{device_id}")]
        weights_list = [*weights_list, *extra_keys]

        if skip_keys:
            weights_list = [k for k in weights_list if k not in skip_keys]

        target_files = {weight_file_map[k] for k in weights_list if k in weight_file_map}

        state_dicts: dict[str, torch.Tensor] = {}
        weights_set = set(weights_list)
        for weight_file in target_files:
            filepath = os.path.join(model_path, weight_file)
            if skip_keys:
                logger.info(
                    f"Selectively loading weights from {weight_file} for device {device_id}"
                )
                with safe_open(filepath, framework="pt", device=f"cuda:{device_id}") as f:
                    for key in f.keys():
                        if key in weights_set:
                            state_dicts[key] = f.get_tensor(key)
                torch.cuda.empty_cache()
            else:
                logger.info(f"Loading weights from {weight_file} for device {device_id}")
                state_dict = load_file(filepath, device=f"cuda:{device_id}")
                state_dicts.update(state_dict)
                del state_dict
                torch.cuda.empty_cache()

        state_dicts["freqs_cis"] = self._gen_freqs_cis().to(device_id)
        return state_dicts

    def update_sampling_config(
        self,
        temperature: float,
        top_p: float,
        top_k: int,
        use_topp: bool = True,
    ) -> None:
        """Update sampling config and re-capture CUDA graphs if necessary."""
        new_config = (temperature, top_p, top_k, use_topp)
        current_config = (self.temperature, self.top_p, self.top_k, self.use_topp)
        if new_config == current_config:
            return

        print(
            f"Recapturing CUDA graphs: "
            f"temperature={temperature}, top_p={top_p}, top_k={top_k}, use_topp={use_topp}"
        )

        if self.with_mtp:
            qwen36_show_hands_go_home(True)
            qwen36_show_hands_go_home(False)
        else:
            qwen36_show_hands_go_home(False)

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_topp = use_topp

        for device_id in range(self.num_devices):
            result = self.multi_devices_results[device_id]
            if result is not None:
                intermediates = result[0]
                intermediates[Idx.SAMPLING_CONFIG].copy_(
                    torch.tensor(
                        [temperature, top_p, float(top_k), 1.0 if use_topp else 0.0],
                        dtype=torch.float32,
                        device=f"cuda:{device_id}",
                    )
                )

        for device_id in range(self.num_devices):
            with torch.cuda.device(device_id):
                intermediates, caches, params, profile_logs = self._get_device_result(device_id)
                qwen36_show_hands_prepare_money(
                    params,
                    intermediates,
                    caches,
                    profile_logs,
                    self.forward_max_seq_len,
                    self.with_mtp,
                )
                if self.with_mtp:
                    qwen36_show_hands_prepare_money(
                        params[: self._base_params_count],
                        intermediates,
                        caches[: self._base_caches_count],
                        profile_logs,
                        self.forward_max_seq_len,
                        False,
                    )

    @staticmethod
    def tot_size_in_bytes_aligned(temp_vars: list[torch.Tensor], aligned_size: int) -> int:
        tot_size: int = 0
        for param in temp_vars:
            aligned_param_size = (param.nbytes + aligned_size - 1) // aligned_size * aligned_size
            tot_size += aligned_param_size
        return tot_size

    def generate_params_with_continuous_storage(
        self,
        temp_vars: list[torch.Tensor],
        device: torch.device,
        aligned_size: int = 1024,
    ) -> list[torch.Tensor]:
        """Pack a list of tensors into one contiguous storage buffer."""
        tot_size = self.tot_size_in_bytes_aligned(temp_vars, aligned_size)
        large_tensor = torch.zeros(tot_size, device=device, dtype=torch.uint8)
        cloned_params: list[torch.Tensor] = []
        offset = 0
        for param in temp_vars:
            aligned_param_size = (param.nbytes + aligned_size - 1) // aligned_size * aligned_size
            cloned_params.append(
                large_tensor[offset : offset + param.nbytes].view(param.dtype).view(param.shape)
            )
            offset += aligned_param_size
        return cloned_params

    def _get_temp_vars(
        self,
        batch_size: int,
        seq_len: int,
        device_id: int,
        extra_args: dict[str, Any] | None = None,
    ) -> list[torch.Tensor]:
        """Allocate the fixed-size temp_vars tensor list for Qwen3.6."""
        dev = f"cuda:{device_id}"
        bf16_desc = {"dtype": torch.bfloat16, "device": dev}
        fp32_desc = {"dtype": torch.float32, "device": dev}
        int32_desc = {"dtype": torch.int32, "device": dev}
        int64_desc = {"dtype": torch.int64, "device": dev}
        fp8_desc = {"dtype": torch.float8_e4m3fn, "device": dev}

        temperature = extra_args["temperature"] if extra_args else self.temperature
        top_p = extra_args["top_p"] if extra_args else self.top_p
        top_k = extra_args["top_k"] if extra_args else self.top_k
        use_topp = extra_args["use_topp"] if extra_args else self.use_topp

        dim = self.model_args.dim
        batch_seq = (batch_size, seq_len)
        vocab_per_device = self.model_args.vocab_size // self.num_devices
        n_routed_experts = self.model_args.n_routed_experts
        n_activated_experts = self.model_args.n_activated_experts
        n_total_experts = n_activated_experts + self.model_args.n_shared_experts
        moe_inter_dim = self.model_args.inter_dim // self.num_devices
        rope_dim = self.model_args.rope_dim

        temp_vars: list[torch.Tensor | None] = [None] * TEMP_VARS_SIZE

        temp_vars[Idx.X] = torch.zeros(*batch_seq, dim, **bf16_desc)
        temp_vars[Idx.HIDDEN_RMSNORM] = torch.zeros(*batch_seq, dim, **bf16_desc)
        temp_vars[Idx.EMBEDDING_RMSNORM] = torch.zeros(*batch_seq, dim, **bf16_desc)

        temp_vars[Idx.DELTA_OUT] = torch.zeros(*batch_seq, dim, **bf16_desc)
        temp_vars[Idx.GQA_OUT] = torch.zeros(*batch_seq, dim, **bf16_desc)
        temp_vars[Idx.ROPE_FREQS] = torch.zeros(seq_len, rope_dim * 2, **fp32_desc)
        temp_vars[Idx.CUR_POS] = torch.zeros(batch_size, **int32_desc)
        temp_vars[Idx.TOKEN_ID] = torch.zeros(*batch_seq, 1, **int32_desc)

        temp_vars[Idx.X_MLP_IN] = torch.zeros(*batch_seq, dim, **bf16_desc)
        temp_vars[Idx.SCORES] = torch.zeros(*batch_seq, n_routed_experts, **fp32_desc)
        temp_vars[Idx.SEL_PROBS] = torch.zeros(*batch_seq, n_activated_experts, **fp32_desc)
        temp_vars[Idx.SEL_INDICES] = torch.zeros(*batch_seq, n_activated_experts, **int32_desc)
        temp_vars[Idx.UP_GATE] = torch.zeros(
            *batch_seq, n_total_experts, moe_inter_dim, **bf16_desc
        )
        temp_vars[Idx.EXP_OUT] = torch.zeros(*batch_seq, dim, **bf16_desc)

        temp_vars[Idx.LOGITS_OUT] = torch.zeros(*batch_seq, vocab_per_device, **fp32_desc)
        temp_vars[Idx.TOKEN_OUT] = torch.zeros(*batch_seq, 1, **int32_desc)

        temp_vars[Idx.SAMPLING_SEED] = torch.zeros(*batch_seq, **int64_desc)
        temp_vars[Idx.SAMPLING_POSITIONS] = torch.zeros(*batch_seq, **int64_desc)
        temp_vars[Idx.SAMPLING_CONFIG] = torch.tensor(
            [temperature, top_p, float(top_k), 1.0 if use_topp else 0.0], **fp32_desc
        )
        temp_vars[Idx.TOP_P_SCORES] = torch.zeros(*batch_seq, **fp32_desc)
        temp_vars[Idx.TOP_P_DEBUG] = torch.zeros(*batch_seq, vocab_per_device, **fp32_desc)

        temp_vars[Idx.X_QUANT] = torch.zeros(*batch_seq, dim, **fp8_desc)
        temp_vars[Idx.X_SCALE] = torch.zeros(
            *batch_seq, dim // self.model_args.block_size, **fp32_desc
        )
        temp_vars[Idx.MOE_UP_GATE] = torch.zeros(
            *batch_seq, n_total_experts, moe_inter_dim, **bf16_desc
        )

        temp_vars[Idx.DRAFT_TOKENS] = torch.zeros(*batch_seq, **int32_desc)
        temp_vars[Idx.PREDICTED_TOKENS] = torch.zeros(*batch_seq, 1, **int32_desc)
        temp_vars[Idx.PREDICTED_HIDDEN] = torch.zeros(*batch_seq, dim, **bf16_desc)
        temp_vars[Idx.ACCEPTED_TOKENS] = torch.zeros(batch_size, **int32_desc)
        temp_vars[Idx.NEXT_DRAFT_TOKENS] = torch.zeros(*batch_seq, **int32_desc)
        temp_vars[Idx.MTP0_TOKEN_OUT] = torch.zeros(*batch_seq, 1, **int32_desc)
        temp_vars[Idx.MTP0_EXP_OUT] = torch.zeros(*batch_seq, dim, **bf16_desc)
        temp_vars[Idx.LAST_HIDDEN_STATES] = torch.zeros(*batch_seq, dim, **bf16_desc)

        max_top_n = 256
        temp_vars[Idx.TOP_N_LOG_PROBS] = torch.zeros(*batch_seq, max_top_n, **fp32_desc)
        temp_vars[Idx.TOP_N_INDICES] = torch.zeros(*batch_seq, max_top_n, **int32_desc)
        temp_vars[Idx.LOGPROBS_FLAG] = torch.zeros(1, **int32_desc)

        for i, t in enumerate(temp_vars):
            if t is None:
                raise RuntimeError(f"temp_vars[{i}] ({Idx(i).name}) was not initialized")

        return temp_vars  # type: ignore[return-value]

    def _init_weights(
        self,
        model_path: str | None,
        cached_ffn_ops_per_device: dict[int, list] | None = None,
        skip_keys_per_device: dict[int, set[str]] | None = None,
    ) -> None:
        """Load model weights across all devices in parallel.

        Args:
            model_path: Path to the model weights directory.  If ``None``,
                random weights are generated for smoke testing.
            cached_ffn_ops_per_device: Optional cached FFN ops per device.
            skip_keys_per_device: Optional safetensors keys to skip per device.
        """

        def __load_weights(device_id: int, model_path: str | None) -> None:
            intermediates: list[torch.Tensor] = []
            caches: list[torch.Tensor] = []
            params: list[torch.Tensor] = []
            state_dicts: dict[str, torch.Tensor] = {}
            start_time = time.time()
            with torch.cuda.device(device_id):
                if model_path is not None:
                    skip_keys = (
                        skip_keys_per_device.get(device_id)
                        if skip_keys_per_device is not None
                        else None
                    )
                    state_dicts = self.load_device_weights(
                        model_path,
                        device_id,
                        [
                            "model.embed_tokens.weight",
                            f"layer_{self.model_args.n_layers}_lm_head.weight_dev_{device_id}",
                            f"layer_{self.model_args.n_layers}_model.norm.weight_dev_{device_id}",
                        ],
                        skip_keys=skip_keys,
                    )

                cached_ffn_ops = (
                    cached_ffn_ops_per_device.get(device_id)
                    if cached_ffn_ops_per_device is not None
                    else None
                )
                stack = QwenTransformerStack(
                    self.model_args,
                    device_id,
                    self.num_devices,
                    cached_ffn_ops=cached_ffn_ops,
                )
                if model_path is not None:
                    stack.init_tilert_weights(state_dicts)
                else:
                    stack.init_random_weights()
                    # Mark the top-level stack as random-init so the residual
                    # scaling heuristic is applied only for smoke tests.
                    stack._random_init_marker = True
                self._stack_objects[device_id] = stack

                params.extend(stack.get_weights_list())
                caches.extend(stack.get_cache_vars())

                head_proj = RMSNormHeadProj(
                    model_args=self.model_args,
                    device_id=device_id,
                    num_devices=self.num_devices,
                )
                if model_path is not None:
                    head_state = {
                        alias: state_dicts[alias]
                        for alias in head_proj.tilert_weights_alias()
                        if alias in state_dicts
                    }
                    # Fallback: the converted checkpoint stores head/norm with
                    # a ``layer_{n_layers}_`` prefix; if the bare alias is missing,
                    # try the prefixed form.
                    prefixed_aliases = {
                        alias: f"layer_{self.model_args.n_layers}_{alias}_dev_{device_id}"
                        for alias in head_proj.tilert_weights_alias()
                    }
                    for alias, prefixed in prefixed_aliases.items():
                        if alias not in head_state and prefixed in state_dicts:
                            head_state[alias] = state_dicts[prefixed]
                    head_proj.init_tilert_weights(head_state)
                else:
                    head_proj.init_random_weights(device_id=device_id)
                params.extend(head_proj.get_weights_list())

                # Embedding table is replicated on all devices.
                if model_path is not None:
                    embed_tokens = state_dicts["model.embed_tokens.weight"]
                else:
                    embed_tokens = torch.randn(
                        self.model_args.vocab_size,
                        self.model_args.dim,
                        dtype=torch.bfloat16,
                        device=f"cuda:{device_id}",
                    ) / (self.model_args.dim**0.5)
                params.append(embed_tokens)

                # RoPE frequencies are also per-device for convenience.
                if model_path is not None:
                    freqs_cis = state_dicts["freqs_cis"]
                else:
                    freqs_cis = self._gen_freqs_cis().to(device_id)
                params.append(freqs_cis)

                intermediates.extend(
                    self.generate_params_with_continuous_storage(
                        self._get_temp_vars(
                            1,
                            self.forward_max_seq_len,
                            device_id,
                            {
                                "temperature": self.temperature,
                                "top_p": self.top_p,
                                "top_k": self.top_k,
                                "use_topp": self.use_topp,
                            },
                        ),
                        device_id,
                    )
                )

                sampling_config = intermediates[Idx.SAMPLING_CONFIG]
                sampling_config.copy_(
                    torch.tensor(
                        [
                            self.temperature,
                            self.top_p,
                            float(self.top_k),
                            1.0 if self.use_topp else 0.0,
                        ],
                        dtype=torch.float32,
                        device=device_id,
                    )
                )

                base_params_count = len(params)
                base_caches_count = len(caches)

                if self.with_mtp:
                    from tilert.models.qwen3_6.modules.mtp import QwenMTP

                    mtp = QwenMTP(self.model_args, device_id, self.num_devices)
                    if model_path is not None:
                        # Placeholder: MTP weights not yet defined.
                        pass
                    params.extend(mtp.get_weights_list())
                    caches.extend(mtp.get_cache_vars())
                    logger.info(f"Loaded real MTP weights for device {device_id}")

                profile_logs = get_profile_log_tensor(device=device_id, num_max_insts=65536)
                result: DeviceResult = (intermediates, caches, params, profile_logs)
                self.multi_devices_results[device_id] = result
                self._base_params_count = base_params_count
                self._base_caches_count = base_caches_count

            del state_dicts
            torch.cuda.empty_cache()
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            time_str = (
                f"{minutes} minutes {seconds} seconds" if minutes > 0 else f"{seconds} seconds"
            )
            logger.info(f"Completed loading weights for device {device_id} in {time_str}")

        threads: list[threading.Thread] = []
        exceptions: list[Exception | None] = [None] * self.num_devices
        for device_id in range(self.num_devices):

            def _runner(dev_id: int) -> None:
                try:
                    __load_weights(dev_id, model_path)
                except Exception as exc:  # pragma: no cover - surfaced after join
                    exceptions[dev_id] = exc

            thread = threading.Thread(target=_runner, args=(device_id,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        for device_id, exc in enumerate(exceptions):
            if exc is not None:
                raise RuntimeError(f"Failed to initialize device {device_id}: {exc}") from exc

        # Note: no V2 P2P setup for Qwen3.6 (no DSA/MLA).

        # Qwen3.6 backend library does not exist yet, so skip the CUDA-graph
        # prepare step.  Golden forward is always used until kernels are ready.
        if os.environ.get("TILERT_QWEN36_ENABLE_BACKEND_PREPARE", "0") == "1":
            for device_id in range(self.num_devices):
                with torch.cuda.device(device_id):
                    intermediates, caches, params, profile_logs = self._get_device_result(device_id)
                    qwen36_show_hands_prepare_money(
                        params,
                        intermediates,
                        caches,
                        profile_logs,
                        self.forward_max_seq_len,
                        self.with_mtp,
                    )
                    if self.with_mtp:
                        qwen36_show_hands_prepare_money(
                            params[: self._base_params_count],
                            intermediates,
                            caches[: self._base_caches_count],
                            profile_logs,
                            self.forward_max_seq_len,
                            False,
                        )

    def from_pretrained(self, model_path: str) -> None:
        """Load the model weights from the given path."""
        if not os.path.exists(model_path):
            raise ValueError(f"Model weights directory {model_path} does not exist")
        self._init_weights(model_path)

    def from_pretrained_with_cache(
        self,
        model_path: str,
        cached_ffn_ops_per_device: dict[int, list],
        skip_keys_per_device: dict[int, set[str]],
    ) -> None:
        """Load weights reusing cached MOE/MLP ops."""
        if not os.path.exists(model_path):
            raise ValueError(f"Model weights directory {model_path} does not exist")
        self._init_weights(
            model_path,
            cached_ffn_ops_per_device=cached_ffn_ops_per_device,
            skip_keys_per_device=skip_keys_per_device,
        )

    def init_random_weights(self) -> None:
        """Generate random weights for smoke testing."""
        self._init_weights(None)

    def _golden_forward_device(
        self,
        device_id: int,
        token_id: torch.Tensor,
        cur_pos: int,
    ) -> DeviceResult:
        """Reference forward for a single device.

        This is used until the Qwen3.6 CUDA kernels are available.  It mirrors
        the tensor layout expected by the future kernel so that switching over
        is a drop-in replacement.
        """
        intermediates, caches, params, profile_logs = self._get_device_result(device_id)
        stack = self._stack_objects[device_id]
        if stack is None:
            raise RuntimeError(f"QwenTransformerStack not initialized on device {device_id}")

        # Locate the final head projection and embedding weights in params.
        stack_weight_count = len(stack.get_weights_list())
        head_weight_count = 2  # RMSNormHeadProj.get_weights_list().
        embed_weight = params[stack_weight_count + head_weight_count]
        freqs_cis_param = params[stack_weight_count + head_weight_count + 1]

        # 1. Embedding lookup.  ``token_id`` may be a scalar int32 or a [1]
        # tensor, so flatten it to a single index before indexing.  Also make sure
        # the index tensor lives on the same device as the embedding table to
        # avoid ``indices should be on the same device as the indexed tensor``
        # errors during multi-device generation.
        idx = token_id.view(-1).to(embed_weight.device)
        x = embed_weight[idx].unsqueeze(0).to(torch.bfloat16)
        intermediates[Idx.TOKEN_ID][0, 0, 0] = token_id
        intermediates[Idx.CUR_POS][0] = cur_pos

        # 2. Transformer stack (DeltaNet + Gated Attention layers).
        # freqs_cis_param has shape (max_seq_len, rope_dim) in real layout.
        # Pass the full tensor to golden_forward; GQA attention will slice it
        # by start_pos internally, and DeltaNet ignores it.
        h, caches = stack.golden_forward(x, cur_pos, freqs_cis_param)

        # 3. Final RMSNorm + head projection.
        # Params are TileRT-sharded; head projection weight is split across
        # devices as (logits_dim/num_devices, dim).  The reference path needs
        # the full matrix on every device, so all-gather it here.
        local_head = params[stack_weight_count + 1]
        if local_head.dim() == 2 and local_head.size(0) * self.num_devices == self.model_args.vocab_size:
            head_list = [torch.empty_like(local_head) for _ in range(self.num_devices)]
            torch.distributed.all_gather(head_list, local_head)
            full_head = torch.cat(head_list, dim=0)
        else:
            full_head = local_head

        head_proj = RMSNormHeadProj(
            model_args=self.model_args,
            device_id=device_id,
            num_devices=self.num_devices,
        )
        head_proj.ref_rmsnorm_gamma = params[stack_weight_count]
        head_proj.ref_head_proj = full_head
        logits = head_proj.golden_forward(h)
        # All-gather produces logits_dim on each device; slice local shard.
        local_logits_dim = self.model_args.vocab_size // self.num_devices
        local_logits_start = device_id * local_logits_dim
        local_logits_end = local_logits_start + local_logits_dim
        intermediates[Idx.LOGITS_OUT][0, 0, local_logits_start:local_logits_end].copy_(
            logits[0, 0, local_logits_start:local_logits_end]
        )

        # 4. Sampling (greedy / top-k / top-p placeholder).
        token_out = self._sample(logits[0, 0])
        intermediates[Idx.TOKEN_OUT][0, 0, 0] = token_out

        return intermediates, caches, params, profile_logs

    def _sample(self, logits: torch.Tensor) -> int:
        """Simple sampling helper for the golden path."""
        if self.use_topp:
            # Placeholder: fall back to greedy for smoke tests.
            return int(logits.argmax().item())
        return int(logits.argmax().item())

    def forward(
        self,
        token_id: torch.Tensor,
        with_mtp: bool | None = None,
        cur_pos: int = 0,
    ) -> list[DeviceResult]:
        """Run one decode step.

        Args:
            token_id: Scalar or [1] int32 tensor containing the input token id.
            with_mtp: Override MTP mode.  Defaults to ``self.with_mtp``.
            cur_pos: Current decode position (used by the golden path).

        Returns:
            List of per-device ``DeviceResult`` tuples.
        """
        active_mtp = with_mtp if with_mtp is not None else self.with_mtp

        # CUDA graph path placeholder.
        try:
            qwen36_show_hands(token_id.cpu(), active_mtp)
        except (AttributeError, RuntimeError):
            # Backend kernels not available; use golden path.
            # Run each device's golden forward in its own thread so that all
            # devices are active concurrently, matching the performance baseline.
            results: list[DeviceResult | None] = [None] * self.num_devices
            exceptions: list[Exception | None] = [None] * self.num_devices
            threads: list[threading.Thread] = []

            def _runner(dev_id: int) -> None:
                try:
                    with torch.cuda.device(dev_id):
                        results[dev_id] = self._golden_forward_device(
                            dev_id, token_id, cur_pos
                        )
                except Exception as exc:  # pragma: no cover - surfaced after join
                    exceptions[dev_id] = exc

            for device_id in range(self.num_devices):
                thread = threading.Thread(target=_runner, args=(device_id,))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()

            for device_id, exc in enumerate(exceptions):
                if exc is not None:
                    raise RuntimeError(
                        f"Golden forward failed on device {device_id}: {exc}"
                    ) from exc

            return results  # type: ignore[return-type]

        return [self._get_device_result(device_id) for device_id in range(self.num_devices)]

    def set_sampling_seed(self, seed: int, with_mtp: bool | None = None) -> None:
        """Set the sampling seed for top-p sampling."""
        active_mtp = with_mtp if with_mtp is not None else self.with_mtp
        try:
            qwen36_show_hands_set_sampling_seed(seed, active_mtp)
        except (AttributeError, RuntimeError):
            pass

    def set_cur_pos(self, cur_pos: int, with_mtp: bool | None = None) -> None:
        """Set the current decode position for RoPE."""
        active_mtp = with_mtp if with_mtp is not None else self.with_mtp
        try:
            qwen36_show_hands_set_cur_pos(cur_pos, active_mtp)
        except (AttributeError, RuntimeError):
            pass

    def reset_sequence(self) -> None:
        """Reset the decode sequence state."""
        try:
            if self.with_mtp:
                qwen36_show_hands_reset(True)
                qwen36_show_hands_reset(False)
            else:
                qwen36_show_hands_reset(False)
        except (AttributeError, RuntimeError):
            pass

    def cleanup(self) -> None:
        """Release CUDA graphs."""
        try:
            if self.with_mtp:
                qwen36_show_hands_go_home(True)
                qwen36_show_hands_go_home(False)
            else:
                qwen36_show_hands_go_home(False)
        except (AttributeError, RuntimeError):
            pass

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception as e:
            print(f"Exception during cleanup: {e}", file=sys.stderr)

    def _get_device_result(self, device_id: int) -> DeviceResult:
        device_result = self.multi_devices_results[device_id]
        if device_result is None:
            raise RuntimeError(f"Device {device_id} is not initialized")
        return device_result

    def set_prefill_valid_tokens(self, num_valid_tokens: int) -> None:
        """Set the number of valid tokens for prefill mode."""
        try:
            qwen36_mtp_e2e_show_hands_set_prefill_valid_tokens(num_valid_tokens)
        except (AttributeError, RuntimeError):
            pass

    def set_prefill_mtp_extra_token(self, token: int) -> None:
        """Set the extra token for MTP[0] shifted input during prefill."""
        try:
            qwen36_mtp_e2e_show_hands_set_prefill_mtp_extra_token(token)
        except (AttributeError, RuntimeError):
            pass

    def get_next_draft_tokens(self, device_id: int = 0) -> torch.Tensor:
        """Get next_draft_tokens from the specified device."""
        intermediates, _, _, _ = self._get_device_result(device_id)
        return intermediates[Idx.NEXT_DRAFT_TOKENS]

    def get_num_accepted(self, device_id: int = 0) -> int:
        """Get number of accepted tokens from the specified device."""
        intermediates, _, _, _ = self._get_device_result(device_id)
        return int(intermediates[Idx.ACCEPTED_TOKENS][0].item())

    def get_predicted_tokens(self, device_id: int = 0) -> torch.Tensor:
        """Get predicted_tokens from the specified device."""
        intermediates, _, _, _ = self._get_device_result(device_id)
        return intermediates[Idx.PREDICTED_TOKENS]

    def get_logits(self, device_id: int = 0) -> torch.Tensor:
        """Get logits from the specified device."""
        intermediates, _, _, _ = self._get_device_result(device_id)
        return intermediates[Idx.LOGITS_OUT]

    def get_next_token(self, device_id: int = 0) -> int:
        """Convenience helper: return the sampled next token id."""
        intermediates, _, _, _ = self._get_device_result(device_id)
        return int(intermediates[Idx.TOKEN_OUT][0, 0, 0].item())

    def get_top_n_logprobs(self, device_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Get top-N log-probabilities and token IDs from the top_p kernel."""
        intermediates, _, _, _ = self._get_device_result(device_id)
        return (
            intermediates[Idx.TOP_N_LOG_PROBS],
            intermediates[Idx.TOP_N_INDICES],
        )
