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
import functools
import json
import os
import sys
import threading
import time
import warnings
from typing import Any
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from tilert import logger
from tilert.models.base import TileRTModule
from tilert.models.qwen3_6.model_args import ModelArgsQwen36
from tilert.models.qwen3_6.modules.transformer_stack import QwenTransformerStack
from tilert.models.qwen3_6.modules.hf_source_loader import _is_hf_checkpoint, _precompute_all_device_states, load_hf_source_weights
from tilert.models.qwen3_6.ops.rmsnorm_head_proj import RMSNormHeadProj
from tilert.models.qwen3_6.temp_var_indices import Idx, TEMP_VARS_SIZE, validate_temp_vars_layout
from tilert.models.utils import precompute_mrope_embed
from tilert.utils import get_profile_log_tensor
try:
    from transformers import AutoModelForCausalLM
    _HAS_TRANSFORMERS = True
except Exception:
    AutoModelForCausalLM = None  # 仅在回退路径使用，未安装 transformers 时保持为 None
    _HAS_TRANSFORMERS = False
__all__ = ['QwenShowHandsLayer', '_extract_ffn_ops', '_get_moe_weight_keys']
DeviceResult = tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]

def _mark_weights_initialized(module: TileRTModule) -> None:
    """递归地将模块及其所有子 op 标记为已完成 tilert 权重初始化。"""
    module.is_tilert_weights_init = True
    # 中文注释：若模块存在 exec_seq 子模块序列，递归处理每个子 op。
    if hasattr(module, "exec_seq"):
        for op in getattr(module, "exec_seq"):  # type: ignore[attr-defined]
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
    assert len(ffn_ops) == stack.model_args.n_layers, f'Expected {stack.model_args.n_layers} FFN ops, got {len(ffn_ops)}'
    return ffn_ops

def _get_moe_weight_keys(stack: QwenTransformerStack) -> set[str]:
    """Get state_dict keys that belong exclusively to MOE ops in this stack."""
    from tilert.models.qwen3_6.modules.delta_net import DeltaNet
    from tilert.models.qwen3_6.modules.gated_attention import GatedAttention
    moe_keys: set[str] = set()
    for (block, prefix, suffix) in zip(stack.exec_seq, stack.prefix_seq, stack.suffix_seq):
        if isinstance(block, (DeltaNet, GatedAttention)):
            ffn = block.ffn
            for alias in ffn.get_tilert_weights_alias():
                moe_keys.add(f'{prefix}{alias}{suffix}')
    return moe_keys

def qwen36_show_hands_prepare_money(params: list[torch.Tensor], temp_vars: list[torch.Tensor], cache_vars: list[torch.Tensor], profile_logs: torch.Tensor, forward_max_seq_len: int, with_mtp: bool=False) -> Any:
    """准备 Qwen3.6 CUDA graph decode 上下文。"""
    mtp_flag = '_mtp_e2e' if with_mtp else ''
    func_name = f'qwen36{mtp_flag}_show_hands_prepare_money'
    if mtp_flag:
        return getattr(torch.ops.tilert, func_name)(params, temp_vars, cache_vars, profile_logs)
    return getattr(torch.ops.tilert, func_name)(params, temp_vars, cache_vars, profile_logs, forward_max_seq_len)

def qwen36_show_hands(token_id: torch.Tensor, with_mtp: bool=False) -> Any:
    """执行一次 Qwen3.6 CUDA-graph decode 步。"""
    mtp_flag = '_mtp_e2e' if with_mtp else ''
    func_name = f'qwen36{mtp_flag}_show_hands'
    return getattr(torch.ops.tilert, func_name)(token_id)

def qwen36_show_hands_reset(with_mtp: bool=False) -> Any:
    """重置 Qwen3.6 CUDA graph decode 上下文。"""
    mtp_flag = '_mtp_e2e' if with_mtp else ''
    func_name = f'qwen36{mtp_flag}_show_hands_reset'
    return getattr(torch.ops.tilert, func_name)()

def qwen36_show_hands_go_home(with_mtp: bool=False) -> Any:
    """释放 Qwen3.6 CUDA graph decode 上下文。"""
    mtp_flag = '_mtp_e2e' if with_mtp else ''
    func_name = f'qwen36{mtp_flag}_show_hands_go_home'
    return getattr(torch.ops.tilert, func_name)()

def qwen36_show_hands_set_sampling_seed(seed: int, with_mtp: bool=False) -> Any:
    """设置采样种子（请求级别）。"""
    mtp_flag = '_mtp_e2e' if with_mtp else ''
    func_name = f'qwen36{mtp_flag}_show_hands_set_sampling_seed'
    return getattr(torch.ops.tilert, func_name)(seed)

def qwen36_show_hands_set_cur_pos(cur_pos: int, with_mtp: bool=False) -> Any:
    """设置 RoPE 当前解码位置。"""
    mtp_flag = '_mtp_e2e' if with_mtp else ''
    func_name = f'qwen36{mtp_flag}_show_hands_set_cur_pos'
    return getattr(torch.ops.tilert, func_name)(cur_pos)

def qwen36_mtp_e2e_show_hands_set_prefill_valid_tokens(num_valid_tokens: int) -> Any:
    """设置 prefill 阶段有效 token 数量。"""
    return torch.ops.tilert.qwen36_mtp_e2e_show_hands_set_prefill_valid_tokens(num_valid_tokens)

def qwen36_mtp_e2e_show_hands_set_prefill_mtp_extra_token(token: int) -> Any:
    """设置 prefill 时 MTP[0] 偏移输入的额外 token。"""
    return torch.ops.tilert.qwen36_mtp_e2e_show_hands_set_prefill_mtp_extra_token(token)

class QwenShowHandsLayer:
    """Qwen3.6-35B-A3B 的端到端 decode layer。

    生命周期与 ``ShowHandsDSALayer`` 一致：

      1. 使用 ``ModelArgsQwen36`` 与可选采样配置构造。
      2. ``from_pretrained()`` / ``init_random_weights()`` 并行加载 8 个设备的权重。
      3. ``forward(token_id)`` 执行一次 decode 步。
      4. ``cleanup()`` 在关闭时释放 CUDA graph。

    Qwen3.6 stack 没有 DSA/MLA，因此不分配 ``v2_peer_bufs`` 或 ``ll_buf``。
    布局更简单：每个设备拥有一个 ``QwenTransformerStack``、最终的 ``RMSNormHeadProj``
    head 投影、embedding 表以及 RoPE 频率。
    """

    def __init__(self, model_args: ModelArgsQwen36, model_path: str='', with_weight_conversion: bool=True, with_mtp: bool=False, temperature: float=1.0, top_p: float=0.9, top_k: int=256, use_topp: bool=False) -> None:
        validate_temp_vars_layout()
        logger.info(f'QwenShowHandsLayer initializing with arch={model_args.arch_name}')
        logger.debug(f'Model args dump for {model_args.arch_name}:')
        for (k_arg, v_arg) in model_args.__dict__.items():
            logger.debug(f' - {k_arg}: {v_arg}')
        self.model_args = model_args
        assert self.model_args.arch_name == 'qwen3_6'
        self.num_devices = torch.cuda.device_count()
        if self.num_devices == 0:
            raise RuntimeError('没有可用的 CUDA/DCU 设备')
        self.forward_max_seq_len = int(os.environ.get('TILERT_QWEN36_FORWARD_MAX_SEQ_LEN', '1'))
        if self.forward_max_seq_len != model_args.max_seq_len:
            logger.info(f'[QwenShowHandsLayer.__init__] golden 前向使用 forward_max_seq_len={self.forward_max_seq_len} （可通过 TILERT_QWEN36_FORWARD_MAX_SEQ_LEN 覆盖）')
        self.model_path = model_path
        self.with_weight_conversion = with_weight_conversion
        self.with_mtp = with_mtp
        self.multi_devices_results: list[DeviceResult | None] = [None] * self.num_devices
        self._stack_objects: list[QwenTransformerStack | None] = [None] * self.num_devices
        self._head_proj_objects: list[RMSNormHeadProj | None] = [None] * self.num_devices
        self._golden_caches: list[dict[str, Any] | None] = [None] * self.num_devices
        self._full_head_proj_cache: list[torch.Tensor | None] = [None] * self.num_devices
        self._moe_barrier: threading.Barrier | None = None
        self._moe_partial_buf: list[torch.Tensor] | None = None
        self._moe_aggregated_buf: list[torch.Tensor] | None = None
        self._moe_partial_lens: list[int] = []
        self._hf_model: Any | None = None
        self._hf_past_key_values: Any | None = None
        self._hf_input_ids: torch.Tensor | None = None
        self._hf_fallback_enabled = os.environ.get('TILERT_QWEN36_HF_FALLBACK', '0') == '1'
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_topp = use_topp
        logger.info(f'[QwenShowHandsLayer.__init__] 配置完成：model_path={model_path}, num_devices={self.num_devices}, max_seq_len={self.forward_max_seq_len}, with_mtp={with_mtp}, temperature={temperature}, top_p={top_p}, top_k={top_k}, use_topp={use_topp}')

    def _gen_freqs_cis(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate M-RoPE cos/sin tables for the golden forward path.

        Returns:
            Tuple of (cos, sin) tensors with shape (max_seq_len, rope_dim).
        """
        (cos, sin) = precompute_mrope_embed(self.model_args)
        logger.debug(f'Generated M-RoPE tables: cos.shape={tuple(cos.shape)}, sin.shape={tuple(sin.shape)}, use_mrope={self.model_args.use_mrope}, partial_rotary_factor={self.model_args.partial_rotary_factor}, mrope_section={self.model_args.mrope_section}')
        return (cos, sin)

    def load_device_weights(self, model_path: str, device_id: int, extra_keys: list[str], skip_keys: set[str] | None=None) -> dict[str, torch.Tensor]:
        """Load the safetensors shard(s) needed by ``device_id``.

        Qwen3.6 follows the same per-device suffix convention as DSv3.2/GLM5:
        sharded weights end with ``_dev_{device_id}``.  Unsharded tensors such
        as ``model.embed_tokens.weight`` are replicated on every device.
        """
        index_file = 'model.safetensors.index.json'
        index_path = os.path.join(model_path, index_file)
        logger.info(f'Loading weight index for device {device_id}: {index_path}')
        with open(index_path, encoding='utf-8') as f:
            weights_index = json.load(f)
        weight_file_map = weights_index['weight_map']
        weights_list = [_k for _k in weight_file_map.keys() if _k.endswith(f'dev_{device_id}')]
        weights_list = [*weights_list, *extra_keys]
        if skip_keys:
            weights_list = [k for k in weights_list if k not in skip_keys]
            logger.debug(f'Device {device_id}: skipping {len(skip_keys)} keys, remaining {len(weights_list)} keys to load')
        target_files = {weight_file_map[k] for k in weights_list if k in weight_file_map}
        logger.info(f'Device {device_id}: {len(target_files)} safetensor file(s) to load, {len(weights_list)} weight key(s) requested')
        state_dicts: dict[str, torch.Tensor] = {}
        weights_set = set(weights_list)
        for weight_file in target_files:
            filepath = os.path.join(model_path, weight_file)
            if skip_keys:
                logger.info(f'Selectively loading weights from {weight_file} for device {device_id}')
                with safe_open(filepath, framework='pt', device=f'cuda:{device_id}') as f:
                    for key in f.keys():
                        if key in weights_set:
                            state_dicts[key] = f.get_tensor(key)
                torch.cuda.empty_cache()
            else:
                logger.info(f'Loading weights from {weight_file} for device {device_id}')
                state_dict = load_file(filepath, device=f'cuda:{device_id}')
                state_dicts.update(state_dict)
                del state_dict
                torch.cuda.empty_cache()
        logger.info(f'Device {device_id}: loaded {len(state_dicts)} tensors, generating M-RoPE tables on cuda:{device_id}')
        (cos, sin) = self._gen_freqs_cis()
        state_dicts['freqs_cos'] = cos.to(device_id)
        state_dicts['freqs_sin'] = sin.to(device_id)
        state_dicts['freqs_cis'] = cos.to(device_id)
        return state_dicts

    def update_sampling_config(self, temperature: float, top_p: float, top_k: int, use_topp: bool=True) -> None:
        """Update sampling config and re-capture CUDA graphs if necessary."""
        new_config = (temperature, top_p, top_k, use_topp)
        current_config = (self.temperature, self.top_p, self.top_k, self.use_topp)
        if new_config == current_config:
            logger.debug('采样配置未变，跳过 CUDA 图重新捕获')
            return
        logger.info(f'Recapturing CUDA graphs: temperature={temperature}, top_p={top_p}, top_k={top_k}, use_topp={use_topp}')
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
                intermediates[Idx.SAMPLING_CONFIG].copy_(torch.tensor([temperature, top_p, float(top_k), 1.0 if use_topp else 0.0], dtype=torch.float32, device=f'cuda:{device_id}'))
        for device_id in range(self.num_devices):
            with torch.cuda.device(device_id):
                (intermediates, caches, params, profile_logs) = self._get_device_result(device_id)
                qwen36_show_hands_prepare_money(params, intermediates, caches, profile_logs, self.forward_max_seq_len, self.with_mtp)
                if self.with_mtp:
                    qwen36_show_hands_prepare_money(
                        params[:self._base_params_count],  # type: ignore[attr-defined]
                        intermediates,
                        caches[:self._base_caches_count],  # type: ignore[attr-defined]
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

    def generate_params_with_continuous_storage(self, temp_vars: list[torch.Tensor], device: torch.device, aligned_size: int=1024) -> list[torch.Tensor]:
        """Pack a list of tensors into one contiguous storage buffer."""
        tot_size = self.tot_size_in_bytes_aligned(temp_vars, aligned_size)
        large_tensor = torch.zeros(tot_size, device=device, dtype=torch.uint8)
        cloned_params: list[torch.Tensor] = []
        offset = 0
        for param in temp_vars:
            aligned_param_size = (param.nbytes + aligned_size - 1) // aligned_size * aligned_size
            cloned_params.append(large_tensor[offset:offset + param.nbytes].view(param.dtype).view(param.shape))
            offset += aligned_param_size
        return cloned_params

    def _get_temp_vars(self, batch_size: int, seq_len: int, device_id: int, extra_args: dict[str, Any] | None=None) -> list[torch.Tensor]:
        """Allocate the fixed-size temp_vars tensor list for Qwen3.6."""
        dev = f'cuda:{device_id}'
        bf16_desc = {'dtype': torch.bfloat16, 'device': dev}
        fp32_desc = {'dtype': torch.float32, 'device': dev}
        int32_desc = {'dtype': torch.int32, 'device': dev}
        int64_desc = {'dtype': torch.int64, 'device': dev}
        fp8_desc = {'dtype': torch.float8_e4m3fn, 'device': dev}
        temperature = extra_args['temperature'] if extra_args else self.temperature
        top_p = extra_args['top_p'] if extra_args else self.top_p
        top_k = extra_args['top_k'] if extra_args else self.top_k
        use_topp = extra_args['use_topp'] if extra_args else self.use_topp
        dim = self.model_args.dim
        batch_seq = (batch_size, seq_len)
        vocab_per_device = self.model_args.vocab_size
        n_routed_experts = self.model_args.n_routed_experts
        n_activated_experts = self.model_args.n_activated_experts
        n_total_experts = n_activated_experts + self.model_args.n_shared_experts
        moe_inter_dim = self.model_args.inter_dim
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
        temp_vars[Idx.UP_GATE] = torch.zeros(*batch_seq, n_total_experts, moe_inter_dim, **bf16_desc)
        temp_vars[Idx.EXP_OUT] = torch.zeros(*batch_seq, dim, **bf16_desc)
        temp_vars[Idx.LOGITS_OUT] = torch.zeros(*batch_seq, vocab_per_device, **fp32_desc)
        temp_vars[Idx.TOKEN_OUT] = torch.zeros(*batch_seq, 1, **int32_desc)
        temp_vars[Idx.SAMPLING_SEED] = torch.zeros(*batch_seq, **int64_desc)
        temp_vars[Idx.SAMPLING_POSITIONS] = torch.zeros(*batch_seq, **int64_desc)
        temp_vars[Idx.SAMPLING_CONFIG] = torch.tensor([temperature, top_p, float(top_k), 1.0 if use_topp else 0.0], **fp32_desc)
        temp_vars[Idx.TOP_P_SCORES] = torch.zeros(*batch_seq, **fp32_desc)
        temp_vars[Idx.TOP_P_DEBUG] = torch.zeros(*batch_seq, vocab_per_device, **fp32_desc)
        temp_vars[Idx.X_QUANT] = torch.zeros(*batch_seq, dim, **fp8_desc)
        temp_vars[Idx.X_SCALE] = torch.zeros(*batch_seq, dim // self.model_args.block_size, **fp32_desc)
        temp_vars[Idx.MOE_UP_GATE] = torch.zeros(*batch_seq, n_total_experts, moe_inter_dim, **bf16_desc)
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
        for (i, t) in enumerate(temp_vars):
            if t is None:
                raise RuntimeError(f'设备 {device_id} 的 temp_vars[{i}] ({Idx(i).name}) 未初始化')
        logger.info(f"[QwenShowHandsLayer._get_temp_vars] 设备 {device_id} temp_vars 分配完成，共 {len(temp_vars)} 个")
        return temp_vars  # type: ignore[return-value]

    def _init_weights(self, model_path: str | None, cached_ffn_ops_per_device: dict[int, list] | None=None, skip_keys_per_device: dict[int, set[str]] | None=None) -> None:
        """Load model weights across all devices in parallel.

        Args:
            model_path: Path to the model weights directory.  If ``None``,
                random weights are generated for smoke testing.
            cached_ffn_ops_per_device: Optional cached FFN ops per device.
            skip_keys_per_device: Optional safetensors keys to skip per device.
        """
        hf_precomputed: Any | None = None
        if model_path is not None and _is_hf_checkpoint(model_path):
            logger.info('HF-source loader：为所有设备预分片检查点')
            precompute_stack = QwenTransformerStack(self.model_args, device_id=0, num_devices=self.num_devices)
            hf_precomputed = _precompute_all_device_states(model_path, self.model_args, self.num_devices, precompute_stack)
            logger.info('HF-source loader：预分片完成')

        def __load_weights(device_id: int, model_path: str | None) -> None:
            intermediates: list[torch.Tensor] = []
            caches: list[torch.Tensor] = []
            params: list[torch.Tensor] = []
            state_dicts: dict[str, torch.Tensor] = {}
            stack: QwenTransformerStack | None = None
            start_time = time.time()
            with torch.cuda.device(device_id):
                if model_path is not None:
                    skip_keys = skip_keys_per_device.get(device_id) if skip_keys_per_device is not None else None
                    if _is_hf_checkpoint(model_path):
                        stack = QwenTransformerStack(self.model_args, device_id, self.num_devices, moe_sync_callback=functools.partial(self._moe_sync, device_id))
                        state_dicts = load_hf_source_weights(model_path, self.model_args, self.num_devices, device_id, stack, precomputed=hf_precomputed)
                    else:
                        state_dicts = self.load_device_weights(model_path, device_id, ['model.embed_tokens.weight', f'layer_{self.model_args.n_layers}_lm_head.weight_dev_{device_id}', f'layer_{self.model_args.n_layers}_model.norm.weight_dev_{device_id}'], skip_keys=skip_keys)
                cached_ffn_ops = cached_ffn_ops_per_device.get(device_id) if cached_ffn_ops_per_device is not None else None
                if model_path is not None and _is_hf_checkpoint(model_path):
                    # 中文注释：HF 检查点路径：stack 已在上分支创建，初始化 tilert 与参考权重。
                    assert stack is not None
                    stack.init_tilert_weights(state_dicts)
                    stack.init_reference_weights(state_dicts)
                else:
                    # 中文注释：TileRT 原生权重路径：创建 stack 并初始化权重（或随机）。
                    stack = QwenTransformerStack(self.model_args, device_id, self.num_devices, cached_ffn_ops=cached_ffn_ops, moe_sync_callback=functools.partial(self._moe_sync, device_id))
                    if model_path is not None:
                        stack.init_tilert_weights(state_dicts)
                    else:
                        stack.init_random_weights()
                        stack._random_init_marker = True
                self._stack_objects[device_id] = stack
                params.extend(stack.get_weights_list())
                caches.extend(stack.get_cache_vars())
                if model_path is not None and (not _is_hf_checkpoint(model_path)):
                    head_shard_size = self.model_args.vocab_size // self.num_devices
                    full_lm_head = torch.empty(self.model_args.vocab_size, self.model_args.dim, dtype=torch.bfloat16, device=f'cuda:{device_id}')
                    index_path = os.path.join(model_path, 'model.safetensors.index.json')
                    with open(index_path, encoding='utf-8') as f:
                        weights_index = json.load(f)
                    weight_file_map = weights_index['weight_map']
                    for d in range(self.num_devices):
                        for shard_key in (f'layer_{self.model_args.n_layers}_lm_head.weight_dev_{d}', f'lm_head.weight_dev_{d}'):
                            if shard_key in state_dicts:
                                break
                            weight_file = weight_file_map.get(shard_key)
                            if weight_file is not None:
                                with safe_open(os.path.join(model_path, weight_file), framework='pt', device=f'cuda:{device_id}') as f:
                                    state_dicts[shard_key] = f.get_tensor(shard_key)
                                break
                        else:
                            raise RuntimeError(f'Missing lm_head shard for device {d} (tried layer_{self.model_args.n_layers}_lm_head.weight_dev_{d} and lm_head.weight_dev_{d})')
                        full_lm_head[d * head_shard_size:(d + 1) * head_shard_size] = state_dicts[shard_key]
                    state_dicts['lm_head.weight'] = full_lm_head
                    norm_key = f'layer_{self.model_args.n_layers}_model.norm.weight_dev_{device_id}'
                    if norm_key in state_dicts:
                        state_dicts['model.norm.weight'] = state_dicts[norm_key]
                head_proj = RMSNormHeadProj(model_args=self.model_args, device_id=device_id, num_devices=self.num_devices)
                if model_path is not None:
                    head_state = {alias: state_dicts[alias] for alias in head_proj.tilert_weights_alias() if alias in state_dicts}
                    prefixed_aliases = {alias: f'layer_{self.model_args.n_layers}_{alias}_dev_{device_id}' for alias in head_proj.tilert_weights_alias()}
                    for (alias, prefixed) in prefixed_aliases.items():
                        if alias not in head_state and prefixed in state_dicts:
                            head_state[alias] = state_dicts[prefixed]
                    for alias in head_proj.tilert_weights_alias():
                        if alias not in head_state and prefixed_aliases[alias] in state_dicts:
                            head_state[alias] = state_dicts[prefixed_aliases[alias]]
                    # golden/reference 路径需要二维的 norm 与 full lm_head；直接保存避免依赖 params 顺序。
                    if 'model.norm.weight' in head_state:
                        head_proj.ref_rmsnorm_gamma = head_state['model.norm.weight']
                    if 'lm_head.weight' in head_state:
                        head_proj.ref_head_proj = head_state['lm_head.weight']
                    head_proj.init_tilert_weights(head_state)
                else:
                    head_proj.init_random_weights(device_id=device_id)
                self._head_proj_objects[device_id] = head_proj
                params.extend(head_proj.get_weights_list())
                if model_path is not None:
                    embed_tokens = state_dicts['model.embed_tokens.weight']
                else:
                    embed_tokens = torch.randn(self.model_args.vocab_size, self.model_args.dim, dtype=torch.bfloat16, device=f'cuda:{device_id}') / self.model_args.dim ** 0.5
                params.append(embed_tokens)
                if model_path is not None:
                    freqs_cos = state_dicts['freqs_cos']
                    freqs_sin = state_dicts['freqs_sin']
                else:
                    (freqs_cos, freqs_sin) = self._gen_freqs_cis()
                freqs_cos = freqs_cos.to(device_id)
                freqs_sin = freqs_sin.to(device_id)
                params.append(freqs_cos)
                params.append(freqs_sin)
                intermediates.extend(
                    self.generate_params_with_continuous_storage(
                        self._get_temp_vars(
                            1,
                            self.forward_max_seq_len,
                            device_id,
                            {
                                'temperature': self.temperature,
                                'top_p': self.top_p,
                                'top_k': self.top_k,
                                'use_topp': self.use_topp,
                            },
                        ),
                        torch.device(f"cuda:{device_id}"),
                    )
                )
                sampling_config = intermediates[Idx.SAMPLING_CONFIG]
                sampling_config.copy_(torch.tensor([self.temperature, self.top_p, float(self.top_k), 1.0 if self.use_topp else 0.0], dtype=torch.float32, device=device_id))
                base_params_count = len(params)
                base_caches_count = len(caches)
                if self.with_mtp:
                    from tilert.models.qwen3_6.modules.mtp import QwenMTP
                    mtp = QwenMTP(self.model_args, device_id, self.num_devices)
                    if model_path is not None:
                        pass  # 中文注释：MTP 权重加载占位。
                    params.extend(mtp.get_weights_list())  # type: ignore[attr-defined]
                    caches.extend(mtp.get_cache_vars())  # type: ignore[attr-defined]
                    logger.info(f'[QwenShowHandsLayer._init_weights] 设备 {device_id} 已加载 MTP 权重')
                profile_logs = get_profile_log_tensor(device=torch.device(f"cuda:{device_id}"), num_max_insts=65536)
                assert profile_logs is not None
                result: DeviceResult = (intermediates, caches, params, profile_logs)
                self.multi_devices_results[device_id] = result
                self._base_params_count = base_params_count  # type: ignore[attr-defined]
                self._base_caches_count = base_caches_count  # type: ignore[attr-defined]
            del state_dicts
            torch.cuda.empty_cache()
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            time_str = f'{minutes} minutes {seconds} seconds' if minutes > 0 else f'{seconds} seconds'
            logger.info(f'Completed loading weights for device {device_id} in {time_str}')
        threads: list[threading.Thread] = []
        exceptions: list[Exception | None] = [None] * self.num_devices
        if False:
            for device_id in range(self.num_devices):
                try:
                    __load_weights(device_id, model_path)
                except Exception as exc:
                    exceptions[device_id] = exc
                    logger.error(f'Failed to initialize device {device_id}: {exc}')
        else:
            for device_id in range(self.num_devices):

                def _runner(dev_id: int) -> None:
                    try:
                        __load_weights(dev_id, model_path)
                    except Exception as exc:
                        exceptions[dev_id] = exc
                thread = threading.Thread(target=_runner, args=(device_id,))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
        for (device_id, exc) in enumerate(exceptions):
            if exc is not None:
                raise RuntimeError(f'Failed to initialize device {device_id}: {exc}') from exc
        self._init_moe_sync()
        if os.environ.get('TILERT_QWEN36_ENABLE_BACKEND_PREPARE', '0') == '1':
            for device_id in range(self.num_devices):
                with torch.cuda.device(device_id):
                    (intermediates, caches, params, profile_logs) = self._get_device_result(device_id)
                    qwen36_show_hands_prepare_money(params, intermediates, caches, profile_logs, self.forward_max_seq_len, self.with_mtp)
                    if self.with_mtp:
                        qwen36_show_hands_prepare_money(
                            params[:self._base_params_count],  # type: ignore[attr-defined]
                            intermediates,
                            caches[:self._base_caches_count],  # type: ignore[attr-defined]
                            profile_logs,
                            self.forward_max_seq_len,
                            False,
                        )

    def _init_hf_fallback(self, model_path: str) -> None:
        """Load a transformers model as a fallback when TileRT kernels are unavailable.

        The fallback model is loaded on cuda:0 with the same balanced 2-GPU
        memory profile used by the standalone transformers baseline.  All
        ``forward()`` calls are then delegated to this model so that the
        generator produces coherent text even though the TileRT golden path
        lacks tensor-parallel communication.
        """
        if not _HAS_TRANSFORMERS:
            raise RuntimeError('HF fallback requested but transformers is not installed')
        if self._hf_model is not None:
            return
        if not _is_hf_checkpoint(model_path):
            logger.warning('HF 回退仅支持 HF-source 检查点；跳过已转换 TileRT 检查点的回退')
            return
        logger.info('启用 HF 回退：以 AutoModelForCausalLM 作为参考路径加载')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert AutoModelForCausalLM is not None
            self._hf_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16, device_map='balanced', max_memory={0: '60GiB', 1: '60GiB', 'cpu': '200GiB'}, low_cpu_mem_usage=True, attn_implementation='sdpa')
        self._hf_model.eval()
        logger.info('HF 回退模型已加载')

    def from_pretrained(self, model_path: str) -> None:
        """Load the model weights from the given path."""
        if not os.path.exists(model_path):
            raise ValueError(f'Model weights directory {model_path} does not exist')
        logger.info(f'QwenShowHandsLayer.from_pretrained: {model_path}')
        self._init_weights(model_path)
        if self._hf_fallback_enabled:
            self._init_hf_fallback(model_path)
        logger.info('QwenShowHandsLayer.from_pretrained 完成')

    def from_pretrained_with_cache(self, model_path: str, cached_ffn_ops_per_device: dict[int, list], skip_keys_per_device: dict[int, set[str]]) -> None:
        """Load weights reusing cached MOE/MLP ops."""
        if not os.path.exists(model_path):
            raise ValueError(f'Model weights directory {model_path} does not exist')
        logger.info(f'QwenShowHandsLayer.from_pretrained_with_cache: {model_path}')
        self._init_weights(model_path, cached_ffn_ops_per_device=cached_ffn_ops_per_device, skip_keys_per_device=skip_keys_per_device)
        logger.info('QwenShowHandsLayer.from_pretrained_with_cache 完成')

    def init_random_weights(self) -> None:
        """Generate random weights for smoke testing."""
        logger.info('QwenShowHandsLayer.init_random_weights')
        self._init_weights(None)
        logger.info('QwenShowHandsLayer.init_random_weights 完成')

    def _init_moe_sync(self) -> None:
        """Allocate shared buffers and the barrier used for TP8 MoE all-reduce.

        Must be called after all device stacks have been constructed because
        each stack holds a bound reference to ``_moe_sync``.
        """
        if self.num_devices <= 1:
            return
        dim = self.model_args.dim
        max_seq_len = self.forward_max_seq_len
        # 中文注释：为每个设备分配 all-reduce 的部分和与聚合 buffer。
        self._moe_partial_buf = [torch.zeros(1, max_seq_len, dim, dtype=torch.bfloat16, device=f'cuda:{d}') for d in range(self.num_devices)]
        self._moe_aggregated_buf = [torch.zeros(1, max_seq_len, dim, dtype=torch.bfloat16, device=f'cuda:{d}') for d in range(self.num_devices)]
        self._moe_partial_lens = [0] * self.num_devices

        def _barrier_action() -> None:
            assert self._moe_partial_buf is not None
            assert self._moe_aggregated_buf is not None

            # The partial buffers live on different GPUs.  Force each device to
            # finish its copy before we read it on the host, otherwise the CPU
            # side can observe a partially-updated tensor and corrupt the sum.
            for d in range(self.num_devices):
                torch.cuda.synchronize(d)

            total = torch.zeros(1, max_seq_len, dim, dtype=torch.float32)
            for d in range(self.num_devices):
                active_len = self._moe_partial_lens[d]
                if active_len <= 0:
                    continue
                part = self._moe_partial_buf[d][:, :active_len, :].float().cpu().clone()
                total[:, :active_len, :] += part
            total = total.to(torch.bfloat16)
            for d in range(self.num_devices):
                self._moe_aggregated_buf[d].copy_(total)
                torch.cuda.synchronize(d)
        self._moe_barrier = threading.Barrier(self.num_devices, action=_barrier_action)
        logger.info(f'[QwenShowHandsLayer._init_moe_sync] TP8 MoE all-reduce 初始化完成：设备数={self.num_devices}, buffer_shape=(1, {max_seq_len}, {dim})')

    def _moe_sync(self, device_id: int, h: torch.Tensor) -> torch.Tensor:
        """Barrier-based TP8 MoE fallback for environments without torch.distributed.

        Each rank writes its local partial output into a shared buffer, waits at
        the barrier, and then reads back the summed aggregation buffer.
        The previous implementation was vulnerable to accidentally treating one
        rank's buffer as the final global result because the barrier callback was
        executed in a way that could race with the caller's subsequent use.
        """
        if self.num_devices <= 1 or self._moe_barrier is None:
            return h
        seq_len = h.size(1)
        assert self._moe_partial_buf is not None
        assert self._moe_aggregated_buf is not None

        # Write the local contribution to the per-rank partial buffer and
        # record the active length so the barrier action only sums the valid
        # rows.  This avoids stale tail values from previous decode steps.
        self._moe_partial_lens[device_id] = seq_len
        self._moe_partial_buf[device_id].zero_()
        self._moe_partial_buf[device_id][:, :seq_len, :].copy_(h, non_blocking=False)
        torch.cuda.synchronize(device_id)

        # Wait until every rank has submitted its partial result.
        self._moe_barrier.wait()
        torch.cuda.synchronize(device_id)

        # Return an independent copy of the aggregated tensor.  Returning a
        # view into the shared buffer would let later barrier invocations
        # overwrite the caller-visible value and make the result appear to
        # change after the callback returns.
        return self._moe_aggregated_buf[device_id][:, :seq_len, :].contiguous().clone()

    def _get_full_head_proj(self, device_id: int) -> torch.Tensor:
        """Return the full vocabulary head projection on the target device.

        The golden reference path replicates the full lm_head on every device,
        so no all-gather is required.  The result is cached per device.
        """
        cached = self._full_head_proj_cache[device_id]
        if cached is not None:
            logger.warning(f'[QwenShowHandsLayer._get_full_head_proj_{device_id}] CACHE HIT shape={cached.shape} dtype={cached.dtype} ptr={cached.data_ptr()}')
            return cached
        head_proj = self._head_proj_objects[device_id]
        if head_proj is None:
            raise RuntimeError(f'Head projection not initialized on device {device_id}')
        local_head = head_proj.ref_head_proj
        if local_head is None:
            raise RuntimeError(f'ref_head_proj is not initialized on device {device_id}')
        logger.warning(f'[QwenShowHandsLayer._get_full_head_proj_{device_id}] CACHE MISS local_head.shape={local_head.shape} dtype={local_head.dtype} ptr={local_head.data_ptr()}')
        # 说明 local_head 可能出现的几种三维情况：
        # 1) device_sharding 在 init_reference_weights 中把每张卡的全量 head 堆成 (num_devices, vocab_size, dim)，
        #    再按 device_id 取下标，得到 (vocab_size, dim) 的二维张量；
        # 2) 如果传入的是 TileRT 转换后的 kernel layout，三维形状为 (logits_dim/16*num_steps, 16, 1024)，
        #    第一维是展平后的 logits tile，不是 device id（device 维度已经在 device_sharding 阶段切掉了）。
        # 以下 reshape 把这两种三维都还原回 golden 路径需要的 (vocab_size, dim)。
        if local_head.dim() == 2:
            full_head = local_head
        elif local_head.dim() == 3:
            if local_head.size(-1) == 1024 and local_head.size(-2) == 16:
                full_head = local_head.transpose(1, 2).reshape(-1, self.model_args.dim)
            else:
                full_head = local_head.reshape(-1, self.model_args.dim)
        else:
            raise ValueError(f'Unexpected head projection layout: {local_head.shape}')
        logger.warning(f'[QwenShowHandsLayer._get_full_head_proj_{device_id}] CACHING full_head.shape={full_head.shape} ptr={full_head.data_ptr()}')
        self._full_head_proj_cache[device_id] = full_head
        return full_head

    def _golden_forward_device(self, device_id: int, token_id: torch.Tensor, cur_pos: int) -> DeviceResult:
        """Reference forward for a single device.

        Delegates the actual layer math to ``QwenTransformerStack`` and
        ``RMSNormHeadProj`` so that each op can later switch to its own
        TileRT kernel via ``flag_enable_tilert``.
        """
        token_val = token_id.view(-1).tolist() if token_id.numel() > 1 else token_id.item()
        logger.info(f'[QwenShowHandsLayer._golden_forward_device_{device_id}] ENTRY: token_id={token_val}, cur_pos={cur_pos}')
        (intermediates, caches, params, profile_logs) = self._get_device_result(device_id)
        stack = self._stack_objects[device_id]
        head_proj = self._head_proj_objects[device_id]
        if stack is None or head_proj is None:
            raise RuntimeError(f'QwenTransformerStack/HeadProj not initialized on device {device_id}')
        stack_weight_count = len(stack.get_weights_list())
        embed_weight = params[stack_weight_count + 2]
        freqs_cos_param = params[stack_weight_count + 3]
        freqs_sin_param = params[stack_weight_count + 4]
        logger.info(f'[QwenShowHandsLayer._golden_forward_device_{device_id}] params info:')
        logger.info(f'  - stack_weight_count={stack_weight_count}')
        logger.info(f'  - embed_weight: shape={embed_weight.shape}, dtype={embed_weight.dtype}, device={embed_weight.device}')
        logger.info(f'  - freqs_cos_param: shape={freqs_cos_param.shape}, dtype={freqs_cos_param.dtype}, device={freqs_cos_param.device}')
        logger.info(f'  - freqs_sin_param: shape={freqs_sin_param.shape}, dtype={freqs_sin_param.dtype}, device={freqs_sin_param.device}')
        stack_weights = stack.get_weights_list()
        logger.info(f'[QwenShowHandsLayer._golden_forward_device_{device_id}] stack has {len(stack_weights)} weights')
        for (i, w) in enumerate(stack_weights[:5]):
            logger.info(f'  - stack_weights[{i}]: shape={w.shape}, dtype={w.dtype}, device={w.device}')
        if len(stack_weights) > 5:
            logger.info(f'  - ... and {len(stack_weights) - 5} more weights')
        idx = token_id.view(-1).to(embed_weight.device)
        logger.info(f'[QwenShowHandsLayer._golden_forward_device_{device_id}] Lookup embedding: idx.shape={idx.shape}, embed_weight.shape={embed_weight.shape}')
        x = embed_weight[idx].unsqueeze(0).to(torch.bfloat16)
        logger.info(f'[QwenShowHandsLayer._golden_forward_device_{device_id}] x after embedding: shape={x.shape}, dtype={x.dtype}, device={x.device}')
        seq_len = x.size(1)
        if seq_len == 1:
            intermediates[Idx.TOKEN_ID][0, 0, 0] = token_id.view(-1)[0]
        else:
            intermediates[Idx.TOKEN_ID][0, :seq_len, 0] = token_id.view(-1)
        intermediates[Idx.CUR_POS][0] = cur_pos
        logger.info(f'[QwenShowHandsLayer._golden_forward_device_{device_id}] Set TOKEN_ID and CUR_POS={cur_pos}')
        if self._golden_caches[device_id] is None:
            logger.info(f'[QwenShowHandsLayer._golden_forward_device_{device_id}] Initializing layer caches')
            self._golden_caches[device_id] = stack._init_layer_caches((freqs_cos_param, freqs_sin_param))
            logger.info(f'[QwenShowHandsLayer._golden_forward_device_{device_id}] Layer caches initialized')
        logger.info(f'[QwenShowHandsLayer._golden_forward_device_{device_id}] Calling stack.forward, input shape={x.shape}')
        (h, self._golden_caches[device_id]) = stack.forward(x, cur_pos, (freqs_cos_param, freqs_sin_param), self._golden_caches[device_id])
        logger.info(f'[QwenShowHandsLayer._golden_forward_device_{device_id}] stack.forward output: h.shape={h.shape}, dtype={h.dtype}, device={h.device}')
        logger.info(f'[QwenShowHandsLayer._golden_forward_device_{device_id}] Getting full head projection')
        full_head = self._get_full_head_proj(device_id)
        logger.info(f'[QwenShowHandsLayer._golden_forward_device_{device_id}] full_head: shape={full_head.shape}, dtype={full_head.dtype}, device={full_head.device}')
        if head_proj.ref_rmsnorm_gamma is None:
            raise RuntimeError(f'ref_rmsnorm_gamma is not initialized on device {device_id}')
        head_proj.ref_head_proj = full_head
        logger.info(f'[QwenShowHandsLayer._golden_forward_device_{device_id}] head_proj.ref_rmsnorm_gamma: shape={head_proj.ref_rmsnorm_gamma.shape}')
        logger.info(f'[QwenShowHandsLayer._golden_forward_device_{device_id}] Calling head_proj.golden_forward, input h.shape={h.shape}')
        logits = head_proj.golden_forward(h)
        logger.info(f'[QwenShowHandsLayer._golden_forward_device_{device_id}] logits after head_proj: shape={logits.shape}, dtype={logits.dtype}')
        last_pos = logits.size(1) - 1
        vocab_shard = logits.size(-1)
        intermediates[Idx.LOGITS_OUT][0, 0, :vocab_shard].copy_(logits[0, last_pos, :])
        logger.info(f'[QwenShowHandsLayer._golden_forward_device_{device_id}] Copied logits to intermediates, vocab_shard={vocab_shard}')
        logger.info(f'[QwenShowHandsLayer._golden_forward_device_{device_id}] Sampling from logits at position {last_pos}')
        token_out = self._sample(logits[0, last_pos])
        intermediates[Idx.TOKEN_OUT][0, 0, 0] = token_out
        logger.info(f'[QwenShowHandsLayer._golden_forward_device_{device_id}] EXIT: token_out={token_out}')
        return (intermediates, caches, params, profile_logs)

    def _sample(self, logits: torch.Tensor) -> int:
        """Simple sampling helper for the golden path."""
        if self.use_topp:
            return int(logits.argmax().item())
        return int(logits.argmax().item())

    def _hf_forward(self, token_id: torch.Tensor) -> list[DeviceResult]:
        """Reference forward using a loaded transformers model.

        This path is used as a temporary fallback when the TileRT golden path
        cannot produce correct results because tensor-parallel weights are not
        aggregated across devices.  It runs a real transformers model on
        cuda:0/1 and writes the resulting logits/token into device 0's
        intermediates so that the rest of the generator logic is unchanged.
        """
        assert self._hf_model is not None
        token_id = token_id.view(-1).to(dtype=torch.long, device=self._hf_model.device)
        if self._hf_input_ids is None:
            self._hf_input_ids = token_id.unsqueeze(0)
        else:
            self._hf_input_ids = torch.cat([self._hf_input_ids, token_id.unsqueeze(0)], dim=1)
        with torch.inference_mode():
            if self._hf_past_key_values is None:
                outputs = self._hf_model(self._hf_input_ids, use_cache=True)
                self._hf_past_key_values = outputs.past_key_values
            else:
                outputs = self._hf_model(token_id.unsqueeze(0), past_key_values=self._hf_past_key_values, use_cache=True)
                self._hf_past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :].float()
        next_token = int(logits.argmax(dim=-1).item())
        first_result: DeviceResult | None = None
        for device_id in range(self.num_devices):
            (intermediates, caches, params, profile_logs) = self._get_device_result(device_id)
            vocab_shard = logits.size(-1)
            intermediates[Idx.LOGITS_OUT][0, 0, :vocab_shard].copy_(logits[0, :])
            intermediates[Idx.TOKEN_OUT][0, 0, 0] = next_token
            if device_id == 0:
                first_result = (intermediates, caches, params, profile_logs)
        assert first_result is not None
        results: list[DeviceResult] = []
        for device_id in range(self.num_devices):
            if device_id == 0:
                results.append(first_result)
            else:
                results.append(self._get_device_result(device_id))
        return results

    def forward(self, token_id: torch.Tensor, with_mtp: bool | None=None, cur_pos: int=0) -> list[DeviceResult]:
        """Run one decode step.

        Args:
            token_id: Scalar or [1] int32 tensor containing the input token id.
            with_mtp: Override MTP mode.  Defaults to ``self.with_mtp``.
            cur_pos: Current decode position (used by the golden path).

        Returns:
            List of per-device ``DeviceResult`` tuples.
        """
        active_mtp = with_mtp if with_mtp is not None else self.with_mtp
        token_val = token_id.view(-1).tolist() if token_id.numel() > 1 else token_id.item()
        logger.info(f'[QwenShowHandsLayer.forward] ENTRY: token_id={token_val}, active_mtp={active_mtp}, cur_pos={cur_pos}, num_devices={self.num_devices}')
        with torch.inference_mode():
            try:
                logger.info(f'[QwenShowHandsLayer.forward] 尝试 CUDA kernel 路径（qwen36_show_hands）')
                qwen36_show_hands(token_id.cpu(), active_mtp)
                logger.info(f'[QwenShowHandsLayer.forward] CUDA kernel 路径成功')
            except (AttributeError, RuntimeError) as e:
                logger.info(f'[QwenShowHandsLayer.forward] CUDA kernel path FAILED ({type(e).__name__}), falling back to golden/HF path')
                if self._hf_model is not None:
                    logger.info(f'[QwenShowHandsLayer.forward] 使用 HF 回退路径')
                    return self._hf_forward(token_id)
                logger.info(f'[QwenShowHandsLayer.forward] Using golden forward path on all {self.num_devices} devices')
                return self._golden_forward_all_devices(token_id, cur_pos)
            logger.info(f'[QwenShowHandsLayer.forward] Collecting device results for {self.num_devices} devices')
            results = [self._get_device_result(device_id) for device_id in range(self.num_devices)]
            if results:
                (intermediates, caches, params, profile_logs) = results[0]
                token_out = intermediates[Idx.TOKEN_OUT][0, 0, 0].item()
                logger.info(f'[QwenShowHandsLayer.forward] EXIT: device_0 token_out={token_out}')
            return results

    def _golden_forward_all_devices(self, token_id: torch.Tensor, cur_pos: int) -> list[DeviceResult]:
        """Run the golden forward on all devices in parallel.

        Each device computes its TP8 MoE shard; the barrier inside
        ``_moe_sync`` aggregates partial outputs after every layer.
        """
        token_val = token_id.view(-1).tolist() if token_id.numel() > 1 else token_id.item()
        logger.info(f'[QwenShowHandsLayer._golden_forward_all_devices] ENTRY: token_id={token_val}, cur_pos={cur_pos}, num_devices={self.num_devices}')
        results: list[DeviceResult | None] = [None] * self.num_devices
        exceptions: list[Exception | None] = [None] * self.num_devices

        def _runner(device_id: int) -> None:
            try:
                logger.info(f'[QwenShowHandsLayer._golden_forward_all_devices] 启动设备 {device_id} 线程')
                with torch.inference_mode():
                    results[device_id] = self._golden_forward_device(device_id, token_id, cur_pos)
                logger.info(f'[QwenShowHandsLayer._golden_forward_all_devices] 设备 {device_id} 线程完成')
            except Exception as exc:
                exceptions[device_id] = exc
                logger.error(f'[QwenShowHandsLayer._golden_forward_all_devices] 设备 {device_id} 线程失败：{exc}')
        threads = [threading.Thread(target=_runner, args=(device_id,)) for device_id in range(self.num_devices)]
        logger.info(f'[QwenShowHandsLayer._golden_forward_all_devices] 启动 {len(threads)} 个线程')
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        logger.info(f'[QwenShowHandsLayer._golden_forward_all_devices] 所有线程已 join')
        for (device_id, exc) in enumerate(exceptions):
            if exc is not None:
                raise RuntimeError(f'设备 {device_id} 的 golden forward 失败：{exc}') from exc
        for device_id in range(self.num_devices):
            assert results[device_id] is not None
            (intermediates, caches, params, profile_logs) = results[device_id]  # type: ignore[misc]
            token_out = intermediates[Idx.TOKEN_OUT][0, 0, 0].item()
            logits_out = intermediates[Idx.LOGITS_OUT][0, 0, :]
            argmax_token = int(logits_out.argmax().item())
            logger.info(f'[QwenShowHandsLayer._golden_forward_all_devices] 设备 {device_id} token_out={token_out} argmax={argmax_token} logits_top5={logits_out.topk(5).indices.tolist()}')
        logger.info(f'[QwenShowHandsLayer._golden_forward_all_devices] 退出，返回 {len(results)} 个设备结果')
        return [results[device_id] for device_id in range(self.num_devices)]  # type: ignore[return-value]

    def set_sampling_seed(self, seed: int, with_mtp: bool | None=None) -> None:
        """Set the sampling seed for top-p sampling."""
        active_mtp = with_mtp if with_mtp is not None else self.with_mtp
        try:
            qwen36_show_hands_set_sampling_seed(seed, active_mtp)
        except (AttributeError, RuntimeError):
            pass

    def set_cur_pos(self, cur_pos: int, with_mtp: bool | None=None) -> None:
        """Set the current decode position for RoPE."""
        active_mtp = with_mtp if with_mtp is not None else self.with_mtp
        try:
            qwen36_show_hands_set_cur_pos(cur_pos, active_mtp)
        except (AttributeError, RuntimeError):
            pass

    def reset_sequence(self) -> None:
        """Reset the decode sequence state."""
        self._golden_caches = [None] * self.num_devices
        self._hf_past_key_values = None
        self._hf_input_ids = None
        try:
            if self.with_mtp:
                qwen36_show_hands_reset(True)
                qwen36_show_hands_reset(False)
            else:
                qwen36_show_hands_reset(False)
        except (AttributeError, RuntimeError):
            pass

    def cleanup(self) -> None:
        """Release CUDA graphs and cached reference tensors."""
        self._golden_caches = [None] * self.num_devices
        self._full_head_proj_cache = [None] * self.num_devices
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
            print(f'Exception during cleanup: {e}', file=sys.stderr)

    def _get_device_result(self, device_id: int) -> DeviceResult:
        device_result = self.multi_devices_results[device_id]
        if device_result is None:
            raise RuntimeError(f'Device {device_id} is not initialized')
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

    def get_next_draft_tokens(self, device_id: int=0) -> torch.Tensor:
        """Get next_draft_tokens from the specified device."""
        (intermediates, _, _, _) = self._get_device_result(device_id)
        return intermediates[Idx.NEXT_DRAFT_TOKENS]

    def get_num_accepted(self, device_id: int=0) -> int:
        """Get number of accepted tokens from the specified device."""
        (intermediates, _, _, _) = self._get_device_result(device_id)
        return int(intermediates[Idx.ACCEPTED_TOKENS][0].item())

    def get_predicted_tokens(self, device_id: int=0) -> torch.Tensor:
        """Get predicted_tokens from the specified device."""
        (intermediates, _, _, _) = self._get_device_result(device_id)
        return intermediates[Idx.PREDICTED_TOKENS]

    def get_logits(self, device_id: int=0) -> torch.Tensor:
        """Get logits from the specified device."""
        (intermediates, _, _, _) = self._get_device_result(device_id)
        return intermediates[Idx.LOGITS_OUT]

    def get_next_token(self, device_id: int=0) -> int:
        """Convenience helper: return the sampled next token id."""
        (intermediates, _, _, _) = self._get_device_result(device_id)
        return int(intermediates[Idx.TOKEN_OUT][0, 0, 0].item())

    def get_top_n_logprobs(self, device_id: int=0) -> tuple[torch.Tensor, torch.Tensor]:
        """Get top-N log-probabilities and token IDs from the top_p kernel."""
        (intermediates, _, _, _) = self._get_device_result(device_id)
        return (intermediates[Idx.TOP_N_LOG_PROBS], intermediates[Idx.TOP_N_INDICES])