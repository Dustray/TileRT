# Qwen3.6-35B-A3B 接入 TileRT 技术方案 (Plan 2, TP8)

请及时回顾此文档

> 从 `docs/qwen36_integration_plan.md` 精简而来。根据本地 vLLM 实测数据，纯 EP 综合性能劣于纯 TP（首次执行 TTFT 差距约 10 倍），因此**当前主路线切换为 TP8（Tensor Parallelism 8）**。本计划保留后续可能复用的 EP 混合路线要点，但优先按 TP8 推进。
> 最近更新：2026-07-24。

## 0. 环境与目标

- **项目**：以TileRT Deepseek模块框架为模板，使Qwen模块（走golden forward流程，非c++闭源库）完成推理目标。
- **目标**：基于原始 HuggingFace checkpoint，在 8 张 DCU 上以 **TP8（Tensor Parallelism 8，无 Expert Parallelism）** 跑通 `QwenShowHandsLayer.golden_forward`，并通过 `scripts/verify_qwen36_generator_official_prompt.py` 生成可读文本。
- **源模型**：`/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master`
  - 原始 checkpoint key 前缀：`model.language_model.*`
  - 26 个 safetensors shard，总大小约 71.9 GB（文本部分）。
- **测试环境**：只能在`tilert-qwen3.6` docker容器内部，不要用宿主机虚拟环境：`docker exec -it tilert-qwen3.6 bash`

```bash
export LD_LIBRARY_PATH=/opt/dtk-26.04/lib:/opt/dtk-26.04/hip/lib:/opt/dtk-26.04/.hyhal/rocm_smi/lib
export PYTHONPATH=/public/home/dinggy/yiny/projects/TileRT:$PYTHONPATH
```

## 1. 关键参数 (ModelArgsQwen36)

```python
vocab_size: int = 248320
dim: int = 2048
inter_dim: int = 512           # MoE / shared expert 中间层
n_layers: int = 40

# full_attention (GQA)
n_heads: int = 16
n_kv_heads: int = 2
qk_head_dim: int = 256
rope_dim: int = 64
partial_rotary_factor: float = 0.25
mrope_section: list[int] = [11, 11, 10]
use_mrope: bool = True

# linear_attention (DeltaNet)
linear_num_key_heads: int = 16
linear_num_value_heads: int = 32
linear_key_head_dim: int = 128
linear_value_head_dim: int = 128
linear_conv_kernel_dim: int = 4

# MoE
n_routed_experts: int = 256
n_activated_experts: int = 8
n_shared_experts: int = 1

# 层结构
n_delta_layers: int = 30        # linear_attention
n_gated_layers: int = 10        # full_attention

max_seq_len: int = 262144
rope_theta: float = 1e7
rms_norm_eps: float = 1e-6
```

## 2. 架构总览

```
10 × [3 × (linear_attention → MoE) → 1 × (full_attention → MoE)]
```

- 40 层异构：`layer_types` 由 HF `text_config.layer_types` 给出，每 4 层为一个 `[linear_attention]×3 + [full_attention]×1` 循环。
- 顶层：`embed_tokens` → 40 层 stack → `model.norm` → `lm_head`。

## 3. TP8 布局设计

**核心原则**：只有 40 层 DecoderLayer 内部的 **routed experts** 做 TP8 切分，其余部分全部完整复制（DP）。

```text
8 张 DCU（device 0..7），每张卡持有：
  - embed_tokens.weight      [248320, 2048]      完整复制（vocab 不切）
  - model.norm.weight          [2048]              完整复制
  - lm_head.weight             [248320, 2048]      完整复制（vocab 不切）
  - 40 层每层：
      - input_layernorm.weight / post_attention_layernorm.weight   [2048]   完整复制
      - self_attn / linear_attn 全部投影矩阵完整复制（Attention 不做 TP）
      - 全部 256 个 routed experts 的 gate_up_proj / down_proj 按 inter_dim 切 8 份
      - shared expert 的 gate_up_proj / down_proj 同样按 inter_dim 切 8 份
      - router.weight / shared_expert_gate.weight                   完整复制
```

- `tp_size = 8`，`num_devices = 8`。
- **Attention / DeltaNet / RMSNorm / Embedding / LM Head**：全部完整复制到每张卡，forward 时不做 allreduce。
- **Routed Experts**：每个 expert 的 `gate_up_proj` / `down_proj` 按 `inter_dim` 切 8 份；每张卡计算本 rank 的 1/8 中间激活，最后通过 `allreduce` 聚合。
- **Shared Expert**：与 routed experts 采用相同的 inter_dim 切分，forward 后同样 allreduce。
- **不启用 EP**：每个 device 拥有完整 256 个 experts 的 1/8 slice，不需要 all-to-all dispatch。
- **复用 allreduce 基础设施**：MoE 输出聚合可复用现有 `PaddedAllReduceAdd`，也可新增轻量 `allreduce + residual add`。

### 3.1 与已有 op 的对应

| 模块 | TP8 切分方式 | 聚合 op |
|------|-------------|---------|
| `EmbedTokens` | 完整复制 | 无 |
| `RMSNorm`（layer/post/lm_head 前） | 完整复制 | 无 |
| `GQAAttention` | 完整复制（q/k/v/o_proj 不切） | 无 |
| `DeltaNet` | 完整复制（in_proj_* / out_proj 不切） | 无 |
| `RMSNormExpertProj` | 完整复制 | 无 |
| `SharedExpert` | gate/up/down 按 `inter_dim/8` 切 | allreduce + residual add |
| `ExpertSelectUpGateSiLU`（routed） | gate/up 按 `inter_dim/8` 切 | 无需 allreduce |
| `ExpertDownAllReduce`（routed） | down 按 `inter_dim/8` 输入切、输出 `dim` 完整 | allreduce + residual add |
| `RMSNormHeadProj` / `LM Head` | 完整复制 | 无 |

### 3.2 保留的 EP 路线（未来备选）

如果后续需要评估 EP/TP 混合，可在本计划基础上重新启用：
- attention 继续做 TP；
- MoE 做 EP（expert 维度分片）+ all-to-all；
- 参考 `docs/perf_summary.md`，上线前必须用真实 workload 对比 EP 与 TP 的总吞吐/TTFT/抖动。

## 4. 关键代码文件清单

| 文件 | 作用 |
|------|------|
| `tilert/models/qwen3_6/model_args.py` | `ModelArgsQwen36` 参数 |
| `tilert/models/qwen3_6/generator.py` | `Qwen36Generator` 文本生成入口 |
| `tilert/models/qwen3_6/temp_var_indices.py` | `QwenTempVarIdx` 35 槽临时变量布局 |
| `tilert/models/qwen3_6/modules/transformer_stack.py` | 40 层异构栈，golden/tilert 双路径 |
| `tilert/models/qwen3_6/modules/delta_net.py` | DeltaNet wrapper + `QwenMoeBlock` 复用 |
| `tilert/models/qwen3_6/modules/gated_attention.py` | GQA wrapper |
| `tilert/models/qwen3_6/modules/moe.py` | MoE block 组合：RMSNormExpertProj + ExpertSelectUpGateSiLU + ExpertDownAllReduce |
| `tilert/models/qwen3_6/modules/end2end.py` | `QwenShowHandsLayer`：多设备加载、forward、golden fallback |
| `tilert/models/qwen3_6/modules/hf_source_loader.py` | 从原始 HF checkpoint 一次性预切分所有 device 的权重 |
| `tilert/models/qwen3_6/ops/gqa_attention.py` | GQA 参考实现与 weight layout |
| `tilert/models/qwen3_6/ops/delta_net.py` | DeltaNet 参考实现与 weight layout |
| `tilert/models/qwen3_6/ops/expert_sel_up_gate_silu.py` | Router + gate/up projection + SiLU |
| `tilert/models/qwen3_6/ops/expert_down_allreduce.py` | Down projection + aggregation |
| `tilert/models/common.py` | `_safe_weight_dequant`、RMSNorm 等通用工具 |
| `tilert/models/utils.py` | M-RoPE precompute / apply |

## 5. TP8 核心模块改造清单

### 5.1 注意力类 op（GQA / DeltaNet）

按最新策略，**Attention 与 DeltaNet 保持 DP 完整复制，不做 TP 切分**。

- 保留当前 EP8 实现中的权重复制逻辑（`device_sharding` 将完整权重 stack 8 份）。
- forward 不需要 allreduce。
- 无需修改 `gqa_attention.py`、`delta_net.py` 中的切分逻辑。
- 若未来要评估 attention TP，再在本计划基础上扩展。

涉及文件（当前无需改动）：
- `tilert/models/qwen3_6/ops/gqa_attention.py`
- `tilert/models/qwen3_6/ops/delta_net.py`
- `tilert/models/qwen3_6/modules/gated_attention.py`
- `tilert/models/qwen3_6/modules/delta_net.py`

### 5.2 MoE 类 op（routed + shared experts）

当前 EP8 实现：按 expert 维度切 8 份，每个 device 32 个完整专家。  
TP8 目标：
- 每个 device 持有全部 256 个 routed experts + 1 shared expert；
- 每个 expert 的 `gate_up_proj` / `down_proj` 按 `inter_dim` 切 8 份；
- `ExpertSelectUpGateSiLU`（routed 与 shared）计算本 rank 的 1/8 中间激活；
- `ExpertDownAllReduce` 计算完整 `dim` 输出，但只使用本 rank 的 1/8 `inter_dim` 输入；最后通过 allreduce 聚合各 rank 的部分和；
- 聚合后做 residual add。

涉及文件：
- `tilert/models/qwen3_6/ops/expert_sel_up_gate_silu.py`
- `tilert/models/qwen3_6/ops/expert_down_allreduce.py`
- `tilert/models/qwen3_6/ops/rmsnorm_expert_proj.py`
- `tilert/models/qwen3_6/modules/moe.py`

### 5.3 加载与端到端

- `tilert/models/qwen3_6/modules/hf_source_loader.py`：预切分逻辑从 EP8 4D stacked 改为 TP8 3D split。
- `tilert/models/qwen3_6/modules/end2end.py`：多设备加载、temp_vars、golden forward 路径需要适配 TP8 的 allreduce 输出。

## 6. 已完成的修复（仍可复用）

### 6.1 通用反量化安全回退

`tilert/models/common.py` 新增 `_safe_weight_dequant(weight, scale)`：
- 单元素 scale 直接广播；
- 满足 `(m//128, n//128)` 形状时走原有 kernel；
- 否则 cast 为 bf16（配合 fake all-ones scale 安全）。

所有 qwen3_6 op 的 reference 路径均已切换。

### 6.2 GQA gated q-projection

- `q_proj` 输出翻倍（query + gate），split 后 chunk；
- 对 q/k 应用 per-head RMSNorm；
- attention output 乘以 `sigmoid(gate.mean(dim=-1))`。

### 6.3 残差缩放条件化

`transformer_stack.py` 仅在 `_is_random_init()` 为真时使用 `residual_scale = 1.0 / n_layers`；真实权重路径关闭该缩放。

### 6.4 真实权重加载预切分（需改为 TP8）

`hf_source_loader.py`：
- 一次性读取完整 CPU state dict；
- `_precompute_all_device_states` 调用各 op 的 `device_sharding`；
- `_unshard_to_per_device` 按 `num_devices` 维提取每个 device 的切片；
- 各 device 线程只做 `tensor.to(cuda:{device_id})` 与 `init_tilert_weights`。

### 6.5 已完成但未在 TP8 中继续使用的 EP8 修复

- `_ensure_per_device` 切片逻辑（`expert_down_allreduce.py`、`expert_sel_up_gate_silu.py`）用于处理 EP8 的 4D stacked 张量；切到 TP8 后 MoE 权重形状会恢复为 3D per-device，该 workaround 可保留但不再关键。
- EP8 相关的 `local_indices = indices % n_local_routed` 在 TP8 中不再需要（所有 device 看到同样的完整 expert 集合的 1/8 slice）。

## 7. 已知问题与风险

| 问题 | 影响 | 应对 |
|------|------|------|
| 当前代码按 EP8 实现 MoE 权重分片，需要回滚/重构成 TP8 | 如果不改，TP8 下每个 expert 只算 1/8 激活但无聚合，结果错误 | 重写 routed/shared expert 的 `device_sharding` 和 `golden_forward`，引入 allreduce |
| attention / DeltaNet 当前是复制而非切分 | 符合最新 TP8 策略（非 bug） | 保持当前复制逻辑，无需改动 |
| `route_scale=2.5` 与 HF 官方 `Qwen3_5MoeTopKRouter` 不一致 | 可能影响 logits 对齐 | 待验证；必要时改为 `1.0` |
| shared expert gate 位置与 HF 不同 | HF 在 down-proj 后乘，TileRT 在 up_gate_silu 后乘；数值等价 | 可保持 |
| CUDA kernel 尚未就绪 | `libtilert_qwen36.so` 未构建 | Python golden 路径 fallback |

## 8. 验证计划

| 步骤 | 脚本/方法 | 标准 | 状态 |
|---|---|---|---|
| 1 | 单 op TP8 单元测试 | GQA / DeltaNet / MoE 在 8 卡上 golden_forward 输出形状与单卡一致 | ⏳ 待实现 |
| 2 | `scripts/verify_qwen36_random_init_forward.py` | TP8 40 层 forward finite、形状正确、next token 合法 | ⏳ 待实现 |
| 3 | `scripts/verify_qwen36_real_weights_forward.py` | 真实 HF 权重加载成功、8 卡内存均衡、连续 4 个 token 有效 | ⏳ 待执行 |
| 4 | `scripts/verify_qwen36_generator_official_prompt.py` | 官方 prompt 生成非空、可读文本 | ⏳ 待执行 |
| 5 | 与 HF `AutoModelForCausalLM` 对比 logits | 相同 prompt top-1 token 一致或误差 < 1e-2 | ⏳ 待执行 |
| 6 | CUDA kernel | `libtilert_qwen36.so`：gqa_attention_op、delta_net_op、qwen36_show_hands* | ⏳ 待实现 |

## 9. 后续工作优先级（TP8，按推理流程自底向上）

按 **Embedding → RMSNorm → Attention → Shared Experts → Routed Experts → LM Head** 的顺序推进。每个模块先完成设计/文档，再实现代码，再跑单模块/局部测试。

1. **P0 - Embedding**：确认 `embed_tokens` 完整复制、加载与 forward 路径。
2. **P0 - RMSNorm**：input / post-attention / final 三种 RMSNorm 完整复制，权重加载正确。
3. **P0 - Attention（GQA / DeltaNet）**：保持 DP 完整复制，验证 golden_forward 输出与单卡一致；必要时加入 device 间同步检查。
4. **P0 - Shared Experts**：实现 inter_dim TP8 切分 + allreduce + residual add。
5. **P0 - Routed Experts**：实现 256 experts 的 inter_dim TP8 切分 + topk 路由 + allreduce + residual add。
6. **P0 - LM Head**：完整复制，确认 logits 输出聚合方式。
7. **P0**：适配 `hf_source_loader.py` 和 `end2end.py`，支持从原始 HF checkpoint 直接产出 TP8 权重。
8. **P0**：更新并运行 `scripts/verify_qwen36_random_init_forward.py` 与 `verify_qwen36_real_weights_forward.py`。
9. **P1**：与 HF `AutoModelForCausalLM` 对比 logits。
10. **P2**：构建 `libtilert_qwen36.so` CUDA kernels。
11. **P3**：长序列 prefill、KV cache 复用、top-p/top-k 采样完整实现。

## 10. 附：vLLM 基准测试洞察（切换 TP8 的依据）

来源：`docs/perf_summary.md`（本地实测，input_len=550，output_len=256，num_prompts=50）。

## 11.其他参考文件

模型配置文件：

/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master/config.json

/public/home/dinggy/yiny/modelscope/models/Qwen--Qwen3.6-35B-A3B/snapshots/master/model.safetensors.index.json

Huggingface对此模型的官方算子：

/public/home/dinggy/yiny/projects/TileRT/docs/modeling_qwen3_5_moe.py

### 关键结论

- **纯 EP 综合表现劣于纯 TP**。同 TP/DP 配置下，带 EP 的输出吞吐、TTFT、P99 抖动普遍更差。
- **首次执行惩罚极重**：带 EP 配置的首次 TTFT 可达 100–240 秒，而无 EP 配置仅 15–27 秒，差距约 10 倍。
- **最佳吞吐配置**：`tp2-dp4`（无 EP），稳定输出吞吐 **989 tok/s**。
- **最佳延迟配置**：`tp8-again`（无 EP），稳定 Mean TPOT **38.25 ms**。

### 为什么先选纯 TP8

TileRT 当前 EP8 方案等价于 **TP1-EP8**（attention/DeltaNet/O-proj 全部复制，只有 expert 维度分片），所有跨设备通信都集中在 MoE 前后两次 all-to-all 上。这比 vLLM 的 `tp2-dp4-ep` 等混合并行方案对 all-to-all 的依赖更重。纯 TP8 的好处：

1. 没有 all-to-all，通信只有 attention/MoE output 的 allreduce，路径确定、好优化。
2. 首次启动没有 EP 的 communicator/weight dispatch 预热惩罚。
3. 与现有 `UnProjOAllReduce`、`PaddedAllReduceAdd` 等基础设施直接复用。

### 如果未来重新评估 EP

EP 并非完全不可用，但需要满足：
- all-to-all 有充分 warmup；
- buffer 按 `max_batch * topk * hidden` 静态预分配；
- 上线指标拆成 `first TTFT`、`warmup TTFT`、`stable TTFT` 分别报告；
- 优先尝试 **TP+EP 混合**（如 TP2-EP4、TP4-EP2），不要锁死纯 EP。

