# Qwen3.6-35B-A3B 接入 TileRT 技术方案

## 0. 背景术语说明

### 0.1 常用缩写

| 缩写 | 全称 | 含义 |
|---|---|---|
| **MLA** | Multi-head Latent Attention | DeepSeek-V3.2/GLM-5 使用的注意力结构。把 Q/K/V 通过低秩 LoRA 压缩到 latent space，显著减少 KV cache。 |
| **MLP** | Multi-Layer Perceptron | 稠密前馈网络（Dense FFN）。在 DeepSeek-V3.2/GLM-5 中，前若干层是 dense MLP，后面才是 MoE。 |
| **MoE** | Mixture of Experts | 专家混合模型。每次只激活 top-k 个 expert，Qwen3.6 是每层都是 MoE，256 专家里激活 8+1。 |
| **GQA** | Grouped Query Attention | Qwen3.6 使用的注意力。Query 头数多（16），KV 头数少（2），通过分组减少 KV cache。 |
| **MTP** | Multi-Token Prediction | DeepSeek-V3.2 的额外模块，一次预测多个 token。Qwen3.6 没有 MTP。 |

### 0.2 DeepSeek-V3.2 / GLM-5 / Qwen3.6-35B-A3B 结构对比

| 特性 | DeepSeek-V3.2 | GLM-5 | Qwen3.6-35B-A3B |
|---|---|---|---|
| 总层数 | 61 | 78 | 40 |
| 层类型 | 前 3 层 dense MLP + 后 58 层 MoE | 前 3 层 dense MLP + 后 75 层 MoE | **每层都是 MoE**，无 dense MLP |
| 注意力 | **MLA** | **MLA**（从 DSv3.2 修改） | **GQA + DeltaNet**（异构） |
| 隐藏维度 | 7168 | 6144 | 2048 |
| MoE inter_dim | 2048 | 2048 | **512** |
| 专家数 | 256 | 256 | 256 |
| 激活专家 | 8 + 1 shared | 8 | 8 + 1 shared |
| MTP | **有**（layer 61） | 无 | 无 |
| 层模式 | 同构（除前 3 层 MLP） | 同构 | **异构**：10 × [3 linear_attention + 1 full_attention] |
| TileRT 移植难度 | 基准 | 小（改 dim/scale dtype） | 大（需新 op + 异构调度 + 真实权重格式与假设不符） |

#### DeepSeek-V3.2
- 核心特点：通过 MLA 把 KV cache 压缩到低维 latent，配合 MTP 做多 token 预测。
- TileRT 适配：需要完整的 `mla_v2` 模块以及大量 MLA 专用 op。

#### GLM-5
- 核心特点：可以看作 DeepSeek-V3.2 架构的"放大/变种"，维度改为 6144，无 MTP。
- TileRT 适配：多数 op 直接从 DSv3.2 复制后改 dim 和 scale dtype。

#### Qwen3.6-35B-A3B
- 核心特点：
  - 不用 MLA，用 **GQA** + 新型线性注意力 **DeltaNet**（checkpoint 中分别叫 `full_attention` 与 `linear_attention`）。
  - 层间异构：30 层 `linear_attention` + 10 层 `full_attention`，由 `text_config.layer_types` 显式指定。
  - 规模更小（2048 dim vs 7168/6144），但**每层都是 MoE**。
  - checkpoint 同时包含 `model.visual.*`（多模态视觉塔）和 `mtp.*`（1 层 MTP），文本权重位于 `model.language_model.*` 下。
- TileRT 适配：
  - 不需要 MLA 相关 op。
  - MoE op 需要支持 Qwen3.6 的**堆叠 expert 权重**格式（`experts.down_proj` / `experts.gate_up_proj`）。
  - 需要新增 `gqa_attention.py`、`delta_net.py` 等 wrapper 及对应 CUDA kernel。

## 1. 架构差异分析

| 特性 | DeepSeek-V3.2 | Qwen3.6-35B-A3B | 适配策略 |
|------|---------------|-----------------|----------|
| 注意力 | MLA (Multi-Head Latent) | **GQA** (`full_attention`) + **DeltaNet** (`linear_attention`) | 需新实现两种 kernel |
| 层结构 | 61层同质 | **40层异质** (30 linear + 10 full) | 需要分层调度 |
| MoE inter_dim | 2048 | **512** | 需适配堆叠 expert 权重格式 |
| 专家数 | 256 | 256 | 可复用路由逻辑 |
| 激活专家 | 8+1 | 8+1 | 可复用 |
| MTP | 有 | **1 层** (`mtp_num_hidden_layers=1`) | 首版可跳过 |
| 多模态 | 否 | 是 (`model.visual.*`) | 首版跳过 |

## 2. Qwen3.6 隐藏层结构

```
10 × [3 × (linear_attention → MoE) → 1 × (full_attention → MoE)]
```

- **linear_attention 层**: DeltaNet 线性注意力 + MoE（30 层）
- **full_attention 层**: GQA + MoE（10 层）
- 循环: 10 次，每次 4 层 = 40 层
- 每层的具体类型由 `text_config.layer_types` 数组给出，例如前 4 层为
  `["linear_attention", "linear_attention", "linear_attention", "full_attention", ...]`

## 3. 关键参数 (ModelArgsQwen36)

```python
vocab_size: int = 248320
dim: int = 2048              # 隐藏层维度
inter_dim: int = 512         # MoE / shared expert 中间层
n_layers: int = 40           # 总层数

# full_attention (GQA)
n_heads: int = 16             # Query 头数
n_kv_heads: int = 2           # KV 头数 (GQA)
qk_head_dim: int = 256        # Q/K 头维度 (self_attn q/k/v/o 的 out dim = 4096, 即 n_heads * head_dim)
rope_dim: int = 64            # partial_rotary_factor * head_dim

# linear_attention (DeltaNet)
linear_num_key_heads: int = 16
linear_num_value_heads: int = 32
linear_key_head_dim: int = 128
linear_value_head_dim: int = 128
linear_conv_kernel_dim: int = 4

# MoE 配置
n_routed_experts: int = 256
n_activated_experts: int = 8
n_shared_experts: int = 1

# 层结构 (由 text_config.layer_types 显式给出)
n_delta_layers: int = 30      # linear_attention 层数
n_gated_layers: int = 10      # full_attention 层数
n_mtp_layers: int = 1        # checkpoint 含 1 层 MTP

# 上下文
max_seq_len: int = 262144
rope_theta: float = 1e7
rms_norm_eps: float = 1e-6
```

## 4. 实现组件清单

### 4.1 Python 层

```
tilert/models/qwen3_6/
├── __init__.py
├── generator.py          # Qwen36Generator
├── model_args.py         # ModelArgsQwen36
├── temp_var_indices.py   # 临时变量索引
├── modules/
│   ├── __init__.py
│   ├── dsa.py            # DSA 层 (Qwen 适配版)
│   ├── gqa.py            # GQA 注意力
│   ├── delta_net.py      # DeltaNet 线性注意力
│   ├── moe.py            # MoE (可复用 DSv3.2)
│   ├── mlp.py            # MLP
│   ├── mtp.py            # MTP (可选)
│   └── end2end.py        # 端到端层
└── ops/                   # CUDA 算子 (需新开发)
    ├── gqa_attention.py
    ├── delta_net.py
    ├── ...
```

### 4.2 C++/CUDA 算子

需要实现的核心算子（基于现有算子修改）:

1. **gqa_attention**: GQA 注意力 kernel (基于 `flash_sparse_mla.py` 改造)
2. **delta_net**: DeltaNet 线性注意力 (新开发)
3. **gated_attention**: 门控注意力机制 (新开发)
4. **expert_sel_up_gate_silu**: 可复用，调整 inter_dim=512
5. **padded_allreduce_add**: 可复用
6. **topk**: 可复用

## 5. 复用可能性分析

### 5.1 可直接复用

- `expert_sel_up_gate_silu`: 只需调整 inter_dim
- `topk`, `sparse_index`: expert routing
- `padded_allreduce_add`: 多卡通信
- `down_allreduce`, `eh_proj_allreduce`: MoE 通信

### 5.2 需要适配

- `flash_sparse_mla.py` → `gqa_attention.py`: MLA → GQA
- `moe.py` 模块: inter_dim 从 2048 改为 512

### 5.3 需要全新开发

- `delta_net.py`: 线性注意力机制
- `gated_attention.py`: 门控注意力

## 6. 代码审查结果 (2025-07-15)

### 6.1 已完成的代码文件

| 文件 | 状态 | 说明 |
|------|------|------|
| `tilert/models/qwen3_6/__init__.py` | ✅ 完成 | 模块导出 |
| `tilert/models/qwen3_6/model_args.py` | ✅ 完成 | 架构参数正确 |
| `tilert/models/qwen3_6/generator.py` | ⚠️ 框架 | 需实现核心解码层 |
| `tilert/models/qwen3_6/modules/__init__.py` | ✅ 完成 | 所有子模块导出 |
| `tilert/models/qwen3_6/modules/dsa.py` | ✅ 实现 | 40 层异构栈 + golden/tilert forward |
| `tilert/models/qwen3_6/modules/delta_net.py` | ✅ 实现 | reference wrapper + MoE；tilert forward 占位 |
| `tilert/models/qwen3_6/modules/gated_attention.py` | ✅ 实现 | GQA reference + op wrapper；tilert forward 占位 |
| `tilert/models/qwen3_6/modules/moe.py` | ✅ 实现 | MoE block 组合 RMSNormExpertProj / ExpertSelectUpGateSiLU / ExpertDownAllReduce |
| `tilert/models/qwen3_6/modules/mlp.py` | ⚠️ 框架 | 占位实现 |
| `tilert/models/qwen3_6/modules/mtp.py` | ⚠️ 框架 | 占位实现 |
| `tilert/__init__.py` | ✅ 完成 | 后端注册 |
| `tilert/generate.py` | ✅ 完成 | 模型类型支持 |

### 6.2 与 DeepSeek 代码对比

#### ModelArgsQwen36 参数验证 ✅

| 参数 | Qwen3.6 值 | 正确性 |
|------|------------|--------|
| vocab_size | 248320 | ✅ |
| dim | 2048 | ✅ |
| inter_dim | 512 | ✅ (远小于 DeepSeek 的 18432) |
| n_layers | 40 | ✅ |
| n_heads | 16 | ✅ |
| n_kv_heads | 2 | ✅ (GQA) |
| max_seq_len | 262144 | ✅ |
| n_routed_experts | 256 | ✅ |
| n_activated_experts | 8 | ✅ |

#### 代码模式 ✅

严格遵循 DeepSeek 的 `golden_forward/tilert_forward` 模式:

```python
def forward(self, *args, **kwargs):
    if self.flag_enable_tilert:
        return self.tilert_forward(*args, **kwargs)
    return self.golden_forward(*args, **kwargs)
```

### 6.3 发现的问题

#### 问题 1: generator.py 缺少核心解码层

**DeepSeek 实现**:
```python
self.decode_layer = ShowHandsDSALayer(
    model_args=self.config,
    model_path=self.model_weights_dir,
    with_mtp=with_mtp,
    ...
)
```

**Qwen3.6 当前**:
```python
self.decode_layer = None  # placeholder
```

**影响**: 无法进行实际推理，需要创建 `QwenShowHandsDSALayer`（Python 层 op 完成后下一步）。

#### 问题 2: dsa.py 子模块未初始化 ✅ 已解决

**当前实现** (异构层):
```python
self.layer_types: list[int] = []
for _ in range(model_args.n_blocks):
    self.layer_types.extend([0, 0, 0, 1])

for layer_idx, layer_type in enumerate(self.layer_types):
    if layer_type == 0:
        block = DeltaNet(...)
    else:
        block = GatedAttention(...)
    self.register_op(block, prefix=f"layer_{layer_idx}_", suffix=f"_dev_{device_id}")
```

**分析**: 40 层异构栈已初始化，支持 golden/tilert 双路径 forward。

### 6.4 后续工作优先级

| 优先级 | 任务 | 说明 |
|--------|------|------|
| P0 | 创建 QwenShowHandsDSALayer | 端到端解码层 |
| P0 | 新增 op wrapper | `gqa_attention.py`、`delta_net.py` ✅ 已完成 |
| P1 | 补全 golden_forward | DeltaNet / Gated Attention 真实参考计算 ✅ 已完成 |
| P2 | 硬编码参数清理 | `expert_down_allreduce.py`、`rmsnorm_up_gate_silu.py` tile/scale 形状适配 2048-dim ✅ 已完成 |
| P3 | 实现 MTP 支持 | 投机解码 |
| P4 | 构建 CUDA kernels | `libtilert_qwen36.so`（`gqa_attention_op`、`delta_net_op`） |

## 7. 工作量估算

| 组件 | 工作量 | 备注 |
|------|--------|------|
| ModelArgs 配置 | 0.5 天 | 参数定义 |
| Generator 接口 | 1 天 | 复用 GLM5 模板 |
| DSA 模块 | 3 天 | 适配异质层结构 |
| GQA 算子 | 3 天 | 基于 MLA 改造 |
| DeltaNet 算子 | 5 天 | 全新开发 |
| MoE 适配 | 1 天 | 调整 inter_dim |
| 端到端集成 | 2 天 | 测试调通 |
| **总计** | **~15 天** | 纯语言版 |

## 7. 下一步行动

1. 确认是否需要 MTP (多 token 预测) 支持
2. 获取模型 config.json 确认具体参数
3. 开始实现基础代码结构

## 8. Python 层 op 适配进展 (2026-07-15)

### 8.1 `tilert/models/qwen3_6/ops/` 目录清理

- 从 `deepseek_v3_2/ops/` 复制全部 op wrapper 后，按 Qwen3.6 架构裁剪。
- 删除 13 个 MLA/Indexer 专用文件：
  - `flash_sparse_mla.py`
  - `layernorm_rope_rotate.py`
  - `projo_wkvb.py`
  - `projq_wqb.py`
  - `projx_wis.py`
  - `projx_wqaki.py`
  - `projx_wqkva.py`
  - `rmsnorm_kv.py`
  - `rmsnorm_projq_wqb.py`
  - `rmsnorm_projq_wqi.py`
  - `rmsnorm_projx_wqakis.py`
  - `rmsnorm_projx_wqkva.py`
  - `sparse_index.py`
- 修复 `__init__.py` 导出，剩余可复用 op：
  - `broadcast_selected_token_ids`, `receive_selected_token_ids`
  - `down_allreduce`
  - `eh_proj_allreduce`
  - `expert_down_allreduce`
  - `expert_sel_up_gate_silu`
  - `padded_allreduce_add`
  - `qkv_rope`
  - `rmsnorm_expert_proj`, `rmsnorm_head_proj`, `rmsnorm_quant`, `rmsnorm_up_gate_silu`
  - `rotate`
  - `topk`
  - `unproj_o_allreduce`

### 8.2 已完成的维度/算法调整

| 文件 | 调整内容 |
|------|----------|
| `rmsnorm_up_gate_silu.py` | 移除 BF16MMA 支持；scale buffer dtype 改为 float32；适配 `inter_dim=512`；`tilert_scales` 形状改为 `(n_experts, inter_dim_per_device/block_size, dim/block_size)` |
| `expert_down_allreduce.py` | 移除 BF16MMA 支持；Qwen3.6 分支仅处理 `dim_per_sm=16` 的 16 行 tile，GLM5 保留原 48+8 拆分；scale padding 改为按 `dim_per_sm * scale_cols` 计算 |
| `unproj_o_allreduce.py` | 仅保留 FP16MMA；`v_head_dim` → `qk_head_dim` |
| `qkv_rope.py` | `qk_rope_head_dim` → `rope_dim` |
| `rotate.py` | `qk_rope_head_dim` → `rope_dim`；`index_n_heads` → `n_heads`；`index_head_dim` → `qk_head_dim` |
| `__init__.py` | 移除已删除文件的 import 与 `__all__` 导出 |

### 8.3 `weight_converter.py` 适配

- 导入 `ModelArgsQwen36`。
- 增加 `self.is_qwen36` 标识，根据 Qwen3.6 结构计算层数：
  - `num_dense_layers = n_delta_layers` (30)
  - `num_moe_layers = n_gated_layers` (10)
  - `num_mtp_layers = 0`
- `transform_attention` 已按 Qwen3.6 调整前缀，但**真实 checkpoint 的 attention 权重名与最初假设不同**：
  - `full_attention` 层：`self_attn.q_proj.weight`、`k_proj.weight`、`v_proj.weight`、`o_proj.weight`，以及 **新增的 `q_norm.weight`、`k_norm.weight`**。
  - `linear_attention` 层：不使用 `self_attn.*`，而是 `linear_attn.in_proj_qkv.weight`、`in_proj_a.weight`、`in_proj_b.weight`、`in_proj_z.weight`、`conv1d.weight`、`dt_bias`、`A_log`、`norm.weight`、`out_proj.weight`。
  - 因此 weight converter 需要按 `layer_types[layer_id]` 分支读取，并返回不同的权重别名。
- `convert_a_layer`：Qwen3.6 每层都是 MoE，统一走 `transform_moe`。
- `transform_moe`：Qwen3.6 的 expert 权重在 checkpoint 中是**按 expert 堆叠的张量**：
  - `mlp.experts.down_proj`: `(256, 2048, 512)`
  - `mlp.experts.gate_up_proj`: `(256, 1024, 2048)`
  - `mlp.gate.weight`: `(256, 2048)`
  - `mlp.shared_expert.down_proj.weight`: `(2048, 512)`
  - `mlp.shared_expert.gate_proj.weight / up_proj.weight`: `(512, 2048)`
  - `mlp.shared_expert_gate.weight`: `(1, 2048)`
  - 需要先把堆叠张量拆成 per-expert 的 `gate_proj / up_proj / down_proj`，再喂给现有 `ExpertSelectUpGateSiLU` / `ExpertDownAllReduce` 进行 device sharding。
- `transform_mlp`：Qwen3.6 不使用，但若被调用则切换为 Qwen3.6 的 `RMSNormUpGateSiLU` / `DownAllReduce`。
- `transform_mtp`：当前返回空字典。但 checkpoint 实际包含 `mtp.*` 和 `mtp.layers.*`（1 层），`text_config.mtp_num_hidden_layers=1`。若后续需要 MTP 投机解码，需补充转换。
- `__process_head_weights`：Qwen3.6 使用 `qwen3_6/ops/rmsnorm_head_proj.py`。
- CLI 新增 `--model_type qwen3_6` 分支。

### 8.4 新增 op 包装（2026-07-16）

- `tilert/models/qwen3_6/ops/gqa_attention.py`：新增 `GQAAttention` / `GQAAttentionWeightsConverter`，
  定义 `gqa_attention_op` 的 Python 调用接口与 weight alias，golden forward 提供标准 GQA 参考实现。
- `tilert/models/qwen3_6/ops/delta_net.py`：新增 `DeltaNetOp` / `DeltaNetWeightsConverter`，
  定义 `delta_net_op` 的 Python 调用接口与 weight alias，golden forward 提供简单线性 attention 参考实现。
- `ops/__init__.py` 导出新增四个公开符号：`delta_net`、`DeltaNetOp`、`gqa_attention`、`GQAAttention`。

### 8.5 模块接入新 op

- `modules/gated_attention.py`：移除 `QKVRoPE` + `Rotate` 占位，改用 `GQAAttentionOp`；
  tilert forward 路径为 `GQAAttentionOp -> UnProjOAllReduce`。
- `modules/delta_net.py`：接入 `DeltaNetOp`；golden/tilert forward 均走 `DeltaNetOp -> RMSNormUpGateSiLU`。
- `modules/dsa.py`：统一 KV cache 共享给所有 GatedAttention 层，DeltaNet  recurrent state
  按层索引保存在 `caches["delta_state"]` 中。

### 8.6 当前限制

- `GQAAttentionOp.tilert_forward` 与 `DeltaNetOp.tilert_forward` 目前会调用尚未注册的
  `torch.ops.tilert.gqa_attention_op` / `delta_net_op`，需等 CUDA kernel 完成后才能真正跑通。
- 当前 conversion 在 `test_mode=True` 下只转换 3 个代表层（0, 30, 39）+ head/embedding；完整 40 层转换需关闭 test mode 并验证内存/磁盘。

### 8.7 验证状态

- 所有 `qwen3_6/ops/*.py` 通过 AST 语法检查。
- 所有 `qwen3_6/modules/*.py` 通过 AST 语法检查。
- `weight_converter.py` 通过 AST 语法检查。
- **2025-07-17**: `WeightConverter(..., test_mode=True).to_tilert_weights()` 在真实 Qwen3.6-35B-A3B checkpoint 上成功完成；CLI `--model_type qwen3_6` 同步可用，输出 9 个 safetensors shard（共 497 张量）。
  - `linear_attention` 层（0, 30）输出 `linear_attn.*` 权重。
  - `full_attention` 层（39）输出 `self_attn.*` 权重。
  - MoE 堆叠 expert 权重被正确拆分并生成 fake float32 `weight_scale_inv`。
  - 缺失的 `mlp.gate.e_score_correction_bias` 默认补 0；缺失的 FP8 scales 由 op 合成全 1。

### 8.8 待办事项

| 优先级 | 任务 | 说明 |
|--------|------|------|
| P0 | 实现 `modules/moe.py` | ✅ 已完成 |
| P0 | 实现 `modules/gated_attention.py` | ✅ 已完成 |
| P0 | 实现 `modules/delta_net.py` | ✅ 已完成 |
| P0 | 实现 `modules/dsa.py` | ✅ 已完成 |
| P1 | 新增 op wrapper | `gqa_attention.py`、`delta_net.py` ✅ 已完成 |
| P2 | 硬编码参数清理 | `rmsnorm_up_gate_silu.py`、`expert_down_allreduce.py` ✅ 已完成 |
| P2 | `transform_attention` 补全 | ✅ 已按 `linear/full_attention` 分支读取真实权重名 |
| P2 | `transform_moe` 补全 | ✅ 已适配堆叠 expert 权重格式 |
| P2 | `transform_mtp` 补全 | ⚠️ 仍为 stub；checkpoint 含 `mtp.*`，但首版可跳过 |
| P3 | 完整 40 层转换 | 关闭 test_mode 验证全量转换 |
| P4 | CUDA kernel 开发 | `gqa_attention_op`、`delta_net_op` |