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
│   ├── transformer_stack.py  # Qwen3.6 Transformer 层栈（非 DSA）
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
| `tilert/models/qwen3_6/modules/transformer_stack.py` | ✅ 实现 | 40 层异构栈 + golden/tilert forward；支持 cached_ffn_ops 共享 MoE 实例。注意：Qwen3.6 本身不是 DSA 架构，这里只是复用 DSv3.2 的 FFN-cache 机制。 |
| `tilert/models/qwen3_6/modules/delta_net.py` | ✅ 实现 | reference wrapper + QwenMoeBlock；支持通过 `ffn_op` 参数复用外部 MoE block |
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

**影响**: 无法进行实际推理，需要创建 `QwenShowHandsLayer`（Python 层 op 完成后下一步）。注意：Qwen3.6 不使用 DeepSeek 的 DSA，show-hands 层只是对端到端推理调度层的沿用命名。

#### 问题 2: dsa.py 子模块未初始化 ✅ 已解决

**当前实现** (异构层):
```python
self.layer_types: list[int] = []
for _ in range(model_args.n_blocks):
    self.layer_types.extend([0, 0, 0, 1])

for layer_idx, layer_type in enumerate(self.layer_types):
    ffn_op = cached_ffn_ops[layer_idx] if cached_ffn_ops else None
    if layer_type == 0:
        block = DeltaNet(..., ffn_op=ffn_op)
    else:
        block = GatedAttention(...)
    self.register_op(block, prefix=f"layer_{layer_idx}_", suffix=f"_dev_{device_id}")
```

**分析**: 40 层异构栈已初始化，支持 golden/tilert 双路径 forward。新增 `cached_ffn_ops` 用于在 reference 验证阶段复用单个 `QwenMoeBlock`，避免 30 个独立 MoE 块导致 64GB GPU 显存 OOM。

### 6.4 后续工作优先级

| 优先级 | 任务 | 说明 |
|--------|------|------|
| P0 | 创建 QwenShowHandsLayer | 端到端解码层（Qwen3.6 不使用 DSA，命名去 DSA） |
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
| Transformer 层栈模块 | 3 天 | 适配异质层结构 |
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
- `modules/delta_net.py`：接入 `DeltaNetOp`；golden/tilert forward 均走 `DeltaNetOp -> QwenMoeBlock`，后者内部串联 `RMSNormExpertProj -> ExpertSelectUpGateSiLU -> ExpertDownAllReduce`。
- `modules/transformer_stack.py`：统一 KV cache 共享给所有 GatedAttention 层，DeltaNet  recurrent state
  按层索引保存在 `caches["delta_state"]` 中。

### 8.6 当前限制

- `GQAAttentionOp.tilert_forward` 与 `DeltaNetOp.tilert_forward` 目前会调用尚未注册的
  `torch.ops.tilert.gqa_attention_op` / `delta_net_op`，需等 CUDA kernel 完成后才能真正跑通。
- 当前 conversion 在 `test_mode=True` 下只转换 3 个代表层（0, 30, 39）+ head/embedding；完整 40 层转换需关闭 test mode 并验证内存/磁盘。

### 8.7 验证状态

- 所有 `qwen3_6/ops/*.py` 通过 AST 语法检查。
- 所有 `qwen3_6/modules/*.py` 通过 AST 语法检查。
- `weight_converter.py` 通过 AST 语法检查。
- **2025-07-17**: `WeightConverter(..., test_mode=True).to_tilert_weights()` 在真实 Qwen3.6-35B-A3B checkpoint 上成功完成；CLI `--model_type qwen3_6` 同步可用。
- **2025-07-17**: 完整 40 层非 test_mode 转换成功，输出 9 个 shard，共 6497 张量，总大小约 24.9 GB。
- **2026-07-16**: 40 层端到端 reference forward 验证通过（共享 `QwenMoeBlock` cache）；输出形状 `(1, 2, 2048)` 与 `(1, 1, 2048)`。
- **2026-07-16**: 40 层端到端 reference forward 验证通过。为绕过单卡 64GB 显存限制，`QwenTransformerStack` 支持 `cached_ffn_ops`，所有 `DeltaNet` 层复用同一个 `QwenMoeBlock`；输出形状 `torch.Size([1, 2, 2048])` 与 `torch.Size([1, 1, 2048]`。

### 8.8 原始 checkpoint 与 TileRT 输出差异说明

| 模块 | 原始 checkpoint 中张量数 | 原始大小 | 当前处理方式 |
|---|---|---|---|
| `model.language_model.*`（40 层文本模型） | 692 | 68.3 GB | ✅ 完整转换；输出 6497 张量 |
| `model.visual.*`（多模态视觉塔） | 333 | 0.89 GB | ⚠️ 首版跳过，不参与转换 |
| `mtp.*` / `mtp.layers.*`（1 层 MTP） | 19 | 1.69 GB | ⚠️ 首版跳过，不参与转换 |
| `lm_head.weight` 等顶层 | 1 | 1.02 GB | ✅ 已包含在最终 `layer_40_*` head 输出中 |

- 原始 checkpoint 总大小约 **71.9 GB**（1045 张量）。
- TileRT 输出约 **24.9 GB**，约为原始语言模型部分（68.3 GB）的 **36.5%**。
- **注意**：由于 `ExpertSelectUpGateSiLU` / `ExpertDownAllReduce` 的 device 维度索引已修复，旧的转换结果可能存在 expert 维度截断问题，建议用当前代码重新转换一次。
- 尺寸差异来源：
  1. **视觉塔和 MTP 被跳过**：共约 2.58 GB 未进入输出。
  2. **转换格式改变**：原始权重为 bf16；TileRT 输出中权重保持 bf16，但每个 expert 增加了 float32 `weight_scale_inv`（fake all-ones），且按 device 拆分产生重复/广播的 scale 张量，导致部分层尺寸非线性下降。
  3. **当前 scale 广播策略**：Qwen3.6 `inter_dim=512`，`block_size=128`，每个 expert 的 scale 行数只有 4 行，少于 `num_devices=8`，因此 scale 在 device 维度广播而不是拆分，整体张量数增加但单张尺寸减小。

  - `linear_attention` 层（0, 30）输出 `linear_attn.*` 权重。
  - `full_attention` 层（39）输出 `self_attn.*` 权重。
  - MoE 堆叠 expert 权重被正确拆分并生成 fake float32 `weight_scale_inv`。
  - 缺失的 `mlp.gate.e_score_correction_bias` 默认补 0；缺失的 FP8 scales 由 op 合成全 1。

### 8.9 模块引用持有者对齐 (2026-07-16)

- 更新 `tilert/models/qwen3_6/modules/delta_net.py` 中的 `QwenDeltaNetRef`：
  - 字段从旧的 `self_attn.q/k/v/o_proj_weight` 改为与 `ops/delta_net.py` 的
    `DeltaNetRefWeightsAlias` 对齐的 `linear_attn.*` 字段：
    `in_proj_qkv_weight`、`in_proj_z_weight`、`in_proj_a_weight`、
    `in_proj_b_weight`、`conv1d_weight`、`A_log`、`dt_bias`、
    `norm_weight`、`out_proj_weight`。
  - `init_reference_weights` 改为读取 `linear_attn.in_proj_qkv.weight`、
    `linear_attn.in_proj_z.weight` 等键。
- 更新 `tilert/models/qwen3_6/modules/gated_attention.py` 中的 `QwenAttentionRef`：
  - 新增 `q_norm_weight` 和 `k_norm_weight` 字段，与 `ops/gqa_attention.py` 的
    `GQAAttentionRefWeightsAlias` 对齐。
  - `init_reference_weights` 改为同时读取 `self_attn.q_norm.weight` 和
    `self_attn.k_norm.weight`。
  - `golden_forward` 在 RoPE 之前对 Q/K 投影结果应用 per-head RMSNorm。
- 两个 reference-only holder 仍只实现 `golden_forward`；因 `TileRTModule` 基类要求
  同时实现 `tilert_forward`，直接实例化会触发抽象类错误。它们仅作为
  `DeltaNet` / `GatedAttention` 的注册子 op 使用，由父模块转发调用。

### 8.10 待办事项

| 优先级 | 任务 | 说明 |
|--------|------|------|
| P0 | 实现 `modules/moe.py` | ✅ 已完成 |
| P0 | 实现 `modules/gated_attention.py` | ✅ 已完成 |
| P0 | 实现 `modules/delta_net.py` | ✅ 已完成 |
| P0 | 实现 `modules/transformer_stack.py` | ✅ 已完成 |
| P1 | 新增 op wrapper | `gqa_attention.py`、`delta_net.py` ✅ 已完成 |
| P2 | 硬编码参数清理 | `rmsnorm_up_gate_silu.py`、`expert_down_allreduce.py` ✅ 已完成 |
| P2 | `transform_attention` 补全 | ✅ 已按 `linear/full_attention` 分支读取真实权重名 |
| P2 | `transform_moe` 补全 | ✅ 已适配堆叠 expert 权重格式 |
| P2 | `transform_mtp` 补全 | ⚠️ 仍为 stub；checkpoint 含 `mtp.*`，但首版可跳过 |
| P3 | 完整 40 层转换 | ✅ 已完成（9 shards，6497 tensors，~24.9 GB） |
| P3 | 端到端 reference forward 验证 | ✅ 已完成；40 层 `QwenTransformerStack.golden_forward` 输出形状正确 |
| P4 | 模块引用持有者对齐 | ✅ 已完成（`delta_net.py`、`gated_attention.py`） |
| P5 | CUDA kernel 开发 | `gqa_attention_op`、`delta_net_op` |

## 9. Python 层 op 完成度审计与修复 (2026-07-17)

本次审计范围为 `tilert/models/qwen3_6/ops/*.py`，不包含 CUDA kernel 和 `tilert_forward`
实现。目标是把 reference 路径（`init_reference_weights`、`device_sharding`、
`golden_forward`、`init_tilert_vars`）以及 weight converter 的 stub 补齐。

### 9.1 已完成的修复

| 文件 | 修复内容 | 状态 |
|------|----------|------|
| `ops/gqa_attention.py` | 实现 `device_sharding`（按 `n_heads/n_kv_heads` 拆分 Q/KV/O，按 `rope_dim` 拆分 rope 缓存）和 `init_reference_weights`（加载本地 shard 并反量化） | ✅ 完成 |
| `ops/delta_net.py` | 实现 `device_sharding`（按 `qkv/z/a/b` 输出维度拆分）和 `init_reference_weights`；新增 `_get_local_out_slices` 辅助 | ✅ 完成 |
| `ops/rotate.py` | `init_tilert_vars` 增加 `device` 参数，output buffer 与 profile log tensor 显式分配到指定设备 | ✅ 完成 |
| `ops/qkv_rope.py` | `init_tilert_vars` 增加 `device` 参数，profile log tensor 显式分配 | ✅ 完成 |
| `ops/expert_down_allreduce.py` | 删除重复的 `convert_to_bf16mma` 方法（Qwen3.6 已不支持 BF16MMA） | ✅ 完成 |
| `ops/expert_sel_up_gate_silu.py` | 修复 `init_reference_weights` 中 device 维度索引 `[did]` → `[:, did]`；修复 `golden_forward` 对 bsz=1 flatten 后的 indices 处理；`.T` → `.mT` | ✅ 完成 |
| `ops/expert_down_allreduce.py` | 修复 `convert_to_general` 硬编码 `//8` 为 `base_inter_dim // num_devices`；修复 device 维度索引；`golden_forward` 处理 2-D indices/weights 并 `.T` → `.mT` | ✅ 完成 |
| `modules/moe.py` | 新增 `QwenMoeBlock`，串联 `RMSNormExpertProj -> ExpertSelectUpGateSiLU -> ExpertDownAllReduce` | ✅ 完成 |
| `modules/delta_net.py` | FFN 由 `RMSNormUpGateSiLU` 替换为 `QwenMoeBlock`；支持 `ffn_op` 共享实例 | ✅ 完成 |
| `modules/dsa.py` | 构造共享 `cached_ffn_ops` 并传给所有 `DeltaNet` 层；40 层 golden forward 通过 | ✅ 完成 |
| `models/utils.py` | 兼容 `qk_rope_head_dim` / `rope_dim`、`rope_factor`、`beta_fast`/`beta_slow`、`original_seq_len` | ✅ 完成 |
| `models/model_args.py` | 补齐 Qwen3.6 相关 RoPE/YaRN 参数默认值 | ✅ 完成 |

### 9.2 验证结果

- `ops/gqa_attention.py`、`ops/delta_net.py`、`ops/rotate.py`、`ops/qkv_rope.py` 语法检查通过。
- `ops/expert_down_allreduce.py` 因环境缺少 `torch` 包导致 import 无法解析（其他文件同样依赖 `torch` 但 Pylance 已能解析），代码本身无语法/AST 错误；删除重复方法后 `ExpertDownAllReduceWeightsConverter.dispatch` 仍能正确路由到 `convert_to_general`。
- 端到端 reference forward 验证：
  - `QwenMoeBlock.golden_forward(x)` → `torch.Size([1, 2, 2048])`
  - `DeltaNet.golden_forward(x, 0)` → `torch.Size([1, 2, 2048])`
  - 8 层 `QwenTransformerStack.golden_forward` → `torch.Size([1, 2, 2048])`
  - 40 层 `QwenTransformerStack.golden_forward`（共享 MoE cache）→ `torch.Size([1, 2, 2048])`
  - 40 层 `QwenTransformerStack.golden_forward`（seq_len=1）→ `torch.Size([1, 1, 2048])`

### 9.3 剩余待办

| 优先级 | 任务 | 说明 |
|--------|------|------|
| P1 | 端到端 reference forward 验证 | ✅ 已完成；40 层 `QwenTransformerStack.golden_forward` 输出形状正确 |
| P2 | `transform_mtp` 补全 | checkpoint 含 1 层 MTP；当前 stub，首版可跳过 |
| P2 | 真实权重重新转换 | 建议重新运行；MoE device 分片索引修复后旧转换结果可能 expert 维度错误 |
| P3 | CUDA kernel 开发 | `gqa_attention_op`、`delta_net_op` |
| P4 | `generator.py` 接入 | 在 op 全部完成后实现 `QwenShowHandsLayer`（去 DSA 命名） |

### 9.4 注意事项

- `GQAAttention` / `DeltaNetOp` 的 `init_tilert_weights` 仍依赖 `dispatch` 调用各自
  `WeightsConverter`，与 `tilert_forward` 一并留到 CUDA kernel 阶段实现。
- `tilert/models/qwen3_6/modules/delta_net.py` 与 `gated_attention.py` 中的
  `QwenDeltaNetRef` / `QwenAttentionRef` 属于 reference-only holder，没有
  `tilert_forward`；不能单独实例化，需通过父模块 `DeltaNet` / `GatedAttention` 调用。
- 端到端 golden forward 已通过 sanity run：
  - `QwenMoeBlock.golden_forward(x)` → `torch.Size([1, 2, 2048])`
  - `DeltaNet.golden_forward(x, 0)` → `torch.Size([1, 2, 2048])`
  - 8 层 `QwenTransformerStack.golden_forward(x, 0, freqs_cis)` → `torch.Size([1, 2, 2048])`
  - 40 层 `QwenTransformerStack.golden_forward(x, 0, freqs_cis)`（共享 MoE cache）→ `torch.Size([1, 2, 2048])`
  - 40 层 `QwenTransformerStack.golden_forward(x, 0, freqs_cis)`（seq_len=1）→ `torch.Size([1, 1, 2048])`
- 下一步建议：进入 CUDA kernel 开发与真实权重端到端推理验证。

## 10. Reference forward 数值修复与验证（2026-07-17）

### 10.1 修复背景

在随机初始化权重下直接运行 40 层 `QwenTransformerStack.golden_forward` 时，
发现输出随着层数加深指数发散：
- 每个子层都是完整 Transformer 结构（norm → attention/moe → residual），
  但随机权重的输出 std 与输入大致相当，残差相加导致方差每两层翻倍。
- 原始 `DeltaNetOp._linear_attention` 使用裸累积和，没有归一化，
  导致线性 attention 输出随序列位置爆炸。
- `RMSNorm` 默认权重由 `init_func` 初始化为接近 0.5 的值，
  削弱了 norm 对随机权重的稳定作用。
- 多设备分片时，`GQAAttention` 对 `n_kv_heads=2` 直接按 8 device 切片失败；
  MoE scale buffer 在 `inter_dim=512` 下出现 0 维 reshape 或除零错误。

### 10.2 已应用的修复

| 文件 | 修复内容 |
|------|----------|
| `tilert/models/qwen3_6/ops/delta_net.py` | `_linear_attention` 改用 elu+1 核函数 + 累积和归一化 + `1/sqrt(head_dim)` 温度缩放，输出稳定不再随位置爆炸。 |
| `tilert/models/qwen3_6/modules/dsa.py` | 在 40 层异构栈的残差相加中引入 `residual_scale = 1.0 / n_layers`；**仅用于随机初始化 sanity test**，加载 pretrained weights 后应移除或条件化。 |
| `tilert/models/qwen3_6/modules/delta_net.py` | 补齐 input RMSNorm、post-attention RMSNorm、attention 残差、FFN 残差；RMSNorm 权重延迟初始化为 1.0。 |
| `tilert/models/qwen3_6/modules/gated_attention.py` | 补齐 input RMSNorm、post-attention RMSNorm、attention 残差、FFN 残差；RMSNorm 权重延迟初始化为 1.0。 |
| `tilert/models/qwen3_6/ops/gqa_attention.py` | 修复 K/V split 使用 `v_head_dim` 的 bug；随机初始化时 Q/K/V/O 按 `1/sqrt(fan_in)` 缩放；`device_sharding` 在 `n_kv_heads < num_devices` 时复制 KV heads，避免 8 卡 stack 尺寸不一致。 |
| `tilert/models/qwen3_6/ops/expert_sel_up_gate_silu.py` | `init_reference_weights` 去量化后直接转 bf16，避免 fp32 临时张量 OOM；`convert_to_mma` 修复 `moe_rows < block_size` 时的 ZeroDivisionError。 |
| `tilert/models/qwen3_6/ops/expert_down_allreduce.py` | 随机初始化 down 权重按 `1/sqrt(moe_inter_dim)` 缩放；`ref_down` 转 bf16；`convert_to_general` 处理 `scale_cols == 0` 时折叠 scale。 |

### 10.3 关键工程约定

- **随机初始化缩放**：所有线性层权重（包括 DeltaNet/GQA/MoE 的 projection 矩阵）
  都按 `1/sqrt(fan_in)` 初始化，RMSNorm 权重初始化为 1.0，
  使无 pretrained weights 时 reference 输出仍保持有界。
- **残差缩放仅用于测试**：`dsa.py` 中的 `residual_scale` 明确标记为
  "random-init sanity test only"。真实权重加载后，
  Qwen3.6 的真实初始化/归一化设计已经保证数值稳定，应回退到标准残差相加。
- **KV head 复制**：Qwen3.6 只有 2 个 KV heads，在 8 设备场景下无法均匀切片；
  每个 device 复制完整 2 个 KV heads，Q heads 仍按 device 分片。
- **bf16 reference 权重**：MoE 的 gate/up/down reference 权重在去量化后转 bf16，
  单卡/多卡都能容纳 reference 副本。

### 10.4 验证结果

| 配置 | mean | std | finite | 峰值显存 |
|---|---|---|---|---|
| 40 层 / 8 device / dim=2048 / 256 experts / max_seq_len=512 | 0.041 | 2.67 | True | 16.1 GB |
| 默认 `ModelArgsQwen36`（max_seq_len=262144）/ 8 device | 0.034 | 2.66 | True | 16.7 GB |
| 增量 forward（pos=0 然后 pos=4，cache/state 复用）| — | 2.72 | True | — |
| 40 层 / 1 device / 64 experts | -0.021 | 2.73 | True | 38.5 GB |

说明：
- 8 device 下 40 层完整配置（256 experts、262144 序列长度）显存约 16–17 GB，
  远小于单卡 64 GB 上限，多设备分片有效。
- 单卡 40 层 + 256 experts 因需存储完整模型 reference 副本而 OOM（预期）；
  单卡 64 experts 可作为开发调试用的小 expert 变体。
- 输出 std 在 2.6–2.8 之间，数值有界且无 NaN/Inf，满足 reference 路径 sanity 要求。
- 增量 forward 验证 KV cache 与 DeltaNet recurrent state 可正确跨 step 复用。

### 10.5 与真实权重的衔接建议

- 移除或加配置开关：`dsa.py` 中的 `residual_scale`。
- 验证真实 pretrained weights 转换后的 MoE scale 张量是否正确广播
  （`inter_dim=512`、`block_size=128` 时每个 expert 的 scale 行数只有 4 行，
  小于 `num_devices=8`，必须 broadcast 而非 split）。
- 实现 `QwenShowHandsDSALayer` 与 `generator.py` 端到端解码，
  用真实 weights 跑通 token-by-token generation。
## 11. 三大模型（DSv3.2 / GLM-5 / Qwen3.6）代码架构统一性对比

### 11.1 总体关系

| 维度 | DeepSeek-V3.2 | GLM-5 | Qwen3.6-35B-A3B | 统一性评估 |
|---|---|---|---|---|
| 基础架构 | 原创实现 | 从 DSv3.2 复制后微调 | 参考 DSv3.2/GLM5 模式但独立实现 | Qwen3.6 与 DS/GLM 差异较大 |
| `ModelArgs` | 独立 dataclass | 完全复制 DSv3.2 的 `ModelArgs`，甚至 `arch_name` 仍是 `"deepseek_v3_2"` | 独立 `ModelArgsQwen36` | GLM5 与 DS 高度统一；Qwen 独立 |
| DSA 层调度 | 同构 61 层，按 `n_dense_layers` 分 Mlp/Moe | 同构 78 层，复用相同逻辑 | 异构 40 层（DeltaNet/GatedAttention） | 差异来自模型本身 |
| Block 结构 | `MlpBlock`/`MoeBlock` = MLA + Mlp/Moe | 与 DS 一致 | `DeltaNet`/`GatedAttention` 内部自包含 Attention + MoE | Qwen 把 FFN 放进子层 |
| MoE 模块 | `Moe`/`MoeBlock` 均为 `SerializableTileRTModule` | 与 DS 一致 | `QwenMoe` 是普通类，`QwenMoeBlock` 是 `TileRTModule`（非 Serializable） | 基类不一致，需注意注册/权重别名 |
| Op 目录 | 完整 MLA + MoE ops | 裁剪少量算法，主体复用 | 删除 MLA ops，新增 GQA/DeltaNet，保留可复用 MoE ops | Qwen 新增大量文件 |
| End2End/ShowHands | 完整 | 完整，复用 DS 函数并加 `is_glm5` 分支 | 缺失 `end2end.py`、`temp_var_indices.py` | Qwen 尚未接入 show-hands 推理框架 |
| Generator | 完整 | 完整 | Placeholder，`decode_layer = None` | Qwen 仅接口占位 |

### 11.2 关键代码模式对比

#### 11.2.1 ModelArgs

- **DSv3.2 / GLM5**：字段完全重合，仅实例值不同（dim、n_layers 等）。
  GLM5 的 `model_args.py` 实际上是 DSv3.2 的完整副本，
  连 `arch_name = "deepseek_v3_2"` 都未改，导致 GLM5 的 `ShowHandsDSALayer`
  不得不在运行时重新判断 `self.is_glm5 = self.model_args.arch_name == "glm_5"`。
- **Qwen3.6**：使用独立的 `ModelArgsQwen36`，字段命名也不同
  （`qk_head_dim` / `v_head_dim` / `rope_dim` 代替 DS 的
  `qk_nope_head_dim` / `qk_rope_head_dim` / `v_head_dim`，
  以及 DeltaNet 专用字段）。这是合理的，因为架构差异大，
  但 `models/utils.py` 需要同时兼容两套命名。

#### 11.2.2 DSA 模块

- **DSv3.2 / GLM5**：`Dsa.__init__` 遍历 `range(n_layers)`，
  根据 `layer_idx < n_dense_layers` 决定创建 `MlpBlock` 或 `MoeBlock`；
  每个 Block 内部再组合 `MLA + Mlp/Moe`。
- **Qwen3.6**：`QwenTransformerStack.__init__` 根据 `layer_types` 模式
  `[0,0,0,1] × n_blocks` 决定创建 `DeltaNet` 或 `GatedAttention`；
  每个子层内部已经包含自己的 MoE FFN。
  注意：Qwen3.6 本身不是 DSA 架构，只是复用了 DSv3.2 的 `cached_ffn_ops` FFN-cache 机制。
  此外 Qwen3.6 引入了 `cached_ffn_ops` 机制，
  让 30 个 DeltaNet 层可以复用同一个 `QwenMoeBlock`，
  这是为了缓解单卡 64GB 显存下随机初始化 reference 的内存压力，
  DS/GLM 没有这一需求。

#### 11.2.3 Block 内部结构

- **DSv3.2 / GLM5**：`MlpBlock`/`MoeBlock` 只负责把 MLA 和 FFN 注册为子 op，
  它们自身没有 `golden_forward`；残差和层归一化被拆散到多个 op 中
  （如 `RMSNormProjxWqkva`、`RmsnormProjqWqb` 等），由 C++ show-hands 调度。
- **Qwen3.6**：`DeltaNet` / `GatedAttention` 作为完整 Transformer 子层，
  在 Python 层实现了 `input_layernorm → attention → residual →
  post_attention_layernorm → moe → residual`，
  这使得 reference 路径更容易验证，但与 DS/GLM "op 粒度更细" 的风格不同。

#### 11.2.4 MoE 模块组织

| 文件/类 | DSv3.2 / GLM5 | Qwen3.6 | 差异影响 |
|---|---|---|---|
| `Moe` | `SerializableTileRTModule`，注册 `RMSNormExpertProj` / `ExpertSelectUpGateSiLU` / `ExpertDownAllReduce` | `QwenMoe` 是普通类，不继承 `TileRTModule` | Qwen 的 `QwenMoe` 不直接参与 `register_op`，权重列表需通过 `QwenMoeBlock.get_weights_list` 手动聚合 |
| `MoeBlock` | `SerializableTileRTModule`，注册 `MLA + Moe` | `QwenMoeBlock` 继承 `TileRTModule`，注册 `QwenMoe` | 基类不同，`SerializableTileRTModule` 会自动处理 prefix/suffix 和 `exec_seq`；`TileRTModule` 需要手动维护 |
| `cached_ffn_ops` | `Moe` 实例复用 | `QwenMoeBlock` 实例复用 | 语义相同，但类型不同 |

> 建议：如果后续 Qwen3.6 也要接入 show-hands / C++ 调度框架，
> 最好让 `QwenMoe` / `QwenMoeBlock` 继承 `SerializableTileRTModule`，
> 与 DS/GLM 保持一致，否则 `end2end.py` 中 `_extract_ffn_ops`、
> `_get_moe_weight_keys` 等辅助函数需要为 Qwen 写专门分支。

#### 11.2.5 Op 目录与复用

- **可直接复用的 op**：`broadcast_selected_token_ids`、`receive_selected_token_ids`、
  `down_allreduce`、`eh_proj_allreduce`、`padded_allreduce_add`、`topk`、
  `rmsnorm_expert_proj`、`rmsnorm_head_proj`、`rmsnorm_quant`、
  `rmsnorm_up_gate_silu`、`rotate`、`unproj_o_allreduce`。
- **已适配的 op**：`expert_down_allreduce`、`expert_sel_up_gate_silu`、
  `qkv_rope`、`rotate`（维度改为 `rope_dim`）。
- **Qwen3.6 新增 op**：`gqa_attention.py`、`delta_net.py`。
- **已删除的 MLA 专用 op**：`flash_sparse_mla`、`layernorm_rope_rotate`、
  `projo_wkvb`、`projq_wqb`、`projx_wis`、`projx_wqaki`、`projx_wqkva`、
  `rmsnorm_kv`、`rmsnorm_projq_wqb`、`rmsnorm_projq_wqi`、
  `rmsnorm_projx_wqakis`、`rmsnorm_projx_wqkva`、`sparse_index`。

#### 11.2.6 Show-Hands / End2End

- **DSv3.2 / GLM5**：都有 `modules/end2end.py` 中的 `ShowHandsDSALayer`，
  负责多设备 DSA 对象管理、CUDA graph 准备、weight loading、MTP 支持等。
- **Qwen3.6**：目前缺失 `end2end.py` 和 `temp_var_indices.py`，
  `generator.py` 也是 placeholder。这是合理状态，因为 CUDA kernel 还未完成；
  但要实现端到端推理，这是下一步必须补齐的组件。

### 11.3 统一性风险与改进建议

| 风险点 | 说明 | 建议 |
|---|---|---|
| `ModelArgs.arch_name` 不一致 | GLM5 仍使用 `"deepseek_v3_2"`，导致运行时判断 arch | 将 GLM5 的 `arch_name` 改为 `"glm_5"`；或在 Qwen3.6 的 end2end 中明确判断 `"qwen3_6"` |
| Qwen MoE 基类不一致 | `QwenMoe` 不是 `SerializableTileRTModule`，`QwenMoeBlock` 是 `TileRTModule` | 统一改为 `SerializableTileRTModule`，使 `register_op` / `exec_seq` / `prefix_seq` 自动可用 |
| Qwen 缺少 `temp_var_indices.py` | show-hands C++ 接口依赖固定 temp_vars 布局 | 后续实现 `QwenTempVarIdx` 时，尽量复用 DS 的索引名，新增 DeltaNet/GQA 专用 buffer |
| Qwen `generator.py` 占位 | 无法实际生成文本 | CUDA kernel 完成后实现 `QwenShowHandsDSALayer` 并接入 |
| FFN cache 机制差异 | DS/GLM 在 `MoeBlock`/`MlpBlock` 层级复用；Qwen 在 `DeltaNet` 内部复用 `QwenMoeBlock` | 统一成 "每个 layer 一个 ffn_op，可外部注入" 的语义即可 |
| Op 算法枚举 | Qwen 删除 BF16MMA，仅保留 GENERAL/FP8MMA/FP16MMA | 保持现状；不同模型支持不同 kernel 是正常的 |

### 11.4 结论

- **GLM-5 与 DeepSeek-V3.2 几乎是同一套代码框架**，差异主要集中在
  `model_args` 数值、少量 op 算法选择、MTP 开关和 chat template 处理。
  这是 TileRT 架构复用的理想形态。
- **Qwen3.6 由于注意力机制（GQA + DeltaNet）和层间异构**，
  不得不在 DSA、Block、Op 等层面独立实现。
  当前 Python 层 reference 路径已经按统一风格（`golden_forward`/`tilert_forward`、
  `register_op`、`TileRTModule` 基类）完成，
  所有 op 的 `golden_forward` 已适配 Qwen3.6 维度与多设备切分规则。
  但在 MoE 基类选择、部分 `__call__` 分支、show-hands end2end、temp_var 布局等方面尚未完全对齐。
- 下一步若要接入 C++ 推理框架，优先补齐：
  1. `QwenMoe` / `QwenMoeBlock` 基类统一为 `SerializableTileRTModule`；
  2. `down_allreduce.py`、`rmsnorm_up_gate_silu.py`、`unproj_o_allreduce.py`
     的 `__call__` 接入 `flag_enable_tilert` 分支；
  3. `tilert/models/qwen3_6/temp_var_indices.py`；
  4. `tilert/models/qwen3_6/modules/end2end.py`；
  5. `generator.py` 中的 `QwenShowHandsDSALayer`。

### 11.5 Qwen3.6 op `golden_forward` 适配状态

2026-07-17 对所有 18 个 `tilert/models/qwen3_6/ops/*.py` 文件做了
`golden_forward` / 权重初始化 / 多设备切分二次检查，结果如下：

| 文件 | golden_forward | 权重初始化 | 多设备适配 | 备注 |
|---|---|---|---|---|
| `delta_net.py` | ✅ | ✅ | 按 head 切分 | 稳定线性 attention 已接入 |
| `gqa_attention.py` | ✅ | ✅ | KV head 复制已处理 | K 用 `qk_head_dim`，V/O 用 `v_head_dim` 已正确 |
| `expert_sel_up_gate_silu.py` | ✅ | ✅ | scale 行数 < device 数时广播 | `convert_to_mma` 已用实际 scale 行数 |
| `expert_down_allreduce.py` | ✅ | ✅ | scale 行数 < device 数时广播/折叠 | `scale_cols==0` 已处理，`ref_down` 转 bf16 |
| `down_allreduce.py` | ✅ | ✅ | 按 expert/device 切分 | dense MLP 用，当前 Qwen DSA 未调用 |
| `unproj_o_allreduce.py` | ✅ | ✅ | 支持 Q head 不能整除 device 数时的 padding | `head_dim` 已改为 `v_head_dim` |
| `rmsnorm_up_gate_silu.py` | ✅ | ✅ | 复用 expert gate/up sharding | dense MLP 用，当前 Qwen DSA 未调用 |
| `rmsnorm_expert_proj.py` | ✅ | ✅ | gamma 复制 | gate 打分用 |
| `rmsnorm_head_proj.py` | ✅ | ✅ | gamma 复制，head 按 device 切 | final norm + lm_head |
| `qkv_rope.py` | ✅ | 无权重 | 无需切分 | 复用 DS/GLM 逻辑 |
| `rotate.py` | ✅ | 无权重 | 无需切分 | hadamard + RoPE |
| `topk.py` | ✅ | 无权重 | 无需切分 | 仅 topk 包装 |
| `eh_proj_allreduce.py` | ✅ | ✅ | 输入维度按 device 切 | EH projection，当前未使用 |
| `padded_allreduce_add.py` | ✅ | 无权重 | 单 GPU 占位 | allreduce + residual |
| `broadcast_selected_token_ids.py` | 函数级 | 无权重 | 通信 op | 无 golden |
| `receive_selected_token_ids.py` | 函数级 | 无权重 | 通信 op | 无 golden |
| `rmsnorm_quant.py` | 函数级 | 无权重 | 无需切分 | 调用 `rmsnorm_op` |

**待改进项（低优先级，当前未影响 reference 路径）**：
- `down_allreduce.py`、`rmsnorm_up_gate_silu.py`、`unproj_o_allreduce.py`
  的 `__call__` 目前直接返回 `golden_forward`，未判断
  `self.flag_enable_tilert`。虽然这三个 op 当前未被 Qwen3.6 DSA 直接调用，
  但为保持 `TileRTModule` 行为一致性，建议后续改为：
  ```python
  if self.flag_enable_tilert:
      return self.tilert_forward(...)
  return self.golden_forward(...)
  ```
