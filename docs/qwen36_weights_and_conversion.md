# Qwen3.6-35B-A3B 权重与模型结构说明

> 本文档独立于 `qwen36_integration_plan.md`，聚焦 Qwen3.6 在**权重布局**、**模型结构**以及与 DeepSeek-V3.2 的差异，并说明 TileRT 中 Qwen3.6 权重转换的实现方式。

---

## 1. 模型结构概览

### 1.1 Qwen3.6-35B-A3B 核心参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 总层数 | 40 | 10 个 block，每个 block 3 层 DeltaNet + 1 层 GQA |
| 隐藏维度 `dim` | 2048 | 远小于 DeepSeek 7168 / GLM-5 6144 |
| MoE 中间维 `inter_dim` | 512 | DeepSeek 2048 / GLM-5 2048 |
| 路由专家数 `n_routed_experts` | 256 | 与 DeepSeek 相同 |
| 共享专家数 `n_shared_experts` | 1 | 与 DeepSeek 相同 |
| 激活专家数 `n_activated_experts` | 8 + 1 shared | 与 DeepSeek 相同 |
| 注意力 | **GQA + DeltaNet** | DeepSeek 为 MLA |
| GQA 头数 | Q=16, KV=2 | `n_heads=16`, `n_kv_heads=2` |
| 头维度 | `qk_head_dim=256`, `v_head_dim=256` | Q/K/V/O 投影输出维度均为 4096 |
| RoPE | **M-RoPE** | `rope_dim=64`, `partial_rotary_factor=0.25`, `mrope_section=[11,11,10]` |
| 序列长度 | max 256K（可 YaRN 扩展到 1M） | `max_seq_len=262144`, `rope_theta=1e7` |
| MTP | 支持，但**默认禁用** | `text_config.mtp_num_hidden_layers=1`，首版 `num_mtp_layers=0` |
| 多模态 | 含视觉塔 `model.visual.*` | 首版跳过不转换 |

### 1.2 层类型模式

Qwen3.6 是**异构**结构，由 `text_config.layer_types` 显式指定：

```python
layer_types = ["linear_attention"] * 3 + ["full_attention"]  # 重复 10 次
```

即每 4 层为一个周期：

```
[DeltaNet, DeltaNet, DeltaNet, GQA] × 10 = 40 层
```

- `linear_attention`：DeltaNet 线性注意力层
- `full_attention`：GQA 全注意力层

---

## 2. 权重布局

### 2.1 顶层权重

| 含义 | Qwen3.6 权重名 | DeepSeek 权重名 |
|------|---------------|-----------------|
| Embedding | `model.language_model.embed_tokens.weight` | `model.embed_tokens.weight` |
| Final RMSNorm | `model.language_model.norm.weight` | `model.norm.weight` |
| LM Head | `lm_head.weight` | `lm_head.weight` |

### 2.2 DeltaNet（`linear_attention`）层权重

```
model.language_model.layers.{i}.
├── input_layernorm.weight
├── post_attention_layernorm.weight
└── linear_attn.
    ├── in_proj_qkv.weight      # Q/K/V 融合投影
    ├── in_proj_z.weight        # gate / value gate
    ├── in_proj_a.weight        # decay gate a
    ├── in_proj_b.weight        # decay gate b
    ├── conv1d.weight           # 1-D causal convolution
    ├── A_log                   # state matrix log
    ├── dt_bias                 # delta time bias
    ├── norm.weight             # output norm
    └── out_proj.weight         # output projection
```

### 2.3 GQA（`full_attention`）层权重

```
model.language_model.layers.{i}.
├── input_layernorm.weight
├── post_attention_layernorm.weight
└── self_attn.
    ├── q_proj.weight       # 输出翻倍：query + gate
    ├── k_proj.weight
    ├── v_proj.weight
    ├── o_proj.weight
    ├── q_norm.weight       # per-head Q RMSNorm
    └── k_norm.weight       # per-head K RMSNorm
```

> 注意：Qwen3.5/3.6 的 `q_proj` 输出维度为 `2 * n_heads * qk_head_dim`，后半部分作为 gate，需 chunk 后通过 `sigmoid(gate.mean(dim=-1))` 对 attention 输出做门控。

### 2.4 MoE 权重

Qwen3.6 **每层都是 MoE**，权重为**按 expert 堆叠的张量**：

```
model.language_model.layers.{i}.mlp.
├── gate.weight                         # (256, 2048) 路由打分
├── experts.gate_up_proj                # (256, 1024, 2048)
├── experts.down_proj                   # (256, 2048, 512)
├── shared_expert.gate_proj.weight      # (512, 2048)
├── shared_expert.up_proj.weight        # (512, 2048)
├── shared_expert.down_proj.weight      # (2048, 512)
└── shared_expert_gate.weight           # (1, 2048) 共享专家门控
```

| 张量 | 形状 | 说明 |
|------|------|------|
| `gate.weight` | `(n_routed_experts, dim)` | 每个专家一个路由向量 |
| `experts.gate_up_proj` | `(n_routed_experts, moe_inter_dim//2, dim)` | 每个专家 gate + up 合并 |
| `experts.down_proj` | `(n_routed_experts, dim, moe_inter_dim)` | 每个专家 down 投影 |
| `shared_expert.gate_proj/up_proj` | `(moe_inter_dim, dim)` | 共享专家 Gate/Up |
| `shared_expert.down_proj` | `(dim, moe_inter_dim)` | 共享专家 Down |
| `shared_expert_gate.weight` | `(1, dim)` | 共享专家混合门控 |

### 2.5 MTP 权重（支持但禁用）

```
mtp.
├── embed_tokens.weight
├── layers.0.
│   ├── ...（与语言模型层结构类似）
│   └── ...
└── ...
```

首版 `num_mtp_layers=0`，`transform_mtp` 返回空字典，不进入转换输出。

---

## 3. Qwen3.6 vs DeepSeek-V3.2 关键差异

| 维度 | DeepSeek-V3.2 | Qwen3.6-35B-A3B |
|------|---------------|-----------------|
| 总层数 | 61 | 40 |
| 层结构 | 前 3 层 Dense MLP + 后 58 层 MoE | 每层都是 MoE，无 Dense MLP |
| 注意力 | MLA（Multi-head Latent Attention） | GQA + DeltaNet 异构 |
| 隐藏维度 | 7168 | 2048 |
| MoE 中间维 | 2048 | 512 |
| RoPE | 标准 1D RoPE | M-RoPE（多模态 RoPE） |
| 权重前缀 | `model.layers.{i}.` | `model.language_model.layers.{i}.` |
| 注意力权重 | `self_attn.wq_a/b`, `wkv_a/b`, `wo` 等 MLA 低秩矩阵 | `self_attn.q/k/v/o_proj` + `q_norm/k_norm` |
| 线性注意力权重 | 无 | `linear_attn.in_proj_*`, `conv1d`, `A_log`, `dt_bias` |
| Expert 权重 | 通常已按 per-expert 拆分 | 堆叠张量 `experts.gate_up_proj` / `experts.down_proj` |
| 共享专家 | 有 | 有，且多一个 `shared_expert_gate` |
| MTP | 有（layer 61） | 支持，默认禁用 |
| 多模态 | 无 | 有视觉塔 `model.visual.*` |

---

## 4. TileRT 权重转换流程

转换入口：`tilert/models/preprocess/weight_converter.py`

### 4.1 输入

- Hugging Face 格式 checkpoint
- 源文件：`model.safetensors.index.json` + 多个 `model-*.safetensors`
- Qwen3.6 文本权重约 **68.3 GB**（692 个张量）

### 4.2 输出

- TileRT 多卡分片格式
- 文件：`model.safetensors-{i:05d}-of-{N:05d}.safetensors` + `model.safetensors.index.json`
- Qwen3.6 输出示例：**9 个 shard，6497 张量，约 24.9 GB**

### 4.3 转换步骤

#### 步骤 1：初始化 `WeightConverter`

```python
is_qwen36 = isinstance(model_args, ModelArgsQwen36)
if is_qwen36:
    layer_types = model_args.layer_types
    num_dense_layers = linear_attention 层数 (30)
    num_moe_layers   = full_attention 层数 (10)
    num_mtp_layers   = 0  # 默认禁用
    layer_prefix     = "model.language_model.layers"
    emb_name         = "model.language_model.embed_tokens.weight"
    norm_name        = "model.language_model.norm.weight"
else:
    # DeepSeek / GLM-5 路径
    num_dense_layers = model_args.n_dense_layers
    num_moe_layers   = n_layers - num_dense_layers
    num_mtp_layers = 1
    layer_prefix   = "model.layers"
```

#### 步骤 2：逐层转换 `convert_a_layer`

Qwen3.6 每层都调用 `transform_moe`（无 Dense MLP）。

#### 步骤 3：`transform_attention` — 按层类型分支

- `linear_attention`：读取 `linear_attn.*` 权重
- `full_attention`：读取 `self_attn.q/k/v/o_proj` + `q_norm/k_norm`

#### 步骤 4：`transform_moe` — 拆分堆叠 expert 权重

1. 从 checkpoint 读取：
   - `mlp.gate.weight`
   - `mlp.experts.gate_up_proj`
   - `mlp.experts.down_proj`
   - `mlp.shared_expert.*`
   - `mlp.shared_expert_gate.weight`

2. 将堆叠张量拆为 per-expert 的 `gate_proj / up_proj / down_proj`：

```python
# gate_up_proj: (256, 1024, 2048) -> gate (256, 512, 2048), up (256, 512, 2048)
gate_proj, up_proj = experts_gate_up_proj.chunk(2, dim=1)
```

3. 喂入 Qwen3.6 的 `ExpertSelectUpGateSiLUQwen36` 与 `ExpertDownAllReduceQwen36` 做 device sharding。

4. `mlp.gate.weight` 直接以 `(n_routed_experts, dim)` 形状传入路由 op，不再做额外 reshape。

5. 每个 device 输出包含：

```python
{
    "unproj_o_gamma": post_attn_norm_weight,
    "exp_proj_weights": mlp_gate_weight,      # 路由 gate
    "exp_bias": ...,
    "exp_gate_weights": ...,
    "exp_gate_scales": ...,
    "exp_up_weights": ...,
    "exp_up_scales": ...,
    "exp_down_weights": ...,
    "exp_down_scales": ...,
    "shared_expert_gate": shared_expert_gate,  # Qwen 特有
}
```

#### 步骤 5：`transform_mtp`（默认跳过）

```python
if is_qwen36:
    mtp_tensors = {}
    for key in weights_hf:
        if key.startswith("mtp."):
            mtp_tensors[key[4:]] = weights_hf[key]
    return {f"dev_{i}": mtp_tensors for i in range(num_devices)}
```

当前返回空字典；后续启用 MTP 时再扩展。

#### 步骤 6：处理 Head 与 Embedding

- `__process_head_weights`：使用 `qwen3_6/ops/rmsnorm_head_proj.py`，输出 `layer_40_lm_head.weight_dev_*` 与 `layer_40_model.norm.weight_dev_*`
- `__process_embedding_weights`：输出 `layer_0_embedding.weight_dev_*`

#### 步骤 7：分片保存

调用 `save_file_sharded`，按 `max_shard_size=5GB` 保存，生成新的 index 文件。

### 4.4 关键实现细节

| 问题 | 处理方式 |
|------|--------|
| `mlp.gate.weight` 形状 | 断言为 `(n_routed_experts, dim)`，直接传入，不做 reshape |
| 堆叠 expert 权重 | `chunk` 拆分为 per-expert `gate_proj/up_proj/down_proj` |
| FP8 scale 缺失 | bf16-only checkpoint 省略 FP8 scales，op 内部合成全 1 scale |
| 路由 bias 缺失 | `e_score_correction_bias` 缺失时 op 默认补 0 |
| 小 scale 广播 | `inter_dim=512`、`block_size=128` 时每个 expert 只有 4 行 scale，小于 `num_devices=8`，在 device 维度广播而非拆分 |
| 视觉塔 | `model.visual.*` 跳过，不参与转换 |
| MTP | `mtp.*` 跳过，不参与转换（默认禁用） |

### 4.5 输出规模与差异说明

| 模块 | 原始大小 | 转换后处理 |
|------|----------|------------|
| `model.language_model.*` | ~68.3 GB | 完整转换，输出 ~24.9 GB |
| `model.visual.*` | ~0.89 GB | 跳过 |
| `mtp.*` / `mtp.layers.*` | ~1.69 GB | 跳过 |
| `lm_head.weight` 等顶层 | ~1.02 GB | 已包含在 head 输出中 |

输出从 68.3 GB 降到 24.9 GB 的主要原因：
1. 视觉塔与 MTP 未进入输出（约 2.58 GB）
2. 按 device 拆分后 scale 张量大量广播
3. 每个 expert 增加 fake float32 `weight_scale_inv`

---

## 5. 与 DeepSeek 权重转换的主要区别

| 方面 | DeepSeek-V3.2 | Qwen3.6-35B-A3B |
|------|---------------|-----------------|
| 层前缀 | `model.layers.{i}` | `model.language_model.layers.{i}` |
| Attention 转换 | `SparseSelectMlaV2` + `PureMlaV2` | 按 `layer_types` 分支读取 `linear_attn.*` / `self_attn.*` |
| MoE 转换 | 直接 per-expert 权重 | 需先 `chunk` 拆分堆叠的 `experts.gate_up_proj/down_proj` |
| Shared Expert | 结构类似 | 多一个 `shared_expert_gate.weight` |
| MTP | 固定 1 层 | 支持但默认禁用（`num_mtp_layers=0`） |
| Head 处理 | `rmsnorm_head_proj.py`（DS 版本） | `qwen3_6/ops/rmsnorm_head_proj.py`（支持 `model.language_model.norm.weight` 回退） |
| 输出层索引 | layer 61 | layer 40 |

---

## 6. 参考文件

- `tilert/models/preprocess/weight_converter.py` — 权重转换入口
- `tilert/models/qwen3_6/model_args.py` — Qwen3.6 架构参数
- `tilert/models/qwen3_6/ops/expert_sel_up_gate_silu.py` — expert up/gate 分片
- `tilert/models/qwen3_6/ops/expert_down_allreduce.py` — expert down 分片
- `tilert/models/qwen3_6/ops/rmsnorm_head_proj.py` — head 权重处理
- `docs/qwen36_integration_plan.md` — 完整接入方案
