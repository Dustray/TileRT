# Qwen3.6-35B-A3B 接入 TileRT 技术方案

## 1. 架构差异分析

| 特性 | DeepSeek-V3.2 | Qwen3.6-35B-A3B | 适配策略 |
|------|---------------|-----------------|----------|
| 注意力 | MLA (Multi-Head Latent) | **GQA** (Grouped Query) | 需新实现 GQA kernel |
| 层结构 | 61层同质 | **40层异质** (DeltaNet + Gated Attn) | 需要分层调度 |
| MoE inter_dim | 2048 | **512** | 可复用，调整参数 |
| 专家数 | 256 | 256 | 可复用 |
| 激活专家 | 8+1 | 8+1 | 可复用 |
| 多模态 | 否 | 是 | 首版跳过 |

## 2. Qwen3.6 隐藏层结构

```
10 × [3 × (DeltaNet → MoE) → 1 × (Gated Attention → MoE)]
```

- **DeltaNet 层**: 线性注意力 + MoE
- **Gated Attention 层**: GQA + MoE
- 循环: 10 次，每次 4 层 = 40 层

## 3. 关键参数 (ModelArgsQwen36)

```python
vocab_size: int = 248320
dim: int = 2048           # 隐藏层维度
inter_dim: int = 512      # MoE 中间层 (远小于 DeepSeek 的 2048)
n_layers: int = 40        # 总层数
n_heads: int = 16         # Query 头数
n_kv_heads: int = 2       # KV 头数 (GQA)
qk_head_dim: int = 256    # 头维度
rope_dim: int = 64        # RoPE 维度

# MoE 配置
n_routed_experts: int = 256
n_activated_experts: int = 8
n_shared_experts: int = 1

# 层结构
n_delta_layers: int = 30   # 3 × 10
n_gated_layers: int = 10   # 1 × 10

# 上下文
max_seq_len: int = 262144
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
| `tilert/models/qwen3_6/modules/dsa.py` | ⚠️ 框架 | 需初始化子模块 |
| `tilert/models/qwen3_6/modules/delta_net.py` | ⚠️ 框架 | 占位实现 |
| `tilert/models/qwen3_6/modules/gated_attention.py` | ⚠️ 框架 | 占位实现 |
| `tilert/models/qwen3_6/modules/moe.py` | ⚠️ 框架 | 占位实现 |
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

**影响**: 无法进行实际推理，需要创建 `QwenShowHandsDSALayer`

#### 问题 2: dsa.py 子模块未初始化

**DeepSeek 实现** (同构层):
```python
for layer_idx in range(model_args.n_layers):
    if layer_idx < model_args.n_dense_layers:
        block = MlpBlock(...)
    else:
        block = MoeBlock(...)
    self.register_op(block, ...)
```

**Qwen3.6 当前** (异构层):
```python
# TODO: Initialize DeltaNet and Gated Attention layers
self.layer_types = [0,0,0,1, 0,0,0,1, ...]  # 40层映射
```

**分析**: 符合当前开发阶段，实现 golden_forward 时需补充

### 6.4 后续工作优先级

| 优先级 | 任务 | 说明 |
|--------|------|------|
| P0 | 创建 QwenShowHandsDSALayer | 端到端解码层 |
| P1 | 实现 golden_forward | 各模块 PyTorch 参考实现 |
| P2 | 初始化 QwenDsa 子模块 | 40层循环初始化 |
| P3 | 实现 MTP 支持 | 投机解码 |
| P4 | 构建 CUDA kernels | libtilert_qwen36.so |

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