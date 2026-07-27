# Qwen3.6 35B A3B 模型 8 GPU 并行设计（文档中关于决定使用EP切分的部分已作废，现在决定FFN部分使用TP切分，其余为DP）

## 1. 模型规格

> 来源：`qwen35b_config.json`（官方配置）

| 参数 | 数值 |
|------|------|
| 模型类型 | 语言模型 |
| 总参数量 | 35 B |
| 激活参数 | 3 B |
| 隐藏层维度 (`hidden_size`) | 2048 |
| 层数 (`num_hidden_layers`) | 40 |
| 层类型分布 | 30 层 `linear_attention` + 10 层 `full_attention` |
| Token 嵌入 / 输出头维度 (`vocab_size`) | 248320（已填充） |
| `tie_word_embeddings` | false |
| 训练 | 采用多步训练 (MTP)，`mtp_num_hidden_layers = 1` |

### 1.1 门控 DeltaNet (`linear_attention`)

| 参数 | 数值 |
|------|------|
| QK 头数量 (`linear_num_key_heads`) | 16 |
| QK 头维度 (`linear_key_head_dim`) | 128 |
| V 头数量 (`linear_num_value_heads`) | 32 |
| V 头维度 (`linear_value_head_dim`) | 128 |
| 卷积核维度 (`linear_conv_kernel_dim`) | 4 |
| 输出门控 (`attn_output_gate`) | true |

### 1.2 门控注意力 (`full_attention`)

| 参数 | 数值 |
|------|------|
| Q 头数量 (`num_attention_heads`) | 16 |
| KV 头数量 (`num_key_value_heads`) | 2 |
| 头维度 (`head_dim`) | 256 |
| 旋转位置嵌入比例 (`partial_rotary_factor`) | 0.25 |
| RoPE base (`rope_theta`) | 10M |
| 输出门控 (`attn_output_gate`) | true |

### 1.3 混合专家 (MoE)

| 参数 | 数值 |
|------|------|
| 专家总数 (`num_experts`) | 256 |
| 每 token 路由专家数 (`num_experts_per_tok`) | 8 |
| 路由专家中间层维度 (`moe_intermediate_size`) | 512 |
| 共享专家中间层维度 (`shared_expert_intermediate_size`) | 512 |
| 实际激活专家数 | 8 路由专家 + 1 共享专家 |

---

## 2. 关键特征分析

### 2.1 层结构是交替而非 macro block

从 `layer_types` 可见，40 层按 `linear_attention × 3 → full_attention × 1` 循环 10 次，不是之前理解的 “10 × (3 DeltaNet-MoE + 1 Attention-MoE)” 嵌套 macro block。每层（无论 linear 还是 full）后都接 MoE FFN。

### 2.2 稀疏度极高

- 总参数 35B，激活参数仅 3B，激活比例约 **8.6%**。
- 每层仅激活 9/256 ≈ **3.5%** 的专家。
- 计算稀疏，通信（MoE all-to-all）容易成为瓶颈。

### 2.3 参数分布极不均衡

按官方配置精确估算：

| 组件 | 参数量 | 占比 |
|------|--------|------|
| Embedding | 0.509 B | 1.5% |
| LM Head | 0.509 B | 1.5% |
| Linear Attention (DeltaNet，30 层) | ~0.87 B | 2.5% |
| Full Attention (GQA，10 层) | ~0.23 B | 0.7% |
| **Attention 合计** | **~1.10 B** | **~3.1%** |
| MoE FFN (256 路由 + 1 共享/层，40 层) | **~32.4 B** | **~92.6%** |
| RMSNorm / gate / bias 等 | ~0.3 B | ~0.9% |
| **总计** | **~35 B** | **100%** |

> **结论**：MoE FFN 是绝对主体，Attention 仅占约 3%。

### 2.4 隐藏层维度小

- `hidden_size = 2048` 属于较小维度。
- 张量并行 (TP) 的 allreduce 开销在小 tensor 上占比高，收益有限。

### 2.5 GQA KV 头数极少

- KV 头仅 2 个，TP4 切分需要 padding，增加 kernel 复杂度。
- TP2 可以整除（每卡 1 个 KV 头）。

### 2.6 DeltaNet 头数可整除

- QK 头 16、V 头 32，均可被 TP2/TP4/TP8 整除。
- 但 DeltaNet 是线性注意力/RNN-like 结构，可能存在跨 step 的隐状态传递。

### 2.7 专家数大

- 256 专家，EP8 时每卡只需承载 32 个专家。
- 专家中间维 512，单独计算量不大，但调用频繁。

---

## 3. 候选并行方案对比

| 方案 | EP 维度 | TP 维度 | 优势 | 劣势 | 适用场景 |
|------|--------|--------|------|------|---------|
| **EP8（推荐）** | 8 | 无 | 实现最简单，无 TP allreduce，all-to-all 只在 MoE 层，GQA 无需 padding | 8 rank all-to-all 延迟相对较高 | 默认首选 |
| EP4 + TP2 | 4 | 2 | 降低 all-to-all rank 数，TP2 可整除 GQA | 增加 TP allreduce，实现更复杂 | attention 实测成瓶颈 |
| EP2 + TP4 | 2 | 4 | all-to-all rank 最少 | KV 头 2 无法被 4 整除，需 padding，不推荐 | — |
| 纯 TP8 | 无 | 8 | — | MoE 无法利用 EP 节省显存和计算，所有卡参与所有专家 | 不推荐 |
| TileRT 式 1+7 异构 | 7（MoE） | 7（Attention） | — | 无 MLA/dense-MLP 的特殊分工，GPU 0 会闲置或成瓶颈 | 不推荐 |

---

## 4. 推荐方案：EP8（无张量并行）

### 4.1 分组方式

```text
GPU 0 ~ GPU 7：每张卡是一个独立的 EP rank
256 专家 ÷ 8 = 32 专家/卡
共享专家复制到每张卡
```

### 4.2 为什么推荐 EP8 无 TP

1. **隐藏维太小，TP 不划算**
   - `2048 / 8 = 256`，虽然能切，但 TP allreduce 的 latency 在小 tensor 上占比高。
   - Attention 和 DeltaNet 在每卡上完整计算，避免了 TP allreduce。

2. **内存充足，不需要 TP 来省显存**
   - `35B / 8 ≈ 4.4B` 参数/卡，加上 KV cache 和 activation，远小于 MI300X 的 192GB。
   - 完全可以每张卡完整保存 attention/DeltaNet 权重。

3. **避免 GQA KV 头切分问题**
   - KV 头只有 2 个，TP2 还能切成每卡 1 个；TP4 需要 padding 到 4，增加 kernel 复杂度。

4. **all-to-all 只在 MoE 层发生**
   - 每个 token 只激活 9 个专家，通信量本身不大。
   - 8 个 rank 的 all-to-all 在 8-GPU baseboard 上通常有专门优化。

5. **Attention 占比仅 3%，为 TP 引入 allreduce 不划算**
   - 即使 attention 成为瓶颈，TP2 的收益也微乎其微。
   - 保持 attention 卡内独立计算更优。

6. **MoE 调用频繁**
   - 40 层每层后都接 MoE，all-to-all 是主要通信来源。
   - 减少额外同步（如 TP allreduce）比单纯降低 all-to-all rank 数更重要。

---

## 5. 备选方案：EP4 + TP2

如果实测 EP8 下 attention/DeltaNet 计算成为瓶颈（概率较小），可以采用：

```text
GPU 0,1  → TP group A
GPU 2,3  → TP group B
GPU 4,5  → TP group C
GPU 6,7  → TP group D

4 个 TP 组作为 4 个 EP rank
每组 2 张卡内部 TP
256 专家 ÷ 4 = 64 专家/EP rank
```

### 5.1 切分方式

| 模块 | TP2 切分 |
|------|----------|
| DeltaNet QK | 16 头 → 每卡 8 头 |
| DeltaNet V | 32 头 → 每卡 16 头 |
| GQA Q | 16 头 → 每卡 8 头 |
| GQA KV | 2 头 → 每卡 1 头 |
| MoE 中间维 512 | 512 → 每卡 256 |

### 5.2 适用条件

- Batch size 较大，attention/DeltaNet 的 GEMM 能吃饱。
- 希望进一步降低 EP all-to-all 的 rank 数。

### 5.3 为什么不推荐 EP4 + TP2 作为默认

- Attention 仅占总参数 **3.1%**，即使它成为瓶颈，TP2 带来的加速也很有限。
- TP2 引入的 allreduce 会叠加到原本就频繁的 MoE 通信上。
- 只有在 profiling 确认 attention/DeltaNet 占用 >30% 总时间时，才考虑启用 TP2。

---

## 6. 国产 AMD GPU 实现要点

### 6.1 EP all-to-all 使用 RCCL

```python
import torch.distributed as dist

# EP group包含全部8个rank
ep_group = dist.new_group(ranks=list(range(8)))

dist.all_to_all_single(output, input, group=ep_group)
```

可尝试调整 RCCL 算法选择：

```bash
export NCCL_ALGO=RING
# 或根据 AMD 文档选择 Infinity Fabric 优化算法
```

### 6.2 拓扑感知 rank 绑定

- 把 EP 相邻 rank 放在同一 XGMI switch / 同一 NUMA 域内。
- 减少跨 NUMA / 跨交换机的 all-to-all 跳数。

### 6.3 DeltaNet 状态连续性

- 线性注意力的隐状态需要跨 step 保留。
- EP 前后的 gather/scatter 不要破坏状态布局。
- 尽量让 DeltaNet 状态常驻本地显存，避免每 step 跨卡拷贝。

### 6.4 交替层结构下的 all-to-all 优化

- 40 层是 `linear_attention × 3 → full_attention × 1` 的交替，不是 macro block。
- 每层的 MoE all-to-all 独立发生，**无法像 TileRT 那样把多层融合成单一 CUDA Graph replay 超级内核**。
- 可考虑：
  - 把相邻 `linear_attention + MoE` 的两次 all-to-all 用 double buffering 重叠；
  - 在 `full_attention` 层前后做通信-计算 overlap，因为 full attention 计算量比 linear attention 大。
- DeltaNet 的 3 个连续 `linear_attention` 层之间没有 attention 类型的切换，适合保持统一的通信节奏。

### 6.5 共享专家处理

- 共享专家 always-on，已存在本地。
- 不需要 all-to-all，避免重复路由。

### 6.6 MTP 模块

- MTP 通常只在主输出之后做多 token 预测。
- 可以在 EP rank 内部独立完成，不跨 EP。

### 6.7 算子融合机会

| 可融合算子 | 说明 |
|-----------|------|
| `RMSNorm + DeltaNet gate/project` | 减少 kernel launch 和访存 |
| `RMSNorm + Attention gate/project` | 同上 |
| `MoE up/gate + SiLU + down` | 专家内全融合 |
| `down projection + all-to-all/reduce` | 与通信融合 |
| `lm_head + sampling` | decode 阶段减少数据回传 |

---

## 7. 与 TileRT 当前架构的差异

| 维度 | TileRT (DeepSeek-V3.2) | Qwen3.6 35B A3B |
|------|------------------------|-----------------|
| 注意力 | MLA (1+7 异构) | GQA + DeltaNet |
| MoE 占比 | 部分层 | 全部 40 层 |
| Dense MLP | 有 | 无 |
| 隐藏维 | 7168 / 6144 | 2048 |
| 专家数 | 256 | 256 |
| 推荐并行 | 1+7 异构 TP | EP8 无 TP |
| 通信核心 | NVLink direct store | RCCL / Infinity Fabric |

### 迁移启示

- Qwen3.6 35B A3B 不需要 TileRT 的 1+7 异构设计。
- 核心工作是从 NCCL/NVLink 的通信原语迁移到 RCCL/Infinity Fabric。
- MoE 的 all-to-all 和专家内 GEMM 的 swizzle/MMA 需要按 AMD Matrix Core 重写。

---

## 8. 总结

| 场景 | 推荐并行策略 |
|------|--------------|
| 默认 | **EP8，无 TP** |
| attention 实测成为瓶颈 | **EP4 + TP2** |
| 避免使用 | EP2 + TP4、纯 TP8、1+7 异构 |

**结论**：根据官方配置精确估算，Qwen3.6 35B A3B 的 **MoE FFN 占 92.6% 参数，Attention 仅占 3.1%**。对于这种小隐藏维、高稀疏度、GQA KV 头极少的模型，**EP8 无张量并行**是最简洁、通信路径最少、最容易在国产 AMD GPU 上实现的方案。
