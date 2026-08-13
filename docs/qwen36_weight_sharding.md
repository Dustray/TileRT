# Qwen3.6 权重 Tensor Parallel / Expert Parallel 切分说明

> 分析范围：`tilert/models/qwen3_6/ops/*.py` 的 `device_sharding` / `process_*_weights` 及相关权重别名。
> 环境假设：8 卡（`num_devices=8`），EP8（Expert Parallel）+ TP8（Tensor Parallel on MoE intermediate dim）。

---

## 1. 总体切分策略

| 组件 | 切分维度 | 切分方式 | 说明 |
|:---|:---|:---|:---|
| **Attention / DeltaNet** | 不切分 | 每张卡复制完整权重 | EP8：只有 MoE 专家在专家维度切分，线性注意力权重全量复制 |
| **MoE routed expert gate/up/down** | `inter_dim` | TP8 | 每个专家的中间维度被切成 `inter_dim // num_devices` |
| **MoE shared expert gate/up/down** | `inter_dim` | TP8 | 共享专家同样按中间维度切分 |
| **MoE gate score projection** | 不切分 | 每张卡完整 `n_routed_experts × dim` | 路由矩阵小，全量复制 |
| **lm_head** | `vocab_size` | TP8 / vocab shard | 每张卡负责部分词表 logits |
| **final RMSNorm gamma** | 不切分 | 复制到每张卡 | `(num_devices, dim)` |

---

## 2. 各 OP 的权重切分细节

### 2.1 `GQAAttention` / `DeltaNetOp`（Attention & Linear Attention）

| OP | 相关权重 | 切分方式 | 形状变化（示例，num_devices=8） |
|:---|:---|:---|:---|
| `GQAAttention` | `q_proj.weight`, `k_proj.weight`, `v_proj.weight` → `qkv_proj_weights` | **不切分，复制** | `(num_devices, (2*n_heads + n_kv + n_kv_v)*head_dim, dim)` |
| | `o_proj.weight` | **不切分，复制** | `(num_devices, dim, n_heads*v_head_dim)` |
| | `q_norm.weight`, `k_norm.weight` | **不切分，复制** | `(num_devices, head_dim)` |
| `DeltaNetOp` | `in_proj_qkv/z/a/b.weight`, `conv1d.weight`, `A_log`, `dt_bias`, `norm.weight`, `out_proj.weight` | **不切分，复制** | 每个张量 stack 8 份 |

**关键代码特征**：
```python
# GQAAttention.device_sharding
torch.stack([qkv_proj_weights for _ in range(self.num_devices)], dim=0)
torch.stack([o_w for _ in range(self.num_devices)], dim=0)
```

> 结论：Attention 和 DeltaNet 在 Qwen3.6 当前实现中**没有 TP 切分**，每张卡都保存完整副本。

---

### 2.2 `RMSNormExpertProj`（MoE 路由前的 Norm + Gate Projection）

| 权重 | 来源 key | 切分方式 | 形状 |
|:---|:---|:---|:---|
| `post_attention_layernorm.weight` | `post_attention_layernorm.weight` | 不切分 | `(dim,)` → 直接返回 |
| `mlp.gate.weight` | `mlp.gate.weight` | 不切分 | `(n_routed_experts, dim)` → 直接返回 |

**说明**：路由矩阵很小（`n_routed_experts × dim`），因此不做 TP 切分，每张卡完整持有。

---

### 2.3 `ExpertSelectUpGateSiLU`（MoE up / gate 投影）

#### Routed experts（堆叠 `gate_up_proj`）

| 权重 | 原始形状 | TP8 切分后形状 | 切分维度 |
|:---|:---|:---|:---|
| `gate_proj_weight` | `(n_experts, 2*inter_dim, dim)` | `(n_experts, num_devices, inter_dim//num_devices, dim)` | 沿 `inter_dim` 切分 |
| `up_proj_weight` | 同上 | 同上 | 沿 `inter_dim` 切分 |
| `gate_proj_scale` | `(n_experts, 2*inter_dim//128, dim//128)` | `(n_experts, num_devices, inter_dim//num_devices//128, dim//128)` | scale 同步切分 |
| `up_proj_scale` | 同上 | 同上 | scale 同步切分 |

#### Shared expert（单独 `gate_proj` / `up_proj`）

| 权重 | 原始形状 | TP8 切分后形状 | 切分维度 |
|:---|:---|:---|:---|
| `shared_gate_proj_weight` | `(inter_dim, dim)` | `(1, num_devices, inter_dim//num_devices, dim)` | 沿 `inter_dim` 切分 |
| `shared_up_proj_weight` | `(inter_dim, dim)` | `(1, num_devices, inter_dim//num_devices, dim)` | 沿 `inter_dim` 切分 |

**关键代码特征**：
```python
# TP8 分支
local_inter_dim = inter_dim // num_devices
gate_proj_weight = gate_proj_weight.reshape(n_experts, num_devices, local_inter_dim, dim)
# 非 TP8（EP8）分支：沿专家维度切分
n_local_experts = n_experts // num_devices
gate_proj_weight = gate_proj_weight.reshape(num_devices, n_local_experts, inter_dim, dim).transpose(0, 1)
```

> 当前 Qwen3.6 实际使用 `tp_mode=True`，因此**按中间维度切分**，不是按专家维度切分。

---

### 2.4 `ExpertDownAllReduce`（MoE down 投影）

| 权重 | 原始形状 | TP8 切分后形状 | 切分维度 |
|:---|:---|:---|:---|
| `down_proj` (routed) | `(n_experts, dim, inter_dim)` | `(n_experts, num_devices, dim, inter_dim//num_devices)` | 沿输入 `inter_dim` 切分 |
| `down_proj` (shared) | `(dim, inter_dim)` | `(1, num_devices, dim, inter_dim//num_devices)` | 沿 `inter_dim` 切分 |
| `down_proj_scale` | 对应 scale | `(n_experts, num_devices, dim//128, inter_dim//num_devices//128)` | scale 同步切分 |

**关键代码特征**：
```python
if tp_mode:
    local_inter_dim = moe_inter_dim // num_devices
    down_proj_weight = down_proj_weight.reshape(n_experts, num_devices, dim, local_inter_dim)
else:
    # EP8：沿专家维度切分
    down_proj_weight = down_proj_weight.reshape(num_devices, n_local_experts, dim, local_inter_dim).transpose(0, 1)
```

> 由于 down 投影的输出是 `dim`，而输入 `inter_dim` 被切分，因此每个卡输出的是部分 `dim` 结果，需要 `all-reduce` 聚合。

---

### 2.5 `RMSNormHeadProj`（最终 Norm + lm_head）

| 权重 | 来源 key | 切分方式 | 形状变化 |
|:---|:---|:---|:---|
| `model.norm.weight` | `model.norm.weight` / `model.language_model.norm.weight` | 不切分，复制 | `(1, dim)` → `(num_devices, dim)` |
| `lm_head.weight`（TileRT native checkpoint） | `lm_head.weight` | 沿 `vocab_size` 切分 | `(vocab_size, dim)` → `(num_devices, vocab_size/num_devices, dim)` |
| `lm_head.weight`（golden_forward 运行时） | `full_head` | **不切分，完整复制到每张卡** | `(vocab_size/num_devices, dim)` → `(vocab_size, dim)` |

**关键代码特征**：
```python
# RMSNormHeadProj.device_sharding：native checkpoint 阶段按 vocab 切分
rmsnorm_gamma = rmsnorm_gamma.repeat(self.num_devices, 1)
if head_proj.dim() == 2 and head_proj.size(0) * self.num_devices == self.logits_dim:
    head_proj = head_proj[None, ...].repeat(self.num_devices, 1, 1)

# QwenShowHandsLayer._get_full_head_proj：golden 路径把 shard 还原成完整 head
if local_head.dim() == 3:
    full_head = local_head.reshape(-1, self.model_args.dim)
self._full_head_proj_cache[device_id] = full_head
```

> `lm_head` 的 **TileRT native checkpoint 权重**是按 `vocab_size` 切分的；但在 **golden_forward 运行时**，`_get_full_head_proj()` 会把每张卡的 shard 重新拼成完整 `(vocab_size, dim)` 并缓存，因此 golden 路径实际做 matmul 时使用的是完整 head。真正的生产 kernel 路径（`tilert_forward`）才会使用 native 的 vocab-sharded layout。

---

## 3. 涉及切分的 OP 汇总

| OP | 权重名 | 切分维度 | 切分类型 | 是否需要 all-reduce |
|:---|:---|:---|:---|:---:|
| `ExpertSelectUpGateSiLU` | `exp_gate_weights` / `exp_up_weights` | `inter_dim` | TP8 | ❌（输入已切，输出保持切分） |
| `ExpertDownAllReduce` | `exp_down_weights` | `inter_dim` | TP8 | ✅（输出 dim 需要 all-reduce） |
| `RMSNormHeadProj` | `lm_head.weight`（native checkpoint） | `vocab_size` | TP8 | ❌（native 路径每张卡出部分 logits；golden 路径会先把 shard 还原成 full head） |
| `RMSNormExpertProj` | `mlp.gate.weight` | 无 | 复制 | ❌ |
| `GQAAttention` | `qkv_proj_weights` / `o_proj_weights` | 无 | 复制 | ❌ |
| `DeltaNetOp` | 所有线性/卷积权重 | 无 | 复制 | ❌ |

---

## 4. EP8 vs TP8 对比

| 模式 | 切分对象 | 每张卡持有 | 通信需求 | Qwen3.6 当前使用 |
|:---|:---|:---|:---|:---:|
| **EP8** | 专家维度 | `n_routed_experts/8` 个完整 inter_dim 的专家 | all-to-all / broadcast token ids | ❌ |
| **TP8** | MoE 中间维度 | 所有专家，但每个专家只持有 `inter_dim/8` | all-reduce after down_proj | ✅ |
| **Vocab TP8** | lm_head 词表维度 | native checkpoint 每个卡持有 `vocab_size/8` 的 head；golden 路径会还原成 full head | native 路径需 all-gather logits；golden 路径直接复制 full head | ✅ |

---

## 5. 相关文件路径

| OP | 文件 |
|:---|:---|
| GQAAttention 权重切分 | `tilert/models/qwen3_6/ops/gqa_attention.py:device_sharding` |
| DeltaNetOp 权重切分 | `tilert/models/qwen3_6/ops/delta_net.py:device_sharding` |
| RMSNormExpertProj 权重切分 | `tilert/models/qwen3_6/ops/rmsnorm_expert_proj.py:device_sharding` |
| ExpertSelectUpGateSiLU 权重切分 | `tilert/models/qwen3_6/ops/expert_sel_up_gate_silu.py:process_gate_up_weights` / `device_sharding` |
| ExpertDownAllReduce 权重切分 | `tilert/models/qwen3_6/ops/expert_down_allreduce.py:process_down_weights` / `device_sharding` |
| RMSNormHeadProj 权重切分 | `tilert/models/qwen3_6/ops/rmsnorm_head_proj.py:device_sharding` |
| HF 检查点预分片 | `tilert/models/qwen3_6/modules/hf_source_loader.py:_precompute_all_device_states` |
