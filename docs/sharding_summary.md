# Qwen3.6 TileRT Device Sharding 日志总结

## 基本信息

- **设备数量**: 8 (TP8)
- **模型**: Qwen3.6 MoE
- **关键参数**:
  - `hidden_dim`: 2048
  - `num_experts`: 257 (256 routed + 1 shared)
  - `inter_dim`: 512 → 每设备 64 (512/8)
  - `num_heads`: 32
  - `num_local_kv_heads`: 8 (GQA: 32/4)
  - `head_dim`: 128
  - `vocab_size`: 248320

---

## 各 OP 的 Sharding 策略

### 1. RMSNormHeadProj

| Key | 原始形状 | 切分后形状 | dtype |
|-----|----------|------------|-------|
| model.language_model.norm.weight | (2048) | (8, 2048) | bfloat16 |
| lm_head.weight | (248320, 2048) | (8, 248320, 2048) | bfloat16 |

**切分策略**: 复制 (Replicate) - 在第 0 维堆叠 8 份

---

### 2. DeltaNet (Linear Attention)

| Key | 原始形状 | 切分后形状 | dtype |
|-----|----------|------------|-------|
| linear_attn.in_proj_qkv.weight | (8192, 2048) | (8, 8192, 2048) | bfloat16 |
| linear_attn.in_proj_z.weight | (4096, 2048) | (8, 4096, 2048) | bfloat16 |
| linear_attn.in_proj_a.weight | (32, 2048) | (8, 32, 2048) | bfloat16 |
| linear_attn.in_proj_b.weight | (32, 2048) | (8, 32, 2048) | bfloat16 |
| linear_attn.conv1d.weight | (8192, 1, 4) | (8, 8192, 1, 4) | bfloat16 |
| linear_attn.A_log | (32) | (8, 32) | bfloat16 |
| linear_attn.dt_bias | (32) | (8, 32) | bfloat16 |
| linear_attn.norm.weight | (128) | (8, 128) | bfloat16 |
| linear_attn.out_proj.weight | (2048, 4096) | (8, 2048, 4096) | bfloat16 |

**切分策略**: 复制 (Replicate) - 在第 0 维堆叠 8 份

---

### 3. RMSNormExpertProj

| Key | 原始形状 | 切分后形状 | dtype |
|-----|----------|------------|-------|
| post_attention_layernorm.weight | (2048) | (2048) | float32 |
| mlp.gate.weight | (256, 2048) | (256, 2048) | bfloat16 |

**切分策略**: 不切分 - 保持原样

---

### 4. ExpertSelUpGateSiLU (MoE Up+Gate 投影)

**切分策略**: 专家维度切分 (Expert TP) - 按中间层 (inter_dim) 切分

**原始输入形状**:
| 权重 Key | 原始形状 | 说明 |
|----------|----------|------|
| `mlp.experts.gate_up_proj` | (256, 1024, 2048) | routed experts: (n_routed, 2×inter_dim, hidden) |
| `mlp.shared_expert.gate_proj.weight` | (512, 2048) | shared expert gate: (inter_dim, hidden) |
| `mlp.shared_expert.up_proj.weight` | (512, 2048) | shared expert up: (inter_dim, hidden) |
| `mlp.shared_expert_gate.weight` | (1, 2048) | shared expert gate 门控 |
| `mlp.gate.e_score_correction_bias` | (256,) | expert 选择偏置 |

**切分后形状**:
| Key | 切分后形状 | 说明 |
|-----|------------|-------|
| exp_bias | (8, 256) | 8 devices × 256 experts |
| exp_gate_weights | (257, 8, 64, 2048) | (experts, devices, inter/8, hidden) |
| exp_gate_scales | (257, 8, 1, 16) | FP8 量化 scale |
| exp_up_weights | (257, 8, 64, 2048) | (experts, devices, inter/8, hidden) |
| exp_up_scales | (257, 8, 1, 16) | FP8 量化 scale |
| shared_expert_gate | (8, 1, 2048) | 复制到 8 devices |

**形状变化示意**:
```
原始 gate_up_proj: (256, 1024, 2048)
                          ↓
                    1024 = 2 × 512 (gate + up)
                          ↓ TP8 切分
切分后: (257, 8, 64, 2048)
        │  │   │    │
        │  │   │    └── hidden_dim = 2048 (完整保留)
        │  │   └────── inter_dim_per_device = 512/8 = 64
        │  └────────── num_devices = 8
        └───────────── n_experts = 257 (256 routed + 1 shared)
```

---

### 5. ExpertDownAllReduce (MoE Down 投影)

| Key | 原始形状 | 切分后形状 | dtype |
|-----|----------|------------|-------|
| down_weights | ([1, 8, 2048, 64], [256, 8, 2048, 64]) | (257, 8, 2048, 64) | bfloat16 |
| down_scales | ([1, 8, 16, 1], [256, 8, 16, 1]) | (257, 8, 16, 1) | float32 |

**切分策略**: 专家维度切分 (Expert TP)

**原始输入形状**:
| 权重 Key | 原始形状 | 说明 |
|----------|----------|------|
| `mlp.shared_expert.down_proj.weight` | (2048, 512) | shared expert down |
| `mlp.experts.down_proj` | (256, 2048, 512) | routed experts down |

**形状变化**:
```
原始: (256, 2048, 512)  →  切分后: (257, 8, 2048, 64)
          ↓                           ↓
    routed experts              expert维度保留
    down_proj                   device切分
    inter=512                   inter=64 per device
```

- 原始 `down_proj`: (256, 2048, 512) = (n_routed, hidden, inter_dim)
- 切分后: (257, 8, 2048, 64) = (n_experts, num_devices, hidden, inter_dim/8)
- 与 Up 切分对应

---

### 6. GQAAttention

| Key | 原始形状 | 切分后形状 | dtype |
|-----|----------|------------|-------|
| self_attn.q_proj.weight | (8192, 2048) | (8, 9216, 2048) | bfloat16 |
| self_attn.o_proj.weight | (2048, 4096) | (8, 2048, 4096) | bfloat16 |
| self_attn.q_norm.weight | (256) | (8, 256) | bfloat16 |
| self_attn.k_norm.weight | (256) | (8, 256) | bfloat16 |

**切分策略**: 复制 (Replicate) - 在第 0 维堆叠 8 份
- QKV 合并为 (q_proj + k_proj + v_proj) = (8192 = 4096+2048+2048)
- 实际 sharded 后是 (8, 9216, 2048)，包含了 QKV 三个投影

---

### 7. UnprojOAllReduce

| Key | 原始形状 | 切分后形状 | dtype |
|-----|----------|------------|-------|
| self_attn.o_proj.weight | (2048, 4096) | (8, 2048, 4096) | bfloat16 |
| o_proj_scale_inv | (16, 32) | (8, 16, 32) | float32 |

**切分策略**: 复制 (Replicate) - 在第 0 维堆叠 8 份

---

## 切分模式总结

```
+-------------------------------------------------------------+
|                  两种 Sharding 策略                         |
+-------------------------------------------------------------+
| 1. 复制 (Replicate) - 用于非 MoE 权重                       |
|    - Attention 权重 (Q/K/V/O, Q/K norm)                    |
|    - DeltaNet 所有权重                                      |
|    - Head projection (lm_head)                             |
|    → 形状: (dim) → (num_devices, dim)                      |
|                                                             |
| 2. 专家切分 (Expert TP) - 用于 MoE 专家权重                  |
|    - gate_up_proj, down_proj                                |
|    → 形状: (n_experts, inter_dim, dim)                     |
|       → (n_experts, num_devices, inter_dim/num_devices, hidden)  |
|       例如: (257, 8, 64, 2048)                              |
|       含义: (expert, device, inter_per_device, hidden)     |
+-------------------------------------------------------------+
```

## 关键观察

1. **Expert TP8**: 256 routed experts + 1 shared expert = 257 experts 总计
2. **inter_dim 切分**: 512 / 8 = 64 每设备
3. **GQA**: 32 heads, 8 local KV heads (每 4 个 Q 共享一个 KV head)
4. **DeltaNet**: 30/40 层使用 DeltaNet (线性注意力变体)
5. **vocab 复制**: lm_head 权重在 8 设备上复制，形状 (8, 248320, 2048)

---

## 各 OP 显存占用 (单设备)

> 注：bfloat16 = 2 字节，float32 = 4 字节

### 1. RMSNormHeadProj

| Key | Shape | 显存 |
|-----|-------|------|
| norm.weight | (8, 2048) | **32 KB** |
| lm_head.weight | (8, 248320, 2048) | **~3.8 GB** |
| **合计** | | **~3.8 GB** |

### 2. DeltaNet (9个权重)

| Key | Shape | 显存 |
|-----|-------|------|
| in_proj_qkv.weight | (8, 8192, 2048) | **~256 MB** |
| in_proj_z.weight | (8, 4096, 2048) | **~128 MB** |
| in_proj_a.weight | (8, 32, 2048) | **~1 MB** |
| in_proj_b.weight | (8, 32, 2048) | **~1 MB** |
| conv1d.weight | (8, 8192, 1, 4) | **~0.5 MB** |
| A_log | (8, 32) | **~0.5 KB** |
| dt_bias | (8, 32) | **~0.5 KB** |
| norm.weight | (8, 128) | **~2 KB** |
| out_proj.weight | (8, 2048, 4096) | **~128 MB** |
| **合计** | | **~514 MB** |

### 3. RMSNormExpertProj

| Key | Shape | 显存 |
|-----|-------|------|
| post_attention_layernorm.weight | (2048) | **8 KB** |
| mlp.gate.weight | (256, 2048) | **~1 MB** |
| **合计** | | **~1 MB** |

### 4. ExpertSelUpGateSiLU (MoE Up+Gate)

| Key | Shape | 显存 |
|-----|-------|------|
| exp_bias | (8, 256) | **8 KB** |
| exp_gate_weights | (257, 8, 64, 2048) | **~518 MB** |
| exp_gate_scales | (257, 8, 1, 16) | **~132 KB** |
| exp_up_weights | (257, 8, 64, 2048) | **~518 MB** |
| exp_up_scales | (257, 8, 1, 16) | **~132 KB** |
| shared_expert_gate | (8, 1, 2048) | **32 KB** |
| **合计** | | **~1.04 GB** |

### 5. ExpertDownAllReduce (MoE Down)

| Key | Shape | 显存 |
|-----|-------|------|
| down_weights | (257, 8, 2048, 64) | **~518 MB** |
| down_scales | (257, 8, 16, 1) | **~132 KB** |
| **合计** | | **~518 MB** |

### 6. GQAAttention

| Key | Shape | 显存 |
|-----|-------|------|
| qkv_proj_weights | (8, 9216, 2048) | **~288 MB** |
| o_proj_weights | (8, 2048, 4096) | **~128 MB** |
| q_norm_weights | (8, 256) | **~4 KB** |
| k_norm_weights | (8, 256) | **~4 KB** |
| **合计** | | **~416 MB** |

### 7. UnprojOAllReduce

| Key | Shape | 显存 |
|-----|-------|------|
| unproj_weights | (8, 2048, 4096) | **~128 MB** |
| unproj_scales | (8, 16, 32) | **~16 KB** |
| **合计** | | **~128 MB** |

---

## 显存汇总

| OP | 显存 (单设备) |
|-----|--------------|
| RMSNormHeadProj | **~3.8 GB** |
| ExpertSelUpGateSiLU | **~1.04 GB** |
| ExpertDownAllReduce | **~518 MB** |
| DeltaNet | **~514 MB** |
| GQAAttention | **~416 MB** |
| UnprojOAllReduce | **~128 MB** |
| RMSNormExpertProj | **~1 MB** |
| **单层总计** | **~6.4 GB** |
| **40层总计** | **~256 GB** |

> 注：以上仅为模型权重显存，不包含激活值、梯度、优化器状态等。