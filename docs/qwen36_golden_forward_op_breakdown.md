# Qwen3.6 golden_forward 完整流程 OP 拆解

> 分析对象：`tilert/models/qwen3_6/modules/end2end.py:_golden_forward_device` 及其子模块 golden 路径。
> 数据来源：源码静态分析 + `qwen_op_call_log.txt` 运行时日志。

---

## 第一部分：完整 golden_forward 调用流程

```
_golden_forward_device(device_id, token_id, cur_pos)
│
├─ 1. embedding lookup
│     x = embed_weight[token_id].unsqueeze(0).to(bfloat16)
│     [op: nn.Embedding / tensor indexing]
│
├─ 2. QwenTransformerStack.forward / golden_forward(x, cur_pos, mrope_embed, caches)
│   │
│   ├─ 2.1 若 seq_len > 1，生成 causal mask
│   │     mask = torch.full(...) + torch.triu(...)
│   │     [op: full, triu]
│   │
│   └─ 2.2 遍历 40 层 heterogeneous stack
│         for layer_idx, block in enumerate(exec_seq):
│             if block is DeltaNet:
│                 out, state = DeltaNet.forward/golden_forward(...)
│             elif block is GatedAttention:
│                 out, k_cache, v_cache = GatedAttention.forward/golden_forward(...)
│             h = out
│
├─ 3. 最终 head 投影与采样
│   │
│   ├─ 3.1 full_head = _get_full_head_proj(device_id)
│   │     [处理 lm_head 权重 reshape / cache]
│   │
│   └─ 3.2 logits = RMSNormHeadProj.golden_forward(h)
│         [op: RMSNormHeadProj]
│
└─ 4. token_out = _sample(logits[0, last_pos])
```

### 1.1 DeltaNet 层详细流程

```
DeltaNet.golden_forward(x, start_pos, state)
│
├─ 1. input_layernorm(x)
│     [op: RMSNorm]
│
├─ 2. attn_out, state = DeltaNetOp.golden_forward(norm_x, start_pos, state)
│   │
│   ├─ 2.1 投影
│   │     mixed_qkv = x @ in_proj_qkv_weights.T
│   │     z         = x @ in_proj_z_weights.T
│   │     b         = x @ in_proj_b_weights.T
│   │     a         = x @ in_proj_a_weights.T
│   │     [op: linear / matmul × 4]
│   │
│   ├─ 2.2 因果卷积
│   │     mixed_qkv = mixed_qkv.transpose(1, 2)
│   │     mixed_qkv, conv_state = _causal_conv1d_update(...)
│   │         torch.cat(conv_state, mixed_qkv)
│   │         F.conv1d(..., groups=conv_dim)
│   │         F.silu(out)
│   │     mixed_qkv = mixed_qkv.transpose(1, 2)
│   │     [op: cat, conv1d, silu]
│   │
│   ├─ 2.3 分割 Q/K/V
│   │     q, k, v = torch.split(mixed_qkv, [...])
│   │     [op: split]
│   │
│   ├─ 2.4 门控参数
│   │     beta = torch.sigmoid(b)
│   │     g    = -A_log.exp() * F.softplus(a + dt_bias)
│   │     [op: sigmoid, exp, softplus]
│   │
│   ├─ 2.5 门控 Delta 注意力
│   │     attn_out, recurrent_state = _gated_delta_attention(q,k,v,beta,g,state)
│   │         [op: transpose, repeat_interleave, pad, cumsum, matmul, tril,
│   │          exp, masked_fill, chunk-loop, reshape, contiguous]
│   │
│   ├─ 2.6 门控 RMSNorm
│   │     attn_out = _rmsnorm_gated(attn_out, z_gate, norm_weights)
│   │         rmsnorm + F.silu(gate) elementwise-mul
│   │     [op: rmsnorm, silu]
│   │
│   └─ 2.7 输出投影
│         out = attn_out @ out_proj_weights.T
│         [op: linear / matmul]
│
├─ 3. h = x + attn_out
│     [op: residual add]
│
├─ 4. post_attention_layernorm(h)
│     [op: RMSNorm]
│
├─ 5. ffn_partial = QwenMoeBlock.golden_forward(norm_h)
│   │
│   ├─ 5.1 h_flat, routing_weights, expert_indices = RMSNormExpertProj.golden_forward(x)
│   │       norm_x = x * rsqrt(mean(x^2) + eps)
│   │       scores = norm_x @ gate_weight.T
│   │       routing_weights, expert_indices = torch.topk(F.softmax(scores), topk)
│   │       [op: RMSNorm, matmul, softmax, topk]
│   │
│   ├─ 5.2 moe_intermediate = ExpertSelectUpGateSiLU.golden_forward(h_flat, routing_weights, expert_indices)
│   │       gather expert gate_up_proj
│   │       up, gate = split(gate_up, [inter_dim, inter_dim])
│   │       act = up * F.silu(gate)
│   │       act = act * routing_weights
│   │       [op: gather/slicing, split, silu, elementwise-mul, scale]
│   │
│   └─ 5.3 moe_out = ExpertDownAllReduce.golden_forward(h_flat, expert_indices, moe_intermediate)
│             expert down projection (matmul)
│             weighted sum over top-k experts
│             moe_sync_callback(...)  # 多卡时 all-reduce / barrier
│             [op: matmul, sum/scale, all-reduce]
│
├─ 6. h = h + ffn_partial
│     [op: residual add]
│
└─ 7. return h, state
```

### 1.2 GatedAttention 层详细流程

```
GatedAttention.golden_forward(x, start_pos, mrope_embed, k_cache, v_cache, mask)
│
├─ 1. input_layernorm(x)
│     [op: RMSNorm]
│
├─ 2. attn_out, k_cache, v_cache = GQAAttention.golden_forward(norm_x, start_pos, mrope_embed, k_cache, v_cache, mask)
│   │
│   ├─ 2.1 QKV 投影
│   │     qkv = x @ qkv_proj_weights.T
│   │     [op: linear / matmul]
│   │
│   ├─ 2.2 分割 Q/K/V + gate
│   │     q_gate, k, v = torch.split(qkv, [...])
│   │     q_full_view = q_gate.view(...)
│   │     q, gate = torch.chunk(q_full_view, 2, dim=-1)
│   │     [op: split, chunk]
│   │
│   ├─ 2.3 头维度 RMSNorm
│   │     q = q * rsqrt(mean(q^2)) * (1 + q_norm_weight)
│   │     k = k * rsqrt(mean(k^2)) * (1 + k_norm_weight)
│   │     [op: rmsnorm]
│   │
│   ├─ 2.4 RoPE (M-RoPE)
│   │     q_pe, q_no_pe = torch.split(q, [rope_dim, no_pe_dim], dim=-1)
│   │     k_pe, k_no_pe = torch.split(k, [rope_dim, no_pe_dim], dim=-1)
│   │     q_pe, k_pe = apply_mrope_embed(q_pe, k_pe, cur_cos, cur_sin)
│   │     q = torch.cat([q_pe, q_no_pe], dim=-1)
│   │     k = torch.cat([k_pe, k_no_pe], dim=-1)
│   │     [op: split, apply_mrope_embed, cat]
│   │
│   ├─ 2.5 写入 KV cache
│   │     k_cache[...] = k.transpose(1, 2)
│   │     v_cache[...] = v.transpose(1, 2)
│   │     [op: transpose, copy]
│   │
│   ├─ 2.6 GQA repeat_interleave
│   │     k_full = k_cache[...].transpose(1, 2)
│   │     v_full = v_cache[...].transpose(1, 2)
│   │     if num_heads != num_kv_heads:
│   │         k_full = k_full.repeat_interleave(reps, dim=1)
│   │         v_full = v_full.repeat_interleave(reps, dim=1)
│   │     [op: repeat_interleave]
│   │
│   ├─ 2.7 注意力计算
│   │     scores = torch.matmul(q, k_full.transpose(-2, -1)) / sqrt(head_dim)
│   │     scores = scores + mask
│   │     attn   = F.softmax(scores, dim=-1)
│   │     o      = torch.matmul(attn, v_full)
│   │     [op: matmul, add, softmax]
│   │
│   └─ 2.8 输出投影 + gate
│         out = o.transpose(1, 2).contiguous().view(...)
│         out = out * torch.sigmoid(gate)
│         out = out @ o_proj_weights.T
│         [op: transpose, view, sigmoid, matmul]
│
├─ 3. h = x + attn_out
│     [op: residual add]
│
├─ 4. post_attention_layernorm(h)
│     [op: RMSNorm]
│
├─ 5. ffn_partial = QwenMoeBlock.golden_forward(norm_h)
│     [同 DeltaNet 步骤 5.1~5.3]
│
├─ 6. h = h + ffn_partial
│     [op: residual add]
│
└─ 7. return h, k_cache, v_cache
```

### 1.3 End-to-End Head 投影流程

```
_golden_forward_device
│
└─ RMSNormHeadProj.golden_forward(h)
   │
   ├─ hidden_rmsnorm = h * rsqrt(mean(h^2) + eps) * (1 + gamma)
   │     [op: rmsnorm]
   │
   ├─ if head_proj.dim() == 3:
   │       head_proj = head_proj.transpose(1, 2).reshape(-1, dim)
   │     [op: transpose, reshape]
   │
   └─ 分 chunk 计算 logits
         for chunk in head_proj:
             chunk_logits = hidden_rmsnorm @ head_chunk.T
             chunks.append(chunk_logits.float())
         result = torch.cat(chunks, dim=-1)
         [op: matmul, cat]
```

---

## 第二部分：OP 汇总表格

### 2.1 独立 OP 文件（TileRTModule 子类）

| 序号 | OP 文件 | 类名 | 调用位置 | 日志中 golden_forward 调用次数 | 主要职责 |
|:---:|:---|:---|:---|:---:|:---|
| 1 | `ops/delta_net.py` | `DeltaNetOp` | `DeltaNet.attn` | 2640 | 线性注意力：投影、因果卷积、门控 Delta 注意力、门控 RMSNorm、输出投影 |
| 2 | `ops/gqa_attention.py` | `GQAAttention` | `GatedAttention.attn` | 880 | GQA 注意力：QKV 投影、头 RMSNorm、M-RoPE、KV cache、GQA、softmax、输出投影 |
| 3 | `ops/rmsnorm_expert_proj.py` | `RMSNormExpertProj` | `QwenMoeBlock.moe` | 3520 | 专家路由：RMSNorm、gate 投影、softmax、topk |
| 4 | `ops/expert_sel_up_gate_silu.py` | `ExpertSelectUpGateSiLU` | `QwenMoeBlock.moe` | 3520 | 专家 up/gate 计算：gather、split、SiLU、加权 |
| 5 | `ops/expert_down_allreduce.py` | `ExpertDownAllReduce` | `QwenMoeBlock.moe` | 3520 | 专家 down 投影 + 聚合 + all-reduce |
| 6 | `ops/rmsnorm_head_proj.py` | `RMSNormHeadProj` | `end2end` | 88 | 最终 RMSNorm + lm_head 投影 |

### 2.2 Modules 层（包裹 OP 的 TileRTModule）

| 序号 | Module 文件 | 类名 | 日志中 golden_forward 调用次数 | 主要职责 |
|:---:|:---|:---|:---:|:---|
| 1 | `modules/transformer_stack.py` | `QwenTransformerStack` | 176 | 40 层遍历、causal mask 生成、cache 管理 |
| 2 | `modules/delta_net.py` | `DeltaNet` | 5280 | 包装 DeltaNetOp + 两层 RMSNorm + residual |
| 3 | `modules/gated_attention.py` | `GatedAttention` | 1760 | 包装 GQAAttention + 两层 RMSNorm + residual |
| 4 | `modules/moe.py` | `QwenMoeBlock` | 3520 | 包装 RMSNormExpertProj / ExpertSelectUpGateSiLU / ExpertDownAllReduce |

### 2.3 隐式 PyTorch 操作（出现在 OP / Module 内部）

| 类别 | 操作名 | 出现位置 | 说明 |
|:---|:---|:---|:---|
| **线性/矩阵** | `matmul` / `@` | 所有 projection | qkv/z/b/a/out/gate/down/head 投影 |
| | `F.linear` | `tilert.models.common.linear` | 部分模块调用 |
| **卷积** | `F.conv1d` | `DeltaNetOp._causal_conv1d_update` | 因果深度可分离卷积 |
| **激活** | `F.silu` | `DeltaNetOp`、`ExpertSelectUpGateSiLU` | SiLU 激活 |
| | `torch.sigmoid` | `DeltaNetOp.beta`、`GQAAttentionOp.gate` | 门控 sigmoid |
| | `F.softplus` | `DeltaNetOp.g` | 衰减门控 softplus |
| | `F.softmax` | `GQAAttentionOp`、MoE routing | 注意力 / 路由概率 |
| **Norm** | `rmsnorm` (rsqrt + mean) | 多处 | 输入/后注意力/头/专家 RMSNorm |
| **分合张量** | `torch.split` | `DeltaNetOp`、`GQAAttentionOp` | 分割 QKV / gate_up |
| | `torch.chunk` | `GQAAttentionOp` | 分割 q_gate 为 q 与 gate |
| | `torch.cat` | `GQAAttentionOp`、`RMSNormHeadProj` | 拼接 RoPE 结果 / head chunk logits |
| | `F.pad` | `DeltaNetOp._gated_delta_attention` | chunk 对齐填充 |
| | `torch.view` / `reshape` / `transpose` | 多处 | 头维度变换、cache layout |
| | `torch.repeat_interleave` | `GQAAttentionOp` | GQA head 复制 |
| **RoPE/Rotate** | `apply_mrope_embed` | `GQAAttentionOp.golden_forward` | M-RoPE 旋转位置编码 |
| **采样/TopK** | `torch.topk` | `RMSNormExpertProj.golden_forward` | 专家选择 topk |
| **掩码/辅助** | `torch.full` + `torch.triu` | `QwenTransformerStack.golden_forward` | causal mask |
| | `torch.cumsum` | `DeltaNetOp._gated_delta_attention` | 门控衰减累积 |
| | `torch.exp` / `torch.tril` / `masked_fill` | `DeltaNetOp._gated_delta_attention` | 衰减矩阵构造 |
| **通信** | `moe_sync_callback` (all-reduce) | `ExpertDownAllReduce.golden_forward` | 多卡 TP8 MoE 聚合 |
| **Embedding** | tensor indexing | `end2end._golden_forward_device` | token → embedding lookup |

### 2.4 DeepSeek-V3.2 同名 OP 对照

| DeepSeek-V3.2 OP 名 | Qwen3.6 对应独立类 | Qwen3.6 中是否以独立 OP 触发 | 未触发时的隐式实现位置 |
|:---|:---|:---:|:---|
| `expert_down_allreduce` | `ExpertDownAllReduce` | ✅ | — |
| `expert_sel_up_gate_silu` | `ExpertSelectUpGateSiLU` | ✅ | — |
| `rmsnorm_expert_proj` | `RMSNormExpertProj` | ✅ | — |
| `rmsnorm_head_proj` | `RMSNormHeadProj` | ✅ | — |
| `qkv_rope` | `QKVRoPE`（存在但未在本次 golden 路径实例化） | ❌ | `GQAAttentionOp.golden_forward` 中 `apply_mrope_embed` |
| `topk` | `TopK`（不存在独立类，但 `torch.topk` 被调用） | ❌ | `RMSNormExpertProj.golden_forward` 中 `torch.topk` |
| `rotate` | `Rotate`（存在但未在本次 golden 路径实例化） | ❌ | `GQAAttentionOp.golden_forward` 中 RoPE / 头旋转 |
| `unproj_o_allreduce` | `UnProjOAllReduce`（实例化于 tilert_forward，golden 未调用） | ❌ | GQA EP8 全权重复制，未做 allreduce |
| `rmsnorm_up_gate_silu` | `RMSNormUpGateSiLU`（存在但未触发） | ❌ | 功能拆分到 `RMSNormExpertProj` + `ExpertSelectUpGateSiLU` |

---

## 附录：关键文件路径

| 层级 | 文件 |
|:---|:---|
| End-to-End | `tilert/models/qwen3_6/modules/end2end.py` |
| Stack | `tilert/models/qwen3_6/modules/transformer_stack.py` |
| DeltaNet 模块 | `tilert/models/qwen3_6/modules/delta_net.py` |
| DeltaNet OP | `tilert/models/qwen3_6/ops/delta_net.py` |
| GatedAttention 模块 | `tilert/models/qwen3_6/modules/gated_attention.py` |
| GQA OP | `tilert/models/qwen3_6/ops/gqa_attention.py` |
| MoE 模块 | `tilert/models/qwen3_6/modules/moe.py` |
| RMSNormExpertProj OP | `tilert/models/qwen3_6/ops/rmsnorm_expert_proj.py` |
| ExpertSelectUpGateSiLU OP | `tilert/models/qwen3_6/ops/expert_sel_up_gate_silu.py` |
| ExpertDownAllReduce OP | `tilert/models/qwen3_6/ops/expert_down_allreduce.py` |
| RMSNormHeadProj OP | `tilert/models/qwen3_6/ops/rmsnorm_head_proj.py` |
