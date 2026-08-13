# TileRT 的 Op 与 Kernel 对应描述

> **数据来源**
> - `d:\Vicold\Tile\cuobjdump.readable.log` —— `cuobjdump` 导出的 fat binary 里所有 `__global__` kernel（640 行）
> - `d:\Vicold\Tile\nm-CD so.md` —— `nm -CD libtilert_dsv32.so` 符号表 + 85 处 `torch.ops.tilert` 调用清单
> - `d:\Github\Tile\TileRT\tilert\models\deepseek_v3_2\ops\` —— 30 个 op 的 Python wrapper
> - `d:\Github\Tile\TileRT\tilert\models\deepseek_v3_2\modules\end2end.py` —— `dsa_show_hands` 系列超级内核入口
>
> **日期**：2026-07-13

---

## 一、三层结构回顾

TileRT 一个"算子"实际跨三层：

| 层 | 位置 | 是什么 | 运行在哪 |
|----|------|--------|---------|
| **L1 Python wrapper** | `ops/*.py` 里的 `xxx()` 函数 | 调 `torch.ops.tilert.xxx_op` 的薄包装 | CPU (Python) |
| **L2 host C++ CustomOp** | `.so` 里 `torch::Library` 注册的 `xxx_op` | 设 grid/block、`cudaLaunchKernel` | CPU (C++) |
| **L3 `__global__` executor** | `.so` fat binary 里 `tilert::core::executor::piped_prefetch::XxxExecutorImpl<...>` | 真正在 GPU 上跑的 kernel | GPU |

**关键判断**：
- **op ≠ kernel**。op（L1/L2）是"扣扳机的人"，kernel（L3）是"飞出去的子弹"。
- `cudaLaunchKernel` 启动的是 L3 的 `__global__` executor，**不是** L2 的 `*_op`。
- capture 时，L2 的 `*_op` 在 CPU 上跑一遍，它内部每次 `cudaLaunchKernel(executor, ...)` 被 runtime 拦截，记录成 graph 里的一个 kernel node。

---

## 二、cuobjdump 导出的 `__global__` kernel 全表

从 `cuobjdump.readable.log`（`libtilert_dsv32.so` + `libtilert_glm5.so` 的 fat binary）导出的 `__global__` kernel，全部位于 `tilert::core::executor::piped_prefetch` 命名空间下，均为模板实例化。

### 2.1 DSV32 专属 kernel

| `__global__` executor kernel | 对应概念 |
|----|----|
| `BroadcastSelectedTokenIdsExecutorImpl` | 广播选中的 token id（P2P） |
| `DownAllreduceExecutorImpl` | 下投影后 AllReduce |
| `EHProjAllReduceExecutorImpl` | EH 投影 AllReduce |
| `ExpertDownAllreduceExecutorImpl` | 专家下投影 AllReduce |
| `ExpertSelectUpGateSiLUDSv32ExecutorImpl` | 专家选择 + up_gate + SiLU（MoE） |
| `FlashSparseMlaDSv32DevBExecutorImpl` | Flash Sparse MLA（device B） |
| `FlashSparseMlaExecutorImpl` | Flash Sparse MLA（device A） |
| `FusedMoeExecutorImpl` | 融合 MoE |
| `HeadProjExecutorImpl` | Head 投影 |
| `LayernormRopeRotateExecutorImpl` | LayerNorm + RoPE + Rotate |
| `MTPPreProcessExecutorImpl` | MTP 预处理 |
| `PaddedAllReduceAddExecutorImpl` | Padded AllReduce Add |
| `ProjOWkvbDevBHMMAExecutorImpl` | O 投影 + Wkvb（HMMA, device B） |
| `ProjQWkvbDevBHMMAExecutorImpl` | Q 投影 + Wkvb（HMMA） |
| `ProjXWisExecutorImpl` | X 投影 Wis |
| `ProjXWqakiExecutorImpl` | X 投影 Wqaki |
| `ProjXWqakisGLM5ExecutorImpl` | X 投影 Wqakis（GLM5 变体） |
| `ProjXWqkvaDSV32ExecutorImpl` | X 投影 Wqkva（DSV32） |
| `QkvRopeExecutorImpl` | QKV RoPE |
| `PureMlaDsv32ExecutorImpl` | Pure MLA（DSV32） |
| `RMSNormExpertProjDsv32ExecutorImpl` | RMSNorm + 专家投影（DSV32） |
| `RMSNormHeadProjExecutorImpl` | RMSNorm + Head 投影 |
| `RmsnormKvExecutorImpl` | RMSNorm KV |
| `RmsnormProjQWqbHMMAExecutorImpl` | RMSNorm + Proj Q Wqb（HMMA） |
| `RmsnormProjQWqiHMMAExecutorImpl` | RMSNorm + Proj Q Wqi（HMMA） |
| `RMSNormExecutorImpl` | RMSNorm |
| `RMSNormQuantExecutorImpl` | RMSNorm + Quant |
| `RMSNormUpGateSiLUDSv32ExecutorImpl` | RMSNorm + UpGate SiLU（DSV32） |
| `ReceiveSelectedTokenIdsExecutorImpl` | 接收选中的 token id（P2P） |
| `RotateExecutorImpl` | Rotate |
| `RotateCompressedExecutorImpl` | Rotate Compressed |
| `SparseIndexExecutorImpl` | Sparse Index |
| `SparseIndexFusedDsv32ExecutorImpl` | Sparse Index Fused（DSV32） |
| `SparseSelectMlaDsv32ExecutorImpl` | Sparse Select MLA（DSV32） |
| `Top1AllreduceExecutorImpl` | Top-1 AllReduce |
| `TopkAccurateExecutorImpl` | Top-K Accurate |
| `TopkAccurate512R4ExecutorImpl` | Top-K Accurate 512 R4 |
| `TopkAccurate1024R4ExecutorImpl` | Top-K Accurate 1024 R4 |
| `TopkAccurateFusedDsv32ExecutorImpl` | Top-K Accurate Fused（DSV32） |
| `TopkApproximateExecutorImpl` | Top-K Approximate |
| `UnprojOAllreduceDSV32DevBExecutorImpl` | Unproj O AllReduce（DSV32, device B） |
| `UnprojOAllreduceDSV32ExecutorImpl` | Unproj O AllReduce（DSV32） |

### 2.2 GLM5 额外/变体 kernel

GLM5 共享同一套 DSA 引擎，但有自己的变体 kernel（后缀 `GLM5`）：

| `__global__` executor kernel | 说明 |
|----|----|
| `DownAllreduceGLM5ExecutorImpl` | 下投影 AllReduce（GLM5） |
| `EHProjAllReduceGLM5ExecutorImpl` | EH 投影 AllReduce（GLM5） |
| `ExpertDownAllreduceGLM5ExecutorImpl` | 专家下投影 AllReduce（GLM5） |
| `ExpertSelectUpGateSiLUGlm5ExecutorImpl` | 专家选择（GLM5） |
| `FlashSparseMlaGLM5H8ExecutorImpl` | Flash Sparse MLA（GLM5 H8） |
| `FlashSparseMlaGLM5ExecutorImpl` | Flash Sparse MLA（GLM5） |
| `HeadProjGlm5ExecutorImpl` | Head 投影（GLM5） |
| `MTPPreProcessGlm5ExecutorImpl` | MTP 预处理（GLM5） |
| `PaddedAllReduceAddGLM5ExecutorImpl` | Padded AllReduce Add（GLM5） |
| `ProjOWkvbDevBGLM5HMMAExecutorImpl` | O 投影（GLM5） |
| `ProjQWkvbDevBGLM5HMMAExecutorImpl` | Q 投影（GLM5） |
| `ProjXWqakiGLM5_136CTAExecutorImpl` | X 投影（GLM5, 136 CTA） |
| `ProjXWqakiGLM5_68CTAExecutorImpl` | X 投影（GLM5, 68 CTA） |
| `ProjXWqkvaGLM5ExecutorImpl` | X 投影 Wqkva（GLM5） |
| `PureMlaExecutorImpl` | Pure MLA（通用） |
| `RMSNormExpertProjGlm5ExecutorImpl` | RMSNorm + 专家投影（GLM5） |
| `RMSNormHeadProjGlm5ExecutorImpl` | RMSNorm + Head 投影（GLM5） |
| `RmsnormProjQWqbGLM5HMMAExecutorImpl` | RMSNorm + Proj Q Wqb（GLM5） |
| `RmsnormProjQWqiGLM5HMMAExecutorImpl` | RMSNorm + Proj Q Wqi（GLM5） |
| `RMSNormGlm5ExecutorImpl` | RMSNorm（GLM5） |
| `RMSNormQuantGlm5ExecutorImpl` | RMSNorm + Quant（GLM5） |
| `RMSNormUpGateSiLUGlm5ExecutorImpl` | RMSNorm + UpGate SiLU（GLM5） |
| `SparseSelectMlaExecutorImpl` | Sparse Select MLA（通用） |
| `SparseIndexGLM5ExecutorImpl` | Sparse Index（GLM5） |
| `SparseIndexFusedGlm5ExecutorImpl` | Sparse Index Fused（GLM5） |
| `Top1AllreduceGLM5ExecutorImpl` | Top-1 AllReduce（GLM5） |
| `TopkAccurateFusedGlm5ExecutorImpl` | Top-K Accurate Fused（GLM5） |
| `UnprojOAllreduceDevBExecutorImpl` | Unproj O AllReduce（DevB） |
| `UnprojOAllreduceExecutorImpl` | Unproj O AllReduce（通用） |

### 2.3 非 executor 的辅助 kernel

| kernel | 用途 |
|----|----|
| `ExecuteTopP<float, ...>`（anonymous namespace） | Top-P 采样 |
| `top1_bf16_singleblock_128b` | Top-1 路由（bf16, single block, 128b） |
| `llm_preprocess::pre_forward_kernel<1024, dim, 64, ...>` | LLM 预处理（dim=2048/6144/7168） |
| `mtp_verifier::verify_kernel<MtpLayout, 3, dim>` | MTP 验证（dim=2048/6144/7168） |
| `cub::_V_300200_SM_1000::detail::EmptyKernel<void>` | CUB 占位空 kernel（用于同步/barrier） |

---

## 三、Op ↔ Kernel 映射表

每个 Python `ops/*.py` 里的 op（L1）→ 调用 `torch.ops.tilert.*_op`（L2 host C++）→ 内部 `cudaLaunchKernel` 启动对应的 `__global__` executor（L3）。

| `ops/*.py`（L1 op） | `torch.ops.tilert.*_op`（L2） | `__global__` executor kernel（L3） |
|----|----|----|
| `broadcast_selected_token_ids.py` | `broadcast_selected_token_ids_op` | `BroadcastSelectedTokenIdsExecutorImpl` |
| `down_allreduce.py` | `down_allreduce_op` | `DownAllreduceExecutorImpl` |
| `eh_proj_allreduce.py` | `eh_proj_allreduce_op` | `EHProjAllReduceExecutorImpl` |
| `expert_down_allreduce.py` | `expert_down_allreduce_op` | `ExpertDownAllreduceExecutorImpl` |
| `expert_sel_up_gate_silu.py` | `expert_select_up_gate_silu_op` | `ExpertSelectUpGateSiLUDSv32ExecutorImpl` |
| `flash_sparse_mla.py` | `flash_sparse_mla_op` | `FlashSparseMlaExecutorImpl` / `FlashSparseMlaDSv32DevBExecutorImpl` |
| `layernorm_rope_rotate.py` | `layernorm_rope_rotate_op` | `LayernormRopeRotateExecutorImpl` |
| `padded_allreduce_add.py` | `padded_allreduce_add_op` | `PaddedAllReduceAddExecutorImpl` |
| `projo_wkvb.py` | `projo_wkvb_op` | `ProjOWkvbDevBHMMAExecutorImpl` |
| `projq_wqb.py` | `projq_wqb_op` | `ProjQWkvbDevBHMMAExecutorImpl` |
| `projq_wqi.py` | `rmsnorm_proj_qi_op` / `projq_wqi_op` | `RmsnormProjQWqiHMMAExecutorImpl` |
| `projx_wis.py` | `proj_w_op` | `ProjXWisExecutorImpl` |
| `projx_wqaki.py` | `projx_wqaki_op` | `ProjXWqakiExecutorImpl` |
| `projx_wqkva.py` | `projx_wqkva_op` | `ProjXWqkvaDSV32ExecutorImpl` |
| `qkv_rope.py` | `qkv_rope_op` | `QkvRopeExecutorImpl` |
| `receive_selected_token_ids.py` | `receive_selected_token_ids_op` | `ReceiveSelectedTokenIdsExecutorImpl` |
| `rmsnorm_expert_proj.py` | `rmsnorm_expert_proj_op` | `RMSNormExpertProjDsv32ExecutorImpl` |
| `rmsnorm_head_proj.py` | `rmsnorm_head_proj_op` | `RMSNormHeadProjExecutorImpl` |
| `rmsnorm_kv.py` | `rmsnorm_kv_op` | `RmsnormKvExecutorImpl` |
| `rmsnorm_projq_wqb.py` | `rmsnorm_proj_qb_op` | `RmsnormProjQWqbHMMAExecutorImpl` |
| `rmsnorm_quant.py` | `rmsnorm_op` / `rmsnorm_quant_op` | `RMSNormExecutorImpl` / `RMSNormQuantExecutorImpl` |
| `rmsnorm_up_gate_silu.py` | `rmsnorm_up_gate_silu_op` | `RMSNormUpGateSiLUDSv32ExecutorImpl` |
| `rotate.py` | `rotate_op` | `RotateExecutorImpl` / `RotateCompressedExecutorImpl` |
| `sparse_index.py` | `sparse_index_op` / `sparse_index_topk_dsv32_op` / `sparse_index_topk_glm5_op` | `SparseIndexExecutorImpl` / `SparseIndexFusedDsv32ExecutorImpl` |
| `topk.py` | `topk_approximate_op` / `topk_accurate_op` | `TopkApproximateExecutorImpl` / `TopkAccurateExecutorImpl` / `TopkAccurate512R4ExecutorImpl` / `TopkAccurate1024R4ExecutorImpl` / `TopkAccurateFusedDsv32ExecutorImpl` |
| `unproj_o_allreduce.py` | `unproj_o_allreduce_op` | `UnprojOAllreduceDSV32ExecutorImpl` / `UnprojOAllreduceDSV32DevBExecutorImpl` |

**超级内核（不对应单个 `ops/*.py`，而是 `dsa_show_hands` 系列）**：
- `dsa_show_hands_prepare_money` → 在 capture 时按顺序调用上面所有 `*_op`，每个 `*_op` 内部 `cudaLaunchKernel` 一个 executor kernel → 全部记进 graph
- `dsa_show_hands` → `cudaGraphLaunch`，一次 replay 整张图（61 层 + 通信 + 采样）

---

## 四、Executor 模板参数解读

每个 executor 都是模板实例化，签名形如：

```cpp
XxxExecutorImpl<
    DefaultSchedule,          // 调度策略（piped_prefetch 流水线预取）
    N_STAGES,                 // pipeline stages（3/4/5）
    SHMEM_SIZE,               // shared memory 大小（32768/36864/40960/...）
    1,                        // batch size（强制为 1）
    N_CTA,                    // CTA 数量 / blocks per grid（1/2/4/8/16）
    KeComputeType             // 计算类型枚举（0/3/4/5/6/7/8）
>(GlbArgs<8, 2>)              // 全局参数（8 个设备）
```

| 模板参数 | 含义 |
|----|----|
| `DefaultSchedule` | 调度策略（piped_prefetch 流水线预取） |
| 第 1 个数字（3/4/5） | pipeline stages 或 CTA 维度 |
| 第 2 个数字（32768/36864/40960/...） | shared memory 大小或 tile size |
| `1` | batch size（强制为 1） |
| 第 3 个数字（1/2/4/8/16） | CTA 数量（blocks per grid） |
| `KeComputeType`（0/3/4/5/6/7/8） | 计算类型枚举（BF16/FP8/FP4 等） |
| `GlbArgs<8, 2>` | 全局参数（8 个设备） |

每个 executor 通常有多个实例化，对应不同的 CTA 数量（1/2/4/8/16）和不同的 compute type（0/3/4/5/6/7/8）。运行时根据张量形状/dtype 选择对应实例。

---

## 五、capture 时这些 kernel 怎么进 graph

```
dsa_show_hands_prepare_money(params, temp_vars, cache_vars, profile_logs)
  │
  ├─ cudaStreamBeginCapture(stream)   ← 进入捕获模式
  │
  │  依次调用 61 层 + 通信 + 采样对应的 *_op：
  │    rmsnorm_quant_op(...) → 内部 cudaLaunchKernel(RMSNormQuantExecutorImpl) → 记成 Node A
  │    projx_wis_op(...)     → 内部 cudaLaunchKernel(ProjXWisExecutorImpl)     → 记成 Node B
  │    flash_sparse_mla_op   → 内部 cudaLaunchKernel(FlashSparseMlaExecutorImpl) → 记成 Node C
  │    ...（61 层每个 op 内部的 launch 都被记下来）
  │
  ├─ cudaStreamEndCapture(stream)     ← 产出 cudaGraph_t
  └─ cudaGraphInstantiate(graph)      ← 生成可执行的 cudaGraphExec_t
```

**关键**：被记录进 graph 的是"一次 `cudaLaunchKernel` 调用"——即「要发射哪个 `__global__` executor kernel + grid/block + 参数指针」。op（`*_op`）这个 host C++ 函数本身**不在 graph 里**，它只是 capture 期间跑了一遍、负责把 launch 指令一条条喊出去的 CPU 驱动代码。

---

## 六、replay 时 op 已经退场

`dsa_show_hands(token_id)` 每 token 调一次，内部只剩 `cudaGraphLaunch(graphExec, stream)`：

```
cudaGraphLaunch()
  ├── Node 1: RMSNormQuantExecutorImpl   （rmsnorm_quant_op 发射的）
  ├── Node 2: ProjXWisExecutorImpl       （projx_wis_op 发射的）
  ├── Node 3: FlashSparseMlaExecutorImpl （flash_sparse_mla_op 发射的）
  └── ...（61 层 + 通信 + 采样的所有 executor kernel node）
```

runtime 按图里记录的「`__global__` 指针 + 参数」把每个 executor kernel 重新 launch 一遍。**这一步没有任何 `*_op` 的 C++ 代码在跑**——op 只在 capture 那一趟当过"扣扳机的人"，replay 时子弹是直接从弹匣（graph）里重新打出去的。

这就是 CUDA Graph 零 CPU 调度开销的来源：capture 时 op 当了一次发射器把 launch 记下来，之后每 token replay 只剩一个 `cudaGraphLaunch`，省掉了 61 层 × 多个 op 的 CPU 调度 + grid/block 重算开销。

---

## 七、一句话总结

| 概念 | 是什么 | 在哪 |
|------|--------|------|
| **op**（`ops/*.py` + `torch.ops.tilert.*_op`） | host C++ 函数，capture 时当发射器 | CPU |
| **kernel**（`__global__` executor） | 真正在 GPU 上跑的设备函数 | GPU |
| **graph node** | capture 时一次 `cudaLaunchKernel` 的记录 | graph 里 |

**op ≠ kernel**。`cudaLaunchKernel` 启动的是 `__global__` executor kernel，不是 `*_op`。capture 时 `*_op` 在 CPU 上跑一遍，它内部每次 `cudaLaunchKernel(executor, ...)` 被 runtime 拦截，记录成 graph 里的一个 kernel node。replay 时只剩 `cudaGraphLaunch`，按记录把每个 executor kernel 重新 launch——`*_op` 已功成身退，不参与 replay。
