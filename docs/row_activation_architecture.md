# Row Activation Model Architecture

本文档描述 `row_activation.py` 模块的架构设计。

## 模块结构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       build_row_activation_model                            │
│                              (主入口)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              ┌─────────┐    ┌─────────┐    ┌─────────┐
              │ Input   │    │ Weight  │    │ Output  │
              │ (t_id=0)│    │ (t_id=1)│    │ (t_id=2)│
              └────┬────┘    └────┬────┘    └────┬────┘
                   │              │              │
                   └──────────────┼──────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: reuse_penalty = Π_{j∈irrelevant} bound_j^{xj}                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────┐    ┌────────────────────────┐                 │
│  │ _build_log_product_expr  │───▶│ _build_exp_var_from_log│                 │
│  │  (irrelevant_dims, "rp") │    │  (reuse_penalty)       │                 │
│  └──────────────────────────┘    └────────────────────────┘                 │
│         │                                                                   │
│         ▼  二进制×二进制线性化                                                │
│    z[j,i] = xj × xb[j,i]  →  log_expr = Σ z[j,i] × log(div_i)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1.5: row_acts_row_aligned = Π_{j} bound_j^{xj}                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────┐    ┌────────────────────────┐                 │
│  │ _build_log_product_expr  │───▶│ _build_exp_var_from_log│                 │
│  │  (all_dims, "aligned")   │    │  (row_acts_row_aligned)│                 │
│  └──────────────────────────┘    └────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 2-3: Sequential DRAM Row Crossing                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────┐                                         │
│  │ _build_sequential_dram_crossing│                                         │
│  └───────────────┬────────────────┘                                         │
│                  │                                                          │
│                  ▼                                                          │
│  ┌────────────────────────────────┐                                         │
│  │ precompute_tile_crossing_info  │  预计算 crossing/non-crossing 次数       │
│  └────────────────────────────────┘                                         │
│                  │                                                          │
│                  ▼                                                          │
│  row_acts = Σ xu[k] × non_cross[k] + 2 × reuse_penalty × Σ xu[k] × cross[k]│
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 3.5: Input Block Crossing (仅 Input)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────┐                                        │
│  │ _build_input_block_crossing_acts│ ◀─── 仅当 t_id == 0 (Input)            │
│  └───────────────┬─────────────────┘                                        │
│                  │                                                          │
│                  ▼                                                          │
│  ┌─────────────────────────────────┐    ┌───────────────────────────────┐   │
│  │ build_input_block_crossing_expr │───▶│precompute_input_block_crossing│   │
│  └─────────────────────────────────┘    │        _table                 │   │
│                                         └───────────────────────────────┘   │
│  crossing_acts = 2 × selected_count × reuse_penalty (McCormick envelope)   │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 4: Layout Conditional Combination                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────┐                                       │
│  │ _build_layout_conditional_acts   │                                       │
│  └──────────────────────────────────┘                                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  if row_aligned_var exists:                                         │    │
│  │    total = (1-row_aligned) × seq + row_aligned × aligned + crossing │    │
│  │  else:                                                              │    │
│  │    total = seq + crossing  (sequential only)                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 5: Row Activation Cycles                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    row_act_cycles = total_acts × macs_scale_factor × activation_latency     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 5.5: Data Transfer Cycles (Row Buffer Bandwidth Model)                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────┐                                       │
│  │ _build_data_transfer_cycles      │                                       │
│  └──────────────────────────────────┘                                       │
│                                                                             │
│  data_bytes = mem_reads × element_bytes                                     │
│  data_transfer_cycles = data_bytes / rowbuffer_bandwidth                    │
│                                                                             │
│  Uses PWL exp() to convert log-space mem_reads_inst to actual reads         │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 6: Total DRAM Cycles                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    total_dram_cycles = row_act_cycles + data_transfer_cycles                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                        ┌─────────────────┐
                        │  total_cycles   │
                        │    (返回值)      │
                        └─────────────────┘
```

## 辅助函数依赖关系

```
_build_log_product_expr ──────┐
       │                      │
       │  (log_expr)          │
       ▼                      │
_build_exp_var_from_log ◀─────┘
       │
       │  (exp_var: reuse_penalty / row_acts_aligned)
       ▼
┌──────┴──────┐
│             │
▼             ▼
_build_sequential_dram_crossing    _build_input_block_crossing_acts
       │                                      │
       │  (row_acts_seq)                      │  (block_crossing_acts)
       └──────────────┬───────────────────────┘
                      ▼
         _build_layout_conditional_acts
                      │
                      │  (total_acts)
                      ▼
                   cycles
```

## 函数职责表

| 函数名 | 职责 | 输入 | 输出 |
|--------|------|------|------|
| `_build_log_product_expr` | 构建 log(Π bound_j^{xj}) | dims, var_prefix | log_expr, max_product |
| `_build_exp_var_from_log` | 从 log 创建 exp 变量 | log_expr, max_value | exp_var |
| `_build_sequential_dram_crossing` | Sequential 模式 DRAM 跨行 | tile_info, reuse_penalty | row_acts, ub |
| `_build_input_block_crossing_acts` | Input Block 跨块开销 | workload, reuse_penalty | crossing_acts, max |
| `_build_layout_conditional_acts` | 布局条件组合 | seq, aligned, crossing | total_acts, ub |
| `_compute_tensor_bytes` | 计算 tensor 字节数 | workload, t_id | bytes |
| `_build_data_transfer_cycles` | Row Buffer 带宽数据传输延迟 | mem_reads_inst, bandwidth | transfer_cycles, ub |

## DRAM 延迟模型

```
Total_DRAM_Cycles = Row_Activation_Cycles + Data_Transfer_Cycles

1. Row Activation Cycles:
   row_act_cycles = row_acts × scale × activation_latency
   其中 activation_latency = tRCD + tRP (典型值 ~25 cycles)

2. Data Transfer Cycles (Row Buffer 带宽模型):
   data_transfer_cycles = data_bytes / rowbuffer_bandwidth
   其中:
   - data_bytes = mem_reads × element_bytes
   - rowbuffer_bandwidth = bytes/cycle (从 RowBuffer 层获取)
```

## Crossing 类型与布局模式

| 布局模式 | DRAM Row Crossing | Input Block Crossing |
|----------|-------------------|---------------------|
| Sequential | ✅ 存在 (tile vs row) | ✅ 仅 Input 存在 |
| Row-aligned | ❌ = 0 (block 对齐) | ✅ 仅 Input 存在 |

## ILP 公式

### Input
```
total = (1 - row_aligned) × DRAM_Row_Crossing_seq 
      + row_aligned × row_acts_row_aligned 
      + Input_Block_Crossing
```

### Weight / Output
```
total = (1 - row_aligned) × DRAM_Row_Crossing_seq 
      + row_aligned × row_acts_row_aligned
```

## 线性化技术

### 1. 二进制×二进制线性化 (无 Big-M)

用于 `xj × xb[j,i]` 乘积：
```
z = xj × xb[j,i]

约束:
  z ≤ xj
  z ≤ xb[j,i]
  z ≥ xj + xb[j,i] - 1
  z ∈ {0, 1}
```

**优势**: LP 松弛紧，数值稳定，求解效率高。

### 2. McCormick Envelope

用于连续变量乘积 `x × y` (如 `selected_count × reuse_penalty`)：
```
设 x ∈ [x_L, x_U], y ∈ [y_L, y_U]
aux = x × y

约束:
  aux ≥ x_L × y + x × y_L - x_L × y_L
  aux ≥ x_U × y + x × y_U - x_U × y_U
  aux ≤ x_U × y + x × y_L - x_U × y_L
  aux ≤ x_L × y + x × y_U - x_L × y_U
```

### 3. PWL (分段线性) 近似

用于 `exp()` 函数：
```python
model.addGenConstrExp(log_var, exp_var, options="FuncPieces=-2 FuncPieceError=0.002")
```
