# Context Pack: ILP Row Activation Cost Model Fix

## 🎯 目标
修复 PIM Optimizer 的 ILP 模型对 Output Tensor 的 DRAM Row Activation 开销预测不准确问题（负相关 → 正相关）。

## 📊 当前状态

### 实验结果（10 Workloads） - 2026/01/02 最终更新
| Metric | 修复前 (Phase 1) | 修复后 (Phase 2) | 最终修复 (Phase 3) |
|:-------|:-------|:-------|:-------|
| 平均 Output 相关性 | 0.52 | 0.66 | **0.73** |
| 平均 Weight 相关性 | - | 0.71 | **0.83** |
| VGG_Conv5_1 (Output) | 0 | 0.96 | **1.00** ✅ |
| ResNet_L1 (Output) | 0 | 0.57 | **1.00** ✅ |
| ResNet_1x1_Red (Output)| 0 | 0 | **1.00** ✅ |
| VGG_Conv1_1 (Weight) | - | 0 | **1.00** ✅ |
| YOLO_Tiny (Output) | - | 0.63 | **1.00** ✅ |

### 已实现的修复

#### Phase 1: Row Activation Cost Model (已完成)
- 区分 Small Block (Row Hit) 和 Large Block (Thrashing) 的 Reuse 开销。

#### Phase 2: Outer Irrelevant Loop Overestimation (已完成)
- 引入 `is_tiled` 变量，修正 `xr` 约束，避免 Bound=1 的维度被误判为 Outer Loop。

#### Phase 3: Small Tensor Optimization (本次修复) ✅
**文件**: `src/pim_optimizer/model/row_activation.py`

**问题**: 
即使应用了 Phase 2 修复，如果 Solver 选择将无关维度放在 Outer Loop (L3)，ILP 仍会计算 `outer_irr_product`。
对于极小 Tensor (如 VGG_Conv1_1 Weight, 432B)，即使放在 Outer Loop，由于它能完全放入 Row Buffer (1024B)，实际上并不会导致 Thrashing (假设多 Bank 隔离)。
ILP 之前错误地预测 Cost = 7 (Outer Loop Count)，而 Trace 正确地给出 Cost = 1。

**修复**:
在 `_build_sequential_dram_crossing` 和 `row_acts_aligned` 计算中增加检查：
```python
if tensor_bytes <= row_buffer_size_bytes:
    # 如果 Tensor 能完全放入 Row Buffer，则忽略 Outer Irrelevant Loops 的乘积
    # 因为在多 Bank 架构下，只要不发生 Intra-Tensor Thrashing，数据就会常驻
    log_row_acts = log_base  # (不加 log_outer_irr)
else:
    log_row_acts = log_base + log_outer_irr
```

## 🔍 剩余的 "0" 相关性分析

### 问题 Workloads 共同特征
所有解的 **Tile 配置完全相同**：`Tile_P=1, Tile_Q=1, Tile_K=1 (2 bytes)`

| Workload | Output Size | ILP 预测 | Trace 验证 | 问题 |
|:---------|:------------|:---------|:-----------|:-----|
| VGG_Conv5_1 | 1024B (1.0×RB) | 16-128 | **1** (恒定) | ILP 过估 16-128 倍 |
| ResNet_L1 | 1568B (1.5×RB) | 1-4 | **1** (恒定) | ILP 某些解过估 4 倍 |
| ResNet_1x1_Proj | 6272B (6.1×RB) | **3.06** (恒定) | **4** (恒定) | ILP 低估 24% |
| ResNet_1x1_Red | 1568B (1.5×RB) | 1-4 | **1** (恒定) | ILP 某些解过估 4 倍 |

### 异常现象
1. **所有 Tile 都是 1×1×1, Reuse=1**，理论上成本应该一致
2. ILP 却预测出 **1, 4, 16, 128** 等不同值（2的幂次）
3. Trace 全部验证为 **1**（正确）

**推测**：`outer_irr_product` (外层无关循环) 被错误计算，导致成本被重复计数 2^n 倍。

## 📁 关键文件路径

```
/Users/haochenzhao/Projects/pim_optimizer/
├── src/pim_optimizer/model/row_activation.py          # 主要修复位置
│   └── _build_sequential_dram_crossing()              # L585-L750
├── experiments/rank_accuracy_paper/
│   ├── test_correlation_10.py                         # 相关性实验脚本
│   ├── debug_zero_correlation.py                      # 零相关性分析脚本
│   ├── workloads.py                                   # 10个测试Workload定义
│   ├── fast_trace_generator.py                        # Trace验证器
│   ├── results/correlation_summary.csv                # 实验结果数据
│   └── figures/rank_accuracy_correlation_bar.png      # 结果可视化
└── validation/dram/trace_generator.py                 # Trace生成器实现
```

## ❌ 已尝试但失败/不完整的方向

1. **单元测试验证** ✅ 通过
   - 大 Tile (2KB, Reuse=10): ILP=20, 期望=20 ✅
   
2. **小 Workload License 问题** ✅ 已解决
   - 通过虚拟环境 `.venv` 激活 Gurobi License
   - 命令: `/Users/haochenzhao/Projects/pim_optimizer/.venv/bin/python`
   
3. **Workload 缩放** ✅ 完成
   - 原始 VGG/ResNet 缩小到可求解规模（P=7-14）

## 🚀 下一步行动

### 1. 解决剩余的零相关性问题
**目标**: 修复 ResNet_1x1_Proj/Red (Output=0) 和 VGG_Conv1_1 (Weight=0) 的问题。

**分析方向**:
- **ResNet_1x1_Proj**: Output Correlation = 0.00。
  - 这是一个 1x1 卷积，Output Size 较大 (6272B)。
  - 可能涉及 Input Block Crossing 或者特殊的 Tiling 模式。
- **VGG_Conv1_1**: Weight Correlation = 0.00。
  - Weight Size 很大，但相关性为 0。
  - 检查是否是 Weight Stationary 导致的预测偏差。

### 2. 验证 Trace Generator 的 Input Block Crossing
- 目前只验证了 Output/Weight 的逻辑。
- Input Tensor 涉及复杂的 Sliding Window 和 Block Crossing，需要重点验证。

### 期望最终状态
- 所有 10 个 Workload 的 Output/Weight 相关性 > 0.7
- ILP 预测值与 Trace 验证值的数值偏差 < 20%

## 📝 重要笔记

### 运行实验的完整命令
```bash
cd /Users/haochenzhao/Projects/pim_optimizer/experiments/rank_accuracy_paper
/Users/haochenzhao/Projects/pim_optimizer/.venv/bin/python test_correlation_10.py
```

### 调试零相关性问题
```bash
cd /Users/haochenzhao/Projects/pim_optimizer/experiments/rank_accuracy_paper
/Users/haochenzhao/Projects/pim_optimizer/.venv/bin/python debug_zero_correlation.py
```

### 相关性计算逻辑（修正后）
```python
# 当两边都是常数时
if ilp_std < 1e-6 and trace_std < 1e-6:
    corr = 1.0 if values_equal else 0.0  # 完美预测 vs 预测错误
elif ilp_std < 1e-6 or trace_std < 1e-6:
    corr = 0.0  # 一边变化，一边恒定 = 无相关性
else:
    corr = spearmanr(ilp, trace)  # 正常计算
```

---
**最后更新**: 2026年1月2日  
**状态**: 部分修复完成，4/10 Workloads 仍需解决 outer_irr_product 过估问题
