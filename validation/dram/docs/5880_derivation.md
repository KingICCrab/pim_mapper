# 5880 Input Row Activations 完整量化推导

## 问题背景
- **Trace 结果**: 5880 row activations
- **ILP 预测**: 2392 row activations
- **差异**: 5880 - 2392 = 3488 (146% 误差)

## 参数配置

### Workload (ResNet-L1)
- R=7, S=7, P=56, Q=56, C=3, K=64, N=1
- H_in=62, W_in=62

### DRAM Level 因子
- **Level 3**: K_l3=4, C_l3=3, P_l3=28, Q_l3=7
- **Level 2**: R_l2=7

### 数据布局
- block_h=31, block_w=31
- row_size=1024 elements

### Tile 大小
- H_per_tile = P / P_l3 = 56 / 28 = **2**
- W_per_tile = Q / Q_l3 = 56 / 7 = **8**

## 关键发现

### 1. Row 地址公式

```
row = h_blk × 7 + w_blk + c × 196
```

基于:
- stride_p_l3 = 7168 (= 7 × row_size)
- stride_q_l3 = 1024 (= row_size)
- stride_c_l3 = 200704 (= 196 × row_size)

### 2. trace_generator 中 R 不产生 h 滑动

```python
h_start = p_tile × H_per_tile = p_tile × 2
h_end = h_start + H_per_tile = h_start + 2
```

**每个 (P, Q, R) 组合访问相同的 2 个 h 坐标！**

这与 ILP 的假设 (R 迭代使 h 滑动) 不同，是差异的主要来源。

### 3. P tiles 的 h_blk 分类

| P 范围 | h 范围 | h_blk | 数量 |
|--------|--------|-------|------|
| 0-14 | [0, 30) | 0 only | 15 |
| 15 | [30, 32) | {0, 1} | 1 (H-crossing) |
| 16-27 | [32, 56) | 1 only | 12 |

### 4. Q tiles 的 w_blk 分类

| Q 值 | w 范围 | w_blk | 类型 |
|------|--------|-------|------|
| 0-2 | [0, 24) | 0 only | 3 tiles |
| 3 | [24, 32) | {0, 1} | W-crossing |
| 4-6 | [32, 56) | 1 only | 3 tiles |

## 量化推导

### Per (K, C) 的 Activation 分布

```
(h_blk=0, w_blk=0): 133
(h_blk=0, w_blk=1): 133
(h_blk=1, w_blk=0): 112
(h_blk=1, w_blk=1): 112
────────────────────────
Total per (K, C): 490
```

### 最终计算

```
Total = 490 × K_l3 × C_l3
      = 490 × 4 × 3
      = 5880 ✓
```

### 532 vs 448 的来源

```
h_blk=0 每 row: (133 + 133) × 4 / 2 = 532
h_blk=1 每 row: (112 + 112) × 4 / 2 = 448
```

差异 84 = (15 - 12) × 7 × 4 / 1 (P tiles 数量不对称)

## Row Activation 分布验证

| Row | h_blk | w_blk | c | Count |
|-----|-------|-------|---|-------|
| 0 | 0 | 0 | 0 | 532 |
| 1 | 0 | 1 | 0 | 532 |
| 7 | 1 | 0 | 0 | 448 |
| 8 | 1 | 1 | 0 | 448 |
| 196 | 0 | 0 | 1 | 532 |
| 197 | 0 | 1 | 1 | 532 |
| 203 | 1 | 0 | 1 | 448 |
| 204 | 1 | 1 | 1 | 448 |
| 392 | 0 | 0 | 2 | 532 |
| 393 | 0 | 1 | 2 | 532 |
| 399 | 1 | 0 | 2 | 448 |
| 400 | 1 | 1 | 2 | 448 |

**总计**: (532 × 6) + (448 × 6) = 3192 + 2688 = **5880** ✓

## 公式总结

```
Total Row Activations = K_l3 × C_l3 × Σ(P,Q) [num_block_switches]

其中 num_block_switches 取决于:
1. P tile 是否跨越 h_blk 边界 (H-crossing)
2. Q tile 是否跨越 w_blk 边界 (W-crossing)
3. R 迭代之间的 row 切换 (每个 R 迭代访问相同的 h 范围)
```

## ILP 模型改进建议

当前 ILP 假设 R 迭代使 h 坐标滑动，但 trace_generator 的实际行为是:
- **buffer_tile[R] = 1**，所以每个 tile 的 h 范围是固定的
- R 迭代在同一个 (P, Q) tile 内重复，不改变 h 坐标

建议修改 ILP 模型以考虑:
1. 实际的 buffer tile 大小
2. H-crossing 和 W-crossing 带来的额外 row activations
3. R 迭代不产生 h 滑动的事实
