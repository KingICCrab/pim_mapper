# Input Row Activation (row_aligned 模式) 修复方案

## 问题背景

- **ILP 预测**: 2392
- **Trace 实际**: 5376
- **Root Cause**: 当前 ILP 模型没有正确处理 R/S 在 Buffer level 的情况，以及不同 crossing 类型的访问次数

---

## 最终方案

### 1. 核心公式

```
row_acts = C_factor × (
    reuse_penalty × (h_non × w_non) × 1 +
    K_factor × (h_crossing × w_non) × 2 +
    K_factor × (h_non × w_crossing) × 2 +
    K_factor × (h_crossing × w_crossing) × 4
)
```

**简化形式**:
```
row_acts = C_factor × (
    reuse_penalty × block_nums +
    2 × K_factor × (h_crossing × w_non + h_non × w_crossing) +
    4 × K_factor × h_crossing × w_crossing
)
```

### 2. 变量说明

| 变量 | 说明 |
|------|------|
| `reuse_penalty` | `K_factor^{xj[K]}`，已有代码正确实现 |
| `xj[K]` | K 是否有 relevant 内层循环：K 外层=1, K 内层=0 |
| `h_non` | H 方向不跨越 block 边界的 tile 数 |
| `h_crossing` | H 方向跨越 block 边界的 tile 数 |
| `w_non` | W 方向不跨越 block 边界的 tile 数 |
| `w_crossing` | W 方向跨越 block 边界的 tile 数 |

### 3. 不同 crossing 类型的 block 访问数

| 类型 | H 跨越 | W 跨越 | blocks 数 | K 内层激活次数 | K 外层激活次数 |
|------|--------|--------|-----------|----------------|----------------|
| Non-crossing | ✗ | ✗ | 1 | 1 | K |
| H-only | ✓ | ✗ | 2 | 2K | 2K |
| W-only | ✗ | ✓ | 2 | 2K | 2K |
| Both | ✓ | ✓ | 4 | 4K | 4K |

**关键洞察**: 只有 non-crossing tile 在 K 内层时可以复用，crossing tile 不管 K 在哪都要乘 K_factor。

### 4. 公式验证

**K 最外层 (reuse_penalty = K_factor)**:
```
= C × (K × h_non × w_non + 2K × h_crossing × w_non + 2K × h_non × w_crossing + 4K × h_crossing × w_crossing)
= C × K × (h_non + 2×h_crossing) × (w_non + 2×w_crossing)
= C × K × total_h_blocks × total_w_blocks  ✓
```

**K 最内层 (reuse_penalty = 1)**:
```
= C × (h_non × w_non + 2K × h_crossing × w_non + 2K × h_non × w_crossing + 4K × h_crossing × w_crossing)
= C × (non_crossing + K × all_crossing_blocks)  ✓
```

---

## 预计算表设计

### 不放入预计算表的变量
- **K_factor**: 在 ILP 中单独处理
- **C_factor**: 在 ILP 中单独处理
- **reuse_penalty**: 已有代码处理

### H 方向预计算表

```python
h_table[i_bh, i_P, i_R] = (h_non, h_crossing)
```

参数:
- `block_h`: H 方向 block 大小
- `P_factor`: P 维度因子 (DRAM level)
- `R_factor_total`: R 维度总因子 = R_dram × R_buffer

### W 方向预计算表

```python
w_table[i_bw, i_Q, i_S] = (w_non, w_crossing)
```

参数:
- `block_w`: W 方向 block 大小
- `Q_factor`: Q 维度因子 (DRAM level)
- `S_factor_total`: S 维度总因子 = S_dram × S_buffer

---

## 关键修正点

### 1. R/S 因子需要同时考虑 DRAM 和 Buffer 两个级别

```
R_factor_total = R_dram × R_buffer
S_factor_total = S_dram × S_buffer
```

**当前 bug**: 只使用 DRAM level 的 `xb` 变量，当 R/S 在 Buffer level 时 DRAM 的 R_factor=1，导致计算错误。

### 2. Tile 尺寸计算

```
P_tile = P / P_factor
R_tile = R / R_factor_total

H_tile = (P_tile - 1) × stride_h + (R_tile - 1) × dilation_h + 1
W_tile = (Q_tile - 1) × stride_w + (S_tile - 1) × dilation_w + 1
```

### 3. GCD 周期方法计算 crossing

使用 `g = gcd(step, block_size)`, `period = block_size / g`，枚举一个周期内的 crossing 位置，然后分解为完整周期 + 余数。

---

## ILP 实现要点

### 需要线性化的乘积项

1. `h_non × w_non`
2. `h_crossing × w_non`
3. `h_non × w_crossing`
4. `h_crossing × w_crossing`

### 变量来源

```python
# H 方向
block_h_var = vars.rowbuf_input_block_h[(w, i)]
P_var = vars.xb[(w, dram_level, s, P_dim, i)]
R_dram_var = vars.xb[(w, dram_level, s, R_dim, i)]
R_buf_var = vars.xb[(w, buf_level, s, R_dim, i)]  # 新增！

# W 方向
block_w_var = vars.rowbuf_input_block_w[(w, i)]
Q_var = vars.xb[(w, dram_level, s, Q_dim, i)]
S_dram_var = vars.xb[(w, dram_level, s, S_dim, i)]
S_buf_var = vars.xb[(w, buf_level, s, S_dim, i)]  # 新增！
```

---

## 测试用例

### ResNet-L1 配置

```
Workload: R=7, S=7, P=56, Q=56, C=3, K=64, N=1
DRAM loops: K=4, C=3, P=28, Q=7, R=7 (Level 2), S=1 (in buffer)
Buffer: P=2, Q=8, R=1, S=7
Block Size: 31 × 31
```

**预期结果**: 5376 row activations
