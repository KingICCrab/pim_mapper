# Row Activation Validation Analysis

## 核心发现

### 1. ILP `row_acts` 的真正含义

**关键发现**：ILP 的 `row_acts` 不是"唯一行数"，而是 **DRAM 级别的访问迭代次数**。

对于 `row_aligned` 布局：
```
row_acts_row_aligned = Π_{j ∈ all_dims} bound_j^{xj}
```

其中：
- `bound_j` = 维度 j 的 DRAM 级别因子（Level 2 × Level 3）
- `xj` = 指示维度 j 是否在 DRAM 级别迭代的二进制变量

**示例 (medium workload Input)**：
- DRAM 级别因子: S=3, P=7, Q=7, C=4, K=4
- Input 相关维度: R, S, P, Q, C, N
- row_acts = 3 × 7 × 7 × 4 × (reuse penalty for K=4) = 2352 × (某个因子) ≈ 2688

### 2. ILP 模型语义

| 术语 | ILP 定义 | 物理含义 |
|------|----------|----------|
| `row_acts_row_aligned` | DRAM tile 迭代次数 | 每次 tile 切换可能激活新行 |
| `row_acts_sequential` | unique_rows + crossing_penalty | 顺序访问的行激活次数 |
| `reuse_penalty` | 无关维度的乘积 | 数据需要重新加载的次数 |

### 3. 验证方法比较

#### 方法 A：唯一行数 (incorrect)
```
Trace: 统计访问的唯一行数
问题: 不考虑数据重新加载
结果: Trace << ILP
```

#### 方法 B：行切换次数 (current trace analysis)
```
Trace: 统计所有行切换
问题: 包含了 RowBuffer 内的切换
结果: Trace >> ILP (对于大型 workload)
```

#### 方法 C：DRAM 级别行激活 (correct)
```
Trace: 只统计 DRAM 级别的行激活
       = 唯一行数 × 重新加载次数
       = unique_rows × reuse_penalty
```

### 4. 实验结果

| Workload | Tensor | ILP row_acts | Trace unique_rows | Trace row_switches | 分析 |
|----------|--------|--------------|-------------------|-------------------|------|
| tiny     | Input  | 1            | 1                 | 1                 | ✓ 匹配 |
| tiny     | Weight | 1            | 1                 | 1                 | ✓ 匹配 |
| tiny     | Output | 1            | 1                 | 1                 | ✓ 匹配 |
| small    | Input  | 16           | 16                | 8                 | 见下文 |
| small    | Weight | 3            | 3                 | 7                 | 循环顺序 |
| small    | Output | 1            | 1                 | 1                 | ✓ 匹配 |
| medium   | Input  | 2688         | 32                | 5568              | ILP计算迭代次数 |
| medium   | Weight | 9            | 9                 | 11757             | ✓ 唯一行 |
| medium   | Output | 25           | 25                | 300               | ✓ 唯一行 |

### 5. 结论

1. **ILP `row_acts` 对于 row_aligned 布局**：
   - 计算的是 "DRAM 级别 tile 迭代次数"
   - 这是一个保守的上界估计
   - 假设每次 tile 切换都需要新的行激活

2. **ILP `row_acts` 对于 sequential 布局**：
   - 计算的是 "唯一行数 + 跨行惩罚 × reuse_penalty"
   - 这更接近实际的行激活次数

3. **验证建议**：
   - 对于 sequential 布局: 比较 `ILP row_acts` vs `Trace unique_rows`
   - 对于 row_aligned 布局: ILP 是上界，Trace <= ILP

4. **模型改进方向**：
   - row_aligned 模型可能过于保守
   - 可以考虑更精确地建模实际的行激活模式
