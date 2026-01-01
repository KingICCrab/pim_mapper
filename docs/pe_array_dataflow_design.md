# PE Array Dataflow 设计文档

## 1. 问题背景

我们需要在 ILP 模型中支持 2D PE Array，并正确处理：
1. **广播 (Broadcast)**: 相同数据复用到多个 PE
2. **规约 (Reduction)**: 多个 PE 的部分和累加为最终结果
3. **带宽约束**: 内存到 PE 阵列的数据传输限制

## 2. 核心概念

### 2.1 卷积的 7 个维度

| 维度 | 符号 | 含义 | 典型范围 |
|-----|-----|------|---------|
| R | FX | 卷积核高度 | 1-7 |
| S | FY | 卷积核宽度 | 1-7 |
| P | OX | 输出高度 | 7-224 |
| Q | OY | 输出宽度 | 7-224 |
| C | IC | 输入通道数 | 3-2048 |
| K | OC | 输出通道数 | 64-2048 |
| N | ON | Batch 大小 | 1-256 |

### 2.2 数据类型与维度相关性矩阵 O[j][t]

```
             Input  Weight  Output
          t=  0      1       2
    R (j=0)   1      1       0     ← R 与 Output 无关 (reduction axis)
    S (j=1)   1      1       0     ← S 与 Output 无关 (reduction axis)
    P (j=2)   1      0       1
    Q (j=3)   1      0       1
    C (j=4)   1      1       0     ← C 与 Output 无关 (reduction axis)
    K (j=5)   0      1       1
    N (j=6)   1      0       1
```

**关键洞察:**
- **O[j][t] = 1**: 维度 j 与数据类型 t 相关 → 不同 j 值需要不同的 t 数据
- **O[j][t] = 0**: 维度 j 与数据类型 t 无关 → 可以广播同一份 t 数据

### 2.3 Reduction Axes (规约轴)

对于 Output (t=2), 规约轴是 O[j][2] = 0 的维度:
- **R, S, C** 是规约轴
- 计算: `Output[n,k,p,q] = Σ_{r,s,c} Input[n,c,p+r,q+s] × Weight[k,c,r,s]`

当 `parallel_for` 在规约轴上时，需要累加多个 PE 的部分和。

## 3. 2D PE Array 模型

### 3.1 PE Array 结构

```
              ←── W direction (columns) ──→
         ┌────┬────┬────┬────┐
     ↑   │PE  │PE  │PE  │PE  │  ← Row 0
     │   │0,0 │0,1 │0,2 │0,3 │
     H   ├────┼────┼────┼────┤
    dir  │PE  │PE  │PE  │PE  │  ← Row 1
     │   │1,0 │1,1 │1,2 │1,3 │
     ↓   └────┴────┴────┴────┘
             Col0 Col1 Col2 Col3
```

### 3.2 广播规则

| 数据映射方向 | 广播方向 | 带宽需求 |
|------------|---------|---------|
| H 方向并行 | W 方向广播 | = H (不是 H×W) |
| W 方向并行 | H 方向广播 | = W (不是 H×W) |
| H 和 W 都并行 | 无法广播 | = H × W |

**例子:** H=16, W=16
- K 映射到 H: 需要 16 个不同的 Weight (按 K 索引)，每个广播到 16 列
- P 映射到 W: 需要 16 个不同的 Input (按 P 索引)，每个广播到 16 行

### 3.3 ILP 变量设计

```python
# 原有变量
xb[w, m, l, j, i]  # 选择 workload w, 内存层 m, 循环层 l, 维度 j 的第 i 个因子

# 新增 2D 空间映射变量
xb_h[w, m, j, i]   # 维度 j 映射到 H 方向的因子
xb_w[w, m, j, i]   # 维度 j 映射到 W 方向的因子

# 约束
xb_h[j] × xb_w[j] = xb_spatial[j]     # H和W的乘积等于总空间并行度
∏_j xb_h[j] ≤ PE_H                     # H方向总并行度不超过阵列高度
∏_j xb_w[j] ≤ PE_W                     # W方向总并行度不超过阵列宽度
```

## 4. 带宽模型

### 4.1 带宽需求计算

对于数据类型 t，从内存层 m 到 PE 阵列的带宽需求:

```
BW_demand[t] = ∏_{j: O[j][t]=1 且 xb_h[j]>1} xb_h[j]   (H方向)
             × ∏_{j: O[j][t]=1 且 xb_w[j]>1} xb_w[j]   (W方向)
```

**简化理解:**
- 只有 **相关维度** 会增加带宽需求
- 无关维度可以广播，不增加带宽

### 4.2 带宽约束

```python
# Log 域约束 (线性化)
Σ_{j: O[j][t]=1} xb_h[j] × log(div[j]) ≤ log(BW_limit)
Σ_{j: O[j][t]=1} xb_w[j] × log(div[j]) ≤ log(BW_limit)
```

## 5. 规约 (Reduction) 模型

### 5.1 何时需要规约?

当 `parallel_for` 在规约轴 (R, S, C) 上时:
- 每个 PE 计算不同 C 值的部分和
- 需要累加所有部分和得到最终 Output

### 5.2 规约策略

| 策略 | 延迟 | 能耗 | 硬件需求 |
|-----|-----|-----|---------|
| **Reduction Tree** | O(log₂ N) | 低 | 需要硬件加法树 |
| **Systolic Flow** | O(N) | 中 | PE 间直连 |
| **Buffer Reduction** | O(2N) | 高 | 只需 buffer |

### 5.3 ILP 约束

```python
# 规约并行度计算 (Log 域)
h_reduction = Σ_{j∈{R,S,C}} xb_h[j] × log(div[j])
w_reduction = Σ_{j∈{R,S,C}} xb_w[j] × log(div[j])

# 如果有硬件规约树 (深度 D):
h_reduction ≤ log(2^D)  # = D × log(2)
w_reduction ≤ log(2^D)
```

### 5.4 规约代价加入目标函数

```python
# 规约延迟代价
reduction_latency = reduction_depth × base_latency
# 其中 reduction_depth = ceil(log₂(reduction_parallelism))
```

## 6. 计算单元类型

### 6.1 三种计算单元

| 类型 | 内部结构 | 规约处理 | 代表架构 |
|-----|---------|---------|---------|
| **Scalar PE** | 1×1 MAC | 外部规约 | Eyeriss |
| **Systolic Array** | N×N MAC, PE间直连 | Systolic flow | TPU |
| **Tensor Core** | M×N×K MAC + reduction tree | 内部规约 | NVIDIA |

### 6.2 Tensor Core 模型

```
Tensor Core (16×16 with 16-way reduction):
  - Input:  A[16×16], B[16×16]
  - Output: C[16×16] = A × B (矩阵乘)
  - 内部: 每个 Output 元素由 16 个 MAC 的结果累加
  - Dataflow: Output Stationary + 内部 Reduction Tree
```

## 7. Dataflow 映射表

### 7.1 单维度映射 → 经典 Dataflow

| H 方向 | W 方向 | 等效 Dataflow | 说明 |
|-------|-------|--------------|------|
| K | P,Q,N | Weight Stationary | Weight 在 H 复用 |
| P,Q,N | K | Output Stationary | Output 在 W 复用 |
| C | K | Input Stationary | Input 在 W 复用 |
| R,S | K | Row Stationary | Filter 行在 H 复用 |

### 7.2 多维度映射

```
H = {K}     :  H 方向按 K 并行
W = {P, Q}  :  W 方向按 P×Q 并行 (2D 输出空间)

结果:
  - Weight: K 相关 → H 方向各不相同，W 方向可广播
  - Output: K,P,Q 相关 → 每个 PE 有唯一的 Output tile
  - Input: P,Q 相关 → W 方向各不相同，H 方向可广播
```

### 7.3 同一维度拆分到 H 和 W

```
例: C = 256, H = 16, W = 16
拆分: C_H = 16 (H方向), C_W = 16 (W方向)

结果:
  - 每个 PE 处理 C/256 = 1 个 channel
  - 需要 2D Reduction: 先 H 方向规约，再 W 方向规约
  - 规约深度 = log₂(16) + log₂(16) = 8 stages
```

## 8. 代码结构

```
src/pim_optimizer/
├── arch/
│   ├── memory.py        # MemoryLevel 带宽参数
│   └── pim_arch.py      # PEArray, ComputeUnit 定义
├── model/
│   ├── bandwidth.py     # 2D 带宽约束和规约处理
│   ├── variables.py     # ILP 变量 (xb_h, xb_w)
│   └── constraints.py   # 总约束
└── workload/
    └── conv.py          # 卷积相关性矩阵 O
```

## 9. 下一步计划

1. **完善 ComputeUnit 类**: 添加 reduction_tree_depth, reduction_latency 参数
2. **集成到 optimizer.py**: 在 ILP 模型构建中调用 bandwidth 约束
3. **添加规约代价到目标函数**: 作为延迟或能耗的一部分
4. **测试 Dataflow 探索**: 验证 ILP 能找到不同的最优 dataflow

## 10. 参考

- **Interstellar-CNN-scheduler**: 用 `para_loop_dim` 和 `access_mode` 处理并行
- **Timeloop**: 用 "temporal" 和 "spatial" 显式区分循环类型
- **MAESTRO**: 用 dataflow directive 直接指定 H/W 映射
