# 混合分区全局优化 ILP 模型 V2

## 完整数学模型

### 1. 符号定义

#### 1.1 集合与索引
- $L = \{1, ..., n\}$: 网络层集合
- $C_l$: 第 $l$ 层的候选混合分区方案集合
- $D = \{\text{OUTP}, \text{OFMP}, \text{BATP}, \text{INPP}\}$: 分区维度集合

#### 1.2 层参数
- $K_l$: 第 $l$ 层输出通道数
- $C_l$: 第 $l$ 层输入通道数
- $H_l, W_l$: 第 $l$ 层输出空间维度
- $R_l, S_l$: 第 $l$ 层卷积核尺寸
- $N$: Batch 大小
- $O_l = N \cdot K_l \cdot H_l \cdot W_l$: 第 $l$ 层输出数据量

#### 1.3 分区方案表示 (对齐 nn_dataflow)
每个混合分区方案 $c \in C_l$ 定义为:
$$c = \{(p_d^h, p_d^w) : d \in D\}$$

其中 $(p_d^h, p_d^w)$ 是维度 $d$ 在物理节点阵列上的 H 和 W 方向分区因子。

**约束**: 总节点数必须等于物理节点阵列大小
$$\prod_{d \in D} (p_d^h \cdot p_d^w) = N_H \cdot N_W$$

### 2. 决策变量

$$x_{l,c} \in \{0, 1\}, \quad \forall l \in L, c \in C_l$$

$x_{l,c} = 1$ 当且仅当第 $l$ 层选择混合分区方案 $c$。

### 3. 约束条件

#### 3.1 唯一性约束
每层恰好选择一个方案:
$$\sum_{c \in C_l} x_{l,c} = 1, \quad \forall l \in L$$

### 4. 目标函数

最小化总成本:
$$\min \sum_{l \in L} \sum_{c \in C_l} \text{Compute}_l(c) \cdot x_{l,c} + \sum_{l=1}^{n-1} \sum_{c_i \in C_l} \sum_{c_j \in C_{l+1}} \text{Redist}_{l,l+1}(c_i, c_j) \cdot x_{l,c_i} \cdot x_{l+1,c_j}$$

### 5. 二次项线性化 (McCormick Envelope)

引入辅助变量 $y_{l,c_i,c_j}$ 表示 $x_{l,c_i} \cdot x_{l+1,c_j}$:

$$\begin{aligned}
y_{l,c_i,c_j} &\leq x_{l,c_i} \\
y_{l,c_i,c_j} &\leq x_{l+1,c_j} \\
y_{l,c_i,c_j} &\geq x_{l,c_i} + x_{l+1,c_j} - 1 \\
y_{l,c_i,c_j} &\geq 0
\end{aligned}$$

### 6. 计算成本模型

$$\text{Compute}_l(c) = \frac{\text{MACs}_l}{\prod_{d \in D} (p_d^h \cdot p_d^w)} \cdot \alpha_{\text{INPP}}(c) \cdot \alpha_{\text{OFMP}}(c)$$

其中:
- $\alpha_{\text{INPP}}(c) = 1 + 0.1 \cdot (p_{\text{INPP}} - 1)$: INPP reduction 开销
- $\alpha_{\text{OFMP}}(c)$: Halo exchange 开销

### 7. 完整重分布成本模型

**关键创新**: 考虑所有维度的传播

$$\text{Redist}_{l,l+1}(c_i, c_j) = R_K(c_i, c_j) + R_{HW}(c_i, c_j) + R_N(c_i, c_j) + R_C(c_i)$$

#### 7.1 K→C 传播成本 ($R_K$)

| $p_{\text{OUTP}}^{(l)}$ | $p_{\text{INPP}}^{(l+1)}$ | 通信模式 | 成本 |
|-------------------------|---------------------------|----------|------|
| 1 | 1 | None | 0 |
| 1 | >1 | Scatter | $O_l \cdot \frac{p-1}{p}$ |
| >1 | =K分区 | Perfect match | 0 |
| >1 | 1 | All-Gather | $O_l \cdot \frac{k-1}{k}$ |
| >1 | ≠K分区 | All-to-All | $O_l \cdot (1 - \frac{1}{\max(k,p)}) \cdot 1.5$ |

#### 7.2 空间传播成本 ($R_{HW}$)

$$R_{HW}(c_i, c_j) = R_{\text{spatial\_mismatch}} + R_{\text{halo}}$$

**空间不匹配**:
$$R_{\text{spatial\_mismatch}} = \begin{cases}
0 & \text{if } (p_H^{(l)}, p_W^{(l)}) = (p_H^{(l+1)}, p_W^{(l+1)}) \\
O_l \cdot 1.5 \cdot (1 - \frac{1}{\max(n_l, n_{l+1})}) & \text{otherwise}
\end{cases}$$

**Halo exchange**:
$$R_{\text{halo}} = N \cdot C_{l+1} \cdot [(R_{l+1}-1) \cdot W_{l+1} + (S_{l+1}-1) \cdot H_{l+1}] \cdot 0.5$$

#### 7.3 Batch 传播成本 ($R_N$)

$$R_N(c_i, c_j) = \begin{cases}
0 & \text{if } p_{\text{BATP}}^{(l)} = p_{\text{BATP}}^{(l+1)} \\
O_l \cdot 1.5 \cdot (1 - \frac{1}{\max(b_l, b_{l+1})}) & \text{otherwise}
\end{cases}$$

#### 7.4 INPP Reduction 成本 ($R_C$)

当前层的 INPP 分区需要 All-Reduce:
$$R_C(c_i) = \begin{cases}
0 & \text{if } p_{\text{INPP}}^{(l)} = 1 \\
O_l \cdot 2 \cdot \frac{p-1}{p} & \text{otherwise (Ring All-Reduce)}
\end{cases}$$

---

## 与 nn_dataflow 的对齐

### PartitionScheme 映射
```python
# nn_dataflow PartitionScheme
scheme = PartitionScheme(
    order=(pe.OFMP, pe.BATP, pe.OUTP, pe.INPP),
    pdims=(PhyDim2(2, 2), PhyDim2(2, 2), PhyDim2(1, 1), PhyDim2(1, 1))
)

# 对应的 HybridPartitionChoice
choice = HybridPartitionChoice({
    PartDim.OUTP: (1, 1),   # pdims[pe.OUTP]
    PartDim.OFMP: (2, 2),   # pdims[pe.OFMP]
    PartDim.BATP: (2, 2),   # pdims[pe.BATP]
    PartDim.INPP: (1, 1),   # pdims[pe.INPP]
})
```

### gen_partition 枚举逻辑
```
For 4×4 array (16 nodes):
16 = 16×1×1×1 or 8×2×1×1 or 4×4×1×1 or 4×2×2×1 or 2×2×2×2 or ...

每种分解对应不同的混合分区方案
```

---

## 算法复杂度分析

设 $n$ 为层数，每层最多 $m$ 个候选方案:

- **变量数**: $O(nm + nm^2) = O(nm^2)$
- **约束数**: $O(n + nm^2) = O(nm^2)$
- **最坏时间**: NP-hard，但实际通过剪枝和求解器可快速求解

---

## 参考文献

1. **LEMON**: Layer-wise Execution Model Optimization for DNNs
2. **COSA**: Communication-Optimal Scheduling Algorithm
3. **nn_dataflow**: Neural Network Dataflow Mapping Framework
