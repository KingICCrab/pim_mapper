# ILP Model Invariants - 不可违反的设计原则

本文档记录 ILP 模型的核心设计原则。**任何代码修改前必须检查是否违反这些原则。**

---

## 1. xj 变量语义

**定义**: `xj[w, t, m, m_, j]` 表示维度 j 是否在某个 position p 有内层循环，且该 position 对 tensor t 有 relevant inner loop。

**关键原则**: 
- ❌ **xj 不依赖于 O[j][t] (relevancy)**
- ✅ xj 只依赖于 permutation 和 xr

**原因**:
```
for Q (irrelevant to Weight)    ← Q 改变
  for S (relevant to Weight)    ← S 重新从 0 开始
    access Weight               ← Weight 被重复访问 Q 次
```

即使 Q 对 Weight 无关，Q 的外层循环会导致内层 S 循环重复执行，Weight 被多次访问。
这就是 **reuse penalty** 的来源。

**错误修改记录**:
- 2024-12-24: 错误地添加了 `if O[j][t] == 0: xj = 0`，已撤销

---

## 2. Row Activation 计算

**公式**:
```
row_acts[t] = Σ_j (f[j] × xj[t,j] × O[j][t])
```

- `f[j]`: 维度 j 的 tiling factor
- `xj[t,j]`: 维度 j 是否有内层循环
- `O[j][t]`: 维度 j 是否对 tensor t relevant

**注意**: O[j][t] 在 row_acts 公式中使用，**不是**在 xj 约束中使用！

---

## 3. Relevancy Matrix O[j][t]

| Dim | Input | Weight | Output |
|-----|-------|--------|--------|
| R   | 1     | 1      | 0      |
| S   | 1     | 1      | 0      |
| P   | 1     | 0      | 1      |
| Q   | 1     | 0      | 1      |
| C   | 1     | 1      | 0      |
| K   | 0     | 1      | 1      |
| N   | 1     | 0      | 1      |

**用途**:
- 计算 tensor 大小
- 计算 row_acts 时过滤无关维度的贡献
- **不用于** xj 约束

---

## 修改检查清单

在修改 ILP 模型代码前，检查：

- [ ] 是否改变了 xj 的语义？
- [ ] 是否错误地将 O[j][t] 用于 xj 约束？
- [ ] 修改是否与本文档的原则一致？
- [ ] 是否需要更新本文档？

---

## 变更历史

| 日期 | 修改 | 状态 |
|------|------|------|
| 2024-12-24 | 错误添加 xj irrelevant 约束 | 已撤销 |
