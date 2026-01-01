# Permutation 对 Row Activation 影响的实验

## 实验目的

验证和分析 Loop Permutation 对 DRAM Row Activation 的影响，为 ILP 模型提供准确的代价函数。

## 实验概述

### 核心发现

#### 1. L3 层 Permutation 影响 (K, C, P, Q)

**关键规则**: K 是唯一的 Non-Input 维度

| K 位置 | Input Tile 切换次数 |
|--------|---------------------|
| K 在最内层 | `P_l3 × Q_l3 × C_l3` |
| K 不在最内层 (且后面有 >1 的 Input dim) | `P_l3 × Q_l3 × C_l3 × K_l3` |

**验证结果** (`final_formula_v4.py`):
- 720 个 mappings，100% 精确匹配
- Spearman ρ = 1.000

#### 2. L2 层 R, S 顺序也影响 Row Activation

即使 R_l3=1, S_l3=1 (R、S 在 L2 层)，L2 内部的遍历顺序仍然影响 row switches。

**CHW 数据布局** (`why_l2_order_matters.py`):
| L2 Order | Row Switches |
|----------|--------------|
| RSC | 36 |
| SRC | 36 |
| CRS | **4** (最优) |

**原因**: 不同遍历顺序产生不同地址访问序列，跳跃可能跨越 DRAM row。

## 实验文件说明

### 核心验证文件

| 文件 | 功能 | 结果 |
|------|------|------|
| `final_formula_v4.py` | **最终公式验证** | ✅ 100% 匹配, ρ=1.0 |
| `formula_summary.py` | 公式总结 | 通用公式推导 |
| `verify_formula_v3.py` | 公式 V3 验证 | 发现边界情况 |

### L3 Permutation 分析

| 文件 | 功能 |
|------|------|
| `analyze_per_permutation.py` | 按 permutation 分组分析 |
| `analyze_k_position.py` | 分析 K 位置影响 |
| `analyze_formula_derivation.py` | 公式推导过程 |
| `analyze_mismatch.py` | 分析特殊情况 (P_l3=1) |
| `permutation_summary.py` | K 位置汇总统计 |

### L2 层 R, S 顺序分析

| 文件 | 功能 |
|------|------|
| `test_rs_order.py` | 测试 R, S 顺序影响 |
| `analyze_rs_position.py` | R 位置影响分析 |
| `analyze_l2_order.py` | L2 层遍历顺序分析 |
| `why_l2_order_matters.py` | 解释为什么 L2 顺序重要 |

### Mapping 生成与验证

| 文件 | 功能 |
|------|------|
| `valid_mapping_test.py` | 生成合法 mappings (满足 ILP 约束) |
| `valid_mapping_with_perm.py` | 添加 permutation 变化的 mappings |
| `multi_mapping_v2.py` | 多 mapping 验证 (精确模拟) |
| `multi_mapping_validation.py` | 初版多 mapping 验证 |

## 最终公式

### L3 层 Input Tile 切换次数

```
Input_dims = {C, P, Q}  (如果 R_l3>1, 还包括 R; S 类似)
Non_input_dims = {K}

计算步骤:
1. base = ∏(所有 Input dims 的 _l3 值)
2. 找 K 的位置 pos_k
3. inner_input_product = ∏(K 之后的 Input dims 的 _l3 值)
4. 如果 inner_input_product == 1:
       tile_changes = base
   否则:
       tile_changes = base × K_l3

Row_switches = tile_changes × rows_per_tile
rows_per_tile = ceil(input_tile_size / block_w)
```

### L2 层最优策略

- **数据布局 CHW**: C 在最外层循环 (CRS 顺序)
- **原则**: 让循环顺序匹配数据在内存中的连续方向

## 实验方法

1. **精确模拟**: 5 层嵌套循环，跟踪 (C, P, Q) 变化
2. **约束验证**: 确保 mappings 满足 PE Array 和 Buffer 约束
3. **统计指标**: Spearman 相关系数、平均误差、精确匹配率

## 运行实验

```bash
cd /Users/haochenzhao/Projects/pim_optimizer/validation/dram/permutation_experiments

# 验证最终公式
python final_formula_v4.py

# 分析 L2 层影响
python why_l2_order_matters.py

# 完整公式总结
python formula_summary.py
```

## 结论

1. **L3 层**: K 的位置是关键，公式已验证 100% 准确
2. **L2 层**: R, S 顺序也会影响 row activation (最多 9x 差异)
3. **ILP 模型**: 需要同时考虑 L3 和 L2 层的循环顺序
