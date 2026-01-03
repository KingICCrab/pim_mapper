# Layout Propagation 模块

基于规约分析的布局传播优化技术。

## 核心原理

通过分析算子是否包含规约（Reduction）操作，将算子分为：
- **布局敏感算子**：有规约计算（Conv, FC, Pool等）
- **布局不敏感算子**：逐元素操作（ReLU, Add, BatchNorm等）

布局不敏感算子可以"透传"上下游的分区方案，无需额外的数据重分布。

## 文件结构

```
layout_propagation/
├── README.md                    # 本文件
├── layout_propagation.tex       # LaTeX 文档
└── (其他相关文件)
```

## 相关源代码

- `../layout_propagation.py` - 核心布局传播实现
- `../partition_propagation.py` - 分区传播算法
- `../test_layout_propagation.py` - 测试脚本
- `../analyze_nns_layout.py` - 网络分析脚本

## 主要功能

1. **规约分析器** (`ReductionAnalyzer`)
   - 分析每个算子的规约特性
   - 判断布局敏感性

2. **布局传播器** (`LayoutPropagator`)
   - 在计算图中传播分区方案
   - 识别传播组

3. **传播组识别**
   - 将可共享分区的算子聚合
   - 减少ILP决策变量

## 优化效果

| 网络 | 总层数 | 传播组数 | 变量缩减 |
|------|--------|----------|----------|
| VGG-16 | 38 | 16 | 57.9% |
| ResNet-50 | 107 | 53 | 50.5% |
| ResNet-152 | 311 | 152 | 51.1% |
| MobileNet-V2 | 155 | 52 | 66.5% |

## 使用方法

```python
from layout_propagation import ReductionAnalyzer, LayoutPropagator

# 分析算子
op_info = ReductionAnalyzer.analyze(layer)
print(f"算子 {op_info.name}: 布局{'敏感' if op_info.has_reduction else '不敏感'}")

# 传播分区
propagator = LayoutPropagator(operators)
groups = propagator.find_propagation_groups()
print(f"识别到 {len(groups)} 个传播组")
```
