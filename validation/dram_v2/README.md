# DRAM Row Activation 验证项目

本项目用于验证 PIM Optimizer 中 ILP 模型的 DRAM Row Activation 计算。

## 目录结构

```
dram_v2/
├── README.md               # 本文档
├── core/                   # 核心数据结构
│   ├── __init__.py         # 导出 TraceGenerator (来自 ../dram/)
│   ├── dram_config.py      # DRAM 配置参数
│   ├── workload.py         # 卷积工作负载定义
│   └── mapping.py          # Mapping 数据结构
│
├── formula/                # Row Activation 公式
│   ├── row_activation.py   # 简化公式实现
│   └── README.md           # 公式推导说明
│
├── experiments/            # 实验脚本
│   ├── validate_formula.py # 公式验证（使用简化生成器）
│   ├── validate_with_original.py # 使用原始 trace_generator.py
│   └── analyze_permutation.py # 排列影响分析
│
├── utils/                  # 工具函数
│   └── trace_analyzer.py   # Trace 分析工具
│
└── tests/                  # 单元测试
    └── test_trace_generator.py
```

## 与原始代码的关系

**TraceGenerator 直接使用原始实现**：`dram/trace_generator.py` (1395行)

本项目**不重写**模拟器，只整理：
- 数据结构：`Workload`, `Mapping`
- 公式推导：`formula/row_activation.py`
- 实验脚本：`experiments/`

| 组件 | 来源 | 说明 |
|------|------|------|
| `TraceGenerator` | `dram/trace_generator.py` | **原始实现，直接 import** |
| `DRAMConfig` | `dram/trace_generator.py` | 原始实现 |
| `Workload` | `dram_v2/core/workload.py` | 新增，便于使用 |
| `Mapping` | `dram_v2/core/mapping.py` | 新增，便于使用 |

## 快速开始

```bash
cd validation/dram_v2

# 运行单元测试
python tests/test_trace_generator.py

# 验证公式（使用原始模拟器）
python experiments/validate_with_original.py

# 分析排列影响
python experiments/analyze_permutation.py
```

## 核心概念

### 1. Memory Hierarchy

```
Level 0: PE (register)        ─┬─ Buffer Tile (SRAM内复用，无DRAM访问)
Level 1: GlobalBuffer (SRAM)  ─┘
Level 2: RowBuffer            ─┬─ DRAM Loops (产生DRAM访问)
Level 3: LocalDRAM            ─┘
```

### 2. Dimension Relevance

| Tensor | 相关维度 | 不相关维度 |
|--------|---------|-----------|
| Input  | R, S, P, Q, C, N | **K** |
| Weight | R, S, C, K | P, Q, N |
| Output | P, Q, K, N | R, S, C |

**关键洞察**: K 是唯一的 Non-Input 维度。

### 3. Permutation 对 Row Activation 的影响

K 在排列中的位置决定 Input tile 切换行为：

```
K→C→P→Q: K 在最外层，K后面有3个Input维度
         → Input tile 在每次K迭代都切换 (×K_l3)
         
C→P→Q→K: K 在最内层，K后面无Input维度  
         → Input tile 不因K迭代而切换
```

### 4. L2 Layer 顺序的影响

即使 L3 factors = 1，L2 循环顺序也会影响 row activation：

```
CHW 数据布局下：
  CRS 顺序 → 4 row switches (最优)
  RSC 顺序 → 36 row switches (9x worse)
```

原因：不同的访问顺序导致不同的地址跳跃模式，可能跨越 DRAM row 边界。

## 公式验证状态

| 层级 | 公式 | 准确率 | 备注 |
|------|------|--------|------|
| L3 | V4 公式 | ~50% | 需要进一步验证 |
| L2 | 待开发 | - | L2顺序影响显著 |

## 待办事项

- [ ] 验证原始 trace_generator 的正确性
- [ ] 推导更准确的 L3 公式
- [ ] 开发 L2 层的 row activation 公式
- [ ] 整合到 ILP 模型
