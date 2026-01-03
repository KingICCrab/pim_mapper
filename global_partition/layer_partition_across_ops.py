#!/usr/bin/env python
"""
不同算子类型之间的 Layer Partition 传播分析

本文档分析 Conv、Pooling、FC、Eltwise 等不同算子之间的数据流和分区兼容性。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("""
================================================================================
                     不同算子之间的 Layer Partition 处理
================================================================================

1. nn_dataflow 中的算子类型
================================================================================

┌─────────────────┬────────────────────────────────────────────────────────────┐
│ 算子类型         │ 数据流特点                                                   │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ ConvLayer       │ C→K 变换，H×W 可能缩小（stride>1）                            │
│                 │ 有 filter: nifm×nofm×R×S                                     │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ FCLayer         │ ConvLayer 的特例，H=W=1                                       │
│                 │ 全连接: 输入展平为向量                                          │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ PoolingLayer    │ K→K 不变（通道数不变）                                         │
│                 │ H×W 缩小（通常 stride=kernel_size）                           │
│                 │ 无 filter (或 filter = 1)                                     │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ EltwiseLayer    │ K→K 不变，H×W 不变                                            │
│                 │ 逐元素操作（如 ReLU, Add）                                     │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ LocalRegionLayer│ 局部区域操作的基类                                             │
│                 │ 如 LRN (Local Response Normalization)                        │
└─────────────────┴────────────────────────────────────────────────────────────┘


2. 层间数据流分析
================================================================================

2.1 Conv → Pooling
──────────────────
数据维度变化:
  Conv output:  [N, K, H, W]
  Pool input:   [N, C=K, H, W]  （C 继承自前一层的 K）
  Pool output:  [N, K, H', W']  （K 不变，H'<H, W'<W）

分区传播:
  • K→C 传播: 天然匹配（K=C）
  • 空间分区: 可能需要调整（Pooling 缩小了空间）

特殊情况:
  • Pooling 没有真正的「INPP 分区」（没有输入通道的 reduction）
  • Pooling 的 OFMP 分区需要考虑 Pooling 窗口的边界

2.2 Conv → Conv
──────────────────
数据维度变化:
  Conv1 output: [N, K1, H1, W1]
  Conv2 input:  [N, C2=K1, H2=H1, W2=W1]
  Conv2 output: [N, K2, H2', W2']

分区传播:
  • K→C 传播: 需要检查 K1 和 C2 的分区因子是否匹配
  • 如果 Conv1 用 OUTP (K 分区)，Conv2 用 INPP (C 分区):
    → 天然继承，无需重分布！
  • 如果 Conv1 用 OUTP，Conv2 也用 OUTP:
    → 需要 All-to-All 重分布

2.3 Conv → FC
──────────────────
数据维度变化:
  Conv output: [N, K, H, W]
  FC input:    [N, C=K×H×W, 1, 1]  ← 关键：空间维度展平到通道！
  FC output:   [N, K', 1, 1]

分区传播 (复杂):
  • Conv 的 OUTP (K) 分区 → FC 的部分 INPP 分区
  • Conv 的 OFMP (H×W) 分区 → FC 的部分 INPP 分区
  • FC 输入: C_fc = K_conv × H_conv × W_conv

特殊处理:
  Conv 输出按 OUTP 分区 (K 分到 p 个节点):
    → 每节点有 K/p 个通道，每个通道完整 H×W
    → FC 看到的是 C/p = K/p × H × W
    → 这相当于 FC 的 INPP 分区！

  Conv 输出按 OFMP 分区 (H×W 分到 p 个节点):
    → 每节点有完整 K 个通道，但只有 H/p × W/q 的空间
    → FC 看到的仍是部分数据
    → 需要 All-Gather 重分布

2.4 Pooling → Conv
──────────────────
数据维度变化:
  Pool output: [N, K, H', W']
  Conv input:  [N, C=K, H', W']

分区传播:
  • 与 Conv → Conv 类似
  • 但 Pooling 没有 INPP 分区概念

2.5 Eltwise (Add for ResNet skip connection)
──────────────────
数据维度变化:
  Input1: [N, K, H, W]  (from main path)
  Input2: [N, K, H, W]  (from skip connection)
  Output: [N, K, H, W]

分区要求:
  • 两个输入必须使用相同的分区！
  • 否则需要额外的重分布


3. 分区兼容性矩阵
================================================================================

                     下一层分区
              ┌────────┬────────┬────────┬────────┐
              │  OUTP  │  OFMP  │  BATP  │  INPP  │
    ┌─────────┼────────┼────────┼────────┼────────┤
    │  OUTP   │ All2All│ All2All│ All2All│   0    │  ← K→C 匹配
上  │─────────┼────────┼────────┼────────┼────────┤
一  │  OFMP   │ All2All│0/Adjust│ All2All│ All2All│  ← 空间匹配
层  │─────────┼────────┼────────┼────────┼────────┤
分  │  BATP   │ All2All│ All2All│   0    │ All2All│  ← Batch 匹配
区  │─────────┼────────┼────────┼────────┼────────┤
    │  INPP   │ A-R+A2A│ A-R+A2A│ A-R+A2A│ A-R+A2A│  ← 先 All-Reduce
    └─────────┴────────┴────────┴────────┴────────┘

    0 = 零成本（天然匹配）
    All2All = All-to-All 通信
    A-R = All-Reduce
    Adjust = 可能需要调整（如 Pooling 改变了空间尺寸）


4. 各算子的分区约束
================================================================================

4.1 ConvLayer
─────────────
有效分区维度:
  • OUTP (K): 有效，但需要复制 filter
  • OFMP (H×W): 有效，需要 Halo exchange
  • BATP (N): 有效，需要复制 filter
  • INPP (C): 有效，需要 All-Reduce

4.2 FCLayer
───────────
有效分区维度:
  • OUTP (K): 有效
  • OFMP: 无效（H=W=1）
  • BATP (N): 有效
  • INPP (C): 有效，需要 All-Reduce

4.3 PoolingLayer
────────────────
有效分区维度:
  • OUTP (K): 有效（逐通道操作）
  • OFMP (H×W): 有效，边界处理简单
  • BATP (N): 有效
  • INPP: 无效（Pooling 不跨通道，C=K）

4.4 EltwiseLayer
────────────────
有效分区维度:
  • OUTP (K): 有效
  • OFMP (H×W): 有效
  • BATP (N): 有效
  • INPP: 取决于具体操作（Add 两个输入必须匹配）


5. 代码实现建议
================================================================================

5.1 层类型检测
""")

# 演示如何检测层类型


def get_layer_type(layer):
    """检测层类型"""
    class_name = layer.__class__.__name__

    # 按类名判断
    if 'Conv' in class_name:
        if hasattr(layer, 'hofm') and layer.hofm == 1 and layer.wofm == 1:
            return 'FC'
        return 'Conv'
    elif 'FC' in class_name:
        return 'FC'
    elif 'Pool' in class_name:
        return 'Pool'
    elif 'Eltwise' in class_name or 'Add' in class_name:
        return 'Eltwise'
    elif 'LocalRegion' in class_name or 'LRN' in class_name:
        return 'LocalRegion'
    else:
        # 默认按特征判断
        if hasattr(layer, 'hfil'):
            return 'Conv'
        return 'Unknown'


print("""
5.2 分区生成器更新
──────────────────
def generate_partitions_for_layer(layer, dim_nodes, batch_size):
    layer_type = get_layer_type(layer)
    
    if layer_type == 'Pool':
        # Pooling 不支持 INPP 分区
        exclude_dims = [PartDim.INPP]
    elif layer_type == 'FC':
        # FC 不支持 OFMP 分区（H=W=1）
        exclude_dims = [PartDim.OFMP]
    elif layer_type == 'Eltwise':
        # Eltwise 需要两个输入分区匹配
        # 可能需要特殊处理
        exclude_dims = []
    else:
        exclude_dims = []
    
    return generate_all_partitions(layer, dim_nodes, batch_size, exclude_dims)


5.3 层间转换成本计算
────────────────────
def compute_transition_cost(layer_src, layer_dst, part_src, part_dst):
    type_src = get_layer_type(layer_src)
    type_dst = get_layer_type(layer_dst)
    
    # 特殊情况: Conv → FC (空间展平)
    if type_src in ['Conv', 'Pool'] and type_dst == 'FC':
        # Conv 的输出空间维度会展平到 FC 的输入通道
        # 需要特殊处理
        return compute_conv_to_fc_transition(layer_src, layer_dst, 
                                              part_src, part_dst)
    
    # 特殊情况: Pooling 改变空间尺寸
    if type_src == 'Pool' or type_dst == 'Pool':
        # 考虑 stride 导致的空间尺寸变化
        return compute_pooling_transition(layer_src, layer_dst,
                                           part_src, part_dst)
    
    # 通用情况
    return compute_general_transition(layer_src, layer_dst,
                                       part_src, part_dst)


5.4 Conv → FC 特殊处理
──────────────────────
Conv 输出: [N, K, H, W]  分区: OUTP=(p_k,), OFMP=(p_h, p_w)
FC 输入:   [N, C=K×H×W]  需要的分区: INPP=(p_c,)

分析:
  • 如果 Conv 只用 OUTP: C_fc 的 K 部分已经分好，但 H×W 需要 gather
  • 如果 Conv 只用 OFMP: C_fc 的 H×W 部分已经分好，但 K 需要 gather  
  • 如果 Conv 用 OUTP+OFMP 混合: 更复杂的映射

简化假设（推荐）:
  • Conv→FC 之间总是需要 All-Gather（除非完全匹配）
  • 因为 FC 的 C 维度 = Conv 的 K × H × W
""")

print("""
6. 实际网络示例
================================================================================

6.1 VGG-16 (全是 Conv + Pool + FC)
──────────────────────────────────
Layer 1-2:   Conv(3→64, 224)   → Conv(64→64, 224)   [K→C 传播]
Layer 3:     Pool(64, 224→112)                       [空间缩小]
Layer 4-5:   Conv(64→128, 112) → Conv(128→128, 112) [K→C 传播]
...
Layer 14-16: FC(512×7×7→4096) → FC(4096→4096) → FC(4096→1000)

分区策略:
  • Conv 层: OUTP/OFMP/INPP 都可以
  • Pool 层: OUTP/OFMP（无 INPP）
  • FC 层:   OUTP/INPP（无 OFMP）
  • Conv→FC: 需要重分布（空间展平）


6.2 ResNet-50 (有 skip connection)
──────────────────────────────────
Residual Block:
  Input ─────────────────────────┐
    │                            │ (skip connection)
    ├→ Conv1 → BN → ReLU         │
    ├→ Conv2 → BN → ReLU         │
    ├→ Conv3 → BN                │
    │                            │
    └→ Add ←─────────────────────┘
         ↓
       ReLU
         ↓
       Output

分区约束:
  • Skip connection 的两端必须使用相同分区！
  • 或者在 Add 之前插入重分布

处理方式:
  1. 约束求解器: 强制 skip 两端分区相同
  2. 显式建模: 在目标函数中加入 skip 重分布成本


6.3 Inception/GoogLeNet (有分支合并)
────────────────────────────────────
Inception Module:
  Input ──┬── 1×1 Conv ──────────────────┬── Concat
          ├── 1×1 → 3×3 Conv ────────────┤
          ├── 1×1 → 5×5 Conv ────────────┤
          └── 3×3 Pool → 1×1 Conv ───────┘

分区约束:
  • Concat 操作要求所有分支的空间分区相同
  • 通道可以不同（Concat 在通道维度）


7. 总结
================================================================================

核心原则:
  1. 不同算子有不同的有效分区维度
  2. Conv→FC 是最复杂的转换（空间展平）
  3. Skip connection 需要分区一致性约束
  4. Pooling 无 INPP（不跨通道计算）

建议实现:
  1. 为每种层类型定义有效分区维度
  2. 在分区生成时排除无效维度
  3. Conv→FC 转换单独建模
  4. Skip connection 通过约束或高成本处理

""")
