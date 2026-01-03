#!/usr/bin/env python
"""
Layout/Partition Propagation 实现

基于论文中的 Algorithm 1: Layout Propagation

核心思想：
- Elementwise 算子（ReLU, Add, BN 等）不关心具体的数据布局/分区
- 它们可以"透传"上游或下游的分区，避免不必要的重分布
- 只有"复杂算子"（Conv, FC, MatMul）才真正约束分区

应用场景：
1. 减少层间重分布成本
2. 简化全局分区优化问题
3. 识别可以融合的算子序列
"""
from enum import Enum
from typing import List, Dict, Set, Tuple, Optional
from collections import deque
from ilp_optimizer_v2 import (
    LayerType, LayerConfig, HybridPartitionChoice, PartDim,
    detect_layer_type
)
import sys
import os
# Add global_partition to path
sys.path.append(os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), 'global_partition'))


# 添加父目录到路径以导入 ilp_optimizer_v2
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class OperatorComplexity(Enum):
    """算子复杂度分类"""
    COMPLEX = "complex"      # 复杂算子: Conv, FC, MatMul - 有特定的分区约束
    ELEMENTWISE = "elementwise"  # Elementwise: ReLU, Add, BN - 可以透传分区
    REDUCTION = "reduction"  # Reduction: Pool, Softmax - 可能改变维度
    UNKNOWN = "unknown"


def classify_operator(layer_type: str) -> OperatorComplexity:
    """
    分类算子复杂度。

    Complex 算子: 有权重矩阵，对分区有特定要求
    Elementwise 算子: 逐元素操作，可以透传分区
    Reduction 算子: 在某些维度上聚合，可能改变形状
    """
    if layer_type in [LayerType.CONV, LayerType.FC]:
        return OperatorComplexity.COMPLEX
    elif layer_type == LayerType.ELTWISE:
        return OperatorComplexity.ELEMENTWISE
    elif layer_type == LayerType.POOL:
        return OperatorComplexity.REDUCTION
    else:
        return OperatorComplexity.UNKNOWN


def can_propagate_partition(src_config: LayerConfig, dst_config: LayerConfig) -> bool:
    """
    检查分区是否可以从 src 传播到 dst。

    条件（来自算法）：
    1. dst 是 elementwise 算子
    2. 形状匹配 (t.shape() == s.shape())
    3. dst 的下游不是复杂算子（这个在外层检查）
    """
    # 条件 1: dst 必须是 elementwise
    dst_complexity = classify_operator(dst_config.layer_type)
    if dst_complexity != OperatorComplexity.ELEMENTWISE:
        return False

    # 条件 2: 形状匹配
    if (src_config.nofm != dst_config.nifm or  # K → C
        src_config.hofm != dst_config.hofm or   # H 匹配
            src_config.wofm != dst_config.wofm):    # W 匹配
        return False

    return True


class PartitionPropagator:
    """
    分区传播器 - 实现 Algorithm 1: Layout Propagation

    在计算图中传播分区方案，识别可以共享分区的算子序列。
    """

    def __init__(self, layer_configs: List[LayerConfig],
                 adjacency: Dict[int, List[int]] = None):
        """
        Args:
            layer_configs: 所有层的配置
            adjacency: 邻接表 {layer_idx: [successor_indices]}
                       如果为 None，假设是线性序列
        """
        self.layer_configs = layer_configs
        self.num_layers = len(layer_configs)

        # 构建邻接关系
        if adjacency is None:
            # 默认线性序列: 0 → 1 → 2 → ...
            self.adjacency = {i: [i+1] for i in range(self.num_layers - 1)}
            self.adjacency[self.num_layers - 1] = []
        else:
            self.adjacency = adjacency

    def propagate(self, start_layer: int, partition: HybridPartitionChoice) -> Dict[int, HybridPartitionChoice]:
        """
        从起始层传播分区。

        实现 Algorithm 1 的核心逻辑。

        Returns:
            Dict[layer_idx, partition]: 所有可以使用该分区的层
        """
        result = {start_layer: partition}

        start_config = self.layer_configs[start_layer]
        start_complexity = classify_operator(start_config.layer_type)

        # 如果起始层是复杂算子，尝试向后传播
        if start_complexity == OperatorComplexity.COMPLEX:
            self._propagate_forward(start_layer, partition, result)

        # 也可以向前传播（反向）
        self._propagate_backward(start_layer, partition, result)

        return result

    def _propagate_forward(self, start_layer: int, partition: HybridPartitionChoice,
                           result: Dict[int, HybridPartitionChoice]):
        """向后（下游）传播分区 - BFS"""
        queue = deque([start_layer])
        visited = {start_layer}

        while queue:
            current = queue.popleft()
            current_config = self.layer_configs[current]

            # 遍历所有后继
            for next_layer in self.adjacency.get(current, []):
                if next_layer in visited:
                    continue

                next_config = self.layer_configs[next_layer]

                # 检查是否可以传播
                if can_propagate_partition(current_config, next_config):
                    # Elementwise 可以透传，继续传播
                    result[next_layer] = partition
                    visited.add(next_layer)
                    queue.append(next_layer)

    def _propagate_backward(self, start_layer: int, partition: HybridPartitionChoice,
                            result: Dict[int, HybridPartitionChoice]):
        """向前（上游）传播分区 - 反向 BFS"""
        # 构建反向邻接表
        reverse_adj = {i: [] for i in range(self.num_layers)}
        for src, dsts in self.adjacency.items():
            for dst in dsts:
                reverse_adj[dst].append(src)

        queue = deque([start_layer])
        visited = {start_layer}

        while queue:
            current = queue.popleft()
            current_config = self.layer_configs[current]

            # 遍历所有前驱
            for prev_layer in reverse_adj.get(current, []):
                if prev_layer in visited:
                    continue

                prev_config = self.layer_configs[prev_layer]

                # 检查前驱是否是 elementwise（可以向上传播）
                prev_complexity = classify_operator(prev_config.layer_type)

                if prev_complexity == OperatorComplexity.ELEMENTWISE:
                    # 形状匹配检查
                    if (prev_config.nofm == current_config.nifm and
                        prev_config.hofm == current_config.hofm and
                            prev_config.wofm == current_config.wofm):
                        result[prev_layer] = partition
                        visited.add(prev_layer)
                        queue.append(prev_layer)

    def find_propagation_groups(self) -> List[Set[int]]:
        """
        找出所有可以共享分区的层组。

        返回不相交的层组列表，每组内的层可以使用相同的分区。
        """
        groups = []
        assigned = set()

        # 从每个复杂算子开始传播
        for i, config in enumerate(self.layer_configs):
            if i in assigned:
                continue

            complexity = classify_operator(config.layer_type)

            if complexity == OperatorComplexity.COMPLEX:
                # 创建一个虚拟分区来传播
                dummy_partition = HybridPartitionChoice({})
                group = self.propagate(i, dummy_partition)

                if len(group) > 1:
                    groups.append(set(group.keys()))
                    assigned.update(group.keys())

        # 添加未分组的层（单独成组）
        for i in range(self.num_layers):
            if i not in assigned:
                groups.append({i})

        return groups

    def get_partition_constraints(self) -> List[Tuple[int, int]]:
        """
        获取分区约束：哪些层必须使用相同的分区。

        Returns:
            List of (layer_i, layer_j) pairs that must have same partition
        """
        constraints = []
        groups = self.find_propagation_groups()

        for group in groups:
            layers = sorted(group)
            # 组内所有层两两约束
            for i in range(len(layers)):
                for j in range(i + 1, len(layers)):
                    constraints.append((layers[i], layers[j]))

        return constraints


def identify_fusion_opportunities(layer_configs: List[LayerConfig]) -> List[List[int]]:
    """
    识别可以融合的算子序列。

    基于 Layout Propagation 的思想：如果一组连续的算子可以共享分区，
    它们就是融合的候选。

    Returns:
        List of fusible layer sequences
    """
    propagator = PartitionPropagator(layer_configs)
    groups = propagator.find_propagation_groups()

    # 过滤出多于一层的组
    fusion_candidates = [sorted(g) for g in groups if len(g) > 1]

    return fusion_candidates


# ============================================================================
# 演示
# ============================================================================

def demo():
    """演示 Layout Propagation"""

    print("=" * 70)
    print("Layout/Partition Propagation 演示")
    print("=" * 70)

    print("""
算法核心思想:
─────────────────────────────────────────────────────────────────────────
                                                                         
  方法 (a) - 显式转换:                                                    
    Conv → [转换] → ReLU → [转换] → Conv                                  
    每次分区变化都需要转换，开销大                                          
                                                                         
  方法 (b) - 分区传播:                                                    
    Conv(p) ← ← ← ReLU(p) ← ← ← Conv(p)                                   
                  ↑                                                       
            透传分区，无需转换！                                            
                                                                         
─────────────────────────────────────────────────────────────────────────
""")

    # 创建模拟网络: Conv → ReLU → BN → Conv → ReLU → Pool → FC
    class MockLayer:
        def __init__(self, layer_type, nifm, nofm, hofm, wofm=None, hfil=1):
            self.nifm = nifm
            self.nofm = nofm
            self.hofm = hofm
            self.wofm = wofm or hofm
            self.hfil = hfil
            self._type = layer_type

        @property
        def __class__(self):
            class FakeClass:
                __name__ = self._type
            return FakeClass()

    layers = [
        ('conv1', MockLayer('ConvLayer', 3, 64, 224, hfil=3)),
        ('relu1', MockLayer('EltwiseLayer', 64, 64, 224)),
        ('bn1', MockLayer('EltwiseLayer', 64, 64, 224)),
        ('conv2', MockLayer('ConvLayer', 64, 128, 112, hfil=3)),
        ('relu2', MockLayer('EltwiseLayer', 128, 128, 112)),
        ('pool1', MockLayer('PoolingLayer', 128, 128, 56)),
        ('conv3', MockLayer('ConvLayer', 128, 256, 56, hfil=3)),
        ('fc1', MockLayer('FCLayer', 256, 1000, 1)),
    ]

    print("模拟网络结构:")
    print("-" * 50)
    for name, layer in layers:
        layer_type = detect_layer_type(layer)
        complexity = classify_operator(layer_type)
        print(
            f"  {name:10s} | Type: {layer_type:10s} | Complexity: {complexity.value}")

    # 构建 LayerConfig
    configs = [LayerConfig(name, layer, i)
               for i, (name, layer) in enumerate(layers)]

    # 创建传播器
    propagator = PartitionPropagator(configs)

    # 找出传播组
    groups = propagator.find_propagation_groups()

    print("\n" + "-" * 50)
    print("分区传播组 (可以共享分区的层):")
    print("-" * 50)
    for i, group in enumerate(groups):
        layer_names = [layers[idx][0] for idx in sorted(group)]
        if len(group) > 1:
            print(f"  组 {i+1}: {' → '.join(layer_names)}")
            print(f"         这些层可以使用相同的分区，无需重分布！")
        else:
            print(f"  组 {i+1}: {layer_names[0]} (独立)")

    # 演示从 conv1 开始传播
    print("\n" + "-" * 50)
    print("从 conv1 开始传播分区:")
    print("-" * 50)

    dummy_partition = HybridPartitionChoice({PartDim.OUTP: (4, 4)})
    propagated = propagator.propagate(0, dummy_partition)

    for idx in sorted(propagated.keys()):
        name = layers[idx][0]
        print(f"  {name} 可以使用 conv1 的分区")

    # 融合机会
    print("\n" + "-" * 50)
    print("潜在的算子融合机会:")
    print("-" * 50)
    fusion_candidates = identify_fusion_opportunities(configs)
    for candidate in fusion_candidates:
        names = [layers[idx][0] for idx in candidate]
        print(f"  可融合: {' + '.join(names)}")

    print("\n" + "=" * 70)
    print("应用到全局分区优化:")
    print("=" * 70)
    print("""
在 ILP/DP 优化中的应用:

1. 减少决策变量:
   - 同一传播组内的层共享一个分区变量
   - 减少优化问题规模

2. 减少重分布成本:
   - 组内层间转换成本 = 0
   - 只需考虑组间转换

3. 识别融合边界:
   - 传播组内的层是融合候选
   - 组边界是潜在的流水线分割点
""")


if __name__ == '__main__':
    demo()
