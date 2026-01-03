#!/usr/bin/env python
"""
Layout Propagation 模块

核心原则：
- 有规约计算的算子 → 布局敏感（Layout Sensitive）
- 无规约的算子（Elementwise）→ 布局不敏感，可以透传分区

规约分析：
- Conv: 在 C (输入通道) 维度规约 → 敏感
- FC/MatMul: 在 K (内积) 维度规约 → 敏感  
- Pool: 在空间窗口规约 → 敏感
- Softmax: 在某个维度规约 → 敏感
- BatchNorm: 在 N (batch) 维度规约统计量 → 敏感（训练时）/ 不敏感（推理时）
- ReLU/Add/Mul: 逐元素操作，无规约 → 不敏感
"""

from collections import deque
from typing import List, Dict, Set, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass


class LayoutSensitivity(Enum):
    """算子的布局敏感性"""
    SENSITIVE = "sensitive"      # 有规约，布局敏感
    INSENSITIVE = "insensitive"  # 无规约，可以透传


@dataclass
class OperatorInfo:
    """算子信息"""
    name: str
    op_type: str
    has_reduction: bool  # 是否有规约计算
    reduction_dims: List[str]  # 在哪些维度上规约
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]

    @property
    def sensitivity(self) -> LayoutSensitivity:
        """根据是否有规约判断布局敏感性"""
        if self.has_reduction:
            return LayoutSensitivity.SENSITIVE
        return LayoutSensitivity.INSENSITIVE


class ReductionAnalyzer:
    """
    规约分析器

    分析每个算子是否包含规约操作，以此判断布局敏感性。
    """

    # 已知算子类型的规约特性
    REDUCTION_PATTERNS = {
        # 有规约的算子
        'Conv': {'has_reduction': True, 'reduction_dims': ['C', 'R', 'S']},
        'ConvLayer': {'has_reduction': True, 'reduction_dims': ['C', 'R', 'S']},
        'FC': {'has_reduction': True, 'reduction_dims': ['C']},
        'FCLayer': {'has_reduction': True, 'reduction_dims': ['C']},
        'MatMul': {'has_reduction': True, 'reduction_dims': ['K']},
        'Gemm': {'has_reduction': True, 'reduction_dims': ['K']},
        'Pool': {'has_reduction': True, 'reduction_dims': ['H', 'W']},
        'PoolingLayer': {'has_reduction': True, 'reduction_dims': ['H', 'W']},
        'MaxPool': {'has_reduction': True, 'reduction_dims': ['H', 'W']},
        'AvgPool': {'has_reduction': True, 'reduction_dims': ['H', 'W']},
        'GlobalAvgPool': {'has_reduction': True, 'reduction_dims': ['H', 'W']},
        'Softmax': {'has_reduction': True, 'reduction_dims': ['C']},
        'ReduceSum': {'has_reduction': True, 'reduction_dims': ['axis']},
        'ReduceMean': {'has_reduction': True, 'reduction_dims': ['axis']},
        'ReduceMax': {'has_reduction': True, 'reduction_dims': ['axis']},

        # 无规约的算子（Elementwise）
        'ReLU': {'has_reduction': False, 'reduction_dims': []},
        'Sigmoid': {'has_reduction': False, 'reduction_dims': []},
        'Tanh': {'has_reduction': False, 'reduction_dims': []},
        'LeakyReLU': {'has_reduction': False, 'reduction_dims': []},
        'Add': {'has_reduction': False, 'reduction_dims': []},
        'Sub': {'has_reduction': False, 'reduction_dims': []},
        'Mul': {'has_reduction': False, 'reduction_dims': []},
        'Div': {'has_reduction': False, 'reduction_dims': []},
        'Concat': {'has_reduction': False, 'reduction_dims': []},
        'Split': {'has_reduction': False, 'reduction_dims': []},
        'Reshape': {'has_reduction': False, 'reduction_dims': []},
        'Transpose': {'has_reduction': False, 'reduction_dims': []},
        'Eltwise': {'has_reduction': False, 'reduction_dims': []},
        'EltwiseLayer': {'has_reduction': False, 'reduction_dims': []},

        # BatchNorm: 推理时无规约，训练时有规约
        'BatchNorm': {'has_reduction': False, 'reduction_dims': []},  # 默认推理
        'BatchNormalization': {'has_reduction': False, 'reduction_dims': []},

        # LocalRegion: 有局部规约
        'LocalRegion': {'has_reduction': True, 'reduction_dims': ['local']},
        'LocalRegionLayer': {'has_reduction': True, 'reduction_dims': ['local']},
        'LRN': {'has_reduction': True, 'reduction_dims': ['local']},
    }

    @classmethod
    def analyze(cls, layer) -> OperatorInfo:
        """
        分析单个算子的规约特性。

        Args:
            layer: 算子对象（nn_dataflow Layer 或类似对象）

        Returns:
            OperatorInfo 包含规约分析结果
        """
        # 获取算子类型
        class_name = layer.__class__.__name__
        op_type = cls._get_op_type(class_name)

        # 获取形状信息
        input_shape = cls._get_input_shape(layer)
        output_shape = cls._get_output_shape(layer)

        # 查找规约模式
        pattern = cls.REDUCTION_PATTERNS.get(op_type, None)

        if pattern is None:
            # 未知算子，尝试启发式判断
            has_reduction, reduction_dims = cls._heuristic_analysis(layer)
        else:
            has_reduction = pattern['has_reduction']
            reduction_dims = pattern['reduction_dims']

        return OperatorInfo(
            name=getattr(layer, 'name', class_name),
            op_type=op_type,
            has_reduction=has_reduction,
            reduction_dims=reduction_dims,
            input_shape=input_shape,
            output_shape=output_shape
        )

    @classmethod
    def _get_op_type(cls, class_name: str) -> str:
        """从类名提取算子类型"""
        # 移除常见后缀
        for suffix in ['Layer', 'Op', 'Operation']:
            if class_name.endswith(suffix) and class_name != suffix:
                return class_name
        return class_name

    @classmethod
    def _get_input_shape(cls, layer) -> Tuple[int, ...]:
        """获取输入形状"""
        if hasattr(layer, 'nifm') and hasattr(layer, 'hifm'):
            return (getattr(layer, 'nifm', 1),
                    getattr(layer, 'hifm', 1),
                    getattr(layer, 'wifm', 1))
        return ()

    @classmethod
    def _get_output_shape(cls, layer) -> Tuple[int, ...]:
        """获取输出形状"""
        if hasattr(layer, 'nofm') and hasattr(layer, 'hofm'):
            return (getattr(layer, 'nofm', 1),
                    getattr(layer, 'hofm', 1),
                    getattr(layer, 'wofm', 1))
        return ()

    @classmethod
    def _heuristic_analysis(cls, layer) -> Tuple[bool, List[str]]:
        """
        启发式分析未知算子。

        规则：
        1. 如果输出元素数 < 输入元素数 → 可能有规约
        2. 如果有 filter/kernel 属性 → 可能有规约
        """
        # 检查是否有 filter（卷积类）
        if hasattr(layer, 'hfil') and getattr(layer, 'hfil', 1) > 1:
            return True, ['C', 'R', 'S']

        # 检查输入输出大小
        input_shape = cls._get_input_shape(layer)
        output_shape = cls._get_output_shape(layer)

        if input_shape and output_shape:
            input_size = 1
            output_size = 1
            for s in input_shape:
                input_size *= s
            for s in output_shape:
                output_size *= s

            # 输出比输入小很多 → 可能有规约
            if output_size < input_size * 0.5:
                return True, ['unknown']

        # 默认无规约
        return False, []


class LayoutPropagator:
    """
    布局/分区传播器

    基于规约分析的布局敏感性，在计算图中传播分区方案。

    规则：
    - 布局敏感算子（有规约）：是传播的起点和终点
    - 布局不敏感算子（无规约）：可以透传上游或下游的分区
    """

    def __init__(self, operators: List[OperatorInfo],
                 adjacency: Dict[int, List[int]] = None):
        """
        Args:
            operators: 算子信息列表
            adjacency: 邻接表 {op_idx: [successor_indices]}
                       如果为 None，假设是线性序列
        """
        self.operators = operators
        self.num_ops = len(operators)

        # 构建邻接关系
        if adjacency is None:
            # 默认线性序列
            self.adjacency = {i: [i+1] for i in range(self.num_ops - 1)}
            self.adjacency[self.num_ops - 1] = []
        else:
            self.adjacency = adjacency

        # 构建反向邻接
        self.reverse_adj = {i: [] for i in range(self.num_ops)}
        for src, dsts in self.adjacency.items():
            for dst in dsts:
                if dst < self.num_ops:
                    self.reverse_adj[dst].append(src)

    def propagate_from(self, start_op: int, partition: Any) -> Dict[int, Any]:
        """
        从指定算子开始传播分区。

        只有布局敏感算子才能作为传播起点。
        布局不敏感算子会透传分区。

        Returns:
            Dict[op_idx, partition]: 所有可以使用该分区的算子
        """
        result = {start_op: partition}

        # 向前传播（下游）
        self._propagate_forward(start_op, partition, result)

        # 向后传播（上游）
        self._propagate_backward(start_op, partition, result)

        return result

    def _propagate_forward(self, start_op: int, partition: Any,
                           result: Dict[int, Any]):
        """向下游传播"""
        queue = deque([start_op])
        visited = {start_op}

        while queue:
            current = queue.popleft()

            for next_op in self.adjacency.get(current, []):
                if next_op in visited or next_op >= self.num_ops:
                    continue

                next_info = self.operators[next_op]

                # 布局不敏感算子（无规约）可以透传
                if next_info.sensitivity == LayoutSensitivity.INSENSITIVE:
                    # 检查形状兼容性
                    if self._shape_compatible(current, next_op):
                        result[next_op] = partition
                        visited.add(next_op)
                        queue.append(next_op)

    def _propagate_backward(self, start_op: int, partition: Any,
                            result: Dict[int, Any]):
        """向上游传播"""
        queue = deque([start_op])
        visited = {start_op}

        while queue:
            current = queue.popleft()

            for prev_op in self.reverse_adj.get(current, []):
                if prev_op in visited:
                    continue

                prev_info = self.operators[prev_op]

                # 布局不敏感算子（无规约）可以透传
                if prev_info.sensitivity == LayoutSensitivity.INSENSITIVE:
                    if self._shape_compatible(prev_op, current):
                        result[prev_op] = partition
                        visited.add(prev_op)
                        queue.append(prev_op)

    def _shape_compatible(self, src_op: int, dst_op: int) -> bool:
        """检查两个算子的形状是否兼容"""
        src_info = self.operators[src_op]
        dst_info = self.operators[dst_op]

        # 输出形状 == 输入形状
        return src_info.output_shape == dst_info.input_shape or \
            not src_info.output_shape or not dst_info.input_shape

    def find_propagation_groups(self) -> List[Set[int]]:
        """
        找出所有可以共享分区的算子组。

        每个布局敏感算子是一个组的"锚点"，不敏感算子附着到相邻的敏感算子。

        规则：
        1. 每个敏感算子形成一个组的核心
        2. 不敏感算子附着到其前驱敏感算子
        3. 组内的层共享相同分区，无需重分布
        """
        groups = []
        assigned = set()

        # 找到所有敏感算子作为锚点
        sensitive_ops = self.get_sensitive_operators()

        for anchor in sensitive_ops:
            if anchor in assigned:
                continue

            # 该组包含锚点和可达的不敏感算子
            group = {anchor}
            assigned.add(anchor)

            # 只向前传播（下游），遇到敏感算子停止
            queue = deque([anchor])
            visited = {anchor}

            while queue:
                current = queue.popleft()

                for next_op in self.adjacency.get(current, []):
                    if next_op in visited or next_op >= self.num_ops:
                        continue

                    next_info = self.operators[next_op]

                    # 只传播到不敏感算子
                    if next_info.sensitivity == LayoutSensitivity.INSENSITIVE:
                        if self._shape_compatible(current, next_op):
                            group.add(next_op)
                            assigned.add(next_op)
                            visited.add(next_op)
                            queue.append(next_op)

            groups.append(group)

        # 处理未分配的不敏感算子（没有前驱敏感算子的情况）
        for i in range(self.num_ops):
            if i not in assigned:
                groups.append({i})
                assigned.add(i)

        return groups

    def get_sensitive_operators(self) -> List[int]:
        """获取所有布局敏感算子的索引"""
        return [i for i, op in enumerate(self.operators)
                if op.sensitivity == LayoutSensitivity.SENSITIVE]

    def get_insensitive_operators(self) -> List[int]:
        """获取所有布局不敏感算子的索引"""
        return [i for i, op in enumerate(self.operators)
                if op.sensitivity == LayoutSensitivity.INSENSITIVE]


def analyze_network_sensitivity(layers: List[Any]) -> List[OperatorInfo]:
    """
    分析整个网络的布局敏感性。

    Args:
        layers: 层对象列表

    Returns:
        每层的算子信息（包含规约分析结果）
    """
    return [ReductionAnalyzer.analyze(layer) for layer in layers]


def find_layout_propagation_groups(layers: List[Any],
                                   adjacency: Dict[int, List[int]] = None) -> List[Set[int]]:
    """
    便捷函数：找出可以共享分区的层组。

    Args:
        layers: 层对象列表
        adjacency: 邻接表（可选）

    Returns:
        层组列表，每组内的层可以共享分区
    """
    op_infos = analyze_network_sensitivity(layers)
    propagator = LayoutPropagator(op_infos, adjacency)
    return propagator.find_propagation_groups()


# ============================================================================
# 演示
# ============================================================================

def demo():
    """演示基于规约分析的 Layout Propagation"""

    print("=" * 70)
    print("Layout Propagation (基于规约分析)")
    print("=" * 70)

    print("""
核心原则:
─────────────────────────────────────────────────────────────────────────
  有规约计算 → 布局敏感（Sensitive）   → 是分区边界
  无规约计算 → 布局不敏感（Insensitive）→ 可透传分区
─────────────────────────────────────────────────────────────────────────

算子规约分析:
┌─────────────┬─────────────┬──────────────┬─────────────────────┐
│ 算子        │ 有规约?     │ 规约维度      │ 布局敏感?           │
├─────────────┼─────────────┼──────────────┼─────────────────────┤
│ Conv        │ ✓           │ C, R, S      │ ✓ Sensitive         │
│ FC/MatMul   │ ✓           │ K            │ ✓ Sensitive         │
│ Pool        │ ✓           │ H, W (窗口)  │ ✓ Sensitive         │
│ Softmax     │ ✓           │ C            │ ✓ Sensitive         │
├─────────────┼─────────────┼──────────────┼─────────────────────┤
│ ReLU        │ ✗           │ -            │ ✗ Insensitive       │
│ Add/Mul     │ ✗           │ -            │ ✗ Insensitive       │
│ BatchNorm*  │ ✗           │ -            │ ✗ Insensitive       │
└─────────────┴─────────────┴──────────────┴─────────────────────┘
* BatchNorm 推理时无规约
""")

    # 创建模拟网络
    class MockLayer:
        def __init__(self, name, layer_type, nifm, nofm, hofm, wofm=None, hfil=1):
            self.name = name
            self.nifm = nifm
            self.nofm = nofm
            self.hofm = hofm
            self.wofm = wofm or hofm
            self.hifm = hofm  # 简化：假设输入输出空间相同
            self.wifm = wofm or hofm
            self.hfil = hfil
            self._type = layer_type

        @property
        def __class__(self):
            class FakeClass:
                pass
            FakeClass.__name__ = self._type
            return FakeClass

    # VGG-style 网络片段
    layers = [
        MockLayer('conv1', 'ConvLayer', 3, 64, 224, hfil=3),
        MockLayer('bn1', 'BatchNorm', 64, 64, 224),
        MockLayer('relu1', 'ReLU', 64, 64, 224),
        MockLayer('conv2', 'ConvLayer', 64, 64, 224, hfil=3),
        MockLayer('bn2', 'BatchNorm', 64, 64, 224),
        MockLayer('relu2', 'ReLU', 64, 64, 224),
        MockLayer('pool1', 'MaxPool', 64, 64, 112),
        MockLayer('conv3', 'ConvLayer', 64, 128, 112, hfil=3),
        MockLayer('relu3', 'ReLU', 128, 128, 112),
        MockLayer('fc1', 'FCLayer', 128, 1000, 1),
    ]

    print("\n示例网络:")
    print("-" * 70)

    # 分析每层
    op_infos = analyze_network_sensitivity(layers)

    for i, (layer, op_info) in enumerate(zip(layers, op_infos)):
        reduction_str = f"规约维度: {op_info.reduction_dims}" if op_info.has_reduction else "无规约"
        sensitivity = "🔴 敏感" if op_info.sensitivity == LayoutSensitivity.SENSITIVE else "🟢 不敏感"
        print(
            f"  {i}: {layer.name:10s} | {op_info.op_type:12s} | {reduction_str:20s} | {sensitivity}")

    # 传播分析
    propagator = LayoutPropagator(op_infos)
    groups = propagator.find_propagation_groups()

    print("\n" + "-" * 70)
    print("传播组 (共享分区):")
    print("-" * 70)

    for i, group in enumerate(groups):
        layer_names = [layers[idx].name for idx in sorted(group)]
        if len(group) > 1:
            print(f"  组 {i+1}: {' → '.join(layer_names)}")
            print(f"         └─ 这些层可以透传分区，无需重分布")
        else:
            idx = list(group)[0]
            sensitivity = op_infos[idx].sensitivity.value
            print(f"  组 {i+1}: {layer_names[0]} ({sensitivity})")

    # 统计
    sensitive_ops = propagator.get_sensitive_operators()
    insensitive_ops = propagator.get_insensitive_operators()

    print("\n" + "-" * 70)
    print("统计:")
    print("-" * 70)
    print(f"  布局敏感算子 (有规约): {len(sensitive_ops)} 个")
    print(f"    → 这些是分区决策点")
    print(f"  布局不敏感算子 (无规约): {len(insensitive_ops)} 个")
    print(f"    → 这些可以透传，减少 {len(insensitive_ops)} 个分区变量")

    print("\n" + "=" * 70)
    print("优化效果:")
    print("=" * 70)
    print(f"""
原始问题: {len(layers)} 个算子，每个独立决策分区
优化后:   {len(sensitive_ops)} 个分区决策点

减少决策变量: {len(layers) - len(sensitive_ops)} 个 ({100*(len(layers)-len(sensitive_ops))/len(layers):.1f}%)
""")


if __name__ == '__main__':
    demo()
