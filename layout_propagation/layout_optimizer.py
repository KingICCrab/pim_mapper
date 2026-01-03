"""
Layout 优化器

基于算子最优 layout 搜索和 activation layout 传播的全局优化框架。

核心思想：
1. 对于每个算子，假设已有工具搜索其最佳 layout
2. 区分输入 tensor 类型：Weight vs Activation
   - Weight: 静态数据，可以预先转换为任意 layout，无运行时代价
   - Activation: 动态数据，需要通过 layout propagation 传递
3. Activation 的 layout 由上游算子决定，通过传播机制统一管理
4. 算子写回时按照传播的 layout 写回，供下游消费

这与 SmartMem 的思路一致：
- Weight 的 layout 自由选择（预处理时转换）
- Activation 的 layout 需要在算子间传播协调
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
from abc import ABC, abstractmethod

import sys
import os

# Add current directory to path to import modules directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from .data_layout import DataLayout
except ImportError:
    from data_layout import DataLayout

# ============================================================================
# 数据类型分类
# ============================================================================


class TensorType(Enum):
    """Tensor 类型"""
    WEIGHT = auto()      # 权重：静态，可预转换
    ACTIVATION = auto()  # 激活：动态，需传播
    CONSTANT = auto()    # 常量：类似 weight
    UNKNOWN = auto()


@dataclass
class TensorInfo:
    """
    Tensor 信息
    """
    name: str
    tensor_type: TensorType
    shape: Tuple[int, ...]
    layout: Optional[DataLayout] = None
    # 来源算子 (None 表示网络输入)
    producer: Optional[str] = None
    # 消费算子列表
    consumers: List[str] = field(default_factory=list)


@dataclass
class OperatorInfo:
    """
    算子信息
    """
    name: str
    op_type: str
    # 输入 tensor 名称列表
    inputs: List[str]
    # 输出 tensor 名称列表
    outputs: List[str]
    # 每个输入的类型 (weight/activation)
    input_types: Dict[str, TensorType] = field(default_factory=dict)
    # 最优 layout 配置 (由外部搜索工具提供)
    optimal_input_layouts: Dict[str, DataLayout] = field(
        default_factory=dict)
    optimal_output_layouts: Dict[str, DataLayout] = field(
        default_factory=dict)
    # 算子属性
    attributes: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Layout 搜索接口 (抽象)
# ============================================================================

class LayoutSearcher:
    """
    Layout 搜索器接口

    假设已有的算子最优 layout 搜索工具实现此接口
    """

    def search_optimal_layout(self,
                              op: OperatorInfo,
                              input_shapes: Dict[str, Tuple[int, ...]],
                              output_shapes: Dict[str, Tuple[int, ...]],
                              constraints: Optional[Dict[str,
                                                         DataLayout]] = None
                              ) -> Tuple[Dict[str, DataLayout], Dict[str, DataLayout]]:
        """
        搜索算子的最优 layout

        Args:
            op: 算子信息
            input_shapes: 输入形状
            output_shapes: 输出形状
            constraints: layout 约束 (来自上游传播)

        Returns:
            (optimal_input_layouts, optimal_output_layouts)
        """
        raise NotImplementedError


class DefaultLayoutSearcher(LayoutSearcher):
    """
    默认 Layout 搜索器 (简单启发式)
    """

    def search_optimal_layout(self,
                              op: OperatorInfo,
                              input_shapes: Dict[str, Tuple[int, ...]],
                              output_shapes: Dict[str, Tuple[int, ...]],
                              constraints: Optional[Dict[str,
                                                         DataLayout]] = None
                              ) -> Tuple[Dict[str, DataLayout], Dict[str, DataLayout]]:
        constraints = constraints or {}
        input_layouts = {}
        output_layouts = {}

        # 辅助函数：创建 NCHW layout
        def create_nchw(shape):
            if len(shape) == 4:
                return DataLayout.from_nchw(shape[0], shape[1], shape[2], shape[3])
            return DataLayout.from_shape(shape)

        # 辅助函数：创建 Row Major layout
        def create_row_major(shape):
            return DataLayout.from_shape(shape)

        # 简单的算子类型判断
        is_conv_like = op.op_type in [
            'Conv', 'ConvLayer', 'Pool', 'PoolingLayer', 'BatchNorm']

        # 处理输入
        for inp in op.inputs:
            shape = input_shapes.get(inp, (1,))
            if not shape:
                shape = (1,)

            if inp in constraints:
                input_layouts[inp] = constraints[inp]
            else:
                if is_conv_like and len(shape) == 4:
                    input_layouts[inp] = create_nchw(shape)
                else:
                    input_layouts[inp] = create_row_major(shape)

        # 处理输出
        for out in op.outputs:
            shape = output_shapes.get(out, (1,))
            if not shape or shape == ():
                if op.inputs and op.inputs[0] in input_shapes:
                    shape = input_shapes[op.inputs[0]]
                else:
                    shape = (1,)

            if is_conv_like and len(shape) == 4:
                output_layouts[out] = create_nchw(shape)
            else:
                output_layouts[out] = create_row_major(shape)

        return input_layouts, output_layouts


# ============================================================================
# Layout 传播优化器
# ============================================================================

class LayoutPropagationOptimizer:
    """
    Layout 传播优化器

    核心流程：
    1. 构建计算图
    2. 识别每个 tensor 的类型 (weight/activation)
    3. 对于 weight: 可自由选择 layout (编译时转换)
    4. 对于 activation: 通过传播确定 layout
    5. 在算子边界处理 layout 转换
    """

    def __init__(self, layout_searcher: Optional[LayoutSearcher] = None):
        self.layout_searcher = layout_searcher or DefaultLayoutSearcher()

        # 图结构
        self.tensors: Dict[str, TensorInfo] = {}
        self.operators: Dict[str, OperatorInfo] = {}
        self.op_order: List[str] = []  # 拓扑顺序

        # 优化结果
        self.final_layouts: Dict[str, DataLayout] = {}
        self.required_transforms: List[Tuple[str,
                                             str, DataLayout, DataLayout]] = []

    def add_tensor(self, name: str, tensor_type: TensorType,
                   shape: Tuple[int, ...],
                   initial_layout: Optional[DataLayout] = None):
        """添加 tensor"""
        self.tensors[name] = TensorInfo(
            name=name,
            tensor_type=tensor_type,
            shape=shape,
            layout=initial_layout
        )

    def add_operator(self, name: str, op_type: str,
                     inputs: List[str], outputs: List[str],
                     input_types: Optional[Dict[str, TensorType]] = None,
                     attributes: Optional[Dict[str, Any]] = None):
        """添加算子"""
        # 推断输入类型
        if input_types is None:
            input_types = {}
            for inp in inputs:
                if inp in self.tensors:
                    input_types[inp] = self.tensors[inp].tensor_type
                else:
                    input_types[inp] = TensorType.ACTIVATION

        op = OperatorInfo(
            name=name,
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            input_types=input_types,
            attributes=attributes or {}
        )
        self.operators[name] = op
        self.op_order.append(name)

        # 更新 tensor 的 producer/consumer 关系
        for inp in inputs:
            if inp in self.tensors:
                self.tensors[inp].consumers.append(name)

        for out in outputs:
            if out not in self.tensors:
                # 自动创建输出 tensor (activation)
                self.tensors[out] = TensorInfo(
                    name=out,
                    tensor_type=TensorType.ACTIVATION,
                    shape=(),  # 形状后续推断
                    producer=name
                )
            else:
                self.tensors[out].producer = name

    def _topological_sort(self) -> List[str]:
        """拓扑排序"""
        in_degree = {op: 0 for op in self.operators}

        # 计算入度
        for op_name, op in self.operators.items():
            for inp in op.inputs:
                if inp in self.tensors:
                    producer = self.tensors[inp].producer
                    if producer and producer in self.operators:
                        in_degree[op_name] += 1

        # BFS
        queue = deque([op for op, deg in in_degree.items() if deg == 0])
        sorted_ops = []

        while queue:
            op_name = queue.popleft()
            sorted_ops.append(op_name)

            op = self.operators[op_name]
            for out in op.outputs:
                if out in self.tensors:
                    for consumer in self.tensors[out].consumers:
                        if consumer in in_degree:
                            in_degree[consumer] -= 1
                            if in_degree[consumer] == 0:
                                queue.append(consumer)

        return sorted_ops

    def optimize(self) -> Dict[str, DataLayout]:
        """
        执行 layout 优化

        核心算法：
        1. 按拓扑顺序遍历算子
        2. 对于每个算子：
           a. 收集 activation 输入的 layout 约束 (来自上游)
           b. 调用 layout 搜索器获取最优 layout
           c. 对于 weight 输入：直接使用搜索结果
           d. 对于 activation 输入：检查是否需要转换
           e. 设置输出 layout，传播给下游
        """
        # 拓扑排序
        sorted_ops = self._topological_sort()

        self.final_layouts = {}
        self.required_transforms = []

        print("=" * 70)
        print("Layout 优化过程")
        print("=" * 70)

        for op_name in sorted_ops:
            op = self.operators[op_name]
            print(f"\n处理算子: {op_name} ({op.op_type})")
            print("-" * 50)

            # 步骤 1: 收集 activation 输入的 layout 约束
            activation_constraints = {}
            for inp in op.inputs:
                inp_type = op.input_types.get(inp, TensorType.ACTIVATION)

                if inp_type == TensorType.ACTIVATION:
                    # Activation: 检查是否有上游传播的 layout
                    if inp in self.final_layouts:
                        activation_constraints[inp] = self.final_layouts[inp]
                        print(
                            f"  输入 {inp} (Activation): 继承上游 layout = {self.final_layouts[inp].format_name}")
                    else:
                        print(f"  输入 {inp} (Activation): 无上游约束")
                else:
                    # Weight: 无约束，可自由选择
                    print(f"  输入 {inp} (Weight): 可自由选择 layout")

            # 步骤 2: 搜索最优 layout
            input_shapes = {inp: self.tensors[inp].shape
                            for inp in op.inputs if inp in self.tensors}
            output_shapes = {out: self.tensors[out].shape
                             for out in op.outputs if out in self.tensors}

            optimal_in, optimal_out = self.layout_searcher.search_optimal_layout(
                op, input_shapes, output_shapes, activation_constraints
            )

            print(f"  搜索结果:")
            print(
                f"    最优输入 layout: {[(k, v.format_name) for k, v in optimal_in.items()]}")
            print(
                f"    最优输出 layout: {[(k, v.format_name) for k, v in optimal_out.items()]}")

            # 步骤 3: 处理每个输入
            for inp in op.inputs:
                inp_type = op.input_types.get(inp, TensorType.ACTIVATION)
                optimal_layout = optimal_in.get(inp)
                if not optimal_layout:
                    shape = input_shapes.get(inp, (1,))
                    optimal_layout = DataLayout.from_shape(shape)

                if inp_type == TensorType.WEIGHT:
                    # Weight: 直接使用最优 layout (编译时预转换)
                    self.final_layouts[inp] = optimal_layout
                    print(
                        f"  → Weight {inp}: 设置为 {optimal_layout.format_name} (预处理转换)")

                elif inp_type == TensorType.ACTIVATION:
                    # Activation: 检查是否需要运行时转换
                    if inp in activation_constraints:
                        upstream_layout = activation_constraints[inp]
                        if not upstream_layout.is_compatible(optimal_layout):
                            # 需要转换！记录下来
                            self.required_transforms.append(
                                (inp, op_name, upstream_layout, optimal_layout)
                            )
                            print(
                                f"  → Activation {inp}: 需要转换 {upstream_layout.format_name} -> {optimal_layout.format_name}")
                            # 使用转换后的 layout
                            self.final_layouts[inp] = optimal_layout
                        else:
                            # 兼容，无需转换
                            self.final_layouts[inp] = upstream_layout
                            print(
                                f"  → Activation {inp}: 保持 {upstream_layout.format_name} (兼容)")
                    else:
                        # 首个输入，设置 layout
                        self.final_layouts[inp] = optimal_layout
                        print(
                            f"  → Activation {inp}: 初始化为 {optimal_layout.format_name}")

            # 步骤 4: 设置输出 layout，用于下游传播
            for out in op.outputs:
                out_layout = optimal_out.get(out)
                if not out_layout:
                    shape = output_shapes.get(out, (1,))
                    out_layout = DataLayout.from_shape(shape)
                self.final_layouts[out] = out_layout
                print(f"  → 输出 {out}: 写回为 {out_layout.format_name} (传播给下游)")

        # 汇总
        print("\n" + "=" * 70)
        print("优化结果汇总")
        print("=" * 70)

        print("\n最终 Layout 分配:")
        for tensor_name, layout in self.final_layouts.items():
            tensor = self.tensors.get(tensor_name)
            if tensor:
                print(
                    f"  {tensor_name} ({tensor.tensor_type.name}): {layout.format_name}")

        if self.required_transforms:
            print(f"\n需要的运行时 Layout 转换 ({len(self.required_transforms)} 个):")
            for tensor, op, src, dst in self.required_transforms:
                print(
                    f"  {tensor} @ {op}: {src.format_name} -> {dst.format_name}")
        else:
            print("\n无需运行时 Layout 转换！")

        return self.final_layouts

    def get_transform_cost(self) -> float:
        """计算 layout 转换的总代价"""
        total_cost = 0.0
        for tensor, _, _, _ in self.required_transforms:
            if tensor in self.tensors:
                # 简单估计：转换代价 = tensor 大小
                import math
                shape = self.tensors[tensor].shape
                if shape:
                    total_cost += math.prod(shape)
        return total_cost


# ============================================================================
# 高级 API: 基于 Activation 传播的 Layout 优化
# ============================================================================

def optimize_network_layout(
    operators: List[Dict],
    tensors: List[Dict],
    layout_searcher: Optional[LayoutSearcher] = None
) -> Dict[str, DataLayout]:
    """
    优化整个网络的 layout

    Args:
        operators: 算子列表，每个元素是 dict:
            {
                'name': str,
                'type': str,
                'inputs': List[str],
                'outputs': List[str],
                'input_types': Dict[str, str]  # 'weight' or 'activation'
            }
        tensors: tensor 列表，每个元素是 dict:
            {
                'name': str,
                'type': str,  # 'weight', 'activation', 'input', 'output'
                'shape': Tuple[int, ...]
            }
        layout_searcher: 可选的 layout 搜索器

    Returns:
        每个 tensor 的最终 layout
    """
    optimizer = LayoutPropagationOptimizer(layout_searcher)

    # 添加 tensors
    type_map = {
        'weight': TensorType.WEIGHT,
        'activation': TensorType.ACTIVATION,
        'input': TensorType.ACTIVATION,
        'output': TensorType.ACTIVATION,
        'constant': TensorType.CONSTANT,
    }

    for t in tensors:
        optimizer.add_tensor(
            name=t['name'],
            tensor_type=type_map.get(
                t.get('type', 'activation'), TensorType.ACTIVATION),
            shape=tuple(t.get('shape', []))
        )

    # 添加 operators
    for op in operators:
        input_types = {}
        for inp, inp_type in op.get('input_types', {}).items():
            input_types[inp] = type_map.get(inp_type, TensorType.ACTIVATION)

        optimizer.add_operator(
            name=op['name'],
            op_type=op['type'],
            inputs=op['inputs'],
            outputs=op['outputs'],
            input_types=input_types
        )

    return optimizer.optimize()


# ============================================================================
# 演示
# ============================================================================

def demo():
    """演示 Layout 传播优化"""

    print("=" * 70)
    print("Layout 传播优化演示")
    print("=" * 70)

    print("""
场景说明：
─────────────────────────────────────────────────────────────────────────

假设已有算子级 layout 搜索工具，本模块解决的问题是：

1. Weight vs Activation 的区别对待：
   - Weight: 静态数据，可在编译时预转换为任意 layout，无运行时代价
   - Activation: 动态数据，layout 需要在算子间传播协调

2. Layout 传播机制：
   - 上游算子的输出 layout → 下游算子的输入约束
   - 避免不必要的运行时 layout 转换

3. 决策逻辑：
   - 尊重 activation 的上游 layout (减少转换)
   - weight 可自由选择最优 layout
   - 仅在必要时插入 layout 转换

─────────────────────────────────────────────────────────────────────────
""")

    # 构建示例网络: Conv -> ReLU -> Conv -> Pool -> FC
    optimizer = LayoutPropagationOptimizer()

    # 添加 tensors
    # 输入
    optimizer.add_tensor('input', TensorType.ACTIVATION, (1, 3, 224, 224))

    # Conv1 的 weight
    optimizer.add_tensor('conv1_weight', TensorType.WEIGHT, (64, 3, 7, 7))
    # Conv1 output (假设 stride=2)
    optimizer.add_tensor('conv1_out', TensorType.ACTIVATION, (1, 64, 112, 112))

    # ReLU1 output
    optimizer.add_tensor('relu1_out', TensorType.ACTIVATION, (1, 64, 112, 112))

    # Conv2 的 weight
    optimizer.add_tensor('conv2_weight', TensorType.WEIGHT, (128, 64, 3, 3))
    # Conv2 output (假设 stride=2)
    optimizer.add_tensor('conv2_out', TensorType.ACTIVATION, (1, 128, 56, 56))

    # Pool output (假设 2x2)
    optimizer.add_tensor('pool_out', TensorType.ACTIVATION, (1, 128, 28, 28))

    # FC 的 weight (128*28*28 = 100352)
    optimizer.add_tensor('fc_weight', TensorType.WEIGHT, (1000, 100352))
    # FC output
    optimizer.add_tensor('output', TensorType.ACTIVATION, (1, 1000))

    # 添加算子
    optimizer.add_operator(
        name='conv1',
        op_type='Conv',
        inputs=['input', 'conv1_weight'],
        outputs=['conv1_out'],
        input_types={'input': TensorType.ACTIVATION,
                     'conv1_weight': TensorType.WEIGHT}
    )

    optimizer.add_operator(
        name='relu1',
        op_type='ReLU',
        inputs=['conv1_out'],
        outputs=['relu1_out'],
        input_types={'conv1_out': TensorType.ACTIVATION}
    )

    optimizer.add_operator(
        name='conv2',
        op_type='Conv',
        inputs=['relu1_out', 'conv2_weight'],
        outputs=['conv2_out'],
        input_types={'relu1_out': TensorType.ACTIVATION,
                     'conv2_weight': TensorType.WEIGHT}
    )

    optimizer.add_operator(
        name='pool',
        op_type='Pool',
        inputs=['conv2_out'],
        outputs=['pool_out'],
        input_types={'conv2_out': TensorType.ACTIVATION}
    )

    optimizer.add_operator(
        name='fc',
        op_type='FC',
        inputs=['pool_out', 'fc_weight'],
        outputs=['output'],
        input_types={'pool_out': TensorType.ACTIVATION,
                     'fc_weight': TensorType.WEIGHT}
    )

    # 执行优化
    layouts = optimizer.optimize()

    # 转换代价
    cost = optimizer.get_transform_cost()
    print(f"\n总 Layout 转换代价: {cost}")

    print("\n" + "=" * 70)
    print("关键洞察")
    print("=" * 70)
    print("""
1. Weight 的 layout 自由选择
   - conv1_weight, conv2_weight, fc_weight 都可以预先转换
   - 这些转换在编译/加载时完成，无运行时开销

2. Activation 的 layout 传播
   - input → conv1_out → relu1_out → conv2_out → pool_out → output
   - 整条链路的 layout 需要协调一致

3. ReLU 等 Elementwise 算子
   - 对 layout 不敏感 (ANY)
   - 自动继承上游的 layout，无需转换

4. FC 层的特殊处理
   - 可能需要 NCHW -> ROW_MAJOR 的转换
   - 但因为 pool_out 是 activation，需要评估转换代价
""")


if __name__ == '__main__':
    demo()
