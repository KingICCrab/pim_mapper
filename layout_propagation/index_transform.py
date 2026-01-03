"""
显式 Layout 变换分析模块

基于 SmartMem (ASPLOS'24) 的索引依赖分析方法，分析 Reshape/Transpose/Permute
等显式 layout 变换算子对分区的影响。

核心概念：
- Split: 一个输入维度拆分为多个输出维度 (j -> j', k')
- Merge: 多个输入维度合并为一个输出维度 (i, j -> i')  
- Identity: 维度直接映射 (k -> l')

索引变换示例 (Figure 3 from SmartMem):
  输入: [2, 256, 4]  索引 (i, j, k)
       ↓ Reshape
  中间: [16, 8, 4, 4]
       ↓ Transpose
  输出: [16, 4, 8, 4]  索引 (i', j', k', l')
  
  变换公式:
    i' = i * 8 + j // (4 * 8)    ← Merge(i) + Split(j)
    j' = j % 4                   ← Split(j)
    k' = j % (4 * 8) // 4        ← Split(j)
    l' = k                       ← Identity(k)
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Union
import math


# ============================================================================
# 索引依赖类型
# ============================================================================

class DependencyType(Enum):
    """索引依赖类型"""
    IDENTITY = auto()   # 一对一映射: out[i'] = in[i], i' = i
    SPLIT = auto()      # 一对多分裂: out[i', j'] = in[i], 多个输出维度依赖一个输入维度
    MERGE = auto()      # 多对一合并: out[i'] = in[i, j], 一个输出维度依赖多个输入维度


@dataclass
class IndexDependency:
    """
    索引依赖关系

    描述输出张量的某个维度如何依赖输入张量的维度
    """
    output_dim: int                     # 输出维度索引
    input_dims: List[int]               # 依赖的输入维度索引列表
    dep_type: DependencyType            # 依赖类型
    transform_expr: Optional[str] = None  # 变换表达式 (用于展示)

    @property
    def is_identity(self) -> bool:
        return self.dep_type == DependencyType.IDENTITY

    @property
    def is_split(self) -> bool:
        return self.dep_type == DependencyType.SPLIT

    @property
    def is_merge(self) -> bool:
        return self.dep_type == DependencyType.MERGE


@dataclass
class IndexTransform:
    """
    完整的索引变换描述

    包含从输入形状到输出形状的所有维度映射关系
    """
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    dependencies: List[IndexDependency] = field(default_factory=list)

    @property
    def input_ndim(self) -> int:
        return len(self.input_shape)

    @property
    def output_ndim(self) -> int:
        return len(self.output_shape)

    def get_input_dims_for_output(self, out_dim: int) -> List[int]:
        """获取输出维度依赖的所有输入维度"""
        for dep in self.dependencies:
            if dep.output_dim == out_dim:
                return dep.input_dims
        return []

    def get_output_dims_for_input(self, in_dim: int) -> List[int]:
        """获取依赖某输入维度的所有输出维度"""
        result = []
        for dep in self.dependencies:
            if in_dim in dep.input_dims:
                result.append(dep.output_dim)
        return result


# ============================================================================
# Layout 变换算子
# ============================================================================

class LayoutOp(Enum):
    """Layout 变换算子类型"""
    RESHAPE = auto()      # 改变形状，不改变数据顺序
    TRANSPOSE = auto()    # 转置/排列维度
    PERMUTE = auto()      # 同 Transpose
    FLATTEN = auto()      # 展平
    SQUEEZE = auto()      # 移除大小为1的维度
    UNSQUEEZE = auto()    # 添加大小为1的维度
    VIEW = auto()         # PyTorch view (同 reshape)


@dataclass
class LayoutTransform:
    """
    Layout 变换操作
    """
    op_type: LayoutOp
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    # 对于 Transpose/Permute，需要指定维度排列
    perm: Optional[Tuple[int, ...]] = None
    # 对于 Squeeze/Unsqueeze，需要指定维度
    dim: Optional[int] = None


# ============================================================================
# 变换分析器
# ============================================================================

class TransformAnalyzer:
    """
    Layout 变换分析器

    分析 Reshape/Transpose 等操作的索引依赖关系，
    以及这些变换对分区传播的影响。
    """

    @staticmethod
    def analyze_reshape(input_shape: Tuple[int, ...],
                        output_shape: Tuple[int, ...]) -> IndexTransform:
        """
        分析 Reshape 操作的索引依赖

        Reshape 不改变数据的物理存储顺序，只改变逻辑形状。
        需要分析哪些输入维度被 Split，哪些被 Merge。

        Args:
            input_shape: 输入形状
            output_shape: 输出形状

        Returns:
            IndexTransform 描述完整的索引变换
        """
        # 验证总元素数相等
        in_total = math.prod(input_shape)
        out_total = math.prod(output_shape)
        if in_total != out_total:
            raise ValueError(
                f"Shape mismatch: {input_shape} ({in_total}) vs {output_shape} ({out_total})")

        transform = IndexTransform(input_shape, output_shape)

        # 计算累积步长 (stride)
        in_strides = TransformAnalyzer._compute_strides(input_shape)
        out_strides = TransformAnalyzer._compute_strides(output_shape)

        # 分析每个输出维度的依赖
        # 使用贪心匹配：根据步长关系确定依赖
        dependencies = TransformAnalyzer._match_dimensions(
            input_shape, output_shape, in_strides, out_strides
        )

        transform.dependencies = dependencies
        return transform

    @staticmethod
    def analyze_transpose(input_shape: Tuple[int, ...],
                          perm: Tuple[int, ...]) -> IndexTransform:
        """
        分析 Transpose/Permute 操作的索引依赖

        Transpose 是纯粹的维度重排，每个输出维度对应一个输入维度 (Identity)

        Args:
            input_shape: 输入形状
            perm: 维度排列，perm[i] 表示输出的第 i 维来自输入的第 perm[i] 维

        Returns:
            IndexTransform
        """
        if len(perm) != len(input_shape):
            raise ValueError(
                f"Perm length {len(perm)} != input ndim {len(input_shape)}")

        output_shape = tuple(input_shape[p] for p in perm)
        transform = IndexTransform(input_shape, output_shape)

        # Transpose 的每个输出维度都是 Identity 依赖
        for out_dim, in_dim in enumerate(perm):
            dep = IndexDependency(
                output_dim=out_dim,
                input_dims=[in_dim],
                dep_type=DependencyType.IDENTITY,
                transform_expr=f"out[{out_dim}] = in[{in_dim}]"
            )
            transform.dependencies.append(dep)

        return transform

    @staticmethod
    def analyze_flatten(input_shape: Tuple[int, ...],
                        start_dim: int = 0,
                        end_dim: int = -1) -> IndexTransform:
        """
        分析 Flatten 操作

        Flatten 是特殊的 Reshape，将连续的多个维度合并为一个
        """
        ndim = len(input_shape)
        if end_dim < 0:
            end_dim = ndim + end_dim

        # 计算输出形状
        pre_dims = input_shape[:start_dim]
        flat_dim = math.prod(input_shape[start_dim:end_dim+1])
        post_dims = input_shape[end_dim+1:]
        output_shape = pre_dims + (flat_dim,) + post_dims

        return TransformAnalyzer.analyze_reshape(input_shape, output_shape)

    @staticmethod
    def _compute_strides(shape: Tuple[int, ...]) -> List[int]:
        """计算 row-major 布局下的步长"""
        strides = []
        stride = 1
        for dim in reversed(shape):
            strides.append(stride)
            stride *= dim
        return list(reversed(strides))

    @staticmethod
    def _match_dimensions(input_shape: Tuple[int, ...],
                          output_shape: Tuple[int, ...],
                          in_strides: List[int],
                          out_strides: List[int]) -> List[IndexDependency]:
        """
        匹配输入输出维度的依赖关系

        基于步长分析确定 Split/Merge/Identity 关系
        """
        dependencies = []

        # 将维度按照步长排序，从大到小处理
        in_dims_by_stride = sorted(range(len(input_shape)),
                                   key=lambda i: in_strides[i], reverse=True)
        out_dims_by_stride = sorted(range(len(output_shape)),
                                    key=lambda i: out_strides[i], reverse=True)

        # 追踪哪些维度已被匹配
        matched_in = set()
        matched_out = set()

        # 记录每个输出维度依赖的输入维度
        out_to_in: Dict[int, List[int]] = {i: []
                                           for i in range(len(output_shape))}

        # 贪心匹配：尝试找到步长匹配的维度
        total_elements = math.prod(input_shape)

        # 构建维度范围映射
        # 每个维度覆盖的元素范围 [start, end)
        in_ranges = []
        pos = 0
        for i, dim in enumerate(input_shape):
            stride = in_strides[i]
            in_ranges.append((pos, pos + dim * stride, i))
            pos = 0  # Reset for next dim analysis

        # 简化版本：基于大小因子关系判断
        in_idx = 0
        out_idx = 0
        in_accum = 1
        out_accum = 1

        current_in_dims = []
        current_out_dims = []

        while in_idx < len(input_shape) or out_idx < len(output_shape):
            if in_idx < len(input_shape) and in_accum <= out_accum:
                in_accum *= input_shape[in_idx]
                current_in_dims.append(in_idx)
                in_idx += 1
            elif out_idx < len(output_shape):
                out_accum *= output_shape[out_idx]
                current_out_dims.append(out_idx)
                out_idx += 1

            # 当累积大小相等时，确定一组依赖
            if in_accum == out_accum and (current_in_dims or current_out_dims):
                if len(current_out_dims) == 1 and len(current_in_dims) == 1:
                    # Identity
                    dep_type = DependencyType.IDENTITY
                elif len(current_out_dims) > 1:
                    # Split: 一个或多个输入维度分裂到多个输出维度
                    dep_type = DependencyType.SPLIT
                else:
                    # Merge: 多个输入维度合并到一个输出维度
                    dep_type = DependencyType.MERGE

                # 为每个输出维度创建依赖
                for o_dim in current_out_dims:
                    dep = IndexDependency(
                        output_dim=o_dim,
                        input_dims=list(current_in_dims),
                        dep_type=dep_type
                    )
                    dependencies.append(dep)

                # 重置
                current_in_dims = []
                current_out_dims = []
                in_accum = 1
                out_accum = 1

        return dependencies


# ============================================================================
# 分区兼容性分析
# ============================================================================

@dataclass
class PartitionCompatibility:
    """分区兼容性分析结果"""
    is_compatible: bool
    reason: str
    suggested_transform: Optional[str] = None
    redistribution_cost: float = 0.0


class PartitionTransformAnalyzer:
    """
    分析 Layout 变换对分区的影响

    核心问题：如果输入按某个维度分区，经过 Reshape/Transpose 后，
    输出的分区方式是什么？是否需要数据重分布？
    """

    @staticmethod
    def analyze_partition_through_reshape(
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        partition_dim: int,
        partition_factor: int
    ) -> PartitionCompatibility:
        """
        分析分区如何通过 Reshape 传播

        Args:
            input_shape: 输入形状
            output_shape: 输出形状
            partition_dim: 输入的分区维度
            partition_factor: 分区因子

        Returns:
            PartitionCompatibility 描述兼容性和代价
        """
        transform = TransformAnalyzer.analyze_reshape(
            input_shape, output_shape)

        # 找到输入分区维度对应的输出维度
        output_dims = transform.get_output_dims_for_input(partition_dim)

        if not output_dims:
            return PartitionCompatibility(
                is_compatible=False,
                reason=f"Input dim {partition_dim} has no mapping in output",
                redistribution_cost=math.prod(input_shape)
            )

        # 检查分区是否可以保持
        input_dim_size = input_shape[partition_dim]

        # 获取依赖类型
        dep = None
        for d in transform.dependencies:
            if partition_dim in d.input_dims:
                dep = d
                break

        if dep is None:
            return PartitionCompatibility(
                is_compatible=False,
                reason="No dependency found",
                redistribution_cost=math.prod(input_shape)
            )

        if dep.dep_type == DependencyType.IDENTITY:
            # Identity: 分区直接传递
            return PartitionCompatibility(
                is_compatible=True,
                reason=f"Identity mapping: input dim {partition_dim} -> output dim {output_dims[0]}",
                redistribution_cost=0.0
            )

        elif dep.dep_type == DependencyType.MERGE:
            # Merge: 检查分区是否与合并兼容
            # 如果分区维度是最内层的合并维度，可能兼容
            merged_size = math.prod(input_shape[d] for d in dep.input_dims)
            out_dim_size = output_shape[dep.output_dim]

            if merged_size == out_dim_size and input_dim_size % partition_factor == 0:
                # 可能兼容，但需要检查数据连续性
                return PartitionCompatibility(
                    is_compatible=True,
                    reason=f"Merge compatible: partition preserved in merged dim {dep.output_dim}",
                    suggested_transform=f"Output partitioned on dim {dep.output_dim}",
                    redistribution_cost=0.0
                )
            else:
                return PartitionCompatibility(
                    is_compatible=False,
                    reason=f"Merge breaks partition continuity",
                    redistribution_cost=math.prod(input_shape) * 0.5
                )

        elif dep.dep_type == DependencyType.SPLIT:
            # Split: 分区会分散到多个维度
            # 检查是否可以在某个输出维度上保持分区
            for out_dim in output_dims:
                out_size = output_shape[out_dim]
                if input_dim_size % out_size == 0 or out_size % partition_factor == 0:
                    return PartitionCompatibility(
                        is_compatible=True,
                        reason=f"Split: partition can be on output dim {out_dim}",
                        suggested_transform=f"Repartition on output dim {out_dim}",
                        redistribution_cost=math.prod(input_shape) * 0.1
                    )

            return PartitionCompatibility(
                is_compatible=False,
                reason="Split breaks partition across multiple dims",
                redistribution_cost=math.prod(input_shape) * 0.8
            )

        return PartitionCompatibility(
            is_compatible=False,
            reason="Unknown dependency type",
            redistribution_cost=math.prod(input_shape)
        )

    @staticmethod
    def analyze_partition_through_transpose(
        input_shape: Tuple[int, ...],
        perm: Tuple[int, ...],
        partition_dim: int,
        partition_factor: int
    ) -> PartitionCompatibility:
        """
        分析分区如何通过 Transpose 传播

        Transpose 只是重排维度顺序，分区总是兼容的，
        只是分区维度的索引会改变。
        """
        # 找到输入分区维度在输出中的位置
        output_partition_dim = perm.index(
            partition_dim) if partition_dim in perm else -1

        if output_partition_dim == -1:
            return PartitionCompatibility(
                is_compatible=False,
                reason=f"Partition dim {partition_dim} not in permutation",
                redistribution_cost=math.prod(input_shape)
            )

        # Transpose 不需要数据移动（只是视图变换），但可能影响内存访问模式
        # 如果分区维度变成非连续维度，可能影响效率
        is_contiguous = (output_partition_dim == len(perm) - 1)

        return PartitionCompatibility(
            is_compatible=True,
            reason=f"Transpose: input dim {partition_dim} -> output dim {output_partition_dim}",
            suggested_transform=f"Partition moved to dim {output_partition_dim}",
            redistribution_cost=0.0 if is_contiguous else math.prod(
                input_shape) * 0.05
        )


# ============================================================================
# 变换序列分析
# ============================================================================

class TransformSequenceAnalyzer:
    """
    分析 Layout 变换序列

    在实际网络中，经常有连续的 Reshape + Transpose 组合，
    需要分析整个序列对分区的累积影响。
    """

    def __init__(self):
        self.transforms: List[LayoutTransform] = []
        self.index_transforms: List[IndexTransform] = []

    def add_reshape(self, input_shape: Tuple[int, ...],
                    output_shape: Tuple[int, ...]):
        """添加 Reshape 变换"""
        self.transforms.append(LayoutTransform(
            op_type=LayoutOp.RESHAPE,
            input_shape=input_shape,
            output_shape=output_shape
        ))
        self.index_transforms.append(
            TransformAnalyzer.analyze_reshape(input_shape, output_shape)
        )

    def add_transpose(self, input_shape: Tuple[int, ...],
                      perm: Tuple[int, ...]):
        """添加 Transpose 变换"""
        output_shape = tuple(input_shape[p] for p in perm)
        self.transforms.append(LayoutTransform(
            op_type=LayoutOp.TRANSPOSE,
            input_shape=input_shape,
            output_shape=output_shape,
            perm=perm
        ))
        self.index_transforms.append(
            TransformAnalyzer.analyze_transpose(input_shape, perm)
        )

    def trace_partition(self, initial_partition_dim: int,
                        partition_factor: int) -> List[Tuple[int, PartitionCompatibility]]:
        """
        追踪分区通过整个变换序列

        Returns:
            List of (current_partition_dim, compatibility) for each step
        """
        results = []
        current_dim = initial_partition_dim

        for i, (transform, idx_transform) in enumerate(
            zip(self.transforms, self.index_transforms)
        ):
            if transform.op_type == LayoutOp.RESHAPE:
                compat = PartitionTransformAnalyzer.analyze_partition_through_reshape(
                    transform.input_shape,
                    transform.output_shape,
                    current_dim,
                    partition_factor
                )
            elif transform.op_type in (LayoutOp.TRANSPOSE, LayoutOp.PERMUTE):
                compat = PartitionTransformAnalyzer.analyze_partition_through_transpose(
                    transform.input_shape,
                    transform.perm,
                    current_dim,
                    partition_factor
                )
            else:
                compat = PartitionCompatibility(
                    is_compatible=False,
                    reason=f"Unsupported op type: {transform.op_type}"
                )

            results.append((current_dim, compat))

            # 更新当前分区维度
            if compat.is_compatible:
                # 从 suggested_transform 中提取新的分区维度
                if "dim" in compat.suggested_transform:
                    import re
                    match = re.search(r'dim (\d+)', compat.suggested_transform)
                    if match:
                        current_dim = int(match.group(1))
                elif transform.op_type == LayoutOp.TRANSPOSE:
                    current_dim = transform.perm.index(current_dim)

        return results

    def get_total_cost(self, initial_partition_dim: int,
                       partition_factor: int) -> float:
        """计算通过整个变换序列的总重分布代价"""
        results = self.trace_partition(initial_partition_dim, partition_factor)
        return sum(r[1].redistribution_cost for r in results)


# ============================================================================
# 演示
# ============================================================================

def demo():
    """演示 Layout 变换分析"""

    print("=" * 70)
    print("显式 Layout 变换分析演示")
    print("=" * 70)

    # 示例 1: SmartMem Figure 3 的例子
    print("\n" + "-" * 70)
    print("示例 1: SmartMem Figure 3 - Reshape + Transpose")
    print("-" * 70)

    print("""
    输入: [2, 256, 4]  索引 (i, j, k)
           ↓ Reshape
    中间: [16, 8, 4, 4]
           ↓ Transpose(0, 2, 1, 3)
    输出: [16, 4, 8, 4]  索引 (i', j', k', l')
    """)

    # Reshape 分析
    input_shape = (2, 256, 4)
    mid_shape = (16, 8, 4, 4)

    print("Reshape [2, 256, 4] -> [16, 8, 4, 4]:")
    reshape_transform = TransformAnalyzer.analyze_reshape(
        input_shape, mid_shape)
    for dep in reshape_transform.dependencies:
        print(
            f"  Output dim {dep.output_dim}: {dep.dep_type.name} from input dims {dep.input_dims}")

    # Transpose 分析
    perm = (0, 2, 1, 3)
    print(f"\nTranspose with perm {perm}:")
    transpose_transform = TransformAnalyzer.analyze_transpose(mid_shape, perm)
    for dep in transpose_transform.dependencies:
        print(
            f"  Output dim {dep.output_dim}: {dep.dep_type.name} from input dim {dep.input_dims}")

    # 示例 2: Transformer 中的 Multi-Head Attention reshape
    print("\n" + "-" * 70)
    print("示例 2: Multi-Head Attention reshape")
    print("-" * 70)

    print("""
    QKV projection output: [Batch, SeqLen, NumHeads * HeadDim]
                              ↓ Reshape
    Multi-head format:      [Batch, SeqLen, NumHeads, HeadDim]
                              ↓ Transpose
    Attention input:        [Batch, NumHeads, SeqLen, HeadDim]
    """)

    batch, seq_len, num_heads, head_dim = 4, 512, 8, 64
    qkv_shape = (batch, seq_len, num_heads * head_dim)
    mh_shape = (batch, seq_len, num_heads, head_dim)
    attn_shape = (batch, num_heads, seq_len, head_dim)

    # 分析完整序列
    seq_analyzer = TransformSequenceAnalyzer()
    seq_analyzer.add_reshape(qkv_shape, mh_shape)
    seq_analyzer.add_transpose(mh_shape, (0, 2, 1, 3))

    print(f"\n追踪分区 (初始分区: dim 2, factor 4):")
    results = seq_analyzer.trace_partition(
        initial_partition_dim=2, partition_factor=4)
    for i, (dim, compat) in enumerate(results):
        op = seq_analyzer.transforms[i].op_type.name
        print(f"  Step {i+1} ({op}): dim {dim}")
        print(f"    Compatible: {compat.is_compatible}")
        print(f"    Reason: {compat.reason}")
        if compat.suggested_transform:
            print(f"    Suggested: {compat.suggested_transform}")
        print(f"    Cost: {compat.redistribution_cost:.0f}")

    total_cost = seq_analyzer.get_total_cost(2, 4)
    print(f"\n  Total redistribution cost: {total_cost:.0f}")

    # 示例 3: 分区兼容性检查
    print("\n" + "-" * 70)
    print("示例 3: 分区兼容性检查")
    print("-" * 70)

    test_cases = [
        # (input_shape, output_shape, partition_dim, partition_factor, description)
        ((16, 256), (16, 16, 16), 1, 4,
         "Split: [16, 256] -> [16, 16, 16], partition on dim 1"),
        ((4, 8, 16), (32, 16), 0, 2,
         "Merge: [4, 8, 16] -> [32, 16], partition on dim 0"),
        ((8, 8, 8), (8, 64), 2, 2,
         "Merge last dims: [8, 8, 8] -> [8, 64], partition on dim 2"),
        ((64, 128), (64, 128), 0, 4,
         "Identity: [64, 128] -> [64, 128], partition on dim 0"),
    ]

    for input_shape, output_shape, p_dim, p_factor, desc in test_cases:
        print(f"\n{desc}")
        compat = PartitionTransformAnalyzer.analyze_partition_through_reshape(
            input_shape, output_shape, p_dim, p_factor
        )
        print(f"  Compatible: {compat.is_compatible}")
        print(f"  Reason: {compat.reason}")
        print(f"  Cost: {compat.redistribution_cost:.0f}")

    print("\n" + "=" * 70)
    print("应用总结")
    print("=" * 70)
    print("""
显式 Layout 变换分析的应用场景：

1. 编译器优化
   - 判断 Reshape/Transpose 能否被消除
   - 找到最优的 layout 选择以最小化变换

2. 分区优化
   - 判断分区能否穿过 layout 变换
   - 计算必要的重分布代价

3. 算子融合
   - 判断 layout 变换能否与相邻算子融合
   - 识别可以消除的冗余变换

4. 与隐式传播分析结合
   - 计算算子: 使用 reduction 敏感性分析
   - Layout 算子: 使用索引依赖分析
   - 统一的分区传播框架
""")


if __name__ == '__main__':
    demo()
