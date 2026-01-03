"""
ILP-based Global Partition Optimizer V2.

完整支持混合分区和所有维度的传播约束。

Key Insights (所有分区维度的传播):
    1. OUTP (K) → C: Layer[l].K_partition → Layer[l+1].input_C_distribution
    2. OFMP (H,W) → Spatial: Layer[l].spatial_partition → Layer[l+1].input_spatial_distribution
    3. BATP (N) → Batch: Layer[l].batch_partition → Layer[l+1].batch_distribution (通常匹配)
    4. INPP (C) → Reduction: Layer内部需要 All-Reduce

混合分区:
    每个 PartitionChoice 可以同时指定多个维度的分区因子，
    例如: OUTP=2, OFMP_H=2, OFMP_W=2 使用 2*2*2=8 个节点
"""

import itertools
from enum import IntEnum
from typing import List, Dict, Tuple, Optional, Set
from math import prod

# Try to import ILP solvers
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


class PartDim(IntEnum):
    """Partition dimensions aligned with nn_dataflow parallel_enum."""
    OUTP = 0    # K - Output channels
    OFMP = 1    # H×W - Output feature map (2D)
    BATP = 2    # N - Batch
    INPP = 3    # C - Input channels (requires reduction)
    NUM = 4


class HybridPartitionChoice:
    """
    混合分区方案 - 与 nn_dataflow 的 PartitionScheme 对齐。

    每个维度都有一个 2D 分区因子 (h, w)，对应物理节点阵列。
    """

    def __init__(self, pdims: Dict[PartDim, Tuple[int, int]], order: Tuple[int, ...] = None):
        """
        Args:
            pdims: {PartDim: (h_factor, w_factor)}
                例如: {OUTP: (2, 2), OFMP: (2, 2)} 表示 K 分到 2×2, 空间分到 2×2
            order: 分区层级顺序 (可选)
        """
        # 确保所有维度都有值，默认 (1, 1)
        self.pdims = {}
        for dim in range(PartDim.NUM):
            self.pdims[dim] = pdims.get(dim, (1, 1))

        # 分区顺序
        self.order = order if order else tuple(range(PartDim.NUM))

    def get_factor(self, dim: PartDim) -> Tuple[int, int]:
        """获取某维度的分区因子 (h, w)。"""
        return self.pdims.get(dim, (1, 1))

    def get_size(self, dim: PartDim) -> int:
        """获取某维度使用的节点数。"""
        h, w = self.get_factor(dim)
        return h * w

    @property
    def total_nodes(self) -> int:
        """总节点数。"""
        result = 1
        for dim in range(PartDim.NUM):
            result *= self.get_size(dim)
        return result

    @property
    def dim_h(self) -> int:
        """物理节点阵列的 H 维度。"""
        h = 1
        for dim in range(PartDim.NUM):
            h *= self.pdims[dim][0]
        return h

    @property
    def dim_w(self) -> int:
        """物理节点阵列的 W 维度。"""
        w = 1
        for dim in range(PartDim.NUM):
            w *= self.pdims[dim][1]
        return w

    def __repr__(self):
        parts = []
        dim_names = ['OUTP', 'OFMP', 'BATP', 'INPP']
        for dim in range(PartDim.NUM):
            h, w = self.pdims[dim]
            if h > 1 or w > 1:
                parts.append(f"{dim_names[dim]}=({h},{w})")
        return f"HybridPartition({', '.join(parts) or 'none'}, nodes={self.total_nodes})"

    def __hash__(self):
        return hash((tuple(self.pdims.items()), self.order))

    def __eq__(self, other):
        if isinstance(other, HybridPartitionChoice):
            return self.pdims == other.pdims and self.order == other.order
        return False


class LayerType:
    """层类型枚举。"""
    CONV = 'Conv'
    FC = 'FC'
    POOL = 'Pool'
    ELTWISE = 'Eltwise'
    LOCAL_REGION = 'LocalRegion'
    UNKNOWN = 'Unknown'


def detect_layer_type(layer) -> str:
    """
    检测层类型。

    根据层的类名和属性判断层类型。
    """
    class_name = layer.__class__.__name__

    # 按类名判断
    if 'FC' in class_name or class_name == 'FCLayer':
        return LayerType.FC
    elif 'Conv' in class_name:
        # 检查是否是 FC (H=W=1 的特殊 Conv)
        if getattr(layer, 'hofm', 1) == 1 and getattr(layer, 'wofm', 1) == 1:
            return LayerType.FC
        return LayerType.CONV
    elif 'Pool' in class_name:
        return LayerType.POOL
    elif 'Eltwise' in class_name or 'Add' in class_name:
        return LayerType.ELTWISE
    elif 'LocalRegion' in class_name or 'LRN' in class_name:
        return LayerType.LOCAL_REGION
    else:
        # 默认按特征判断
        if hasattr(layer, 'hfil') and getattr(layer, 'hfil', 1) > 1:
            return LayerType.CONV
        return LayerType.UNKNOWN


class LayerConfig:
    """层配置信息。"""

    def __init__(self, layer_name: str, layer, layer_idx: int, batch_size: int = 1):
        self.name = layer_name
        self.layer = layer
        self.idx = layer_idx
        self.batch_size = batch_size

        # 检测层类型
        self.layer_type = detect_layer_type(layer)

        # 提取层维度
        self.nifm = getattr(layer, 'nifm', 1)   # C - 输入通道
        self.nofm = getattr(layer, 'nofm', 1)   # K - 输出通道
        self.hofm = getattr(layer, 'hofm', 1)   # H - 输出高度
        self.wofm = getattr(layer, 'wofm', 1)   # W - 输出宽度
        self.hfil = getattr(layer, 'hfil', 1)   # R - 卷积核高度
        self.wfil = getattr(layer, 'wfil', 1)   # S - 卷积核宽度

        # 计算有效的分区因子
        self.valid_factors = self._compute_valid_factors()

        # 计算该层类型支持的分区维度
        self.valid_dims = self._get_valid_partition_dims()

    def _get_divisors(self, n: int, max_val: int = 32) -> List[int]:
        """获取 n 的所有因子（限制最大值）。"""
        divisors = []
        for i in range(1, min(int(n**0.5) + 1, max_val + 1)):
            if n % i == 0:
                divisors.append(i)
                if i != n // i and n // i <= max_val:
                    divisors.append(n // i)
        return sorted(set(divisors))

    def _compute_valid_factors(self) -> Dict[PartDim, List[int]]:
        """计算每个维度的有效分区因子。"""
        return {
            PartDim.OUTP: self._get_divisors(self.nofm),
            PartDim.OFMP: self._get_divisors(self.hofm) + self._get_divisors(self.wofm),
            PartDim.BATP: self._get_divisors(self.batch_size),
            PartDim.INPP: self._get_divisors(self.nifm),
        }

    def _get_valid_partition_dims(self) -> List[int]:
        """
        获取该层类型支持的分区维度。

        不同层类型有不同的有效分区维度：
        - Conv: OUTP, OFMP, BATP, INPP 都有效
        - FC: OUTP, BATP, INPP 有效 (OFMP 无效因为 H=W=1)
        - Pool: OUTP, OFMP, BATP 有效 (INPP 无效因为不跨通道计算)
        - Eltwise: OUTP, OFMP, BATP 有效
        """
        if self.layer_type == LayerType.FC:
            # FC 层: H=W=1, OFMP 分区无意义
            return [PartDim.OUTP, PartDim.BATP, PartDim.INPP]
        elif self.layer_type == LayerType.POOL:
            # Pooling 层: 不跨通道计算，INPP 分区无效
            return [PartDim.OUTP, PartDim.OFMP, PartDim.BATP]
        elif self.layer_type == LayerType.ELTWISE:
            # Eltwise 层: 逐元素操作
            return [PartDim.OUTP, PartDim.OFMP, PartDim.BATP]
        else:
            # Conv 和其他: 所有维度都有效
            return [PartDim.OUTP, PartDim.OFMP, PartDim.BATP, PartDim.INPP]

    def supports_partition_dim(self, dim: int) -> bool:
        """检查该层是否支持指定的分区维度。"""
        return dim in self.valid_dims

    @property
    def output_size(self) -> int:
        """输出数据大小。"""
        return self.batch_size * self.nofm * self.hofm * self.wofm

    @property
    def macs(self) -> int:
        """MAC 操作数。"""
        return (self.batch_size * self.nifm * self.nofm *
                self.hofm * self.wofm * self.hfil * self.wfil)


class RedistributionCostModel:
    """
    重分布成本模型（精确公式）。

    处理所有维度的传播：
    1. K→C: OUTP 分区传播到下一层的 C 分布
    2. Spatial: OFMP 分区传播到下一层的空间分布
    3. Batch: BATP 分区传播（通常匹配）
    4. INPP: 需要层内 All-Reduce

    精确通信量公式：
    - All-Gather: (n-1)/n × D  （每个节点发送 D/n，接收 (n-1)×D/n）
    - Scatter: (n-1)/n × D  （根节点发送 D，其他接收 D/n）
    - All-to-All: (n-1)/n × D  （每个节点发送/接收 (n-1)/n 的数据）
    - Ring All-Reduce: 2×(n-1)/n × D  （Reduce-Scatter + All-Gather）
    """

    # 单位数据传输成本（可调）
    COMM_COST_PER_BYTE = 1.0

    @classmethod
    def compute_redistribution_cost(cls,
                                    config_src: LayerConfig,
                                    config_dst: LayerConfig,
                                    choice_src: HybridPartitionChoice,
                                    choice_dst: HybridPartitionChoice) -> float:
        """
        计算从 src 到 dst 的总重分布成本。

        特殊处理：
        - Conv/Pool → FC: 空间维度展平到通道，需要特殊处理
        - Pool 层: 空间尺寸缩小，需要调整
        """
        total_cost = 0.0
        output_size = config_src.output_size

        # 特殊情况: Conv/Pool → FC (空间展平)
        if (config_src.layer_type in [LayerType.CONV, LayerType.POOL] and
                config_dst.layer_type == LayerType.FC):
            return cls._conv_to_fc_cost(config_src, config_dst,
                                        choice_src, choice_dst, output_size)

        # 1. K→C 传播成本
        total_cost += cls._k_to_c_cost(config_src, config_dst,
                                       choice_src, choice_dst, output_size)

        # 2. 空间传播成本
        total_cost += cls._spatial_cost(config_src, config_dst,
                                        choice_src, choice_dst, output_size)

        # 3. Batch 传播成本
        total_cost += cls._batch_cost(choice_src, choice_dst, output_size)

        # 注意：INPP All-Reduce 是层内成本，已移到 ComputeCostModel
        # 不在层间重分布成本中计算

        return total_cost

    @classmethod
    def _conv_to_fc_cost(cls, config_src, config_dst,
                         choice_src, choice_dst, data_size) -> float:
        """
        Conv/Pool → FC 的特殊转换成本。

        Conv 输出: [N, K, H, W]
        FC 输入:   [N, C=K×H×W, 1, 1]  ← 空间维度展平到通道

        分析:
        - Conv 的 OUTP (K 分区) → FC 的部分 INPP
        - Conv 的 OFMP (H×W 分区) → FC 的部分 INPP
        - 如果 Conv 同时有 OUTP 和 OFMP，映射更复杂

        简化处理:
        - 如果 Conv 只用 OUTP 且 FC 用 INPP 分区相同: 部分匹配
        - 否则需要 All-Gather 重分布
        """
        src_outp = choice_src.get_size(PartDim.OUTP)
        src_ofmp_h, src_ofmp_w = choice_src.get_factor(PartDim.OFMP)
        src_ofmp = src_ofmp_h * src_ofmp_w
        dst_inpp = choice_dst.get_size(PartDim.INPP)

        # Conv 输出的 C_fc = K × H × W
        # 数据分布情况：
        # - 如果 src 只用 OUTP (K 分区)，每个节点有 K/p 个通道，完整 H×W
        # - 如果 src 只用 OFMP (空间分区)，每个节点有完整 K，但只有 H×W/p 的空间
        # - 如果混合，每个节点有 K/p_k 通道，H×W/p_s 空间

        # 简化: 如果任何一方有分区，都需要重分布
        total_src_factor = src_outp * src_ofmp

        if total_src_factor == 1:
            # src 没有分区，每个节点有完整数据
            return 0.0

        if src_ofmp > 1:
            # Conv 有空间分区，FC 需要完整空间数据
            # 必须 All-Gather 空间数据
            # 通信量: (p-1)/p × D
            spatial_cost = data_size * cls.COMM_COST_PER_BYTE * \
                (src_ofmp - 1) / src_ofmp
        else:
            spatial_cost = 0.0

        if src_outp > 1:
            # Conv 有 K 分区
            if dst_inpp == src_outp and src_ofmp == 1:
                # 特殊情况: Conv 只有 OUTP，FC 有匹配的 INPP
                # K 维度的分布可以直接映射到 FC 的 C 分区
                k_cost = 0.0
            else:
                # 需要重分布 K 维度
                k_cost = data_size * cls.COMM_COST_PER_BYTE * \
                    (src_outp - 1) / src_outp
        else:
            k_cost = 0.0

        # Batch 成本
        batch_cost = cls._batch_cost(choice_src, choice_dst, data_size)

        return spatial_cost + k_cost + batch_cost

    @classmethod
    def _k_to_c_cost(cls, config_src, config_dst,
                     choice_src, choice_dst, data_size) -> float:
        """
        K→C 传播成本（精确公式）。

        src 的 OUTP 分区决定了 dst 的输入 C 分布。
        如果 dst 需要不同的分区，需要重分布。

        关键理解：
        - src 的 K 输出已经被 OUTP 分区分布到多个节点
        - dst 的 INPP 分区决定了它如何消费 C 维度
        - 只有当 dst.INPP == src.OUTP 时，数据才能直接流动

        精确通信量：
        - All-Gather: (k-1)/k × D  （k 个节点收集完整数据）
        - All-to-All: (n-1)/n × D （n = max(k, p) 节点间重分布）
        """
        k_factor = choice_src.get_size(PartDim.OUTP)  # src 的 K 分区

        # dst 期望的 C 分区
        dst_inpp = choice_dst.get_size(PartDim.INPP)  # dst 的 C 分区

        if k_factor == 1:
            # src 没有 K 分区，每个节点有完整的 K 输出
            # 无论 dst 如何分区 C，每个节点可以本地选择需要的部分
            return 0.0

        # src 有 K 分区 (k > 1)
        # src 的输出被分布到 k 个节点，每个节点只有 K/k 个通道

        if dst_inpp == k_factor:
            # 完美匹配：dst 按 C 分区，正好匹配 src 按 K 分区的输出
            # 数据可以直接流动，无需通信
            return 0.0
        elif dst_inpp == 1:
            # dst 不分区 C，每个节点需要完整的 C 数据
            # 需要 All-Gather: 每个节点收集所有 k 个分片
            return data_size * cls.COMM_COST_PER_BYTE * (k_factor - 1) / k_factor
        else:
            # dst 的 C 分区与 src 的 K 分区不匹配
            # 需要 All-to-All 重分布
            n = max(k_factor, dst_inpp)
            return data_size * cls.COMM_COST_PER_BYTE * (n - 1) / n

    @classmethod
    def _spatial_cost(cls, config_src, config_dst,
                      choice_src, choice_dst, data_size) -> float:
        """
        空间 (OFMP) 传播成本（精确公式）。

        src 的空间分区影响 dst 的输入空间分布。
        还需要考虑 halo exchange（卷积核边界数据）。

        精确 Halo Exchange 公式:
        Halo_per_tile = N × C × [2×halo_h×w_tile + 2×halo_w×h_tile + 4×halo_h×halo_w]
        其中：
        - halo_h = floor((R-1)/2)
        - halo_w = floor((S-1)/2)
        - h_tile = ceil(H/p_h), w_tile = ceil(W/p_w)
        """
        src_ofmp = choice_src.get_factor(PartDim.OFMP)  # (h, w)
        dst_ofmp = choice_dst.get_factor(PartDim.OFMP)

        src_h, src_w = src_ofmp
        dst_h, dst_w = dst_ofmp

        cost = 0.0

        # 检查空间分区是否匹配
        if src_h != dst_h or src_w != dst_w:
            # 空间分区不匹配，需要重分布: (n-1)/n × D
            n = max(src_h * src_w, dst_h * dst_w)
            cost += data_size * cls.COMM_COST_PER_BYTE * (n - 1) / n

        # Halo exchange 成本（精确公式）
        if dst_h > 1 or dst_w > 1:
            # halo 宽度（单侧）
            halo_h = (config_dst.hfil - 1) // 2 if config_dst.hfil > 1 else 0
            halo_w = (config_dst.wfil - 1) // 2 if config_dst.wfil > 1 else 0

            if halo_h > 0 or halo_w > 0:
                # 每个 tile 的大小
                import math
                h_tile = math.ceil(config_dst.hofm / dst_h)
                w_tile = math.ceil(config_dst.wofm / dst_w)

                # 精确 halo 数据量：上下边界 + 左右边界 + 四个角
                halo_data_per_tile = config_dst.batch_size * config_dst.nifm * (
                    2 * halo_h * w_tile +      # 上下边界
                    2 * halo_w * h_tile +      # 左右边界
                    4 * halo_h * halo_w        # 四个角
                )
                cost += halo_data_per_tile * cls.COMM_COST_PER_BYTE

        return cost

    @classmethod
    def _batch_cost(cls, choice_src, choice_dst, data_size) -> float:
        """
        Batch (BATP) 传播成本（精确公式）。

        Batch 维度通常在整个网络中保持一致。
        All-to-All 通信量: (n-1)/n × D
        """
        src_batp = choice_src.get_size(PartDim.BATP)
        dst_batp = choice_dst.get_size(PartDim.BATP)

        if src_batp == dst_batp:
            # Batch 分区匹配，无成本
            return 0.0

        # Batch 分区不匹配（罕见情况）: (n-1)/n × D
        n = max(src_batp, dst_batp)
        return data_size * cls.COMM_COST_PER_BYTE * (n - 1) / n

    @classmethod
    def _inpp_reduction_cost(cls, choice_src, data_size) -> float:
        """
        INPP (C) 分区的 reduction 成本（精确公式）。

        如果 src 使用 INPP 分区，每个节点只计算部分和，
        需要 All-Reduce 来汇总结果。

        Ring All-Reduce 精确通信量：
        - Reduce-Scatter 阶段: (n-1)/n × D
        - All-Gather 阶段: (n-1)/n × D
        - 总计: 2 × (n-1)/n × D
        """
        inpp_factor = choice_src.get_size(PartDim.INPP)

        if inpp_factor <= 1:
            return 0.0

        # Ring All-Reduce: 2 × (n-1)/n × D
        return data_size * cls.COMM_COST_PER_BYTE * 2 * (inpp_factor - 1) / inpp_factor


class ComputeCostModel:
    """
    计算成本模型（精确公式）。

    层内成本 LayerCost_l(c) = Compute + R_C (INPP All-Reduce) + R_halo

    关键理解：
    - OFMP 空间分区：Halo 只增加通信量，不增加计算量
    - INPP 输入通道分区：需要 All-Reduce 来汇总部分和
    - INPP All-Reduce 是层内成本，不是层间转换成本！
    """

    @classmethod
    def compute_cost(cls, config: LayerConfig, choice: HybridPartitionChoice) -> float:
        """
        计算某分区方案的计算成本。

        Halo 只增加通信量（在 RedistributionCostModel 中建模），不增加计算量。
        每个节点计算的输出像素数量是固定的。
        """
        base_macs = config.macs
        total_nodes = choice.total_nodes

        if total_nodes == 0:
            return float('inf')

        # 每节点工作量（基础）
        macs_per_node = base_macs / total_nodes

        # INPP 负载不均衡因子
        # 当 C 不能被 p_INPP 整除时，有 padding 导致负载不均
        inpp_factor = choice.get_size(PartDim.INPP)
        if inpp_factor > 1 and config.nifm % inpp_factor != 0:
            import math
            # α_INPP = ceil(C/p) / (C/p)
            c_per_node_ideal = config.nifm / inpp_factor
            c_per_node_actual = math.ceil(config.nifm / inpp_factor)
            inpp_overhead = c_per_node_actual / c_per_node_ideal
        else:
            inpp_overhead = 1.0

        # INPP All-Reduce 成本（层内成本）
        # 如果使用 INPP 分区，每个节点只计算部分和，需要 All-Reduce
        inpp_reduce_cost = 0.0
        if inpp_factor > 1:
            # Ring All-Reduce: 2 × (n-1)/n × D
            output_size = config.batch_size * config.nofm * config.hofm * config.wofm
            inpp_reduce_cost = output_size * 2 * \
                (inpp_factor - 1) / inpp_factor

        # Halo 交换成本（层内成本）
        # OFMP 空间分区时，需要从邻居获取边界数据
        halo_cost = 0.0
        ofmp_h, ofmp_w = choice.get_factor(PartDim.OFMP)
        if ofmp_h > 1 or ofmp_w > 1:
            halo_h = (config.hfil - 1) // 2 if config.hfil > 1 else 0
            halo_w = (config.wfil - 1) // 2 if config.wfil > 1 else 0
            if halo_h > 0 or halo_w > 0:
                import math
                h_tile = math.ceil(config.hofm / ofmp_h)
                w_tile = math.ceil(config.wofm / ofmp_w)
                # 每个 tile 的 halo 数据量
                halo_per_tile = config.batch_size * config.nifm * (
                    2 * halo_h * w_tile + 2 * halo_w * h_tile + 4 * halo_h * halo_w
                )
                halo_cost = halo_per_tile

        return macs_per_node * inpp_overhead + inpp_reduce_cost + halo_cost


class GlobalPartitionILPOptimizerV2:
    """
    混合分区全局 ILP 优化器 V2。

    完整支持：
    - 混合分区（多维度同时分区）
    - 所有维度的传播约束
    - K→C, Spatial, Batch 传播
    - INPP reduction
    """

    def __init__(self, network, resource, batch_size: int = 1,
                 max_factor: int = 16, solver: str = 'auto'):
        """
        Args:
            network: nn_dataflow Network 对象
            resource: Resource 对象（包含节点信息）
            batch_size: Batch 大小
            max_factor: 每个维度的最大分区因子
            solver: 'gurobi', 'pulp', 'dp', 或 'auto'
        """
        self.network = network
        self.resource = resource
        self.batch_size = batch_size
        self.max_factor = max_factor
        self.solver = self._select_solver(solver)

        # 提取节点阵列维度
        self.dim_nodes = self._get_dim_nodes()
        self.total_nodes = self.dim_nodes[0] * self.dim_nodes[1]

        # 构建层配置
        self.layer_configs = self._build_layer_configs()

        # 生成分区方案
        self.partition_choices = self._generate_partition_choices()

        # 成本缓存
        self.compute_costs = {}
        self.redistribution_costs = {}

    def _select_solver(self, solver: str) -> str:
        """选择求解器。"""
        if solver == 'auto':
            if HAS_GUROBI:
                return 'gurobi'
            elif HAS_PULP:
                return 'pulp'
            else:
                # 使用动态规划作为备选
                print("No ILP solver available, using DP solver.")
                return 'dp'
        elif solver == 'gurobi' and not HAS_GUROBI:
            raise ImportError("Gurobi not available")
        elif solver == 'pulp' and not HAS_PULP:
            raise ImportError("PuLP not available")
        return solver

    def _get_dim_nodes(self) -> Tuple[int, int]:
        """获取节点阵列维度 (H, W)。"""
        if hasattr(self.resource, 'dim_nodes'):
            return (self.resource.dim_nodes.h, self.resource.dim_nodes.w)
        elif hasattr(self.resource, 'proc_region'):
            dim = self.resource.proc_region.dim
            return (dim.h, dim.w)
        return (4, 4)  # 默认 4×4

    def _build_layer_configs(self) -> List[LayerConfig]:
        """构建层配置。"""
        configs = []
        for idx, layer_name in enumerate(self.network):
            layer = self.network[layer_name]
            config = LayerConfig(layer_name, layer, idx, self.batch_size)
            configs.append(config)
        return configs

    def _generate_partition_choices(self) -> List[List[HybridPartitionChoice]]:
        """
        为每层生成所有有效的混合分区方案。

        类似 nn_dataflow 的 gen_partition 逻辑。
        """
        all_choices = []
        dim_h, dim_w = self.dim_nodes

        for config in self.layer_configs:
            layer_choices = []
            seen = set()

            # 枚举所有可能的因子分解
            # h_factors[dim] 和 w_factors[dim] 的乘积 = dim_h 和 dim_w
            for ph in self._factorize_to_dims(dim_h, PartDim.NUM):
                for pw in self._factorize_to_dims(dim_w, PartDim.NUM):
                    pdims = {}
                    valid = True

                    for dim in range(PartDim.NUM):
                        pdims[dim] = (ph[dim], pw[dim])

                        # 检查分区是否有效
                        factor = ph[dim] * pw[dim]
                        if not self._is_valid_factor(config, dim, factor):
                            valid = False
                            break

                    if not valid:
                        continue

                    # 创建分区方案
                    choice = HybridPartitionChoice(pdims)

                    # 验证总节点数
                    if choice.total_nodes != self.total_nodes:
                        continue

                    if choice not in seen:
                        layer_choices.append(choice)
                        seen.add(choice)

            # 确保至少有一个选择
            if not layer_choices:
                # 添加默认的单节点分区
                default = HybridPartitionChoice({})
                layer_choices.append(default)

            all_choices.append(layer_choices)
            print(
                f"Layer {config.name}: {len(layer_choices)} partition choices")

        return all_choices

    def _factorize_to_dims(self, n: int, num_dims: int) -> List[Tuple[int, ...]]:
        """
        将 n 分解为 num_dims 个因子的乘积。

        返回所有可能的分解方式。
        """
        if num_dims == 1:
            return [(n,)]

        results = []
        for f in self._get_divisors(n):
            for rest in self._factorize_to_dims(n // f, num_dims - 1):
                results.append((f,) + rest)
        return results

    def _get_divisors(self, n: int) -> List[int]:
        """获取 n 的所有因子。"""
        divisors = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divisors.append(i)
                if i != n // i:
                    divisors.append(n // i)
        return sorted(divisors)

    def _is_valid_factor(self, config: LayerConfig, dim: PartDim, factor: int) -> bool:
        """检查某维度的分区因子是否有效。"""
        if factor == 1:
            return True

        # 检查该层类型是否支持此分区维度
        if not config.supports_partition_dim(dim):
            # 如果不支持，只允许因子为 1
            return factor == 1

        if dim == PartDim.OUTP:
            return config.nofm % factor == 0 or self._approx_dividable(config.nofm, factor)
        elif dim == PartDim.OFMP:
            # FC 层不支持 OFMP (H=W=1)
            if config.layer_type == LayerType.FC:
                return factor == 1
            return (self._approx_dividable(config.hofm, factor) or
                    self._approx_dividable(config.wofm, factor))
        elif dim == PartDim.BATP:
            return config.batch_size % factor == 0
        elif dim == PartDim.INPP:
            # Pool 层不支持 INPP（不跨通道计算）
            if config.layer_type == LayerType.POOL:
                return factor == 1
            return config.nifm % factor == 0 or self._approx_dividable(config.nifm, factor)

        return True

    def _approx_dividable(self, n: int, d: int, threshold: float = 0.1) -> bool:
        """检查 n 是否近似可被 d 整除。"""
        if d == 0:
            return False
        remainder = n % d
        return remainder == 0 or remainder <= threshold * d or (d - remainder) <= threshold * d

    def _compute_all_costs(self):
        """预计算所有成本。"""
        print("Computing costs...")

        # 计算每个分区方案的计算成本
        for layer_idx, (config, choices) in enumerate(
                zip(self.layer_configs, self.partition_choices)):
            for choice_idx, choice in enumerate(choices):
                cost = ComputeCostModel.compute_cost(config, choice)
                self.compute_costs[(layer_idx, choice_idx)] = cost

        # 计算相邻层之间的重分布成本
        for layer_idx in range(len(self.layer_configs) - 1):
            config_src = self.layer_configs[layer_idx]
            config_dst = self.layer_configs[layer_idx + 1]
            choices_src = self.partition_choices[layer_idx]
            choices_dst = self.partition_choices[layer_idx + 1]

            for ci, choice_src in enumerate(choices_src):
                for cj, choice_dst in enumerate(choices_dst):
                    cost = RedistributionCostModel.compute_redistribution_cost(
                        config_src, config_dst, choice_src, choice_dst)
                    self.redistribution_costs[(layer_idx, ci, cj)] = cost

        print(f"Computed {len(self.compute_costs)} compute costs, "
              f"{len(self.redistribution_costs)} redistribution costs")

    def optimize(self, time_limit: int = 300, verbose: bool = True) -> List[Tuple[str, HybridPartitionChoice]]:
        """
        运行 ILP 优化。

        Returns:
            List of (layer_name, HybridPartitionChoice)
        """
        self._compute_all_costs()

        if self.solver == 'gurobi':
            return self._optimize_gurobi(time_limit, verbose)
        elif self.solver == 'dp':
            return self._optimize_dp(verbose)
        else:
            return self._optimize_pulp(time_limit, verbose)

    def _optimize_dp(self, verbose: bool = True):
        """使用动态规划求解（无需 ILP 求解器）。"""
        print("Using Dynamic Programming solver...")

        num_layers = len(self.layer_configs)

        # dp[l][c] = (最优成本, 前一层选择)
        INF = float('inf')
        dp = [[INF for _ in range(len(self.partition_choices[l]))]
              for l in range(num_layers)]
        prev = [[-1 for _ in range(len(self.partition_choices[l]))]
                for l in range(num_layers)]

        # 初始化第一层
        for c in range(len(self.partition_choices[0])):
            dp[0][c] = self.compute_costs[(0, c)]

        # 动态规划
        for l in range(1, num_layers):
            for cj in range(len(self.partition_choices[l])):
                compute_cost = self.compute_costs[(l, cj)]
                for ci in range(len(self.partition_choices[l-1])):
                    redist_cost = self.redistribution_costs.get(
                        (l-1, ci, cj), 0)
                    total = dp[l-1][ci] + compute_cost + redist_cost
                    if total < dp[l][cj]:
                        dp[l][cj] = total
                        prev[l][cj] = ci

        # 回溯找最优解
        best_cost = INF
        best_last = -1
        for c in range(len(self.partition_choices[-1])):
            if dp[-1][c] < best_cost:
                best_cost = dp[-1][c]
                best_last = c

        # 回溯路径
        path = []
        c = best_last
        for l in range(num_layers - 1, -1, -1):
            path.append(c)
            c = prev[l][c]
        path = path[::-1]

        # 构建解
        solution = []
        total_compute = 0
        total_redist = 0

        for l, c in enumerate(path):
            choice = self.partition_choices[l][c]
            solution.append((self.layer_configs[l].name, choice))
            total_compute += self.compute_costs[(l, c)]
            if l > 0:
                total_redist += self.redistribution_costs.get(
                    (l-1, path[l-1], c), 0)

        if verbose:
            print(f"\nOptimal solution found!")
            print(f"Total compute cost: {total_compute:,.0f}")
            print(f"Total redistribution cost: {total_redist:,.0f}")
            print(f"Total cost: {best_cost:,.0f}")

        return solution

    def _optimize_pulp(self, time_limit: int, verbose: bool):
        """使用 PuLP 求解。"""
        if not HAS_PULP:
            raise ImportError("PuLP not available")

        print("Building ILP model with PuLP...")
        prob = pulp.LpProblem("GlobalPartitionV2", pulp.LpMinimize)

        num_layers = len(self.layer_configs)

        # 决策变量: x[l, c] = 1 如果第 l 层使用方案 c
        x = {}
        for l in range(num_layers):
            for c in range(len(self.partition_choices[l])):
                x[l, c] = pulp.LpVariable(f"x_{l}_{c}", cat='Binary')

        # 约束 1: 每层恰好选择一个方案
        for l in range(num_layers):
            prob += pulp.lpSum(x[l, c] for c in range(len(self.partition_choices[l]))) == 1, \
                f"select_one_layer_{l}"

        # 辅助变量 y[l, ci, cj] 用于线性化
        y = {}
        for l in range(num_layers - 1):
            for ci in range(len(self.partition_choices[l])):
                for cj in range(len(self.partition_choices[l + 1])):
                    y[l, ci, cj] = pulp.LpVariable(f"y_{l}_{ci}_{cj}",
                                                   lowBound=0, upBound=1, cat='Continuous')
                    # 线性化约束
                    prob += y[l, ci, cj] <= x[l, ci], f"lin1_{l}_{ci}_{cj}"
                    prob += y[l, ci, cj] <= x[l + 1, cj], f"lin2_{l}_{ci}_{cj}"
                    prob += y[l, ci, cj] >= x[l, ci] + \
                        x[l + 1, cj] - 1, f"lin3_{l}_{ci}_{cj}"

        # 目标函数: 最小化计算成本 + 重分布成本
        compute_cost = pulp.lpSum(
            x[l, c] * self.compute_costs[(l, c)]
            for l in range(num_layers)
            for c in range(len(self.partition_choices[l]))
        )

        redist_cost = pulp.lpSum(
            y[l, ci, cj] * self.redistribution_costs[(l, ci, cj)]
            for l in range(num_layers - 1)
            for ci in range(len(self.partition_choices[l]))
            for cj in range(len(self.partition_choices[l + 1]))
        )

        prob += compute_cost + redist_cost, "total_cost"

        # 求解
        print("Solving ILP...")
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=verbose)
        prob.solve(solver)

        # 提取解
        if prob.status != pulp.LpStatusOptimal:
            print(f"Warning: Solution status = {pulp.LpStatus[prob.status]}")

        solution = []
        total_compute = 0
        total_redist = 0

        for l in range(num_layers):
            for c in range(len(self.partition_choices[l])):
                if pulp.value(x[l, c]) > 0.5:
                    choice = self.partition_choices[l][c]
                    solution.append((self.layer_configs[l].name, choice))
                    total_compute += self.compute_costs[(l, c)]
                    break

        for l in range(num_layers - 1):
            for ci in range(len(self.partition_choices[l])):
                for cj in range(len(self.partition_choices[l + 1])):
                    val = pulp.value(y[l, ci, cj])
                    if val is not None and val > 0.5:
                        total_redist += self.redistribution_costs[(l, ci, cj)]

        print(f"\nOptimal solution found!")
        print(f"Total compute cost: {total_compute:,.0f}")
        print(f"Total redistribution cost: {total_redist:,.0f}")
        print(f"Total cost: {total_compute + total_redist:,.0f}")

        return solution

    def _optimize_gurobi(self, time_limit: int, verbose: bool):
        """使用 Gurobi 求解。"""
        if not HAS_GUROBI:
            raise ImportError("Gurobi not available")

        print("Building ILP model with Gurobi...")
        model = gp.Model("GlobalPartitionV2")

        if not verbose:
            model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', time_limit)

        num_layers = len(self.layer_configs)

        # 决策变量
        x = {}
        for l in range(num_layers):
            for c in range(len(self.partition_choices[l])):
                x[l, c] = model.addVar(vtype=GRB.BINARY, name=f"x_{l}_{c}")

        # 约束 1: 每层选一个
        for l in range(num_layers):
            model.addConstr(
                gp.quicksum(x[l, c] for c in range(
                    len(self.partition_choices[l]))) == 1,
                f"select_one_{l}")

        # 辅助变量
        y = {}
        for l in range(num_layers - 1):
            for ci in range(len(self.partition_choices[l])):
                for cj in range(len(self.partition_choices[l + 1])):
                    y[l, ci, cj] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                                name=f"y_{l}_{ci}_{cj}")
                    model.addConstr(y[l, ci, cj] <= x[l, ci])
                    model.addConstr(y[l, ci, cj] <= x[l + 1, cj])
                    model.addConstr(
                        y[l, ci, cj] >= x[l, ci] + x[l + 1, cj] - 1)

        # 目标函数
        compute_cost = gp.quicksum(
            x[l, c] * self.compute_costs[(l, c)]
            for l in range(num_layers)
            for c in range(len(self.partition_choices[l]))
        )

        redist_cost = gp.quicksum(
            y[l, ci, cj] * self.redistribution_costs[(l, ci, cj)]
            for l in range(num_layers - 1)
            for ci in range(len(self.partition_choices[l]))
            for cj in range(len(self.partition_choices[l + 1]))
        )

        model.setObjective(compute_cost + redist_cost, GRB.MINIMIZE)

        # 求解
        print("Solving ILP...")
        model.optimize()

        # 提取解
        solution = []
        for l in range(num_layers):
            for c in range(len(self.partition_choices[l])):
                if x[l, c].X > 0.5:
                    choice = self.partition_choices[l][c]
                    solution.append((self.layer_configs[l].name, choice))
                    break

        print(f"\nOptimal solution found!")
        print(f"Objective value: {model.ObjVal:,.0f}")

        return solution

    def print_solution(self, solution: List[Tuple[str, HybridPartitionChoice]]):
        """打印解决方案。"""
        print("\n" + "="*70)
        print("OPTIMAL PARTITION SCHEME")
        print("="*70)

        for layer_name, choice in solution:
            print(f"\n{layer_name}:")
            print(f"  {choice}")
            print(f"  OUTP(K): {choice.get_factor(PartDim.OUTP)}")
            print(f"  OFMP(HW): {choice.get_factor(PartDim.OFMP)}")
            print(f"  BATP(N): {choice.get_factor(PartDim.BATP)}")
            print(f"  INPP(C): {choice.get_factor(PartDim.INPP)}")


# 测试代码
if __name__ == "__main__":
    print("Testing GlobalPartitionILPOptimizerV2...")

    # 创建简单的测试网络
    class SimpleLayer:
        def __init__(self, nifm, nofm, hofm, wofm, hfil=3, wfil=3):
            self.nifm = nifm
            self.nofm = nofm
            self.hofm = hofm
            self.wofm = wofm
            self.hfil = hfil
            self.wfil = wfil

    class SimpleNetwork(dict):
        pass

    class SimpleResource:
        def __init__(self, h, w):
            self.dim_nodes = type('DimNodes', (), {'h': h, 'w': w})()

    # 创建 VGG-like 网络
    network = SimpleNetwork()
    network['conv1'] = SimpleLayer(3, 64, 224, 224)
    network['conv2'] = SimpleLayer(64, 64, 224, 224)
    network['conv3'] = SimpleLayer(64, 128, 112, 112)
    network['conv4'] = SimpleLayer(128, 128, 112, 112)
    network['conv5'] = SimpleLayer(128, 256, 56, 56)

    resource = SimpleResource(4, 4)  # 4×4 = 16 nodes

    # 运行优化
    optimizer = GlobalPartitionILPOptimizerV2(
        network=network,
        resource=resource,
        batch_size=4,
        solver='auto'  # 自动选择可用的求解器
    )

    solution = optimizer.optimize(time_limit=60, verbose=True)
    optimizer.print_solution(solution)
