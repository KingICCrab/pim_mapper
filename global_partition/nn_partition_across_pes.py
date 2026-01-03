#!/usr/bin/env python
"""
NN Partitioning across Processing Elements

本模块实现神经网络层在多处理单元间的分区策略。
类似于 TETRIS 在 3D 内存栈多 vault 间的分区，我们在加速器阵列的
多个处理单元间划分 NN 层以实现并行处理。

===============================================================================
4.2 NN Partitioning across Processing Elements
===============================================================================

处理单元阵列包含多个 PE，每个有独立的本地存储。除了在不同 PE 处理
不同的 NN 或层外，我们还可以将大的 NN 层分割到多个 PE 并行处理。

本节首先给出不同分区方案的分类（Figure 7），然后系统地探索并找到
最优方案。

-------------------------------------------------------------------------------
4.2.1 分区方案分类 (Taxonomy of Partitioning Schemes)
-------------------------------------------------------------------------------

┌─────────────────────────────────────────────────────────────────────────────┐
│                        Layer Partitioning Taxonomy                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Batch Partitioning (BATP)          Fmap Partitioning (OFMP)               │
│  ┌───┬───┬───┬───┐                 ┌─────────────────────┐                 │
│  │ N0│ N1│ N2│ N3│                 │ ┌───┬───┐ ┌───┬───┐ │                 │
│  │   │   │   │   │                 │ │H0 │H1 │ │H0 │H1 │ │                 │
│  │img│img│img│img│                 │ │W0 │W0 │ │W1 │W1 │ │                 │
│  └───┴───┴───┴───┘                 │ └───┴───┘ └───┴───┘ │                 │
│  各PE处理不同batch                   └─────────────────────┘                 │
│                                     各PE处理不同空间区域                     │
│                                                                             │
│  Output Partitioning (OUTP)         Input Partitioning (INPP)              │
│  ┌───────────────────┐             ┌───────────────────┐                   │
│  │ K0  K1  K2  K3    │             │ C0  C1  C2  C3    │                   │
│  │ ↓   ↓   ↓   ↓     │             │ ↓   ↓   ↓   ↓     │                   │
│  │PE0 PE1 PE2 PE3    │             │PE0 PE1 PE2 PE3    │                   │
│  └───────────────────┘             └───────────────────┘                   │
│  各PE处理不同输出通道                 各PE处理不同输入通道                     │
│  ifmap需广播                         部分和需要规约                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

-------------------------------------------------------------------------------
4.2.2 各分区方案详细分析
-------------------------------------------------------------------------------

**Batch Partitioning (数据并行):**

最简单的方案是使用多个 PE 并行处理多个输入图像，有效地将一个 batch
分配到各 PE [数据并行]。虽然对吞吐量有利，但要求 NN 模型在每个 PE
复制，这对大型 NN 是显著的容量挑战。此外，并行度受 batch size 限制。
对延迟敏感的实时应用不太有吸引力，因为它不能改善单张图像的推理延迟。

    优点: 实现简单，无通信开销
    缺点: 需要复制模型权重，并行度受限于 batch size
    适用: 吞吐量优先的离线推理

**Fmap (Image) Partitioning (空间并行):**

如果 fmap 较大（如 112×112），可以将其分割成更小的 tile。更小的
fmap tile 能更好地适配每个 PE 的阵列，减少 row stationary 数据流
中的折叠需求。此外，如果 CONV 层的 ifmap 和 ofmap 使用相同的 fmap
分区，由于 2D 卷积的局部性，大多数数据访问将是本地的。但是，filter
需要在所有 PE 间复制。

    优点: 最小化远程访问（ifmap/ofmap 局部性）
    缺点: filter 需要复制，权重重用降低
    适用: 前几层 CONV（fmap 大，权重相对小）

**Output Partitioning (输出通道并行):**

由于每层通常有多个 ofmap，可以将 ofmap 分配到各 PE。例如，将 No 个
ofmap 分成 16 组，每个 PE 处理 No/16 个 ofmap。由于每个 ofmap 使用
不同的 filter，filter 权重可以完全分区。由于所有 ifmap 贡献到所有
ofmap，所有 ifmap 必须发送到所有 PE，需要远程访问。

    优点: 权重完全分区，最大化权重重用
    缺点: ifmap 需要广播（远程访问）
    适用: 后几层 CONV 和 FC（权重大，fmap 小）

**Input Partitioning (输入通道并行):**

类似输出分区，也可以将 Ni 个 ifmap 分配到各 PE。区别在于计算发生的
位置。如 Section 3.3 讨论，ofmap 访问同时产生读写流量，因此比 ifmap
访问更关键。因此，使用避免 ofmap 远程访问的输出分区，优于避免 ifmap
远程访问的输入分区。

    优点: ifmap 完全分区
    缺点: **需要 All-Reduce 规约部分和** ← 关键开销
    适用: 特殊情况（ifmap 通道数很大）

-------------------------------------------------------------------------------
4.2.3 混合分区方案 (Hybrid Partitioning)
-------------------------------------------------------------------------------

Neurocube 使用简单启发式最小化远程访问：CONV 层用 fmap 分区，FC 层
用输出分区。这个启发式不一定带来最佳性能和能效。除了远程访问数量，
还应考虑分区方案对总内存访问数和 PE 寄存器数据重用的影响。

因为 fmap 分区会切分 fmap，每个 PE 需要加载相同的 filter 到寄存器，
而只在更小的 fmap tile 上重用。相反，输出分区将整个 fmap 保持在一个
PE，一次加载 filter 并在整个 fmap 上使用，带来更高的 filter 重用。
此外，输出分区可以结合 OW bypass 排序，只需一轮 ifmap 读取，最小化
ifmap 远程访问成本。

因此，我们考虑一个**混合分区方案**，在 fmap 分区的优势（最小化远程
访问）和输出分区的优势（更好的片上数据重用以最小化总 DRAM 访问）间
取得平衡。

这个混合方案也符合 Neurocube 启发式的意图：
- 对于常见 CNN 的前几层 CONV，大 fmap 访问占主导，权重重用不显著，
  应主要使用 **fmap 分区**
- 对于 FC 层，filter 权重远大于 fmap，**输出分区**更好以最大化权重重用

-------------------------------------------------------------------------------
4.2.4 成本模型 (Cost Model)
-------------------------------------------------------------------------------

为找到最优分区方案，我们提出一个简单的成本模型（总内存访问能耗）：

    E_access = A_DRAM × e × (1 + β × r)                    --- Eq. (5)

其中:
    e    = 一次本地 DRAM 访问的能耗
    β    = 远程访问的能耗惩罚因子
    r    = 远程访问的比例
    A_DRAM = 总主存访问次数

Fmap 分区最小化 r 但导致更大的 A_DRAM；
输出分区有更小的 A_DRAM 但牺牲 r。

**扩展: INPP 分区的 All-Reduce 成本:**

当使用 INPP（输入通道分区）时，每个 PE 计算部分和，需要 All-Reduce
操作汇聚最终结果：

    E_inpp = D_output × 2 × (p-1)/p × e_comm             --- Eq. (6)

其中:
    D_output = 输出数据量
    p        = INPP 分区因子
    e_comm   = 单位数据通信能耗
    
Ring All-Reduce 需要 2×(p-1)/p × D 的通信量（Reduce-Scatter + All-Gather）。

-------------------------------------------------------------------------------
4.2.5 分区探索策略 (Partitioning Exploration)
-------------------------------------------------------------------------------

每个 NN 层可以使用 fmap 分区和输出分区的组合来并行化。**相邻层是耦合
的，因为前一层的分区方案决定下一层 ifmap 的布局**（见 Figure 7）。

假设有 L 层和每层 C 种分区组合，需要考虑 C^L 种场景的总内存访问能耗
成本。由于 L 可能是 20~100，C 可能是 4~8，成本是禁止性的。

我们利用两个策略减少分区探索难度：

**策略 1: 贪心算法**

探索第 i 层的分区选项时不回溯，假设最优方案中前 i-1 层的分区独立于
后续层。第一个 CONV 层假设只使用 fmap 分区。这将选择数减少到 C × L，
在常见情况下大约几千种。

**策略 2: Layout Propagation（布局传播）**

基于规约分析减少分区决策变量：

    - 有规约的算子 → 布局敏感 → 需要独立分区决策
    - 无规约的算子 → 布局不敏感 → 可透传上游分区

这允许我们将决策变量从 L 层减少到 G 组（G ≤ L），其中每组内的层共享
相同分区。

    原始变量: L × C
    优化后:   G × C  （其中 G 是传播组数）

例如对于 ResNet-50，G/L ≈ 78%，可减少 22% 的决策变量。

**策略 3: 动态规划 / ILP 优化**

对于中等规模问题，使用动态规划或整数线性规划（ILP）求解全局最优：

    min  Σ_l E_compute(l) + Σ_{l,l+1} E_redistribute(l, l+1)
    s.t. 分区因子约束
         总节点数约束
         层类型约束

-------------------------------------------------------------------------------
4.2.6 层类型特定约束 (Layer-Type Specific Constraints)
-------------------------------------------------------------------------------

不同层类型有不同的有效分区维度：

┌─────────────┬──────┬──────┬──────┬──────┬─────────────────────────┐
│ 层类型       │ OUTP │ OFMP │ BATP │ INPP │ 说明                    │
├─────────────┼──────┼──────┼──────┼──────┼─────────────────────────┤
│ Conv        │  ✓   │  ✓   │  ✓   │  ✓   │ 所有维度有效             │
│ FC          │  ✓   │  ✗   │  ✓   │  ✓   │ OFMP 无效 (H=W=1)        │
│ Pool        │  ✓   │  ✓   │  ✓   │  ✗   │ INPP 无效 (无跨通道计算)  │
│ Eltwise     │  ✓   │  ✓   │  ✓   │  ✗   │ 无规约，透传分区         │
└─────────────┴──────┴──────┴──────┴──────┴─────────────────────────┘

-------------------------------------------------------------------------------
4.2.7 跨层传播分析 (Cross-Layer Propagation)
-------------------------------------------------------------------------------

分区方案的跨层影响：

1. **OUTP (K) → C:**
   Layer[l].K_partition → Layer[l+1].input_C_distribution
   
   如果第 l 层用 OUTP=4，输出 K 维度分成 4 份；
   第 l+1 层的输入 C 维度也分成 4 份；
   如果第 l+1 层也用 OUTP 或 BATP，无需重分布；
   如果第 l+1 层用 INPP=4，正好匹配，无需重分布；
   否则需要 All-Gather 收集完整 C。

2. **OFMP (H,W) → Spatial:**
   Layer[l].spatial_partition → Layer[l+1].input_spatial_distribution
   
   空间分区需要考虑 halo 区域（卷积边界）。
   如果 stride=2，还需要处理尺寸变化。

3. **BATP (N) → Batch:**
   Layer[l].batch_partition → Layer[l+1].batch_distribution
   
   Batch 维度通常匹配，无需重分布。

4. **INPP (C) → Reduction:**
   Layer 内部需要 All-Reduce，不影响下一层布局。

===============================================================================
"""

from enum import IntEnum
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import math


# =============================================================================
# 4.2.1 分区维度定义
# =============================================================================

class PartitionDim(IntEnum):
    """
    分区维度枚举。

    与 nn_dataflow 的 parallel_enum 对齐。
    """
    OUTP = 0    # K - Output channels (输出通道分区)
    OFMP = 1    # H×W - Output feature map (空间分区)
    BATP = 2    # N - Batch (批量分区)
    INPP = 3    # C - Input channels (输入通道分区，需要规约)
    NUM = 4


@dataclass
class PartitionScheme:
    """
    分区方案。

    支持混合分区：每个维度独立指定分区因子。
    """
    outp: Tuple[int, int] = (1, 1)  # K 分区 (h_factor, w_factor)
    ofmp: Tuple[int, int] = (1, 1)  # H×W 空间分区
    batp: Tuple[int, int] = (1, 1)  # Batch 分区
    inpp: Tuple[int, int] = (1, 1)  # C 分区 (需要 All-Reduce)

    def get_factor(self, dim: PartitionDim) -> Tuple[int, int]:
        """获取指定维度的分区因子。"""
        return [self.outp, self.ofmp, self.batp, self.inpp][dim]

    def get_size(self, dim: PartitionDim) -> int:
        """获取指定维度使用的节点数。"""
        h, w = self.get_factor(dim)
        return h * w

    @property
    def total_nodes(self) -> int:
        """总节点数。"""
        return (self.get_size(PartitionDim.OUTP) *
                self.get_size(PartitionDim.OFMP) *
                self.get_size(PartitionDim.BATP) *
                self.get_size(PartitionDim.INPP))

    @property
    def has_inpp(self) -> bool:
        """是否使用 INPP 分区（需要 All-Reduce）。"""
        return self.get_size(PartitionDim.INPP) > 1

    def __repr__(self):
        parts = []
        names = ['OUTP', 'OFMP', 'BATP', 'INPP']
        factors = [self.outp, self.ofmp, self.batp, self.inpp]
        for name, (h, w) in zip(names, factors):
            if h > 1 or w > 1:
                parts.append(f"{name}=({h},{w})")
        return f"Partition({', '.join(parts) or 'none'}, nodes={self.total_nodes})"


# =============================================================================
# 4.2.4 成本模型
# =============================================================================

class PartitionCostModel:
    """
    分区成本模型。

    实现 Eq. (5) 和 Eq. (6) 的成本计算。
    """

    # 能耗参数（可配置）
    E_LOCAL_ACCESS = 1.0      # 本地 DRAM 访问能耗 (e)
    E_REMOTE_PENALTY = 0.5    # 远程访问惩罚因子 (β)
    E_COMM_PER_BYTE = 0.1     # 单位通信能耗 (e_comm)

    @classmethod
    def compute_layer_cost(cls,
                           layer_info: dict,
                           scheme: PartitionScheme,
                           prev_scheme: Optional[PartitionScheme] = None) -> dict:
        """
        计算单层的分区成本。

        Returns:
            dict: {
                'compute_cost': 计算成本,
                'memory_cost': 内存访问成本,
                'inpp_allreduce_cost': INPP All-Reduce 成本,
                'redistribute_cost': 跨层重分布成本,
                'total_cost': 总成本
            }
        """
        costs = {}

        # 1. 计算成本（与分区无关，但并行度影响延迟）
        costs['compute_cost'] = layer_info.get('macs', 0) / scheme.total_nodes

        # 2. 内存访问成本 (Eq. 5)
        # A_DRAM 取决于分区方案（影响数据重用）
        a_dram = cls._estimate_dram_accesses(layer_info, scheme)
        r = cls._estimate_remote_ratio(layer_info, scheme)
        costs['memory_cost'] = a_dram * cls.E_LOCAL_ACCESS * \
            (1 + cls.E_REMOTE_PENALTY * r)

        # 3. INPP All-Reduce 成本 (Eq. 6)
        if scheme.has_inpp:
            p = scheme.get_size(PartitionDim.INPP)
            d_output = layer_info.get('output_size', 0)
            costs['inpp_allreduce_cost'] = d_output * \
                2 * (p - 1) / p * cls.E_COMM_PER_BYTE
        else:
            costs['inpp_allreduce_cost'] = 0.0

        # 4. 跨层重分布成本
        if prev_scheme is not None:
            costs['redistribute_cost'] = cls._compute_redistribute_cost(
                layer_info, prev_scheme, scheme)
        else:
            costs['redistribute_cost'] = 0.0

        # 总成本
        costs['total_cost'] = sum([
            costs['compute_cost'],
            costs['memory_cost'],
            costs['inpp_allreduce_cost'],
            costs['redistribute_cost']
        ])

        return costs

    @classmethod
    def _estimate_dram_accesses(cls, layer_info: dict, scheme: PartitionScheme) -> float:
        """
        估算 DRAM 访问次数。

        分区方案影响数据重用：
        - OFMP 分区: filter 需要复制，重用降低
        - OUTP 分区: filter 完全分区，重用不变
        - INPP 分区: 部分和需要写回/读取
        """
        ifmap_size = layer_info.get('ifmap_size', 0)
        ofmap_size = layer_info.get('output_size', 0)
        weight_size = layer_info.get('weight_size', 0)

        # 基础访问量
        base_accesses = ifmap_size + ofmap_size + weight_size

        # OFMP 分区的 filter 复制惩罚
        ofmp_factor = scheme.get_size(PartitionDim.OFMP)
        if ofmp_factor > 1:
            # filter 需要在所有空间分区复制
            weight_penalty = weight_size * (ofmp_factor - 1)
        else:
            weight_penalty = 0

        return base_accesses + weight_penalty

    @classmethod
    def _estimate_remote_ratio(cls, layer_info: dict, scheme: PartitionScheme) -> float:
        """
        估算远程访问比例。

        - OFMP 分区: 低远程比例（空间局部性）
        - OUTP 分区: ifmap 需要广播，高远程比例
        - INPP 分区: 部分和需要 All-Reduce
        """
        ofmp_factor = scheme.get_size(PartitionDim.OFMP)
        outp_factor = scheme.get_size(PartitionDim.OUTP)

        # OFMP 分区：大部分访问是本地的
        if ofmp_factor > 1 and outp_factor == 1:
            return 0.1  # 只有 halo 区域是远程

        # OUTP 分区：ifmap 需要广播
        if outp_factor > 1:
            ifmap_size = layer_info.get('ifmap_size', 0)
            total_size = layer_info.get('total_data_size', 1)
            return ifmap_size / total_size * (outp_factor - 1) / outp_factor

        return 0.0

    @classmethod
    def _compute_redistribute_cost(cls,
                                   layer_info: dict,
                                   prev_scheme: PartitionScheme,
                                   curr_scheme: PartitionScheme) -> float:
        """
        计算跨层重分布成本。

        根据前一层的输出布局和当前层的输入需求计算。
        """
        data_size = layer_info.get('ifmap_size', 0)
        cost = 0.0

        # K→C 传播
        prev_outp = prev_scheme.get_size(PartitionDim.OUTP)
        curr_inpp = curr_scheme.get_size(PartitionDim.INPP)

        if prev_outp > 1:
            if curr_inpp != prev_outp:
                # 需要 All-Gather 然后可能 Scatter
                cost += data_size * (prev_outp - 1) / \
                    prev_outp * cls.E_COMM_PER_BYTE

        # 空间传播
        prev_ofmp = prev_scheme.get_size(PartitionDim.OFMP)
        curr_ofmp = curr_scheme.get_size(PartitionDim.OFMP)

        if prev_ofmp != curr_ofmp:
            # 空间重分布
            cost += data_size * 0.5 * cls.E_COMM_PER_BYTE  # 简化估算

        return cost


# =============================================================================
# 4.2.5 分区探索
# =============================================================================

class PartitionExplorer:
    """
    分区方案探索器。

    实现贪心搜索和全局优化。
    """

    def __init__(self,
                 num_nodes: int,
                 layer_infos: List[dict],
                 layout_groups: Optional[List[Set[int]]] = None):
        """
        Args:
            num_nodes: 总节点数
            layer_infos: 每层的信息
            layout_groups: Layout Propagation 分组（可选）
        """
        self.num_nodes = num_nodes
        self.layer_infos = layer_infos
        self.num_layers = len(layer_infos)
        self.layout_groups = layout_groups

        # 生成候选分区方案
        self.candidates = self._generate_candidates()

    def _generate_candidates(self) -> List[PartitionScheme]:
        """生成所有有效的分区候选方案。"""
        candidates = []

        # 枚举所有因子组合
        factors = self._get_factors(self.num_nodes)

        for outp_h, outp_w in factors:
            for ofmp_h, ofmp_w in factors:
                for batp_h, batp_w in factors:
                    for inpp_h, inpp_w in factors:
                        scheme = PartitionScheme(
                            outp=(outp_h, outp_w),
                            ofmp=(ofmp_h, ofmp_w),
                            batp=(batp_h, batp_w),
                            inpp=(inpp_h, inpp_w)
                        )
                        if scheme.total_nodes <= self.num_nodes:
                            candidates.append(scheme)

        return candidates

    def _get_factors(self, n: int) -> List[Tuple[int, int]]:
        """获取 n 的所有 2D 因子分解。"""
        factors = []
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                factors.append((i, n // i))
                if i != n // i:
                    factors.append((n // i, i))
        # 添加单维度因子
        for i in range(1, n + 1):
            if (i, 1) not in factors:
                factors.append((i, 1))
            if (1, i) not in factors:
                factors.append((1, i))
        return list(set(factors))

    def greedy_search(self) -> Tuple[List[PartitionScheme], float]:
        """
        贪心搜索。

        对每层独立选择最优分区，不回溯。
        复杂度: O(L × C)
        """
        result = []
        total_cost = 0.0

        for l in range(self.num_layers):
            layer_info = self.layer_infos[l]
            prev_scheme = result[-1] if result else None

            best_scheme = None
            best_cost = float('inf')

            for scheme in self.candidates:
                # 检查层类型约束
                if not self._is_valid_for_layer(scheme, layer_info):
                    continue

                costs = PartitionCostModel.compute_layer_cost(
                    layer_info, scheme, prev_scheme)

                if costs['total_cost'] < best_cost:
                    best_cost = costs['total_cost']
                    best_scheme = scheme

            result.append(best_scheme)
            total_cost += best_cost

        return result, total_cost

    def dp_search(self) -> Tuple[List[PartitionScheme], float]:
        """
        动态规划搜索。

        考虑跨层传播，找全局最优。
        复杂度: O(L × C²)
        """
        num_candidates = len(self.candidates)

        # dp[l][c] = (最小成本, 前驱选择)
        dp = [[float('inf')] * num_candidates for _ in range(self.num_layers)]
        parent = [[-1] * num_candidates for _ in range(self.num_layers)]

        # 初始化第一层
        for c, scheme in enumerate(self.candidates):
            if self._is_valid_for_layer(scheme, self.layer_infos[0]):
                costs = PartitionCostModel.compute_layer_cost(
                    self.layer_infos[0], scheme, None)
                dp[0][c] = costs['total_cost']

        # 动态规划
        for l in range(1, self.num_layers):
            layer_info = self.layer_infos[l]

            for c, scheme in enumerate(self.candidates):
                if not self._is_valid_for_layer(scheme, layer_info):
                    continue

                for pc, prev_scheme in enumerate(self.candidates):
                    if dp[l-1][pc] == float('inf'):
                        continue

                    costs = PartitionCostModel.compute_layer_cost(
                        layer_info, scheme, prev_scheme)

                    new_cost = dp[l-1][pc] + costs['total_cost']
                    if new_cost < dp[l][c]:
                        dp[l][c] = new_cost
                        parent[l][c] = pc

        # 回溯找最优路径
        best_final = min(range(num_candidates), key=lambda c: dp[-1][c])
        total_cost = dp[-1][best_final]

        result = []
        c = best_final
        for l in range(self.num_layers - 1, -1, -1):
            result.append(self.candidates[c])
            c = parent[l][c]

        return result[::-1], total_cost

    def _is_valid_for_layer(self, scheme: PartitionScheme, layer_info: dict) -> bool:
        """检查分区方案是否对该层有效。"""
        layer_type = layer_info.get('type', 'Conv')

        # FC 层不支持 OFMP
        if layer_type == 'FC' and scheme.get_size(PartitionDim.OFMP) > 1:
            return False

        # Pool 层不支持 INPP
        if layer_type == 'Pool' and scheme.get_size(PartitionDim.INPP) > 1:
            return False

        return True


# =============================================================================
# 演示
# =============================================================================

def demo():
    """演示分区方案探索。"""
    print("=" * 70)
    print("NN Partitioning across Processing Elements")
    print("=" * 70)

    # 模拟 VGG-style 网络
    layer_infos = [
        {'name': 'conv1', 'type': 'Conv', 'macs': 1e9, 'ifmap_size': 150528,
         'output_size': 802816, 'weight_size': 1728, 'total_data_size': 955072},
        {'name': 'conv2', 'type': 'Conv', 'macs': 3.7e9, 'ifmap_size': 802816,
         'output_size': 802816, 'weight_size': 36864, 'total_data_size': 1642496},
        {'name': 'pool1', 'type': 'Pool', 'macs': 0, 'ifmap_size': 802816,
         'output_size': 200704, 'weight_size': 0, 'total_data_size': 1003520},
        {'name': 'conv3', 'type': 'Conv', 'macs': 1.8e9, 'ifmap_size': 200704,
         'output_size': 401408, 'weight_size': 73728, 'total_data_size': 675840},
        {'name': 'fc1', 'type': 'FC', 'macs': 1e8, 'ifmap_size': 25088,
         'output_size': 4096, 'weight_size': 102760448, 'total_data_size': 102789632},
    ]

    print("\n网络结构:")
    print("-" * 70)
    for info in layer_infos:
        print(
            f"  {info['name']:10s} | {info['type']:6s} | MACs: {info['macs']:.1e}")

    # 创建探索器
    explorer = PartitionExplorer(num_nodes=16, layer_infos=layer_infos)
    print(f"\n候选分区方案数: {len(explorer.candidates)}")

    # 贪心搜索
    print("\n" + "-" * 70)
    print("贪心搜索结果:")
    print("-" * 70)
    greedy_result, greedy_cost = explorer.greedy_search()
    for info, scheme in zip(layer_infos, greedy_result):
        print(f"  {info['name']:10s}: {scheme}")
    print(f"  总成本: {greedy_cost:.2e}")

    # DP 搜索
    print("\n" + "-" * 70)
    print("动态规划搜索结果:")
    print("-" * 70)
    dp_result, dp_cost = explorer.dp_search()
    for info, scheme in zip(layer_infos, dp_result):
        print(f"  {info['name']:10s}: {scheme}")
    print(f"  总成本: {dp_cost:.2e}")

    # 比较
    print("\n" + "-" * 70)
    print("比较:")
    print("-" * 70)
    improvement = (greedy_cost - dp_cost) / greedy_cost * 100
    print(f"  贪心成本: {greedy_cost:.2e}")
    print(f"  DP 成本:  {dp_cost:.2e}")
    print(f"  DP 优于贪心: {improvement:.2f}%")


if __name__ == '__main__':
    demo()
