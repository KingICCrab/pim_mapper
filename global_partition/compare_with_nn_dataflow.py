#!/usr/bin/env python
"""
比较全局分区优化器与 nn_dataflow 原始方法的区别分析
"""

from ilp_optimizer_v2 import (
    HybridPartitionChoice, PartDim, LayerConfig,
    ComputeCostModel, RedistributionCostModel,
    GlobalPartitionILPOptimizerV2 as GlobalPartitionOptimizer
)
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))


# 简单的层定义
class SimpleConvLayer:
    """简化的卷积层"""

    def __init__(self, nifm, nofm, hofm, hfil, wofm=None, wfil=None):
        self.nifm = nifm  # input channels (C)
        self.nofm = nofm  # output channels (K)
        self.hofm = hofm  # output height
        self.wofm = wofm if wofm else hofm  # output width
        self.hfil = hfil  # filter height (R)
        self.wfil = wfil if wfil else hfil  # filter width
        self.type = 'Conv'

    def total_ops(self, batch_size=1):
        return batch_size * self.nofm * self.hofm * self.wofm * self.nifm * self.hfil * self.wfil


def compare_approaches():
    """
    比较我们的全局优化器与 nn_dataflow 原始方法的关键区别
    """

    print("=" * 80)
    print("全局分区优化器 vs nn_dataflow 原始方法 对比分析")
    print("=" * 80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                           方法论对比                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  nn_dataflow 原始方法:                                                       │
│  ────────────────────                                                       │
│  • 逐层贪心 (Layer-by-Layer Greedy)                                          │
│  • 每层独立选择最优分区，不考虑层间转换成本                                        │
│  • schedule_search() 对每层枚举所有 partition，选 top-n                        │
│  • 只在单层内优化 (compute + memory access)                                   │
│  • 层间数据布局重排由 DataLayout 隐式处理                                        │
│                                                                             │
│  我们的全局优化器:                                                             │
│  ────────────────                                                           │
│  • 全局最优 (Global Optimal via ILP/DP)                                       │
│  • 显式建模层间重分布成本 (K→C, Spatial, Batch)                                 │
│  • 联合优化所有层的分区选择                                                      │
│  • 目标: min Σ(compute_cost + transition_cost)                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                         关键区别                                             │
├─────────────────────────────────────────────────────────────────────────────┤

1. 优化范围 (Optimization Scope)
   ┌──────────────────┬───────────────────────────────────────────────────────┐
   │ nn_dataflow      │ 单层优化，每层独立决策                                   │
   ├──────────────────┼───────────────────────────────────────────────────────┤
   │ 我们的优化器      │ 跨层全局优化，考虑层间依赖                                │
   └──────────────────┴───────────────────────────────────────────────────────┘

2. 成本模型 (Cost Model)
   ┌──────────────────┬───────────────────────────────────────────────────────┐
   │ nn_dataflow      │ Cost = ops × unit_cost + mem_access × unit_cost       │
   │                  │ 侧重单层的计算和访存成本                                 │
   ├──────────────────┼───────────────────────────────────────────────────────┤
   │ 我们的优化器      │ LayerCost = Compute + R_C(INPP All-Reduce) + R_halo   │
   │                  │ Transition = R_K + R_spatial + R_N                     │
   │                  │ 显式建模层间通信成本                                     │
   └──────────────────┴───────────────────────────────────────────────────────┘

3. 分区选择逻辑
   ┌──────────────────┬───────────────────────────────────────────────────────┐
   │ nn_dataflow      │ 每层选本层最优的 partition                              │
   │                  │ 可能导致相邻层 partition 不兼容，产生隐式重排              │
   ├──────────────────┼───────────────────────────────────────────────────────┤
   │ 我们的优化器      │ DP: dp[l][c] = min_{c'} { dp[l-1][c'] + trans(c',c) } │
   │                  │                   + compute[l][c]                      │
   │                  │ 考虑所有可能的分区序列，找全局最优                         │
   └──────────────────┴───────────────────────────────────────────────────────┘

└─────────────────────────────────────────────────────────────────────────────┘
""")

    # 创建测试网络
    print("\n" + "=" * 80)
    print("具体示例: 四层卷积网络 (模拟 VGG 片段)")
    print("=" * 80)

    layers = [
        ('conv1', SimpleConvLayer(64, 128, 56, 3)),   # C=64 -> K=128
        ('conv2', SimpleConvLayer(128, 128, 56, 3)),  # C=128 -> K=128
        ('conv3', SimpleConvLayer(128, 256, 28, 3)),  # C=128 -> K=256
        ('conv4', SimpleConvLayer(256, 256, 28, 3)),  # C=256 -> K=256
    ]

    dim_nodes = (4, 4)  # 16 nodes
    batch_size = 1

    print(f"\n网络结构:")
    for name, layer in layers:
        print(
            f"  {name}: Conv(C={layer.nifm}, K={layer.nofm}, H={layer.hofm}, R={layer.hfil})")
    print(
        f"节点阵列: {dim_nodes[0]} x {dim_nodes[1]} = {dim_nodes[0]*dim_nodes[1]} nodes")

    # 创建 mock resource 和 network
    class MockPhyDim2:
        def __init__(self, h, w):
            self.h = h
            self.w = w

        def size(self):
            return self.h * self.w

    class MockProcRegion:
        def __init__(self, dim):
            self.dim = dim

    class MockResource:
        def __init__(self, dim_nodes):
            self.proc_region = MockProcRegion(dim_nodes)

    class MockNetwork:
        def __init__(self, layers):
            self._layers = {name: layer for name, layer in layers}
            self._order = [name for name, _ in layers]

        def __iter__(self):
            return iter(self._order)

        def __getitem__(self, key):
            return self._layers[key]

    mock_dim = MockPhyDim2(dim_nodes[0], dim_nodes[1])
    mock_resource = MockResource(mock_dim)
    mock_network = MockNetwork(layers)

    # 创建优化器
    optimizer = GlobalPartitionOptimizer(
        network=mock_network,
        resource=mock_resource,
        batch_size=batch_size
    )

    # 首先调用 optimize 让优化器计算所有成本
    print("\n计算成本并优化...")
    solution = optimizer.optimize(verbose=False)

    # ====== 方法 1: 贪心 (逐层选最优) ======
    print("\n" + "-" * 60)
    print("方法 1: 贪心方法 (逐层选最优，不考虑转换)")
    print("-" * 60)

    greedy_choices = []
    for layer_idx, config in enumerate(optimizer.layer_configs):
        best_choice = None
        best_cost = float('inf')
        for choice_idx, choice in enumerate(optimizer.partition_choices[layer_idx]):
            cost = optimizer.compute_costs[(layer_idx, choice_idx)]
            if cost < best_cost:
                best_cost = cost
                best_choice = (choice_idx, choice)
        greedy_choices.append(best_choice)
        print(f"  {config.name}: {best_choice[1]} (计算成本={best_cost:,.0f})")

    # 计算贪心总成本
    greedy_total_compute = sum(
        optimizer.compute_costs[(i, c[0])] for i, c in enumerate(greedy_choices)
    )
    greedy_total_trans = 0
    for i in range(len(layers) - 1):
        key = (i, greedy_choices[i][0], greedy_choices[i+1][0])
        if key in optimizer.redistribution_costs:
            greedy_total_trans += optimizer.redistribution_costs[key]

    print(f"\n贪心方法总成本:")
    print(f"  计算成本: {greedy_total_compute:,.0f}")
    print(f"  转换成本: {greedy_total_trans:,.0f}")
    print(f"  总成本: {greedy_total_compute + greedy_total_trans:,.0f}")

    # ====== 方法 2: 全局 DP 优化 ======
    print("\n" + "-" * 60)
    print("方法 2: 全局 DP 优化 (联合考虑所有层)")
    print("-" * 60)

    # solution 已在上面获取
    # 计算全局优化成本
    total_compute = 0
    total_redist = 0
    global_indices = []

    for layer_idx, (layer_name, choice) in enumerate(solution):
        # 找到 choice 在 partition_choices 中的索引
        choice_idx = None
        for ci, c in enumerate(optimizer.partition_choices[layer_idx]):
            if c == choice:
                choice_idx = ci
                break
        global_indices.append(choice_idx)
        total_compute += optimizer.compute_costs[(layer_idx, choice_idx)]
        if layer_idx > 0:
            prev_idx = global_indices[layer_idx - 1]
            key = (layer_idx - 1, prev_idx, choice_idx)
            total_redist += optimizer.redistribution_costs.get(key, 0)

    # 输出结果
    print(f"\n全局优化结果:")
    for layer_name, choice in solution:
        print(f"  {layer_name}: {choice}")

    print(f"\n全局优化总成本:")
    print(f"  计算成本: {total_compute:,.0f}")
    print(f"  转换成本: {total_redist:,.0f}")
    print(f"  总成本: {total_compute + total_redist:,.0f}")

    # ====== 比较 ======
    print("\n" + "=" * 60)
    print("对比分析")
    print("=" * 60)

    greedy_total = greedy_total_compute + greedy_total_trans
    global_total = total_compute + total_redist

    if global_total < greedy_total:
        improvement = (greedy_total - global_total) / greedy_total * 100
        print(f"\n✓ 全局优化比贪心方法好 {improvement:.2f}%")
        print(f"  贪心总成本: {greedy_total:,.0f}")
        print(f"  全局总成本: {global_total:,.0f}")
        print(f"  节省: {greedy_total - global_total:,.0f}")
    elif global_total == greedy_total:
        print(f"\n= 两种方法结果相同 (都达到最优)")
    else:
        print(f"\n(本例中贪心恰好是全局最优)")

    # 分析差异原因
    print("\n" + "-" * 40)
    print("差异分析:")
    print("-" * 40)

    # 检查分区选择是否不同
    diff_layers = []
    for i in range(len(layers)):
        greedy_idx = greedy_choices[i][0]
        global_idx = global_indices[i]
        if greedy_idx != global_idx:
            diff_layers.append(i)

    if diff_layers:
        print(f"\n不同分区选择的层: {[layers[i][0] for i in diff_layers]}")
        for i in diff_layers:
            g_choice = greedy_choices[i][1]
            o_choice = optimizer.partition_choices[i][global_indices[i]]
            g_cost = optimizer.compute_costs[(i, greedy_choices[i][0])]
            o_cost = optimizer.compute_costs[(i, global_indices[i])]
            print(f"\n  {layers[i][0]}:")
            print(f"    贪心选择: {g_choice} (计算={g_cost:,.0f})")
            print(f"    全局选择: {o_choice} (计算={o_cost:,.0f})")

            if i > 0:
                # 显示前一层到当前层的转换成本差异
                prev_g = greedy_choices[i-1][0]
                prev_o = global_indices[i-1]
                curr_g = greedy_choices[i][0]
                curr_o = global_indices[i]

                trans_g = optimizer.redistribution_costs.get(
                    (i-1, prev_g, curr_g), 0)
                trans_o = optimizer.redistribution_costs.get(
                    (i-1, prev_o, curr_o), 0)
                print(f"    转换成本(从上一层): 贪心={trans_g:,.0f}, 全局={trans_o:,.0f})")
    else:
        print("\n所有层的分区选择相同")

    # 总结
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    print("""
全局分区优化器的优势:

1. 显式考虑层间转换成本
   • 贪心方法只看单层最优，忽略转换开销
   • 全局方法可能选择「单层次优但转换友好」的分区

2. 避免频繁的分区切换
   • 相邻层使用兼容分区可减少 All-to-All 重排
   • K→C 传播时，INPP 策略天然继承上一层的 OUTP 输出

3. 特别适用于:
   • 深度网络 (层数多，转换累积)
   • 通信受限系统 (通信成本高)
   • 通道数变化剧烈的网络

4. 与 nn_dataflow 兼容:
   • 使用相同的 PartitionScheme 格式
   • 可集成到现有工作流程
""")


def main():
    compare_approaches()


if __name__ == '__main__':
    main()
