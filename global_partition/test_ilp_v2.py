"""
测试混合分区全局优化器 V2。

展示完整的分区传播和重分布成本建模。
"""

from global_partition.ilp_optimizer_v2 import (
    GlobalPartitionILPOptimizerV2,
    HybridPartitionChoice,
    PartDim,
    LayerConfig,
    RedistributionCostModel
)
import sys
sys.path.insert(0, '.')


class SimpleLayer:
    """简单层类用于测试。"""

    def __init__(self, nifm, nofm, hofm, wofm, hfil=3, wfil=3):
        self.nifm = nifm
        self.nofm = nofm
        self.hofm = hofm
        self.wofm = wofm
        self.hfil = hfil
        self.wfil = wfil


class SimpleNetwork(dict):
    """简单网络类用于测试。"""
    pass


class SimpleResource:
    """简单资源类用于测试。"""

    def __init__(self, h, w):
        self.dim_nodes = type('DimNodes', (), {'h': h, 'w': w})()


def test_redistribution_cost():
    """测试重分布成本计算。"""
    print("="*70)
    print("测试重分布成本模型")
    print("="*70)

    # 创建两个相邻层
    layer1 = SimpleLayer(64, 128, 56, 56)   # C=64, K=128
    layer2 = SimpleLayer(128, 128, 56, 56)  # C=128 (= layer1.K), K=128

    config1 = LayerConfig('conv1', layer1, 0, batch_size=4)
    config2 = LayerConfig('conv2', layer2, 1, batch_size=4)

    # 测试不同分区组合
    test_cases = [
        # (choice1, choice2, expected_behavior)
        (
            HybridPartitionChoice({PartDim.OUTP: (4, 4)}),  # 16-way K 分区
            HybridPartitionChoice({PartDim.INPP: (4, 4)}),  # 16-way C 分区
            "完美匹配: K分区 → C分区"
        ),
        (
            HybridPartitionChoice({PartDim.OUTP: (4, 4)}),  # 16-way K 分区
            HybridPartitionChoice({PartDim.OUTP: (4, 4)}),  # 16-way K 分区
            "需要 All-Gather: layer1 输出分散"
        ),
        (
            HybridPartitionChoice({PartDim.OFMP: (4, 4)}),  # 空间分区
            HybridPartitionChoice({PartDim.OFMP: (4, 4)}),  # 空间分区
            "空间匹配 + Halo exchange"
        ),
        (
            HybridPartitionChoice({PartDim.OFMP: (4, 4)}),  # 空间分区
            HybridPartitionChoice({PartDim.OUTP: (4, 4)}),  # K 分区
            "需要空间 All-to-All"
        ),
        (
            HybridPartitionChoice({PartDim.BATP: (4, 4)}),  # Batch 分区
            HybridPartitionChoice({PartDim.BATP: (4, 4)}),  # Batch 分区
            "Batch 匹配: 无开销"
        ),
        (
            HybridPartitionChoice({PartDim.INPP: (4, 4)}),  # INPP 分区
            HybridPartitionChoice({PartDim.OUTP: (4, 4)}),  # OUTP 分区
            "需要 All-Reduce (INPP)"
        ),
    ]

    print(f"\n层1: {config1.name}, K={config1.nofm}, C={config1.nifm}")
    print(f"层2: {config2.name}, K={config2.nofm}, C={config2.nifm}")
    print(f"输出数据量: {config1.output_size:,}")
    print()

    for choice1, choice2, desc in test_cases:
        cost = RedistributionCostModel.compute_redistribution_cost(
            config1, config2, choice1, choice2)
        print(f"{desc}")
        print(f"  层1: {choice1}")
        print(f"  层2: {choice2}")
        print(f"  重分布成本: {cost:,.0f}")
        print()


def test_hybrid_choices():
    """测试混合分区方案生成。"""
    print("="*70)
    print("测试混合分区方案生成")
    print("="*70)

    network = SimpleNetwork()
    network['conv1'] = SimpleLayer(3, 64, 112, 112)
    network['conv2'] = SimpleLayer(64, 128, 56, 56)

    resource = SimpleResource(4, 4)  # 16 nodes

    optimizer = GlobalPartitionILPOptimizerV2(
        network=network,
        resource=resource,
        batch_size=4,
        solver='dp'
    )

    print(
        f"\n节点阵列: {optimizer.dim_nodes[0]}×{optimizer.dim_nodes[1]} = {optimizer.total_nodes} 节点")

    for l, config in enumerate(optimizer.layer_configs):
        choices = optimizer.partition_choices[l]
        print(f"\n层 {config.name}: {len(choices)} 个候选方案")

        # 显示一些示例
        print("  示例方案:")
        for i, choice in enumerate(choices[:5]):
            print(f"    {i+1}. {choice}")


def test_propagation_impact():
    """测试分区传播对全局优化的影响。"""
    print("="*70)
    print("测试分区传播对全局优化的影响")
    print("="*70)

    # 创建一个有不同维度的网络
    network = SimpleNetwork()
    # 通道逐渐增加，空间逐渐减小 (典型 CNN)
    network['conv1'] = SimpleLayer(3, 64, 224, 224)
    network['pool1'] = SimpleLayer(64, 64, 112, 112, hfil=1, wfil=1)
    network['conv2'] = SimpleLayer(64, 128, 112, 112)
    network['pool2'] = SimpleLayer(128, 128, 56, 56, hfil=1, wfil=1)
    network['conv3'] = SimpleLayer(128, 256, 56, 56)
    network['pool3'] = SimpleLayer(256, 256, 28, 28, hfil=1, wfil=1)
    network['conv4'] = SimpleLayer(256, 512, 28, 28)

    resource = SimpleResource(4, 4)  # 16 nodes

    optimizer = GlobalPartitionILPOptimizerV2(
        network=network,
        resource=resource,
        batch_size=8,
        solver='dp'
    )

    print(f"\n网络结构:")
    for config in optimizer.layer_configs:
        print(f"  {config.name}: C={config.nifm} → K={config.nofm}, "
              f"H×W={config.hofm}×{config.wofm}")

    print(f"\n候选方案数量:")
    for l, config in enumerate(optimizer.layer_configs):
        print(f"  {config.name}: {len(optimizer.partition_choices[l])} 个")

    print("\n运行优化...")
    solution = optimizer.optimize(verbose=True)
    optimizer.print_solution(solution)

    # 分析传播
    print("\n传播分析:")
    for i in range(len(solution) - 1):
        name1, choice1 = solution[i]
        name2, choice2 = solution[i + 1]

        # 检查 K→C
        k1 = choice1.get_size(PartDim.OUTP)
        c2 = choice2.get_size(PartDim.INPP)

        print(f"\n{name1} → {name2}:")
        print(f"  K分区: {k1}, 下一层C分区: {c2}", end="")
        if k1 == c2 or (k1 == 1 and c2 == 1):
            print(" ✓ (匹配)")
        else:
            print(" ✗ (需要重分布)")

        # 检查空间
        ofmp1 = choice1.get_factor(PartDim.OFMP)
        ofmp2 = choice2.get_factor(PartDim.OFMP)
        print(f"  空间分区: {ofmp1} → {ofmp2}", end="")
        if ofmp1 == ofmp2:
            print(" ✓ (匹配)")
        else:
            print(" ✗ (需要重分布)")


def main():
    """运行所有测试。"""
    test_redistribution_cost()
    print("\n")
    test_hybrid_choices()
    print("\n")
    test_propagation_impact()


if __name__ == "__main__":
    main()
