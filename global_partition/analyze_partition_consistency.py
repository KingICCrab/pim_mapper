"""
详细分析 ILP 优化结果与 nn_dataflow 行为的一致性。

主要关注点：
1. OUTP 分区是否合理（K 通道分区）
2. 最后一层 INPP 的选择是否合理
3. 重分布成本为 0 的原因分析
"""

from ilp_optimizer_v2 import (
    GlobalPartitionILPOptimizerV2, LayerConfig, HybridPartitionChoice,
    PartDim, RedistributionCostModel, ComputeCostModel
)
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))


def analyze_k_to_c_propagation():
    """分析 K→C 传播和重分布成本。"""
    print("=" * 70)
    print("K→C 传播分析")
    print("=" * 70)

    # 创建简单的测试案例
    class SimpleLayer:
        def __init__(self, nifm, nofm, hofm, wofm, hfil=3, wfil=3):
            self.nifm = nifm
            self.nofm = nofm
            self.hofm = hofm
            self.wofm = wofm
            self.hfil = hfil
            self.wfil = wfil

    # 测试案例 1: 连续 OUTP 分区（VGG 风格）
    print("\n案例 1: 连续 OUTP 分区（VGG conv 风格）")
    print("-" * 50)

    layer1 = SimpleLayer(64, 128, 56, 56)   # conv3
    layer2 = SimpleLayer(128, 128, 56, 56)  # conv4

    config1 = LayerConfig("conv3", layer1, 0, batch_size=1)
    config2 = LayerConfig("conv4", layer2, 1, batch_size=1)

    # 两层都用 OUTP=(4,4)
    choice_outp = HybridPartitionChoice({
        PartDim.OUTP: (4, 4),
        PartDim.OFMP: (1, 1),
        PartDim.BATP: (1, 1),
        PartDim.INPP: (1, 1)
    })

    # 层1 的 K 分区 = 16，层2 的 C = 128
    # 层1 输出 K=128 被分到 16 个节点，每节点 8 通道
    # 层2 输入 C=128，如果也用 OUTP 分区，C 不分区
    # 但是层1 的 K 输出已经分布了！

    print(f"Layer 1: C={layer1.nifm}, K={layer1.nofm}, 分区: OUTP=(4,4)")
    print(f"Layer 2: C={layer2.nifm}, K={layer2.nofm}, 分区: OUTP=(4,4)")
    print(f"  Layer 1 输出 K=128 分布在 16 节点，每节点 8 通道")
    print(f"  Layer 2 需要 C=128 作为输入")

    # 计算重分布成本
    cost = RedistributionCostModel.compute_redistribution_cost(
        config1, config2, choice_outp, choice_outp
    )

    print(f"\n重分布成本 (K→C): {cost:,.0f}")

    # 解释
    k_factor = choice_outp.get_size(PartDim.OUTP)  # 16
    dst_inpp = choice_outp.get_size(PartDim.INPP)  # 1

    print(f"\n分析:")
    print(f"  Layer 1 OUTP 因子 k = {k_factor}")
    print(f"  Layer 2 INPP 因子 p = {dst_inpp}")
    print(f"  根据公式: k > 1 且 p = 1 → 需要 All-Gather")
    print(f"  All-Gather 成本 = O_l × (k-1)/k")

    output_size = config1.output_size  # batch * K * H * W
    expected_cost = output_size * (k_factor - 1) / k_factor

    print(f"  O_l = {output_size:,}")
    print(f"  期望成本 = {expected_cost:,.0f}")

    # 测试案例 2: 层1 不分 K，层2 用 INPP
    print("\n" + "=" * 70)
    print("案例 2: 层1 不分 K（OFMP），层2 分 C（INPP）")
    print("-" * 50)

    choice_ofmp = HybridPartitionChoice({
        PartDim.OUTP: (1, 1),
        PartDim.OFMP: (4, 4),
        PartDim.BATP: (1, 1),
        PartDim.INPP: (1, 1)
    })

    choice_inpp = HybridPartitionChoice({
        PartDim.OUTP: (1, 1),
        PartDim.OFMP: (1, 1),
        PartDim.BATP: (1, 1),
        PartDim.INPP: (4, 4)
    })

    print(f"Layer 1: 分区 OFMP=(4,4) - 空间分区，K 完整")
    print(f"Layer 2: 分区 INPP=(4,4) - C 通道分区")

    cost2 = RedistributionCostModel.compute_redistribution_cost(
        config1, config2, choice_ofmp, choice_inpp
    )

    print(f"\n重分布成本: {cost2:,.0f}")

    k_factor2 = choice_ofmp.get_size(PartDim.OUTP)  # 1
    dst_inpp2 = choice_inpp.get_size(PartDim.INPP)  # 16

    print(f"\n分析:")
    print(f"  Layer 1 OUTP 因子 k = {k_factor2}")
    print(f"  Layer 2 INPP 因子 p = {dst_inpp2}")
    print(f"  根据公式: k = 1 → 成本为 0（K 完整，本地选择）")

    # 但可能有空间重分布成本
    src_ofmp = choice_ofmp.get_factor(PartDim.OFMP)
    dst_ofmp = choice_inpp.get_factor(PartDim.OFMP)
    print(f"  Layer 1 OFMP = {src_ofmp}")
    print(f"  Layer 2 OFMP = {dst_ofmp}")
    if src_ofmp != dst_ofmp:
        print(f"  空间分区不匹配，需要空间重分布！")

    # 测试案例 3: 为什么 VGG 全用 OUTP 时重分布成本为 0？
    print("\n" + "=" * 70)
    print("案例 3: 为什么全用 OUTP 时重分布成本为 0？")
    print("-" * 50)

    # VGG conv2 → conv3
    vgg_layer1 = SimpleLayer(64, 64, 224, 224, 3, 3)   # conv2
    vgg_layer2 = SimpleLayer(64, 128, 112, 112, 3, 3)  # conv3 (after pooling)

    vgg_config1 = LayerConfig("conv2", vgg_layer1, 0, batch_size=1)
    vgg_config2 = LayerConfig("conv3", vgg_layer2, 1, batch_size=1)

    cost3 = RedistributionCostModel.compute_redistribution_cost(
        vgg_config1, vgg_config2, choice_outp, choice_outp
    )

    print(f"conv2: C=64, K=64, H=224")
    print(f"conv3: C=64, K=128, H=112")
    print(f"两层都用 OUTP=(4,4)")
    print(f"重分布成本: {cost3:,.0f}")

    # 关键分析
    print(f"\n关键分析:")
    print(f"  conv2 输出: K=64 分到 16 节点 → 每节点 4 通道")
    print(f"  conv3 输入: C=64，但 conv3 用 OUTP 分区")
    print(f"  conv3 的 INPP = 1，意味着每个节点需要完整的 C=64 输入")
    print(f"\n  问题: conv2 的 K=64 已经分布，conv3 需要完整 C")
    print(f"  这应该需要 All-Gather！")

    # 检查实际计算
    k = choice_outp.get_size(PartDim.OUTP)  # 16
    p = choice_outp.get_size(PartDim.INPP)  # 1

    print(f"\n  k (src OUTP) = {k}")
    print(f"  p (dst INPP) = {p}")
    print(f"  按照文档: k > 1 且 p = 1 → All-Gather 成本 = O × (k-1)/k")

    # 让我检查 _k_to_c_cost 的实际实现
    print("\n" + "=" * 70)
    print("检查代码实现...")
    print("=" * 70)

    # 直接调用底层函数
    k_to_c = RedistributionCostModel._k_to_c_cost(
        vgg_config1, vgg_config2, choice_outp, choice_outp, vgg_config1.output_size
    )
    spatial = RedistributionCostModel._spatial_cost(
        vgg_config1, vgg_config2, choice_outp, choice_outp, vgg_config1.output_size
    )
    batch = RedistributionCostModel._batch_cost(
        choice_outp, choice_outp, vgg_config1.output_size)
    inpp_reduce = RedistributionCostModel._inpp_reduction_cost(
        choice_outp, vgg_config1.output_size)

    print(f"K→C 成本: {k_to_c:,.0f}")
    print(f"Spatial 成本: {spatial:,.0f}")
    print(f"Batch 成本: {batch:,.0f}")
    print(f"INPP Reduce 成本: {inpp_reduce:,.0f}")
    print(f"总计: {k_to_c + spatial + batch + inpp_reduce:,.0f}")


def analyze_why_outp_is_preferred():
    """分析为什么 OUTP 分区被优先选择。"""
    print("\n" + "=" * 70)
    print("为什么 OUTP 分区被优先选择？")
    print("=" * 70)

    class SimpleLayer:
        def __init__(self, nifm, nofm, hofm, wofm, hfil=3, wfil=3):
            self.nifm = nifm
            self.nofm = nofm
            self.hofm = hofm
            self.wofm = wofm
            self.hfil = hfil
            self.wfil = wfil

    layer = SimpleLayer(128, 256, 56, 56)
    config = LayerConfig("conv", layer, 0, batch_size=1)

    # 比较不同分区策略的计算成本
    choices = {
        'OUTP': HybridPartitionChoice({PartDim.OUTP: (4, 4)}),
        'OFMP': HybridPartitionChoice({PartDim.OFMP: (4, 4)}),
        'INPP': HybridPartitionChoice({PartDim.INPP: (4, 4)}),
    }

    print(
        f"\n层: C={layer.nifm}, K={layer.nofm}, H={layer.hofm}, R={layer.hfil}")
    print(f"总 MACs: {config.macs:,}")
    print(f"输出大小: {config.output_size:,}")
    print(f"\n各分区策略的计算成本:")

    for name, choice in choices.items():
        cost = ComputeCostModel.compute_cost(config, choice)
        inpp = choice.get_size(PartDim.INPP)

        # 详细分解
        macs_per_node = config.macs / 16
        if inpp > 1:
            # INPP 需要 All-Reduce
            reduce_cost = config.output_size * 2 * (inpp - 1) / inpp
        else:
            reduce_cost = 0

        print(f"  {name}:")
        print(f"    MACs/node: {macs_per_node:,.0f}")
        print(f"    INPP All-Reduce: {reduce_cost:,.0f}")
        print(f"    总计算成本: {cost:,.0f}")

    print(f"\n结论:")
    print(f"  - OUTP 和 OFMP 的计算成本相同（没有 All-Reduce）")
    print(f"  - INPP 有额外的 All-Reduce 成本")
    print(f"  - 选择 OUTP 的原因: 传播到下一层时成本更低")


def main():
    """主分析函数。"""
    analyze_k_to_c_propagation()
    analyze_why_outp_is_preferred()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
1. 全用 OUTP 分区时重分布成本为 0 的原因需要检查：
   - 可能是代码实现有问题
   - 或者模型假设与 nn_dataflow 不同

2. nn_dataflow 默认行为：
   - partition_hybrid=False 时：CONV 层只用 OFMP，FC 层只用 OUTP
   - partition_hybrid=True 时：可以混合使用

3. 我们的 ILP 模型选择全用 OUTP 的原因：
   - 计算成本：OUTP 和 OFMP 相同
   - 重分布成本：全用 OUTP 可以避免 K→C 不匹配（如果代码正确）
   
4. 潜在问题：
   - K→C 传播成本计算可能有问题
   - 需要验证连续 OUTP 分区时的实际重分布成本
""")


if __name__ == '__main__':
    main()
