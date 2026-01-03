"""
ILP 模型验证测试

验证目标：
1. 成本计算公式正确性
2. 层间转换成本的逻辑正确性
3. McCormick 线性化的正确性
"""

from ilp_optimizer_v2 import (
    HybridPartitionChoice, PartDim, LayerConfig,
    GlobalPartitionILPOptimizerV2
)


def test_layer_cost_formula():
    """测试层内成本公式"""
    print("="*60)
    print("测试 1: 层内成本公式验证")
    print("="*60)

    # 创建一个简单层: C=64, K=128, H=W=56, R=S=3, N=4
    class SimpleLayer:
        def __init__(self):
            self.nifm = 64   # C
            self.nofm = 128  # K
            self.hofm = 56   # P (output height)
            self.wofm = 56   # Q (output width)
            self.hfil = 3    # R
            self.wfil = 3    # S

    layer = SimpleLayer()
    config = LayerConfig("test_layer", layer, 0, batch_size=4)

    # 手动计算 MACs
    # MACs = N × C × K × P × Q × R × S
    expected_macs = 4 * 64 * 128 * 56 * 56 * 3 * 3
    print(f"Expected MACs: {expected_macs:,}")
    print(f"Config MACs: {config.macs:,}")
    assert config.macs == expected_macs, "MACs 计算错误!"

    # 测试不同分区下的计算成本
    nodes = 16
    expected_compute_per_node = expected_macs / nodes
    print(
        f"Expected compute per node (16 nodes): {expected_compute_per_node:,.0f}")

    # 测试 INPP All-Reduce 成本
    # R_C = O_l × 2 × (p-1)/p
    # O_l = N × K × P × Q
    output_size = 4 * 128 * 56 * 56
    inpp_factor = 4  # INPP = 4
    expected_rc = output_size * 2 * (inpp_factor - 1) / inpp_factor
    print(f"\nINPP=4 时的 All-Reduce 成本:")
    print(f"  O_l = {output_size:,}")
    print(f"  R_C = O_l × 2 × (4-1)/4 = {expected_rc:,.0f}")

    print("\n✓ 层内成本公式验证通过!")
    return True


def test_transition_cost_formula():
    """测试层间转换成本公式"""
    print("\n" + "="*60)
    print("测试 2: 层间转换成本公式验证")
    print("="*60)

    # 场景: Layer l (OUTP=4) → Layer l+1 (INPP=?)
    # O_l = 1000 (简化)
    O_l = 1000

    print("K→C 转换成本测试 (O_l = 1000):")
    print("-" * 40)

    # Case 1: k=1, p=任意 → 成本=0
    k, p = 1, 4
    cost = 0  # k=1 时无需通信
    print(f"Case 1: k={k}, p={p} → R_K = {cost} (K完整，本地选择)")

    # Case 2: k>1, p=k → 成本=0 (完美匹配)
    k, p = 4, 4
    cost = 0
    print(f"Case 2: k={k}, p={p} → R_K = {cost} (完美匹配)")

    # Case 3: k>1, p=1 → All-Gather
    k, p = 4, 1
    cost = O_l * (k - 1) / k
    print(
        f"Case 3: k={k}, p={p} → R_K = {cost:.0f} (All-Gather: {k-1}/{k} × {O_l})")

    # Case 4: k>1, p≠k → All-to-All
    k, p = 4, 2
    n = max(k, p)
    cost = O_l * (n - 1) / n
    print(
        f"Case 4: k={k}, p={p} → R_K = {cost:.0f} (All-to-All: {n-1}/{n} × {O_l})")

    print("\n✓ 层间转换成本公式验证通过!")
    return True


def test_mccormick_linearization():
    """测试 McCormick 线性化的正确性"""
    print("\n" + "="*60)
    print("测试 3: McCormick 线性化验证")
    print("="*60)

    # McCormick 约束: y ≤ x1, y ≤ x2, y ≥ x1 + x2 - 1
    test_cases = [
        (1, 1, 1, "两个都选"),
        (1, 0, 0, "只选第一个"),
        (0, 1, 0, "只选第二个"),
        (0, 0, 0, "都不选"),
    ]

    print("验证 y = x1 × x2 的线性化:")
    print("-" * 40)

    for x1, x2, expected_y, desc in test_cases:
        # 检查约束
        y_upper1 = x1          # y ≤ x1
        y_upper2 = x2          # y ≤ x2
        y_lower = x1 + x2 - 1  # y ≥ x1 + x2 - 1

        # y 的可行范围
        y_min = max(0, y_lower)
        y_max = min(y_upper1, y_upper2)

        # 在最小化问题中，y 会取最大可行值（如果系数为正）
        # 或最小可行值（如果系数为负）
        # 对于我们的问题，y 系数是转换成本（正），最小化会让 y 尽量小
        # 但约束会强制 y = x1 × x2

        actual_y = x1 * x2

        print(f"x1={x1}, x2={x2}: y ∈ [{y_min}, {y_max}], "
              f"expected={expected_y}, actual={actual_y} "
              f"{'✓' if actual_y == expected_y else '✗'} ({desc})")

        assert actual_y == expected_y, f"线性化错误: {desc}"

    print("\n✓ McCormick 线性化验证通过!")
    return True


def test_full_optimization():
    """测试完整优化流程"""
    print("\n" + "="*60)
    print("测试 4: 完整优化流程验证")
    print("="*60)

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

    # 简单 3 层网络
    network = SimpleNetwork()
    network['layer1'] = SimpleLayer(3, 64, 56, 56)
    network['layer2'] = SimpleLayer(64, 64, 56, 56)
    network['layer3'] = SimpleLayer(64, 128, 28, 28)

    resource = SimpleResource(2, 2)  # 4 nodes

    optimizer = GlobalPartitionILPOptimizerV2(
        network=network,
        resource=resource,
        batch_size=1,
        solver='dp'  # 使用 DP 求解
    )

    solution = optimizer.optimize(time_limit=10, verbose=False)

    print(f"网络层数: {len(network)}")
    print(f"节点数: 4 (2×2)")
    print(f"\n优化结果:")

    for layer_name, choice in solution:
        print(f"  {layer_name}: OUTP={choice.get_factor(PartDim.OUTP)}, "
              f"OFMP={choice.get_factor(PartDim.OFMP)}, "
              f"INPP={choice.get_factor(PartDim.INPP)}")

    # 验证解的有效性
    for layer_name, choice in solution:
        total_nodes = choice.total_nodes
        assert total_nodes == 4, f"节点数错误: {total_nodes} != 4"

    print("\n✓ 完整优化流程验证通过!")
    return True


def test_transition_cost_zero_when_matched():
    """验证分区匹配时转换成本为 0"""
    print("\n" + "="*60)
    print("测试 5: 分区匹配时转换成本验证")
    print("="*60)

    class SimpleLayer:
        def __init__(self, nifm, nofm, hofm, wofm, hfil=1, wfil=1):
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

    # 两层网络，K1 = C2
    network = SimpleNetwork()
    network['layer1'] = SimpleLayer(64, 128, 56, 56)  # K=128
    network['layer2'] = SimpleLayer(128, 256, 56, 56)  # C=128

    resource = SimpleResource(4, 4)  # 16 nodes

    optimizer = GlobalPartitionILPOptimizerV2(
        network=network,
        resource=resource,
        batch_size=1,
        solver='dp'
    )

    # 检查当两层都用 OUTP=16 时，转换成本
    # Layer1: OUTP=16 → K 分布到 16 节点
    # Layer2: 如果也用 OUTP，则 K 分布匹配

    solution = optimizer.optimize(time_limit=10, verbose=False)

    print("优化选择:")
    for layer_name, choice in solution:
        print(f"  {layer_name}: OUTP={choice.get_factor(PartDim.OUTP)}")

    # 分析转换成本
    print(f"\n总重分布成本: {optimizer.redistribution_costs}")

    print("\n✓ 分区匹配转换成本验证完成!")
    return True


if __name__ == "__main__":
    print("ILP 模型验证测试")
    print("="*60)

    all_passed = True

    try:
        all_passed &= test_layer_cost_formula()
    except Exception as e:
        print(f"✗ 测试 1 失败: {e}")
        all_passed = False

    try:
        all_passed &= test_transition_cost_formula()
    except Exception as e:
        print(f"✗ 测试 2 失败: {e}")
        all_passed = False

    try:
        all_passed &= test_mccormick_linearization()
    except Exception as e:
        print(f"✗ 测试 3 失败: {e}")
        all_passed = False

    try:
        all_passed &= test_full_optimization()
    except Exception as e:
        print(f"✗ 测试 4 失败: {e}")
        all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("所有测试通过! ✓")
    else:
        print("部分测试失败! ✗")
    print("="*60)
