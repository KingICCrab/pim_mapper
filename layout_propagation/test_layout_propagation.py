#!/usr/bin/env python
"""
测试 Layout Propagation 模块

基于规约分析判断布局敏感性：
- 有规约计算 → 布局敏感 → 分区边界
- 无规约计算 → 布局不敏感 → 可透传分区
"""

from layout_propagation import (
    ReductionAnalyzer, LayoutPropagator, LayoutSensitivity, OperatorInfo,
    analyze_network_sensitivity, find_layout_propagation_groups
)
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class MockLayer:
    """模拟 nn_dataflow 层"""

    def __init__(self, name, layer_type, nifm=64, nofm=64, hofm=56, wofm=None, hfil=1):
        self.name = name
        self.nifm = nifm
        self.nofm = nofm
        self.hofm = hofm
        self.wofm = wofm or hofm
        self.hifm = hofm
        self.wifm = wofm or hofm
        self.hfil = hfil
        self._layer_type = layer_type

    @property
    def __class__(self):
        class FakeClass:
            pass
        FakeClass.__name__ = self._layer_type
        return FakeClass


def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def test_reduction_analysis():
    """测试1: 规约分析"""
    print_header("测试1: 算子规约分析")

    test_cases = [
        # (算子类型, 期望有规约, 期望规约维度)
        ("ConvLayer", True, ['C', 'R', 'S']),
        ("FCLayer", True, ['C']),
        ("PoolingLayer", True, ['H', 'W']),
        ("MaxPool", True, ['H', 'W']),
        ("Softmax", True, ['C']),
        ("ReduceSum", True, ['axis']),
        ("ReLU", False, []),
        ("BatchNorm", False, []),
        ("Add", False, []),
        ("Concat", False, []),
    ]

    print()
    print("%-20s | %-10s | %-20s | %s" % ("算子类型", "有规约?", "规约维度", "敏感性"))
    print("-" * 70)

    all_pass = True
    for layer_type, expect_reduction, expect_dims in test_cases:
        layer = MockLayer("test", layer_type)
        op_info = ReductionAnalyzer.analyze(layer)

        # 验证
        reduction_ok = op_info.has_reduction == expect_reduction
        dims_ok = op_info.reduction_dims == expect_dims

        status = "✓" if (reduction_ok and dims_ok) else "✗"
        if not (reduction_ok and dims_ok):
            all_pass = False

        has_red = "是" if op_info.has_reduction else "否"
        sens = "敏感" if op_info.sensitivity == LayoutSensitivity.SENSITIVE else "不敏感"
        print("%-20s | %-10s | %-20s | %-10s %s" % (
            layer_type, has_red, str(op_info.reduction_dims), sens, status
        ))

    print()
    print("测试结果:", "全部通过 ✓" if all_pass else "存在失败 ✗")
    return all_pass


def test_vgg_style_network():
    """测试2: VGG风格网络传播"""
    print_header("测试2: VGG风格网络 Layout Propagation")

    # VGG-style: Conv-BN-ReLU 块
    layers = [
        # Block 1
        MockLayer('conv1_1', 'ConvLayer', 3, 64, 224, hfil=3),
        MockLayer('bn1_1', 'BatchNorm', 64, 64, 224),
        MockLayer('relu1_1', 'ReLU', 64, 64, 224),
        MockLayer('conv1_2', 'ConvLayer', 64, 64, 224, hfil=3),
        MockLayer('bn1_2', 'BatchNorm', 64, 64, 224),
        MockLayer('relu1_2', 'ReLU', 64, 64, 224),
        MockLayer('pool1', 'MaxPool', 64, 64, 112),

        # Block 2
        MockLayer('conv2_1', 'ConvLayer', 64, 128, 112, hfil=3),
        MockLayer('bn2_1', 'BatchNorm', 128, 128, 112),
        MockLayer('relu2_1', 'ReLU', 128, 128, 112),
        MockLayer('conv2_2', 'ConvLayer', 128, 128, 112, hfil=3),
        MockLayer('relu2_2', 'ReLU', 128, 128, 112),
        MockLayer('pool2', 'MaxPool', 128, 128, 56),

        # FC layers
        MockLayer('fc1', 'FCLayer', 128, 4096, 1),
        MockLayer('relu_fc1', 'ReLU', 4096, 4096, 1),
        MockLayer('fc2', 'FCLayer', 4096, 1000, 1),
    ]

    # 分析
    op_infos = []
    print()
    print("网络结构分析:")
    print("-" * 70)
    print("%4s | %-12s | %-12s | %-6s | %s" % ("序号", "层名", "类型", "规约?", "敏感性"))
    print("-" * 70)

    for i, layer in enumerate(layers):
        op_info = ReductionAnalyzer.analyze(layer)
        op_info.name = layer.name
        op_infos.append(op_info)

        has_red = "Y" if op_info.has_reduction else "N"
        sens = "SENS" if op_info.sensitivity == LayoutSensitivity.SENSITIVE else "PASS"
        print("%4d | %-12s | %-12s | %-6s | %s" % (
            i, layer.name, op_info.op_type, has_red, sens
        ))

    # 传播分析
    propagator = LayoutPropagator(op_infos)
    groups = propagator.find_propagation_groups()

    print()
    print("-" * 70)
    print("传播组 (同组层共享分区，无需重分布):")
    print("-" * 70)

    for i, group in enumerate(groups):
        layer_names = [layers[idx].name for idx in sorted(group)]
        anchor_idx = min(group)
        is_sensitive = op_infos[anchor_idx].sensitivity == LayoutSensitivity.SENSITIVE

        if len(group) > 1:
            print("  组 %d: %s" % (i+1, " -> ".join(layer_names)))
            print("         └─ 锚点: %s, 透传: %d 层" %
                  (layer_names[0], len(group)-1))
        else:
            print("  组 %d: %s (独立锚点)" % (i+1, layer_names[0]))

    # 统计
    sensitive = propagator.get_sensitive_operators()
    insensitive = propagator.get_insensitive_operators()

    print()
    print("-" * 70)
    print("优化统计:")
    print("-" * 70)
    print("  总层数:              %d" % len(layers))
    print("  敏感算子 (有规约):   %d 个 → 需要独立决策分区" % len(sensitive))
    print("  不敏感算子 (无规约): %d 个 → 可透传，无需独立决策" % len(insensitive))
    print()
    print("  原始优化变量:        %d × P (P = 分区候选数)" % len(layers))
    print("  优化后变量:          %d × P" % len(sensitive))
    print("  减少变量:            %d 个 (%.1f%%)" % (
        len(insensitive), 100*len(insensitive)/len(layers)
    ))


def test_resnet_style_network():
    """测试3: ResNet风格网络 (带跳跃连接)"""
    print_header("测试3: ResNet风格网络 (带Add跳跃连接)")

    # ResNet basic block:
    # x -> conv1 -> bn1 -> relu1 -> conv2 -> bn2 -> Add(x) -> relu2
    layers = [
        MockLayer('conv1', 'ConvLayer', 64, 64, 56, hfil=3),
        MockLayer('bn1', 'BatchNorm', 64, 64, 56),
        MockLayer('relu1', 'ReLU', 64, 64, 56),
        MockLayer('conv2', 'ConvLayer', 64, 64, 56, hfil=3),
        MockLayer('bn2', 'BatchNorm', 64, 64, 56),
        MockLayer('add', 'Add', 64, 64, 56),  # 跳跃连接
        MockLayer('relu2', 'ReLU', 64, 64, 56),
        MockLayer('conv3', 'ConvLayer', 64, 128, 28, hfil=3),
        MockLayer('pool', 'MaxPool', 128, 128, 14),
    ]

    # 非线性邻接关系 (有跳跃连接)
    # 0->1->2->3->4->5->6->7->8
    #             ↑_____|  (Add 接收 conv2 输出和原始输入)
    adjacency = {
        0: [1],
        1: [2],
        2: [3],
        3: [4],
        4: [5],
        5: [6],
        6: [7],
        7: [8],
        8: [],
    }

    # 分析
    op_infos = []
    print()
    print("网络结构:")
    print("  conv1 -> bn1 -> relu1 -> conv2 -> bn2 -> add -> relu2 -> conv3 -> pool")
    print("                                      ↑")
    print("                           (跳跃连接从conv1输出)")
    print()
    print("-" * 70)

    for i, layer in enumerate(layers):
        op_info = ReductionAnalyzer.analyze(layer)
        op_info.name = layer.name
        op_infos.append(op_info)

        has_red = "有规约" if op_info.has_reduction else "无规约"
        sens = "敏感" if op_info.sensitivity == LayoutSensitivity.SENSITIVE else "透传"
        print("  %d: %-8s (%s) - %s, %s" % (
            i, layer.name, op_info.op_type, has_red, sens
        ))

    # 传播
    propagator = LayoutPropagator(op_infos, adjacency)
    groups = propagator.find_propagation_groups()

    print()
    print("-" * 70)
    print("传播组:")
    print("-" * 70)

    for i, group in enumerate(groups):
        layer_names = [layers[idx].name for idx in sorted(group)]
        print("  组 %d: %s" % (i+1, " -> ".join(layer_names)))

    print()
    print("关键发现:")
    print("  - Add 是 elementwise 操作，无规约，可透传")
    print("  - conv1 的分区可以传播到 bn1, relu1")
    print("  - conv2 的分区可以传播到 bn2, add, relu2")
    print("  - 这意味着跳跃连接处不需要额外重分布!")


def test_transformer_style():
    """测试4: Transformer风格网络"""
    print_header("测试4: Transformer风格网络 (Attention)")

    # Simplified Transformer block
    layers = [
        MockLayer('q_proj', 'MatMul', 512, 512, 1),  # Query projection
        MockLayer('k_proj', 'MatMul', 512, 512, 1),  # Key projection
        MockLayer('v_proj', 'MatMul', 512, 512, 1),  # Value projection
        MockLayer('qk_matmul', 'MatMul', 512, 512, 1),  # Q @ K^T
        MockLayer('softmax', 'Softmax', 512, 512, 1),  # Attention weights
        MockLayer('attn_v', 'MatMul', 512, 512, 1),  # Attention @ V
        MockLayer('out_proj', 'MatMul', 512, 512, 1),  # Output projection
        MockLayer('residual', 'Add', 512, 512, 1),  # Residual connection
        MockLayer('layernorm', 'BatchNorm', 512, 512, 1),  # Layer norm (类似BN)
        MockLayer('ffn1', 'MatMul', 512, 2048, 1),  # FFN layer 1
        MockLayer('relu', 'ReLU', 2048, 2048, 1),
        MockLayer('ffn2', 'MatMul', 2048, 512, 1),  # FFN layer 2
        MockLayer('residual2', 'Add', 512, 512, 1),  # Residual connection
    ]

    op_infos = []
    print()
    print("Transformer Block 分析:")
    print("-" * 70)

    for i, layer in enumerate(layers):
        op_info = ReductionAnalyzer.analyze(layer)
        op_info.name = layer.name
        op_infos.append(op_info)

        sens = "SENS" if op_info.sensitivity == LayoutSensitivity.SENSITIVE else "PASS"
        print("  %2d: %-12s | %-10s | %s" % (
            i, layer.name, op_info.op_type, sens
        ))

    propagator = LayoutPropagator(op_infos)
    groups = propagator.find_propagation_groups()

    print()
    print("-" * 70)
    print("传播组:")
    for i, group in enumerate(groups):
        layer_names = [layers[idx].name for idx in sorted(group)]
        print("  组 %d: %s" % (i+1, " -> ".join(layer_names)))

    sensitive = propagator.get_sensitive_operators()
    print()
    print("统计: %d/%d 层需要独立分区决策" % (len(sensitive), len(layers)))


def test_sensitivity_api():
    """测试5: 敏感性分析 API"""
    print_header("测试5: 便捷 API 测试")

    layers = [
        MockLayer('conv', 'ConvLayer', 3, 64, 224, hfil=3),
        MockLayer('bn', 'BatchNorm', 64, 64, 224),
        MockLayer('relu', 'ReLU', 64, 64, 224),
        MockLayer('pool', 'MaxPool', 64, 64, 112),
    ]

    # 方式1: 直接获取传播组
    groups = find_layout_propagation_groups(layers)

    print()
    print("便捷 API: find_layout_propagation_groups()")
    print("-" * 70)
    print("输入: [Conv, BN, ReLU, Pool]")
    print("输出传播组:")
    for i, group in enumerate(groups):
        layer_names = [layers[idx].name for idx in sorted(group)]
        print("  组 %d: %s" % (i+1, " -> ".join(layer_names)))

    # 方式2: 获取详细信息
    op_infos = analyze_network_sensitivity(layers)

    print()
    print("详细 API: analyze_network_sensitivity()")
    print("-" * 70)
    for op_info in op_infos:
        print("  %s: has_reduction=%s, sensitivity=%s" % (
            op_info.name, op_info.has_reduction, op_info.sensitivity.value
        ))


def main():
    """运行所有测试"""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " Layout Propagation 测试套件 ".center(68) + "║")
    print("║" + " 基于规约分析的布局敏感性判断 ".center(66) + "║")
    print("╚" + "═" * 68 + "╝")

    print("""
核心原理:
┌─────────────────────────────────────────────────────────────────────┐
│  有规约计算 (如 Σ, max, mean)  →  布局敏感  →  分区决策边界        │
│  无规约计算 (逐元素操作)        →  布局不敏感 →  可透传分区         │
└─────────────────────────────────────────────────────────────────────┘
""")

    # 运行测试
    test_reduction_analysis()
    test_vgg_style_network()
    test_resnet_style_network()
    test_transformer_style()
    test_sensitivity_api()

    print()
    print("=" * 70)
    print("所有测试完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
