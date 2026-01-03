"""
神经网络 Layout 分析

分析多个神经网络的:
1. Layout 敏感性分布
2. 相邻层间的 layout 协商策略
3. 融合 layout transform 到 memory hierarchy 的收益估算
"""

import sys
import os

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import math

# 导入 layout propagation 模块
from layout_propagation import (
    LayoutSensitivity,
    OperatorInfo,
    ReductionAnalyzer,
)

from fused_layout_transform import (
    MemoryLevel,
    MemoryConfig,
    FusedLayoutAddressGenerator,
    FusedLayoutMemoryCopy,
    OperatorMemoryInterface,
    OperatorLayoutNegotiator,
)


# ============================================================================
# 自定义网络定义 (避免依赖 nn_dataflow 复杂导入)
# ============================================================================

@dataclass
class LayerDef:
    """层定义"""
    name: str
    layer_type: str  # 'conv', 'fc', 'pool', 'eltwise'
    nifm: int        # 输入通道数
    nofm: int        # 输出通道数
    hofm: int        # 输出高度
    wofm: int        # 输出宽度
    hfil: int = 1    # 卷积核高度
    wfil: int = 1    # 卷积核宽度
    stride: int = 1


def create_vgg16() -> List[LayerDef]:
    """VGG-16 网络定义"""
    layers = [
        LayerDef('conv1_1', 'conv', 3, 64, 224, 224, 3, 3),
        LayerDef('conv1_2', 'conv', 64, 64, 224, 224, 3, 3),
        LayerDef('pool1', 'pool', 64, 64, 112, 112, 2, 2, 2),

        LayerDef('conv2_1', 'conv', 64, 128, 112, 112, 3, 3),
        LayerDef('conv2_2', 'conv', 128, 128, 112, 112, 3, 3),
        LayerDef('pool2', 'pool', 128, 128, 56, 56, 2, 2, 2),

        LayerDef('conv3_1', 'conv', 128, 256, 56, 56, 3, 3),
        LayerDef('conv3_2', 'conv', 256, 256, 56, 56, 3, 3),
        LayerDef('conv3_3', 'conv', 256, 256, 56, 56, 3, 3),
        LayerDef('pool3', 'pool', 256, 256, 28, 28, 2, 2, 2),

        LayerDef('conv4_1', 'conv', 256, 512, 28, 28, 3, 3),
        LayerDef('conv4_2', 'conv', 512, 512, 28, 28, 3, 3),
        LayerDef('conv4_3', 'conv', 512, 512, 28, 28, 3, 3),
        LayerDef('pool4', 'pool', 512, 512, 14, 14, 2, 2, 2),

        LayerDef('conv5_1', 'conv', 512, 512, 14, 14, 3, 3),
        LayerDef('conv5_2', 'conv', 512, 512, 14, 14, 3, 3),
        LayerDef('conv5_3', 'conv', 512, 512, 14, 14, 3, 3),
        LayerDef('pool5', 'pool', 512, 512, 7, 7, 2, 2, 2),

        LayerDef('fc6', 'fc', 512*7*7, 4096, 1, 1),
        LayerDef('fc7', 'fc', 4096, 4096, 1, 1),
        LayerDef('fc8', 'fc', 4096, 1000, 1, 1),
    ]
    return layers


def create_resnet50() -> List[LayerDef]:
    """ResNet-50 网络定义 (简化版)"""
    layers = [
        LayerDef('conv1', 'conv', 3, 64, 112, 112, 7, 7, 2),
        LayerDef('pool1', 'pool', 64, 64, 56, 56, 3, 3, 2),
    ]

    # Stage 2: 3 blocks, 56x56
    for i in range(3):
        c_in = 64 if i == 0 else 256
        layers.extend([
            LayerDef(f'conv2_{i}_a', 'conv', c_in, 64, 56, 56, 1, 1),
            LayerDef(f'conv2_{i}_b', 'conv', 64, 64, 56, 56, 3, 3),
            LayerDef(f'conv2_{i}_c', 'conv', 64, 256, 56, 56, 1, 1),
            LayerDef(f'conv2_{i}_res', 'eltwise', 256, 256, 56, 56),
        ])

    # Stage 3: 4 blocks, 28x28
    for i in range(4):
        c_in = 256 if i == 0 else 512
        h = 28
        layers.extend([
            LayerDef(f'conv3_{i}_a', 'conv', c_in, 128,
                     h, h, 1, 1, 2 if i == 0 else 1),
            LayerDef(f'conv3_{i}_b', 'conv', 128, 128, h, h, 3, 3),
            LayerDef(f'conv3_{i}_c', 'conv', 128, 512, h, h, 1, 1),
            LayerDef(f'conv3_{i}_res', 'eltwise', 512, 512, h, h),
        ])

    # Stage 4: 6 blocks, 14x14
    for i in range(6):
        c_in = 512 if i == 0 else 1024
        h = 14
        layers.extend([
            LayerDef(f'conv4_{i}_a', 'conv', c_in, 256,
                     h, h, 1, 1, 2 if i == 0 else 1),
            LayerDef(f'conv4_{i}_b', 'conv', 256, 256, h, h, 3, 3),
            LayerDef(f'conv4_{i}_c', 'conv', 256, 1024, h, h, 1, 1),
            LayerDef(f'conv4_{i}_res', 'eltwise', 1024, 1024, h, h),
        ])

    # Stage 5: 3 blocks, 7x7
    for i in range(3):
        c_in = 1024 if i == 0 else 2048
        h = 7
        layers.extend([
            LayerDef(f'conv5_{i}_a', 'conv', c_in, 512,
                     h, h, 1, 1, 2 if i == 0 else 1),
            LayerDef(f'conv5_{i}_b', 'conv', 512, 512, h, h, 3, 3),
            LayerDef(f'conv5_{i}_c', 'conv', 512, 2048, h, h, 1, 1),
            LayerDef(f'conv5_{i}_res', 'eltwise', 2048, 2048, h, h),
        ])

    layers.extend([
        LayerDef('pool5', 'pool', 2048, 2048, 1, 1, 7, 7),
        LayerDef('fc', 'fc', 2048, 1000, 1, 1),
    ])

    return layers


def create_alexnet() -> List[LayerDef]:
    """AlexNet 网络定义"""
    layers = [
        LayerDef('conv1', 'conv', 3, 96, 55, 55, 11, 11, 4),
        LayerDef('pool1', 'pool', 96, 96, 27, 27, 3, 3, 2),

        LayerDef('conv2', 'conv', 96, 256, 27, 27, 5, 5),
        LayerDef('pool2', 'pool', 256, 256, 13, 13, 3, 3, 2),

        LayerDef('conv3', 'conv', 256, 384, 13, 13, 3, 3),
        LayerDef('conv4', 'conv', 384, 384, 13, 13, 3, 3),
        LayerDef('conv5', 'conv', 384, 256, 13, 13, 3, 3),
        LayerDef('pool5', 'pool', 256, 256, 6, 6, 3, 3, 2),

        LayerDef('fc6', 'fc', 256*6*6, 4096, 1, 1),
        LayerDef('fc7', 'fc', 4096, 4096, 1, 1),
        LayerDef('fc8', 'fc', 4096, 1000, 1, 1),
    ]
    return layers


def create_mobilenetv2() -> List[LayerDef]:
    """MobileNetV2 网络定义 (简化版)"""
    layers = [
        LayerDef('conv1', 'conv', 3, 32, 112, 112, 3, 3, 2),
    ]

    # Inverted residual blocks (简化)
    configs = [
        # (expansion, out_channels, num_blocks, stride, h)
        (1, 16, 1, 1, 112),
        (6, 24, 2, 2, 56),
        (6, 32, 3, 2, 28),
        (6, 64, 4, 2, 14),
        (6, 96, 3, 1, 14),
        (6, 160, 3, 2, 7),
        (6, 320, 1, 1, 7),
    ]

    in_ch = 32
    for idx, (exp, out_ch, num_blocks, stride, h) in enumerate(configs):
        for b in range(num_blocks):
            mid_ch = in_ch * exp
            s = stride if b == 0 else 1
            layers.extend([
                LayerDef(f'block{idx}_{b}_expand', 'conv',
                         in_ch, mid_ch, h, h, 1, 1),
                LayerDef(f'block{idx}_{b}_dw', 'conv', mid_ch,
                         mid_ch, h, h, 3, 3, s),  # depthwise
                LayerDef(f'block{idx}_{b}_proj', 'conv',
                         mid_ch, out_ch, h, h, 1, 1),
            ])
            in_ch = out_ch

    layers.extend([
        LayerDef('conv_last', 'conv', 320, 1280, 7, 7, 1, 1),
        LayerDef('pool', 'pool', 1280, 1280, 1, 1, 7, 7),
        LayerDef('fc', 'fc', 1280, 1000, 1, 1),
    ])

    return layers


# 导入 layout propagation 模块


# ============================================================================
# 网络分析器
# ============================================================================

@dataclass
class LayerAnalysis:
    """单层分析结果"""
    name: str
    layer_type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    weight_shape: Optional[Tuple[int, ...]]
    layout_sensitivity: LayoutSensitivity
    data_size_kb: float
    weight_size_kb: float
    mac_ops: int


@dataclass
class TransitionAnalysis:
    """层间过渡分析"""
    producer: str
    consumer: str
    tensor_shape: Tuple[int, ...]
    producer_layout: Tuple[int, ...]  # 假设的 layout
    consumer_layout: Tuple[int, ...]  # 假设的 layout
    needs_transform: bool
    separate_transform_energy_mj: float
    fused_transform_energy_mj: float
    energy_saving_percent: float


class NetworkLayoutAnalyzer:
    """神经网络 Layout 分析器"""

    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size
        self.reduction_analyzer = ReductionAnalyzer()

        # Memory hierarchy 配置
        self.mem_configs = {
            MemoryLevel.DRAM: MemoryConfig(
                level=MemoryLevel.DRAM,
                capacity=4 * 1024 * 1024 * 1024,
                bandwidth=32.0,
                latency=100,
                energy_per_access=20.0  # pJ/byte
            ),
            MemoryLevel.SRAM: MemoryConfig(
                level=MemoryLevel.SRAM,
                capacity=256 * 1024,
                bandwidth=128.0,
                latency=1,
                energy_per_access=1.0
            ),
        }

        self.fused_copy = FusedLayoutMemoryCopy(self.mem_configs)

    def analyze_layer_def(self, layer: LayerDef) -> LayerAnalysis:
        """分析单个层 (使用 LayerDef)"""

        # 计算输入形状
        if layer.layer_type == 'fc':
            input_shape = (self.batch_size, layer.nifm)
            output_shape = (self.batch_size, layer.nofm)
        else:
            # 根据 stride 计算输入大小
            hifm = layer.hofm * layer.stride if layer.stride > 1 else layer.hofm
            wifm = layer.wofm * layer.stride if layer.stride > 1 else layer.wofm
            input_shape = (self.batch_size, layer.nifm, hifm, wifm)
            output_shape = (self.batch_size, layer.nofm,
                            layer.hofm, layer.wofm)

        # 权重形状
        weight_shape = None
        if layer.layer_type == 'conv':
            weight_shape = (layer.nofm, layer.nifm, layer.hfil, layer.wfil)
        elif layer.layer_type == 'fc':
            weight_shape = (layer.nofm, layer.nifm)

        # Layout 敏感性
        op_info = self._layerdef_to_op_info(layer)
        sensitivity = op_info.sensitivity

        # 数据量计算
        data_size_kb = math.prod(output_shape) * 4 / 1024
        weight_size_kb = 0
        if weight_shape:
            weight_size_kb = math.prod(weight_shape) * 4 / 1024

        # MAC 操作数
        mac_ops = self._compute_mac_ops_def(layer, output_shape, weight_shape)

        return LayerAnalysis(
            name=layer.name,
            layer_type=layer.layer_type,
            input_shape=input_shape,
            output_shape=output_shape,
            weight_shape=weight_shape,
            layout_sensitivity=sensitivity,
            data_size_kb=data_size_kb,
            weight_size_kb=weight_size_kb,
            mac_ops=mac_ops
        )

    def _layerdef_to_op_info(self, layer: LayerDef) -> OperatorInfo:
        """将 LayerDef 转换为 OperatorInfo"""

        # 根据 layout_propagation.py 中的 OperatorInfo 定义
        if layer.layer_type == 'conv':
            return OperatorInfo(
                name=layer.name,
                op_type='conv',
                has_reduction=True,
                reduction_dims=['C', 'R', 'S'],
                input_shape=(1, layer.nifm, layer.hofm, layer.wofm),
                output_shape=(1, layer.nofm, layer.hofm, layer.wofm)
            )
        elif layer.layer_type == 'fc':
            return OperatorInfo(
                name=layer.name,
                op_type='fc',
                has_reduction=True,
                reduction_dims=['C'],
                input_shape=(1, layer.nifm),
                output_shape=(1, layer.nofm)
            )
        elif layer.layer_type == 'pool':
            return OperatorInfo(
                name=layer.name,
                op_type='pool',
                has_reduction=True,
                reduction_dims=['H', 'W'],
                input_shape=(1, layer.nifm, layer.hofm *
                             layer.stride, layer.wofm*layer.stride),
                output_shape=(1, layer.nofm, layer.hofm, layer.wofm)
            )
        elif layer.layer_type == 'eltwise':
            return OperatorInfo(
                name=layer.name,
                op_type='eltwise',
                has_reduction=False,
                reduction_dims=[],
                input_shape=(1, layer.nifm, layer.hofm, layer.wofm),
                output_shape=(1, layer.nofm, layer.hofm, layer.wofm)
            )
        else:
            return OperatorInfo(
                name=layer.name,
                op_type='unknown',
                has_reduction=False,
                reduction_dims=[],
                input_shape=(1, layer.nifm, layer.hofm, layer.wofm),
                output_shape=(1, layer.nofm, layer.hofm, layer.wofm)
            )

    def _compute_mac_ops_def(self, layer: LayerDef, output_shape, weight_shape):
        """计算 MAC 操作数"""
        if layer.layer_type == 'conv' and weight_shape:
            N = output_shape[0]
            K, P, Q = layer.nofm, layer.hofm, layer.wofm
            C, R, S = layer.nifm, layer.hfil, layer.wfil
            return N * K * P * Q * C * R * S
        elif layer.layer_type == 'fc' and weight_shape:
            N = output_shape[0]
            K, C = weight_shape
            return N * K * C
        elif layer.layer_type == 'pool':
            N = output_shape[0]
            C, P, Q = layer.nofm, layer.hofm, layer.wofm
            return N * C * P * Q * layer.hfil * layer.wfil
        elif layer.layer_type == 'eltwise':
            return math.prod(output_shape)
        return 0

    def analyze_transition(
        self,
        producer: LayerAnalysis,
        consumer: LayerAnalysis,
        producer_prefers_nhwc: bool = False,
        consumer_prefers_nhwc: bool = True
    ) -> TransitionAnalysis:
        """分析层间 layout 过渡"""

        # 确定 tensor 形状 (producer 的输出)
        tensor_shape = producer.output_shape

        # 假设的 layout 偏好
        nchw = (0, 1, 2, 3)
        nhwc = (0, 2, 3, 1)

        producer_layout = nhwc if producer_prefers_nhwc else nchw
        consumer_layout = nhwc if consumer_prefers_nhwc else nchw

        needs_transform = (producer_layout != consumer_layout)

        # 计算变换代价 (只对 4D tensor)
        if needs_transform and len(tensor_shape) == 4:
            comparison = self.fused_copy.compare_with_separate_transform(
                tensor_shape, producer_layout, consumer_layout
            )
            separate_energy = comparison['separate_transform']['energy_pJ'] / 1e6
            fused_energy = comparison['fused_transform']['energy_pJ'] / 1e6
            saving = (1 - fused_energy / separate_energy) * \
                100 if separate_energy > 0 else 0
        else:
            separate_energy = 0
            fused_energy = 0
            saving = 0
            needs_transform = False

        return TransitionAnalysis(
            producer=producer.name,
            consumer=consumer.name,
            tensor_shape=tensor_shape,
            producer_layout=producer_layout,
            consumer_layout=consumer_layout,
            needs_transform=needs_transform,
            separate_transform_energy_mj=separate_energy,
            fused_transform_energy_mj=fused_energy,
            energy_saving_percent=saving
        )

    def analyze_network_def(self, layers: List[LayerDef], network_name: str) -> Dict[str, Any]:
        """分析整个网络 (使用 LayerDef 列表)"""

        layer_analyses = []
        transitions = []

        for layer in layers:
            analysis = self.analyze_layer_def(layer)
            layer_analyses.append(analysis)

        # 分析层间过渡 (模拟不同层的 layout 偏好)
        for i in range(len(layer_analyses) - 1):
            producer = layer_analyses[i]
            consumer = layer_analyses[i + 1]

            # 模拟场景: 部分 Conv 层可能使用不同的 layout
            # 例如: 某些层使用 NCHW (GPU优化), 某些使用 NHWC (CPU/TPU优化)
            producer_nhwc = (producer.layer_type == 'conv' and i % 5 == 0)
            consumer_nhwc = (consumer.layer_type == 'conv' and (i+1) % 5 != 0)

            transition = self.analyze_transition(
                producer, consumer, producer_nhwc, consumer_nhwc
            )
            transitions.append(transition)

        # 统计
        sensitivity_counts = defaultdict(int)
        for la in layer_analyses:
            sensitivity_counts[la.layout_sensitivity.name] += 1

        total_activation_kb = sum(la.data_size_kb for la in layer_analyses)
        total_weight_kb = sum(la.weight_size_kb for la in layer_analyses)
        total_mac_ops = sum(la.mac_ops for la in layer_analyses)

        # 变换统计
        transforms_needed = sum(1 for t in transitions if t.needs_transform)
        total_separate_energy = sum(
            t.separate_transform_energy_mj for t in transitions)
        total_fused_energy = sum(
            t.fused_transform_energy_mj for t in transitions)

        return {
            'network_name': network_name,
            'num_layers': len(layer_analyses),
            'layer_analyses': layer_analyses,
            'transitions': transitions,
            'sensitivity_distribution': dict(sensitivity_counts),
            'total_activation_kb': total_activation_kb,
            'total_weight_mb': total_weight_kb / 1024,
            'total_gflops': total_mac_ops / 1e9,
            'transforms_needed': transforms_needed,
            'total_separate_energy_mj': total_separate_energy,
            'total_fused_energy_mj': total_fused_energy,
            'energy_saving_mj': total_separate_energy - total_fused_energy,
        }


def print_network_analysis(result: Dict[str, Any]):
    """打印网络分析结果"""

    print("\n" + "=" * 80)
    print(f"网络: {result['network_name']}")
    print("=" * 80)

    print(f"\n【基本信息】")
    print(f"  层数: {result['num_layers']}")
    print(f"  总激活数据: {result['total_activation_kb']:.1f} KB")
    print(f"  总权重: {result['total_weight_mb']:.1f} MB")
    print(f"  总计算量: {result['total_gflops']:.2f} GFLOPs")

    print(f"\n【Layout 敏感性分布】")
    for sens, count in result['sensitivity_distribution'].items():
        pct = count / result['num_layers'] * 100
        print(f"  {sens}: {count} 层 ({pct:.1f}%)")

    print(f"\n【层间 Layout 变换分析】")
    print(
        f"  需要变换的过渡: {result['transforms_needed']} / {len(result['transitions'])}")

    if result['transforms_needed'] > 0:
        print(
            f"\n  单独 Transform 总能耗: {result['total_separate_energy_mj']:.2f} mJ")
        print(f"  融合 Transform 总能耗: {result['total_fused_energy_mj']:.2f} mJ")
        print(f"  节省能耗: {result['energy_saving_mj']:.2f} mJ " +
              f"({result['energy_saving_mj']/result['total_separate_energy_mj']*100:.1f}%)")

    # 显示部分层详情
    print(f"\n【层详情 (前 10 层)】")
    print("-" * 80)
    print(f"{'层名':<20} {'类型':<15} {'输出形状':<20} {'敏感性':<15} {'数据KB':<10}")
    print("-" * 80)

    for la in result['layer_analyses'][:10]:
        shape_str = str(la.output_shape)
        print(f"{la.name:<20} {la.layer_type:<15} {shape_str:<20} {la.layout_sensitivity.name:<15} {la.data_size_kb:<10.1f}")

    if len(result['layer_analyses']) > 10:
        print(f"... 还有 {len(result['layer_analyses']) - 10} 层 ...")

    # 显示需要变换的过渡
    transform_transitions = [
        t for t in result['transitions'] if t.needs_transform]
    if transform_transitions:
        print(f"\n【需要 Layout 变换的过渡】")
        print("-" * 80)
        print(f"{'Producer':<15} → {'Consumer':<15} {'形状':<20} {'节省':<10}")
        print("-" * 80)

        for t in transform_transitions[:5]:
            shape_str = str(t.tensor_shape)
            print(
                f"{t.producer:<15} → {t.consumer:<15} {shape_str:<20} {t.energy_saving_percent:.1f}%")

        if len(transform_transitions) > 5:
            print(f"... 还有 {len(transform_transitions) - 5} 个过渡 ...")


def analyze_all_networks():
    """分析所有网络"""

    print("=" * 80)
    print("神经网络 Layout 分析报告")
    print("分析 Layout Propagation 和 Fused Layout Transform 的效果")
    print("=" * 80)

    analyzer = NetworkLayoutAnalyzer(batch_size=1)

    # 使用自定义网络定义
    networks = [
        (create_vgg16(), "VGG-16"),
        (create_resnet50(), "ResNet-50"),
        (create_alexnet(), "AlexNet"),
        (create_mobilenetv2(), "MobileNetV2"),
    ]

    all_results = []

    for layers, name in networks:
        try:
            result = analyzer.analyze_network_def(layers, name)
            all_results.append(result)
            print_network_analysis(result)
        except Exception as e:
            print(f"\n分析 {name} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 总结比较
    print("\n" + "=" * 80)
    print("【网络比较总结】")
    print("=" * 80)
    print(
        f"\n{'网络':<15} {'层数':<8} {'权重MB':<10} {'GFLOPs':<10} {'变换数':<10} {'节省能耗mJ':<12}")
    print("-" * 80)

    for r in all_results:
        saving = r['energy_saving_mj'] if r['transforms_needed'] > 0 else 0
        print(f"{r['network_name']:<15} {r['num_layers']:<8} {r['total_weight_mb']:<10.1f} "
              f"{r['total_gflops']:<10.2f} {r['transforms_needed']:<10} {saving:<12.2f}")

    # 融合效果统计
    print("\n" + "-" * 80)
    print("【融合 Layout Transform 到 Memory Hierarchy 的效果】")
    print("-" * 80)
    print("""
策略说明:
  - 避免单独的 Layout Transform 算子 (DRAM → DRAM)
  - 将变换融合到必需的 Memory Copy (DRAM → SRAM)
  - 典型节省: 约 50% cycles, 约 47.5% energy
  
适用场景:
  - Producer 输出 NCHW, Consumer 期望 NHWC (或反之)
  - 需要数据搬移到 SRAM 进行计算
  - 硬件支持灵活的地址生成

实现方式:
  1. PRODUCER_WRITE_TRANSFORM: 写入时按目标 layout 生成地址
  2. CONSUMER_READ_TRANSFORM: 读取时按源 layout 生成地址
  3. FUSED_COPY_TRANSFORM: 在 Memory Copy DMA 中完成变换
""")


if __name__ == '__main__':
    analyze_all_networks()
