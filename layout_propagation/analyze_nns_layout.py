#!/usr/bin/env python
"""
分析 nn_dataflow/nns 目录下所有网络的 Layout Propagation

基于规约分析：有规约 → 布局敏感，无规约 → 可透传
"""

from layout_propagation import LayoutSensitivity, LayoutPropagator, OperatorInfo
import os
import re
import sys
from collections import OrderedDict
from typing import Dict, List, Tuple, Set

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))


# ============================================================================
# 层类型的规约特性
# ============================================================================

LAYER_REDUCTION_INFO = {
    # 有规约的层
    'ConvLayer': {'has_reduction': True, 'reduction_dims': ['C', 'R', 'S']},
    'FCLayer': {'has_reduction': True, 'reduction_dims': ['C']},
    'PoolingLayer': {'has_reduction': True, 'reduction_dims': ['H', 'W']},
    'LocalRegionLayer': {'has_reduction': True, 'reduction_dims': ['local']},

    # 无规约的层 (Elementwise)
    'EltwiseLayer': {'has_reduction': False, 'reduction_dims': []},
    'InputLayer': {'has_reduction': False, 'reduction_dims': []},
}


def parse_layer_call(line: str) -> Tuple[str, str, Dict]:
    """
    解析层定义调用，返回 (层名, 层类型, 参数)

    示例:
        NN.add('conv1', ConvLayer(3, 64, 224, 3))
        -> ('conv1', 'ConvLayer', {'nifm': 3, 'nofm': 64, 'hofm': 224, 'sfil': 3})
    """
    # 匹配 NN.add('name', LayerType(...)) - 支持多行和复杂格式
    # 先尝试简单匹配
    pattern = r"NN\.add\(['\"]([^'\"]+)['\"],\s*(\w+)\(([^)]*)\)"
    match = re.search(pattern, line)

    # 如果没匹配到，尝试更宽松的模式（处理换行情况）
    if not match:
        pattern2 = r"NN\.add\(['\"]([^'\"]+)['\"]"
        match2 = re.search(pattern2, line)
        if match2:
            # 查找层类型
            type_match = re.search(
                r"(ConvLayer|FCLayer|PoolingLayer|EltwiseLayer|LocalRegionLayer)\(([^)]*)\)", line)
            if type_match:
                return match2.group(1), type_match.group(1), {}

    # 处理 format 字符串的情况，如 'conv2_{}_a'.format(i)
    if not match:
        pattern3 = r"NN\.add\(['\"]([^'\"{}]+)"
        match3 = re.search(pattern3, line)
        if match3:
            type_match = re.search(
                r"(ConvLayer|FCLayer|PoolingLayer|EltwiseLayer|LocalRegionLayer)\(([^)]*)\)", line)
            if type_match:
                # 简化名称
                name = match3.group(1).rstrip("_")
                return name, type_match.group(1), {}

    if not match:
        return None, None, None

    layer_name = match.group(1)
    layer_type = match.group(2)
    args_str = match.group(3)

    # 解析参数
    params = {}
    if args_str.strip():
        # 简单解析数字参数
        args = [a.strip() for a in args_str.split(',')]
        nums = []
        for arg in args:
            # 跳过关键字参数
            if '=' in arg:
                continue
            try:
                nums.append(int(arg))
            except ValueError:
                pass

        # 根据层类型分配参数
        if layer_type == 'ConvLayer':
            if len(nums) >= 4:
                params = {'nifm': nums[0], 'nofm': nums[1],
                          'hofm': nums[2], 'sfil': nums[3]}
            elif len(nums) >= 3:
                params = {'nifm': nums[0], 'nofm': nums[1], 'hofm': nums[2]}
        elif layer_type == 'FCLayer':
            if len(nums) >= 2:
                params = {'nifm': nums[0], 'nofm': nums[1]}
                if len(nums) >= 3:
                    params['hofm'] = nums[2]
                else:
                    params['hofm'] = 1
        elif layer_type == 'PoolingLayer':
            if len(nums) >= 2:
                params = {'nofm': nums[0], 'hofm': nums[1]}
        elif layer_type == 'EltwiseLayer':
            if len(nums) >= 2:
                params = {'nofm': nums[0], 'hofm': nums[1]}

    return layer_name, layer_type, params


def parse_network_file(filepath: str) -> Tuple[str, List[Tuple[str, str, Dict]]]:
    """
    解析网络定义文件，返回 (网络名, 层列表)
    支持循环生成的层（如 ResNet）
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取网络名
    name_match = re.search(r"Network\(['\"]([^'\"]+)['\"]", content)
    net_name = name_match.group(
        1) if name_match else os.path.basename(filepath)

    # 尝试执行文件来获取实际的层（处理循环等动态生成）
    layers = []

    # 方法1: 静态解析所有 NN.add 调用
    for line in content.split('\n'):
        line = line.strip()
        if 'NN.add(' in line:
            layer_name, layer_type, params = parse_layer_call(line)
            if layer_name and layer_type:
                layers.append((layer_name, layer_type, params))

    # 方法2: 对于 ResNet 等使用循环的网络，模拟展开
    if 'resnet152' in filepath.lower():
        layers = parse_resnet152_layers(content)
    elif 'resnet' in filepath.lower():
        layers = parse_resnet_layers(content)
    elif 'googlenet' in filepath.lower():
        layers = parse_googlenet_layers(content)

    return net_name, layers


def parse_resnet_layers(content: str) -> List[Tuple[str, str, Dict]]:
    """
    手动展开 ResNet 的循环结构

    ResNet-50 结构:
    - conv1, pool1
    - conv2: 3 个 residual block (每个有 a,b,c 三层 + res)
    - conv3: 4 个 residual block
    - conv4: 6 个 residual block  
    - conv5: 3 个 residual block
    - pool5, fc
    """
    layers = [
        ('conv1', 'ConvLayer', {}),
        ('pool1', 'PoolingLayer', {}),
    ]

    # Stage 2: 3 blocks
    for i in range(3):
        layers.append(('conv2_%d_a' % i, 'ConvLayer', {}))
        layers.append(('conv2_%d_b' % i, 'ConvLayer', {}))
        layers.append(('conv2_%d_c' % i, 'ConvLayer', {}))
        if i == 0:
            layers.append(('conv2_br', 'ConvLayer', {}))
        layers.append(('conv2_%d_res' % i, 'EltwiseLayer', {}))

    # Stage 3: 4 blocks
    for i in range(4):
        layers.append(('conv3_%d_a' % i, 'ConvLayer', {}))
        layers.append(('conv3_%d_b' % i, 'ConvLayer', {}))
        layers.append(('conv3_%d_c' % i, 'ConvLayer', {}))
        if i == 0:
            layers.append(('conv3_br', 'ConvLayer', {}))
        layers.append(('conv3_%d_res' % i, 'EltwiseLayer', {}))

    # Stage 4: 6 blocks
    for i in range(6):
        layers.append(('conv4_%d_a' % i, 'ConvLayer', {}))
        layers.append(('conv4_%d_b' % i, 'ConvLayer', {}))
        layers.append(('conv4_%d_c' % i, 'ConvLayer', {}))
        if i == 0:
            layers.append(('conv4_br', 'ConvLayer', {}))
        layers.append(('conv4_%d_res' % i, 'EltwiseLayer', {}))

    # Stage 5: 3 blocks
    for i in range(3):
        layers.append(('conv5_%d_a' % i, 'ConvLayer', {}))
        layers.append(('conv5_%d_b' % i, 'ConvLayer', {}))
        layers.append(('conv5_%d_c' % i, 'ConvLayer', {}))
        if i == 0:
            layers.append(('conv5_br', 'ConvLayer', {}))
        layers.append(('conv5_%d_res' % i, 'EltwiseLayer', {}))

    layers.append(('pool5', 'PoolingLayer', {}))
    layers.append(('fc', 'FCLayer', {}))

    return layers


def parse_resnet152_layers(content: str) -> List[Tuple[str, str, Dict]]:
    """ResNet-152 的层展开"""
    layers = [
        ('conv1', 'ConvLayer', {}),
        ('pool1', 'PoolingLayer', {}),
    ]

    # Stage 2: 3 blocks, Stage 3: 8 blocks, Stage 4: 36 blocks, Stage 5: 3 blocks
    stage_blocks = [(2, 3), (3, 8), (4, 36), (5, 3)]

    for stage, num_blocks in stage_blocks:
        for i in range(num_blocks):
            layers.append(('conv%d_%d_a' % (stage, i), 'ConvLayer', {}))
            layers.append(('conv%d_%d_b' % (stage, i), 'ConvLayer', {}))
            layers.append(('conv%d_%d_c' % (stage, i), 'ConvLayer', {}))
            if i == 0:
                layers.append(('conv%d_br' % stage, 'ConvLayer', {}))
            layers.append(('conv%d_%d_res' % (stage, i), 'EltwiseLayer', {}))

    layers.append(('pool5', 'PoolingLayer', {}))
    layers.append(('fc', 'FCLayer', {}))

    return layers


def parse_googlenet_layers(content: str) -> List[Tuple[str, str, Dict]]:
    """GoogLeNet/Inception 的层展开"""
    layers = [
        ('conv1', 'ConvLayer', {}),
        ('pool1', 'PoolingLayer', {}),
        ('conv2_3x3_reduce', 'ConvLayer', {}),
        ('conv2_3x3', 'ConvLayer', {}),
        ('pool2', 'PoolingLayer', {}),
    ]

    # Inception modules (简化)
    for stage in ['3a', '3b', '4a', '4b', '4c', '4d', '4e', '5a', '5b']:
        layers.append(('inception_%s_1x1' % stage, 'ConvLayer', {}))
        layers.append(('inception_%s_3x3_reduce' % stage, 'ConvLayer', {}))
        layers.append(('inception_%s_3x3' % stage, 'ConvLayer', {}))
        layers.append(('inception_%s_5x5_reduce' % stage, 'ConvLayer', {}))
        layers.append(('inception_%s_5x5' % stage, 'ConvLayer', {}))
        layers.append(('inception_%s_pool_proj' % stage, 'ConvLayer', {}))
        # concat 是无规约的
        layers.append(('inception_%s_output' % stage, 'EltwiseLayer', {}))

    layers.append(('pool3', 'PoolingLayer', {}))
    layers.append(('pool4', 'PoolingLayer', {}))
    layers.append(('pool5', 'PoolingLayer', {}))
    layers.append(('fc', 'FCLayer', {}))

    return layers


def analyze_network(net_name: str, layers: List[Tuple[str, str, Dict]]) -> Dict:
    """分析单个网络的 Layout Propagation"""

    # 创建 OperatorInfo 列表
    op_infos = []
    for layer_name, layer_type, params in layers:
        info = LAYER_REDUCTION_INFO.get(layer_type,
                                        {'has_reduction': True, 'reduction_dims': ['unknown']})

        # 获取形状
        nofm = params.get('nofm', 0)
        hofm = params.get('hofm', 1)

        op_info = OperatorInfo(
            name=layer_name,
            op_type=layer_type,
            has_reduction=info['has_reduction'],
            reduction_dims=info['reduction_dims'],
            input_shape=(params.get('nifm', nofm), hofm, hofm),
            output_shape=(nofm, hofm, hofm)
        )
        op_infos.append(op_info)

    # 传播分析
    propagator = LayoutPropagator(op_infos)
    groups = propagator.find_propagation_groups()

    # 统计
    sensitive_ops = propagator.get_sensitive_operators()
    insensitive_ops = propagator.get_insensitive_operators()

    return {
        'name': net_name,
        'total_layers': len(layers),
        'sensitive_count': len(sensitive_ops),
        'insensitive_count': len(insensitive_ops),
        'num_groups': len(groups),
        'groups': groups,
        'layers': layers,
        'op_infos': op_infos,
        'reduction_pct': 100 * len(insensitive_ops) / len(layers) if layers else 0
    }


def print_network_analysis(result: Dict, verbose: bool = False):
    """打印网络分析结果"""
    print()
    print("=" * 70)
    print("网络: %s" % result['name'])
    print("=" * 70)

    layers = result['layers']
    op_infos = result['op_infos']

    if verbose:
        print()
        print("层级分析:")
        print("-" * 70)
        print("%4s | %-20s | %-15s | %-6s | %s" %
              ("序号", "层名", "类型", "规约?", "敏感性"))
        print("-" * 70)

        for i, (layer_name, layer_type, _) in enumerate(layers):
            op_info = op_infos[i]
            has_red = "Y" if op_info.has_reduction else "N"
            sens = "SENS" if op_info.sensitivity == LayoutSensitivity.SENSITIVE else "PASS"
            print("%4d | %-20s | %-15s | %-6s | %s" % (
                i, layer_name[:20], layer_type, has_red, sens
            ))

    # 打印传播组
    print()
    print("-" * 70)
    print("传播组 (同组共享分区):")
    print("-" * 70)

    groups = result['groups']
    for i, group in enumerate(groups):
        layer_names = [layers[idx][0] for idx in sorted(group)]
        if len(group) > 1:
            # 截断显示
            if len(layer_names) > 5:
                display = " -> ".join(layer_names[:3]) + \
                    " -> ... -> " + layer_names[-1]
            else:
                display = " -> ".join(layer_names)
            print("  组 %2d: %s (%d层)" % (i+1, display, len(group)))
        else:
            print("  组 %2d: %s (锚点)" % (i+1, layer_names[0]))

    # 统计
    print()
    print("-" * 70)
    print("统计:")
    print("-" * 70)
    print("  总层数:              %d" % result['total_layers'])
    print("  敏感算子 (有规约):   %d 个 → 分区决策点" % result['sensitive_count'])
    print("  不敏感算子 (无规约): %d 个 → 可透传" % result['insensitive_count'])
    print("  传播组数:            %d" % result['num_groups'])
    print("  优化变量减少:        %.1f%%" % result['reduction_pct'])


def main():
    """分析所有 nn_dataflow 网络"""

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " nn_dataflow 网络 Layout Propagation 分析 ".center(68) + "║")
    print("║" + " 基于规约分析判断布局敏感性 ".center(66) + "║")
    print("╚" + "═" * 68 + "╝")

    # 网络文件目录
    nns_dir = os.path.join(os.path.dirname(__file__),
                           '..', 'nn_dataflow', 'nns')
    nns_dir = os.path.abspath(nns_dir)

    print()
    print("扫描目录: %s" % nns_dir)

    # 层类型规约说明
    print()
    print("层类型规约分析规则:")
    print("-" * 70)
    print("  ConvLayer:      有规约 (在 C, R, S 维度求和)     → 敏感")
    print("  FCLayer:        有规约 (在 C 维度内积)           → 敏感")
    print("  PoolingLayer:   有规约 (在空间窗口 max/avg)     → 敏感")
    print("  EltwiseLayer:   无规约 (逐元素操作)             → 不敏感")

    # 收集所有网络文件
    network_files = []
    for fname in os.listdir(nns_dir):
        if fname.endswith('.py') and fname != '__init__.py':
            network_files.append(os.path.join(nns_dir, fname))

    network_files.sort()

    print()
    print("发现 %d 个网络文件" % len(network_files))

    # 分析每个网络
    results = []
    for filepath in network_files:
        try:
            net_name, layers = parse_network_file(filepath)
            if layers:
                result = analyze_network(net_name, layers)
                results.append(result)
        except Exception as e:
            print("  跳过 %s: %s" % (os.path.basename(filepath), str(e)))

    # 打印详细分析（选择几个典型网络）
    typical_nets = ['VGG', 'AlexNet', 'ResNet', 'GoogLeNet']
    for result in results:
        if any(t in result['name'] for t in typical_nets):
            print_network_analysis(result, verbose=True)

    # 打印其他网络的摘要
    print()
    print()
    print("=" * 70)
    print("所有网络汇总")
    print("=" * 70)
    print()
    print("%-15s | %6s | %6s | %6s | %6s | %8s" % (
        "网络", "总层数", "敏感", "透传", "组数", "优化%"
    ))
    print("-" * 70)

    total_layers = 0
    total_sensitive = 0
    total_insensitive = 0

    for result in sorted(results, key=lambda x: x['total_layers'], reverse=True):
        print("%-15s | %6d | %6d | %6d | %6d | %7.1f%%" % (
            result['name'][:15],
            result['total_layers'],
            result['sensitive_count'],
            result['insensitive_count'],
            result['num_groups'],
            result['reduction_pct']
        ))
        total_layers += result['total_layers']
        total_sensitive += result['sensitive_count']
        total_insensitive += result['insensitive_count']

    print("-" * 70)
    avg_reduction = 100 * total_insensitive / total_layers if total_layers else 0
    print("%-15s | %6d | %6d | %6d | %6s | %7.1f%%" % (
        "总计/平均", total_layers, total_sensitive, total_insensitive, "-", avg_reduction
    ))

    # 结论
    print()
    print("=" * 70)
    print("结论")
    print("=" * 70)
    print("""
Layout Propagation 优化效果:

1. 敏感算子 (有规约): Conv, FC, Pool
   - 这些是分区决策的"锚点"
   - 需要独立优化分区方案
   
2. 不敏感算子 (无规约): Eltwise (ReLU, Add, BN等)
   - 可以透传上游分区
   - 无需独立分区决策，减少优化变量

3. 平均优化效果: %.1f%% 的层可以透传
   - 这些层不需要独立的分区变量
   - 同传播组内的层间无需数据重分布

4. 对 ILP 优化器的意义:
   - 原始: 每层一组分区变量 → O(L × P) 变量
   - 优化后: 每组一组分区变量 → O(G × P) 变量
   - 其中 L = 层数, G = 传播组数, P = 分区候选数
""" % avg_reduction)


if __name__ == '__main__':
    main()
