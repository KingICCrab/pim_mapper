"""
对所有 nn_dataflow/nns 中的神经网络模型运行 ILP 全局分区优化，
并分析结果。

直接解析 nn_dataflow 网络定义，不依赖 nn_dataflow 的安装。
"""

from ilp_optimizer_v2 import (
    GlobalPartitionILPOptimizerV2 as GlobalPartitionOptimizer,
    LayerConfig, HybridPartitionChoice, PartDim,
    RedistributionCostModel
)
import sys
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))


# 简单的层定义类
@dataclass
class SimpleConvLayer:
    """简化的卷积层。"""
    nifm: int    # 输入通道
    nofm: int    # 输出通道
    hofm: int    # 输出高度
    wofm: int    # 输出宽度
    hfil: int    # 卷积核高度
    wfil: int    # 卷积核宽度
    strd: int = 1

    def total_ops(self, batch_size=1):
        return batch_size * self.nifm * self.nofm * self.hofm * self.wofm * self.hfil * self.wfil


@dataclass
class SimpleFCLayer:
    """简化的全连接层。"""
    nifm: int    # 输入大小
    nofm: int    # 输出大小
    hofm: int    # hofm (通常为 1 或 input size)
    wofm: int = 1
    hfil: int = 1
    wfil: int = 1

    def total_ops(self, batch_size=1):
        return batch_size * self.nifm * self.nofm * self.hofm * self.wofm


@dataclass
class SimplePoolingLayer:
    """简化的池化层。"""
    nofm: int
    hofm: int
    wofm: int
    pool_size: int
    strd: int = 1

    @property
    def nifm(self):
        return self.nofm

    @property
    def hfil(self):
        return self.pool_size

    @property
    def wfil(self):
        return self.pool_size


def parse_network_file(filepath: str) -> List[Tuple[str, object]]:
    """
    解析网络定义文件，提取层信息。
    """
    with open(filepath, 'r') as f:
        content = f.read()

    layers = []

    # 解析 ConvLayer: ConvLayer(nifm, nofm, sofm, sfil, strd=1)
    conv_pattern = r"add\(['\"](\w+)['\"],\s*ConvLayer\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)(?:,\s*(\d+))?\)"
    for match in re.finditer(conv_pattern, content):
        name = match.group(1)
        nifm = int(match.group(2))
        nofm = int(match.group(3))
        sofm = int(match.group(4))
        sfil = int(match.group(5))
        strd = int(match.group(6)) if match.group(6) else 1

        layer = SimpleConvLayer(
            nifm=nifm, nofm=nofm,
            hofm=sofm, wofm=sofm,
            hfil=sfil, wfil=sfil,
            strd=strd
        )
        layers.append((name, layer, 'Conv'))

    # 解析 FCLayer: FCLayer(nifm, nofm, sofm=1)
    fc_pattern = r"add\(['\"](\w+)['\"],\s*FCLayer\((\d+),\s*(\d+)(?:,\s*(\d+))?\)"
    for match in re.finditer(fc_pattern, content):
        name = match.group(1)
        nifm = int(match.group(2))
        nofm = int(match.group(3))
        sofm = int(match.group(4)) if match.group(4) else 1

        layer = SimpleFCLayer(
            nifm=nifm, nofm=nofm,
            hofm=sofm, wofm=sofm
        )
        layers.append((name, layer, 'FC'))

    # 解析 PoolingLayer: PoolingLayer(nofm, sofm, pool, strd=pool)
    pool_pattern = r"add\(['\"](\w+)['\"],\s*PoolingLayer\((\d+),\s*(\d+),\s*(\d+)(?:,\s*(?:strd=)?(\d+))?\)"
    for match in re.finditer(pool_pattern, content):
        name = match.group(1)
        nofm = int(match.group(2))
        sofm = int(match.group(3))
        pool = int(match.group(4))
        strd = int(match.group(5)) if match.group(5) else pool

        layer = SimplePoolingLayer(
            nofm=nofm, hofm=sofm, wofm=sofm,
            pool_size=pool, strd=strd
        )
        layers.append((name, layer, 'Pool'))

    return layers


def get_all_networks() -> List[str]:
    """获取所有网络名。"""
    nns_dir = os.path.join(os.path.dirname(__file__),
                           '..', 'nn_dataflow', 'nns')
    networks = []
    for f in os.listdir(nns_dir):
        if f.endswith('.py') and not f.startswith('__'):
            networks.append(f[:-3])
    return sorted(networks)


def analyze_network(network_name: str, batch_size: int = 1,
                    node_h: int = 4, node_w: int = 4,
                    verbose: bool = True):
    """
    分析一个神经网络的分区优化结果。
    """
    print(f"\n{'='*70}")
    print(f"网络: {network_name}")
    print(f"{'='*70}")

    # 解析网络文件
    nns_dir = os.path.join(os.path.dirname(__file__),
                           '..', 'nn_dataflow', 'nns')
    filepath = os.path.join(nns_dir, network_name + '.py')

    if not os.path.exists(filepath):
        print(f"  [ERROR] 网络文件不存在: {filepath}")
        return None

    try:
        layers = parse_network_file(filepath)
    except Exception as e:
        print(f"  [ERROR] 解析网络失败: {e}")
        return None

    if not layers:
        print(f"  [WARNING] 未解析到任何层")
        return None

    # 统计网络信息
    total_macs = 0
    total_params = 0

    print(f"\n网络统计:")
    print(f"  解析到的层数: {len(layers)}")
    print(f"  Batch Size: {batch_size}")
    print(f"  节点阵列: {node_h} x {node_w} = {node_h * node_w}")

    # 打印层详情
    if verbose:
        print(f"\n层详情:")
        print(f"  {'层名':<20} {'类型':<8} {'维度':<40} {'MACs':<15}")
        print(f"  {'-'*85}")

    for name, layer, layer_type in layers:
        macs = layer.total_ops(batch_size) if hasattr(
            layer, 'total_ops') else 0
        total_macs += macs

        if verbose:
            if hasattr(layer, 'hfil'):
                dims = f"C={layer.nifm}, K={layer.nofm}, H={layer.hofm}, R={layer.hfil}"
            else:
                dims = f"K={layer.nofm}, H={layer.hofm}"
            print(f"  {name:<20} {layer_type:<8} {dims:<40} {macs:>15,}")

    print(f"\n  总 MACs: {total_macs:,}")

    # 过滤出可分区的层（卷积层和全连接层）
    partitionable_layers = [(n, l, t)
                            for n, l, t in layers if t in ('Conv', 'FC')]

    print(f"  可分区层数: {len(partitionable_layers)}")

    if len(partitionable_layers) == 0:
        print("  没有可分区的层，跳过优化")
        return {
            'network': network_name,
            'num_layers': len(layers),
            'total_macs': total_macs,
            'partitionable_layers': 0,
            'status': 'no_partitionable_layers'
        }

    # 创建 LayerConfig 列表
    layer_configs = []
    for i, (name, layer, _) in enumerate(partitionable_layers):
        config = LayerConfig(name, layer, i, batch_size)
        layer_configs.append(config)

    # 运行 ILP 优化
    print(f"\n开始 ILP 全局分区优化...")

    try:
        # 创建简单的网络和资源对象
        class SimpleNetwork(dict):
            def __iter__(self):
                return iter(self.keys())

        class SimpleResource:
            def __init__(self, h, w):
                self.dim_nodes = type('DimNodes', (), {'h': h, 'w': w})()

        # 构建网络对象
        network = SimpleNetwork()
        for name, layer, _ in partitionable_layers:
            network[name] = layer

        resource = SimpleResource(node_h, node_w)

        optimizer = GlobalPartitionOptimizer(
            network=network,
            resource=resource,
            batch_size=batch_size,
            solver='dp'  # Use DP for faster solution on linear chains
        )

        result = optimizer.optimize(verbose=False)

        if result is None:
            print("  [ERROR] 优化失败")
            return {
                'network': network_name,
                'num_layers': len(layers),
                'total_macs': total_macs,
                'status': 'optimization_failed'
            }

        # result 是 List[(layer_name, choice)]
        solution = result
        best_choices = [choice for _, choice in solution]

        # 获取 layer_configs
        layer_configs = optimizer.layer_configs

        # --- Global Solution Costs ---
        global_compute = sum(optimizer.compute_costs[(i, optimizer.partition_choices[i].index(choice))]
                             for i, (_, choice) in enumerate(solution))

        global_redist = 0
        for i in range(len(solution) - 1):
            ci = optimizer.partition_choices[i].index(solution[i][1])
            cj = optimizer.partition_choices[i+1].index(solution[i+1][1])
            global_redist += optimizer.redistribution_costs.get((i, ci, cj), 0)

        global_total = global_compute + global_redist

        # --- Greedy Solution Costs ---
        # For each layer, pick the choice with minimum compute cost
        greedy_choices_indices = []
        greedy_compute = 0

        for i in range(len(layer_configs)):
            # Find min compute cost for layer i
            min_cost = float('inf')
            best_idx = -1
            for c_idx in range(len(optimizer.partition_choices[i])):
                cost = optimizer.compute_costs.get((i, c_idx), float('inf'))
                if cost < min_cost:
                    min_cost = cost
                    best_idx = c_idx

            greedy_choices_indices.append(best_idx)
            greedy_compute += min_cost

        # Calculate redistribution cost for greedy sequence
        greedy_redist = 0
        for i in range(len(layer_configs) - 1):
            ci = greedy_choices_indices[i]
            cj = greedy_choices_indices[i+1]
            greedy_redist += optimizer.redistribution_costs.get((i, ci, cj), 0)

        greedy_total = greedy_compute + greedy_redist

        print(f"\n优化结果对比:")
        print(
            f"  Global Total: {global_total:,.0f} (Compute: {global_compute:,.0f}, Redist: {global_redist:,.0f})")
        print(
            f"  Greedy Total: {greedy_total:,.0f} (Compute: {greedy_compute:,.0f}, Redist: {greedy_redist:,.0f})")
        print(
            f"  Improvement: {(greedy_total - global_total) / greedy_total * 100:.2f}%")

        # 分析分区选择
        partition_summary = {
            'OUTP': 0,  # 纯 K 分区
            'OFMP': 0,  # 纯空间分区
            'BATP': 0,  # 纯 Batch 分区
            'INPP': 0,  # 纯 C 分区
            'HYBRID': 0,  # 混合分区
            'NONE': 0   # 无分区
        }

        print(f"\n各层分区选择:")
        for i, (layer_name, choice) in enumerate(solution):
            # 统计哪些维度被使用
            used_dims = []
            for dim in range(PartDim.NUM):
                if choice.get_size(dim) > 1:
                    used_dims.append(['OUTP', 'OFMP', 'BATP', 'INPP'][dim])

            if len(used_dims) == 0:
                partition_type = 'NONE'
            elif len(used_dims) == 1:
                partition_type = used_dims[0]
            else:
                partition_type = 'HYBRID'

            partition_summary[partition_type] += 1

            if verbose:
                print(f"  {layer_name:<20} {choice!r:<50} -> {partition_type}")

        print(f"\n分区策略统计:")
        for key, count in partition_summary.items():
            if count > 0:
                pct = count / len(solution) * 100
                print(f"  {key}: {count} 层 ({pct:.1f}%)")

        return {
            'network': network_name,
            'num_layers': len(layers),
            'partitionable_layers': len(partitionable_layers),
            'total_macs': total_macs,
            'global_total': global_total,
            'global_compute': global_compute,
            'global_redist': global_redist,
            'greedy_total': greedy_total,
            'greedy_compute': greedy_compute,
            'greedy_redist': greedy_redist,
            'partition_summary': partition_summary,
            'solver': optimizer.solver,
            'status': 'success'
        }

    except Exception as e:
        import traceback
        print(f"  [ERROR] 优化过程出错: {e}")
        traceback.print_exc()
        return {
            'network': network_name,
            'status': 'error',
            'error': str(e)
        }


def main():
    """运行所有网络的分析。"""
    print("=" * 70)
    print("nn_dataflow 神经网络全局分区优化分析")
    print("=" * 70)

    # 配置
    batch_size = 1
    node_h = 4
    node_w = 4

    # 获取所有网络
    networks = get_all_networks()
    print(f"\n发现 {len(networks)} 个网络: {networks}")

    # 过滤掉 LSTM 网络（结构较复杂，需要特殊处理）
    cnn_networks = [n for n in networks if not n.startswith('lstm')]

    print(f"CNN/MLP 网络: {cnn_networks}")

    # 运行分析
    results = []

    # 测试几个主要网络
    test_networks = ['alex_net', 'vgg_net',
                     'googlenet', 'resnet50', 'zfnet', 'vgg19_net']

    for network_name in test_networks:
        if network_name in networks:
            result = analyze_network(
                network_name,
                batch_size=batch_size,
                node_h=node_h,
                node_w=node_w,
                verbose=True
            )
            if result:
                results.append(result)

    # 汇总报告
    print("\n" + "=" * 70)
    print("汇总报告")
    print("=" * 70)
    print(f"\n{'网络':<15} {'Global Cost':<15} {'Greedy Cost':<15} {'Improvement':<12}")
    print("-" * 85)

    plot_data = {'Network': [], 'Global': [], 'Greedy': []}

    for r in results:
        if r['status'] == 'success':
            imp = (r['greedy_total'] - r['global_total']) / \
                r['greedy_total'] * 100
            print(
                f"{r['network']:<15} {r['global_total']:<15,.0f} {r['greedy_total']:<15,.0f} {imp:<12.2f}%")

            plot_data['Network'].append(r['network'])
            plot_data['Global'].append(r['global_total'])
            plot_data['Greedy'].append(r['greedy_total'])
        else:
            print(f"{r['network']:<15} {'-':<15} {'-':<15} {'-':<12} {r['status']}")

    # --- Plotting ---
    if plot_data['Network']:
        networks = plot_data['Network']
        x = np.arange(len(networks))
        width = 0.35

        fig_width = max(10, len(networks) * 1.0)
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        # Greedy: White with diagonal hatch
        rects1 = ax.bar(x - width/2, plot_data['Greedy'], width,
                        label='Greedy (Layer-wise)', color='white', edgecolor='black', hatch='///')

        # Global: White with dot hatch
        rects2 = ax.bar(x + width/2, plot_data['Global'], width,
                        label='Global (ILP)', color='white', edgecolor='black', hatch='...')

        ax.set_ylabel('Total Energy Cost (Normalized)')
        ax.set_title('Global Partitioning vs Greedy Strategy')
        ax.set_xticks(x)
        ax.set_xticklabels(networks, rotation=45, ha='right')
        ax.legend()

        # Add labels (Improvement %)
        for i, (g, l) in enumerate(zip(plot_data['Global'], plot_data['Greedy'])):
            if l > 0:
                imp = (l - g) / l * 100
                if imp > 1.0:  # Only show if significant
                    ax.annotate(f'-{imp:.0f}%',
                                xy=(x[i] + width/2, g),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)

        fig.tight_layout()
        output_path = os.path.join(os.path.dirname(
            __file__), 'global_partition_results.png')
        plt.savefig(output_path)
        print(f"\nChart saved to: {output_path}")

    # 分区策略分析
    print("\n分区策略分布:")
    all_partitions = {'OUTP': 0, 'OFMP': 0,
                      'BATP': 0, 'INPP': 0, 'HYBRID': 0, 'NONE': 0}
    total_layers = 0

    for r in results:
        if r['status'] == 'success' and 'partition_summary' in r:
            for key, count in r['partition_summary'].items():
                all_partitions[key] += count
            total_layers += r['partitionable_layers']

    if total_layers > 0:
        for key, count in all_partitions.items():
            if count > 0:
                pct = count / total_layers * 100
                print(f"  {key}: {count} 层 ({pct:.1f}%)")

    print("\n分析完成!")
    return results


if __name__ == '__main__':
    main()
