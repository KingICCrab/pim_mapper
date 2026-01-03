"""
Extended DRAM Layout Analysis for Neural Network Accelerators

Comprehensive analysis of layout strategies across different neural networks
and accelerator configurations.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum
import json


# Import base classes from dram_layout_analysis
from dram_layout_analysis import (
    Layout, Dataflow, DRAMConfig, ConvLayer, 
    DRAMAccessAnalyzer, LayoutStrategyAnalyzer
)


# HBM Configuration (closer to accelerator reality)
HBM2_CONFIG = DRAMConfig(
    bandwidth_gbps=256.0,  # GB/s total (8 channels)
    row_buffer_size=2048,  # bytes (2KB per pseudo-channel)
    burst_length=4,
    data_width=256,        # bits
    tRCD=14.0,
    tRP=14.0,
    tCAS=14.0,
)

DDR4_CONFIG = DRAMConfig(
    bandwidth_gbps=19.2,   # GB/s per channel
    row_buffer_size=8192,  # bytes (8KB)
    burst_length=8,
    data_width=64,
    tRCD=13.75,
    tRP=13.75,
    tCAS=13.75,
)


def create_resnet50_layers() -> List[ConvLayer]:
    """Create representative conv layers from ResNet-50"""
    return [
        # Stage 1 - after initial conv
        ConvLayer(batch=1, in_channels=64, out_channels=64, 
                  in_height=56, in_width=56, kernel_size=1),
        ConvLayer(batch=1, in_channels=64, out_channels=64, 
                  in_height=56, in_width=56, kernel_size=3, padding=1),
        ConvLayer(batch=1, in_channels=64, out_channels=256, 
                  in_height=56, in_width=56, kernel_size=1),
        
        # Stage 2
        ConvLayer(batch=1, in_channels=256, out_channels=128, 
                  in_height=28, in_width=28, kernel_size=1),
        ConvLayer(batch=1, in_channels=128, out_channels=128, 
                  in_height=28, in_width=28, kernel_size=3, padding=1),
        ConvLayer(batch=1, in_channels=128, out_channels=512, 
                  in_height=28, in_width=28, kernel_size=1),
        
        # Stage 3
        ConvLayer(batch=1, in_channels=512, out_channels=256, 
                  in_height=14, in_width=14, kernel_size=1),
        ConvLayer(batch=1, in_channels=256, out_channels=256, 
                  in_height=14, in_width=14, kernel_size=3, padding=1),
        ConvLayer(batch=1, in_channels=256, out_channels=1024, 
                  in_height=14, in_width=14, kernel_size=1),
        
        # Stage 4
        ConvLayer(batch=1, in_channels=1024, out_channels=512, 
                  in_height=7, in_width=7, kernel_size=1),
        ConvLayer(batch=1, in_channels=512, out_channels=512, 
                  in_height=7, in_width=7, kernel_size=3, padding=1),
        ConvLayer(batch=1, in_channels=512, out_channels=2048, 
                  in_height=7, in_width=7, kernel_size=1),
    ]


def create_vgg16_layers() -> List[ConvLayer]:
    """Create conv layers from VGG-16"""
    return [
        # Block 1
        ConvLayer(batch=1, in_channels=3, out_channels=64, 
                  in_height=224, in_width=224, kernel_size=3, padding=1),
        ConvLayer(batch=1, in_channels=64, out_channels=64, 
                  in_height=224, in_width=224, kernel_size=3, padding=1),
        
        # Block 2 (after pooling)
        ConvLayer(batch=1, in_channels=64, out_channels=128, 
                  in_height=112, in_width=112, kernel_size=3, padding=1),
        ConvLayer(batch=1, in_channels=128, out_channels=128, 
                  in_height=112, in_width=112, kernel_size=3, padding=1),
        
        # Block 3
        ConvLayer(batch=1, in_channels=128, out_channels=256, 
                  in_height=56, in_width=56, kernel_size=3, padding=1),
        ConvLayer(batch=1, in_channels=256, out_channels=256, 
                  in_height=56, in_width=56, kernel_size=3, padding=1),
        ConvLayer(batch=1, in_channels=256, out_channels=256, 
                  in_height=56, in_width=56, kernel_size=3, padding=1),
        
        # Block 4
        ConvLayer(batch=1, in_channels=256, out_channels=512, 
                  in_height=28, in_width=28, kernel_size=3, padding=1),
        ConvLayer(batch=1, in_channels=512, out_channels=512, 
                  in_height=28, in_width=28, kernel_size=3, padding=1),
        ConvLayer(batch=1, in_channels=512, out_channels=512, 
                  in_height=28, in_width=28, kernel_size=3, padding=1),
        
        # Block 5
        ConvLayer(batch=1, in_channels=512, out_channels=512, 
                  in_height=14, in_width=14, kernel_size=3, padding=1),
        ConvLayer(batch=1, in_channels=512, out_channels=512, 
                  in_height=14, in_width=14, kernel_size=3, padding=1),
        ConvLayer(batch=1, in_channels=512, out_channels=512, 
                  in_height=14, in_width=14, kernel_size=3, padding=1),
    ]


def create_mobilenet_layers() -> List[ConvLayer]:
    """Create representative layers from MobileNet (depthwise separable)"""
    # Standard convs only - depthwise would need separate handling
    return [
        ConvLayer(batch=1, in_channels=3, out_channels=32, 
                  in_height=224, in_width=224, kernel_size=3, stride=2, padding=1),
        # After depthwise
        ConvLayer(batch=1, in_channels=32, out_channels=64, 
                  in_height=112, in_width=112, kernel_size=1),
        ConvLayer(batch=1, in_channels=64, out_channels=128, 
                  in_height=56, in_width=56, kernel_size=1),
        ConvLayer(batch=1, in_channels=128, out_channels=128, 
                  in_height=56, in_width=56, kernel_size=1),
        ConvLayer(batch=1, in_channels=128, out_channels=256, 
                  in_height=28, in_width=28, kernel_size=1),
        ConvLayer(batch=1, in_channels=256, out_channels=256, 
                  in_height=28, in_width=28, kernel_size=1),
        ConvLayer(batch=1, in_channels=256, out_channels=512, 
                  in_height=14, in_width=14, kernel_size=1),
        ConvLayer(batch=1, in_channels=512, out_channels=512, 
                  in_height=14, in_width=14, kernel_size=1),
        ConvLayer(batch=1, in_channels=512, out_channels=1024, 
                  in_height=7, in_width=7, kernel_size=1),
    ]


def generate_layout_preferences(n_layers: int, pattern: str) -> List[Layout]:
    """
    Generate layout preferences based on accelerator pattern
    
    Patterns:
    - 'uniform_nchw': All layers prefer NCHW
    - 'uniform_nhwc': All layers prefer NHWC  
    - 'alternating': Alternate between NCHW and NHWC
    - 'random_like': Pseudo-random pattern (deterministic for reproducibility)
    - 'block_3': Changes every 3 layers
    """
    if pattern == 'uniform_nchw':
        return [Layout.NCHW] * n_layers
    elif pattern == 'uniform_nhwc':
        return [Layout.NHWC] * n_layers
    elif pattern == 'alternating':
        return [Layout.NCHW if i % 2 == 0 else Layout.NHWC for i in range(n_layers)]
    elif pattern == 'random_like':
        # Deterministic pseudo-random based on index
        return [Layout.NCHW if (i * 7 + 3) % 11 < 6 else Layout.NHWC for i in range(n_layers)]
    elif pattern == 'block_3':
        return [Layout.NCHW if (i // 3) % 2 == 0 else Layout.NHWC for i in range(n_layers)]
    else:
        return [Layout.NCHW] * n_layers


def compute_transformation_cost_breakdown(layers: List[ConvLayer], 
                                           from_layout: Layout, 
                                           to_layout: Layout,
                                           config: DRAMConfig) -> Dict:
    """
    Compute detailed transformation costs between layers
    """
    analyzer = DRAMAccessAnalyzer(config)
    costs = []
    
    for i, layer in enumerate(layers):
        if from_layout == Layout.NCHW:
            shape = (layer.batch, layer.in_channels, layer.in_height, layer.in_width)
        else:
            shape = (layer.batch, layer.in_height, layer.in_width, layer.in_channels)
        
        cost = analyzer.estimate_layout_transform_cost(shape, from_layout, to_layout)
        costs.append({
            'layer_idx': i,
            'shape': shape,
            'total_bytes': math.prod(shape) * 4,
            'transform_latency_ns': cost['total_latency_ns'],
            'overhead_factor': cost['transform_overhead_factor']
        })
    
    return costs


def analyze_network_strategies(network_name: str, layers: List[ConvLayer], 
                                config: DRAMConfig) -> Dict:
    """
    Comprehensive analysis of all strategies for a network
    """
    analyzer = LayoutStrategyAnalyzer(config)
    
    results = {
        'network': network_name,
        'num_layers': len(layers),
        'dram_config': {
            'bandwidth_gbps': config.bandwidth_gbps,
            'row_buffer_size': config.row_buffer_size,
        },
        'strategies': {}
    }
    
    # Test different preference patterns
    patterns = ['uniform_nchw', 'uniform_nhwc', 'alternating', 'random_like', 'block_3']
    
    for pattern in patterns:
        prefs = generate_layout_preferences(len(layers), pattern)
        
        # Propagation with NCHW
        prop_nchw = analyzer.analyze_propagation_strategy(layers, Layout.NCHW, prefs)
        # Propagation with NHWC
        prop_nhwc = analyzer.analyze_propagation_strategy(layers, Layout.NHWC, prefs)
        # Adaptation starting from NCHW
        adapt = analyzer.analyze_adaptation_strategy(layers, Layout.NCHW, prefs)
        
        results['strategies'][pattern] = {
            'preferences': [p.value for p in prefs],
            'propagation_nchw': {
                'total_latency_ns': prop_nchw['total_latency_ns'],
                'num_transforms': prop_nchw['num_transforms'],
            },
            'propagation_nhwc': {
                'total_latency_ns': prop_nhwc['total_latency_ns'],
                'num_transforms': prop_nhwc['num_transforms'],
            },
            'adaptation': {
                'total_latency_ns': adapt['total_latency_ns'],
                'num_transforms': adapt['num_transforms'],
            },
            'best_strategy': 'adaptation' if adapt['total_latency_ns'] < min(
                prop_nchw['total_latency_ns'], prop_nhwc['total_latency_ns']
            ) else ('prop_nchw' if prop_nchw['total_latency_ns'] < prop_nhwc['total_latency_ns'] else 'prop_nhwc'),
        }
    
    return results


def print_analysis_report(results: Dict):
    """Pretty print analysis results"""
    print(f"\n{'='*70}")
    print(f"Network: {results['network']} ({results['num_layers']} layers)")
    print(f"DRAM: {results['dram_config']['bandwidth_gbps']} GB/s, "
          f"{results['dram_config']['row_buffer_size']} byte row buffer")
    print(f"{'='*70}")
    
    for pattern, data in results['strategies'].items():
        print(f"\n  Pattern: {pattern}")
        print(f"    Propagation (NCHW): {data['propagation_nchw']['total_latency_ns']/1e6:.2f} ms")
        print(f"    Propagation (NHWC): {data['propagation_nhwc']['total_latency_ns']/1e6:.2f} ms")
        print(f"    Adaptation:         {data['adaptation']['total_latency_ns']/1e6:.2f} ms "
              f"({data['adaptation']['num_transforms']} transforms)")
        print(f"    Best: {data['best_strategy']}")


def analyze_fusion_opportunity(layers: List[ConvLayer], 
                                preferences: List[Layout],
                                fusion_groups: List[Tuple[int, int]],
                                config: DRAMConfig) -> Dict:
    """
    Analyze layout transformation opportunities with operator fusion
    
    When consecutive operations are fused, intermediate layout transforms
    can potentially be eliminated.
    """
    analyzer = DRAMAccessAnalyzer(config)
    
    # Without fusion - transform at every boundary
    no_fusion_cost = 0
    transforms_without_fusion = 0
    current_layout = preferences[0]
    
    for i, pref in enumerate(preferences[1:], 1):
        if current_layout != pref:
            if current_layout == Layout.NCHW:
                shape = (layers[i-1].batch, layers[i-1].in_channels, 
                        layers[i-1].in_height, layers[i-1].in_width)
            else:
                shape = (layers[i-1].batch, layers[i-1].in_height, 
                        layers[i-1].in_width, layers[i-1].in_channels)
            cost = analyzer.estimate_layout_transform_cost(shape, current_layout, pref)
            no_fusion_cost += cost['total_latency_ns']
            transforms_without_fusion += 1
            current_layout = pref
    
    # With fusion - transforms only at fusion group boundaries
    fusion_cost = 0
    transforms_with_fusion = 0
    
    # Build set of layers that are fusion boundaries
    fusion_boundaries = set()
    for start, end in fusion_groups:
        fusion_boundaries.add(start)
        fusion_boundaries.add(end)
    
    current_layout = preferences[0]
    for i, pref in enumerate(preferences[1:], 1):
        # Only transform if at a fusion boundary
        if i in fusion_boundaries and current_layout != pref:
            if current_layout == Layout.NCHW:
                shape = (layers[i-1].batch, layers[i-1].in_channels, 
                        layers[i-1].in_height, layers[i-1].in_width)
            else:
                shape = (layers[i-1].batch, layers[i-1].in_height, 
                        layers[i-1].in_width, layers[i-1].in_channels)
            cost = analyzer.estimate_layout_transform_cost(shape, current_layout, pref)
            fusion_cost += cost['total_latency_ns']
            transforms_with_fusion += 1
            current_layout = pref
    
    return {
        'no_fusion': {
            'transforms': transforms_without_fusion,
            'total_cost_ns': no_fusion_cost
        },
        'with_fusion': {
            'transforms': transforms_with_fusion,
            'total_cost_ns': fusion_cost,
            'fusion_groups': fusion_groups
        },
        'savings_ns': no_fusion_cost - fusion_cost,
        'savings_pct': (no_fusion_cost - fusion_cost) / no_fusion_cost * 100 if no_fusion_cost > 0 else 0
    }


def run_comprehensive_analysis():
    """Run analysis on multiple networks with different DRAM configs"""
    
    networks = {
        'ResNet-50': create_resnet50_layers(),
        'VGG-16': create_vgg16_layers(),
        'MobileNet': create_mobilenet_layers(),
    }
    
    configs = {
        'DDR4': DDR4_CONFIG,
        'HBM2': HBM2_CONFIG,
    }
    
    all_results = []
    
    for net_name, layers in networks.items():
        for config_name, config in configs.items():
            print(f"\nAnalyzing {net_name} with {config_name}...")
            results = analyze_network_strategies(net_name, layers, config)
            results['dram_type'] = config_name
            all_results.append(results)
            print_analysis_report(results)
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: Best Strategy per Network and DRAM Type")
    print("="*80)
    print(f"{'Network':<15} {'DRAM':<8} {'Best Pattern':<15} {'Strategy':<12} {'Latency (ms)':<15}")
    print("-"*80)
    
    for result in all_results:
        for pattern, data in result['strategies'].items():
            if data['best_strategy'] == 'adaptation':
                latency = data['adaptation']['total_latency_ns']
            elif data['best_strategy'] == 'prop_nchw':
                latency = data['propagation_nchw']['total_latency_ns']
            else:
                latency = data['propagation_nhwc']['total_latency_ns']
            
            # Only print if this pattern shows adaptation wins
            if data['best_strategy'] == 'adaptation':
                print(f"{result['network']:<15} {result['dram_type']:<8} {pattern:<15} "
                      f"{data['best_strategy']:<12} {latency/1e6:.3f}")
    
    return all_results


if __name__ == "__main__":
    results = run_comprehensive_analysis()
    
    # Fusion analysis example
    print("\n" + "="*70)
    print("FUSION ANALYSIS EXAMPLE (ResNet-50, HBM2)")
    print("="*70)
    
    layers = create_resnet50_layers()
    prefs = generate_layout_preferences(len(layers), 'alternating')
    
    # Define fusion groups (consecutive layers that are fused)
    fusion_groups = [(0, 2), (3, 5), (6, 8), (9, 11)]  # Bottleneck blocks
    
    fusion_result = analyze_fusion_opportunity(layers, prefs, fusion_groups, HBM2_CONFIG)
    
    print(f"\nWithout fusion: {fusion_result['no_fusion']['transforms']} transforms, "
          f"{fusion_result['no_fusion']['total_cost_ns']/1e6:.3f} ms")
    print(f"With fusion:    {fusion_result['with_fusion']['transforms']} transforms, "
          f"{fusion_result['with_fusion']['total_cost_ns']/1e6:.3f} ms")
    print(f"Savings:        {fusion_result['savings_pct']:.1f}%")
