#!/usr/bin/env python3
"""
Analyze all nn_dataflow networks using the PIM Cost Model.

This script:
1. Extracts all compute layers from nn_dataflow networks
2. Estimates cycles using the validated PIM Cost Model
3. Reports theoretical peak performance and efficiency

The PIM Cost Model was validated against UniNDP with ~3.5% error.
"""

import sys
from pathlib import Path
import json
from dataclasses import dataclass, field
from typing import List, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Import PIM Cost Model - it's in src/golden_model
sys.path.insert(0, str(SRC_DIR / 'golden_model'))
from pim_cost_model import (
    PIMCostModel, PIMArchConfig, WorkloadSpec, TilingConfig
)

# Import nn_dataflow networks
from nn_dataflow.nns import all_networks, import_network
from nn_dataflow.core import ConvLayer, FCLayer


@dataclass
class LayerAnalysis:
    """Analysis result for one layer."""
    network: str
    layer_name: str
    layer_type: str
    K: int  # Input dimension
    L: int  # Output dimension
    MACs: int
    cycles: float
    compute_cycles: float
    memory_cycles: float
    efficiency: float
    throughput_gops: float
    
    def to_dict(self):
        return {
            'network': self.network,
            'layer_name': self.layer_name,
            'layer_type': self.layer_type,
            'K': self.K,
            'L': self.L,
            'MACs': self.MACs,
            'cycles': self.cycles,
            'compute_cycles': self.compute_cycles,
            'memory_cycles': self.memory_cycles,
            'efficiency': self.efficiency,
            'throughput_gops': self.throughput_gops
        }


@dataclass
class NetworkSummary:
    """Summary for one network."""
    name: str
    total_layers: int
    total_MACs: int
    total_cycles: float
    avg_efficiency: float
    throughput_gops: float
    layers: List[LayerAnalysis] = field(default_factory=list)


def analyze_layer(net_name: str, layer_name: str, layer_type: str, 
                  K: int, L: int, arch: PIMArchConfig, model: PIMCostModel,
                  output_positions: int = 1) -> LayerAnalysis:
    """Analyze one layer using PIM Cost Model."""
    # MVM: M=1, K=input_dim, N=output_dim
    workload = WorkloadSpec(M=1, N=L, K=K)
    result = model.estimate_gemm(workload)
    
    # For convolution, total MACs = K * L * output_positions
    # The cost model computes per-output-position cycles, so we multiply
    MACs = K * L * output_positions
    total_cycles = result.total_cycles * output_positions
    compute_cycles = result.compute_cycles * output_positions
    memory_cycles = result.memory_cycles * output_positions
    
    efficiency = compute_cycles / total_cycles if total_cycles > 0 else 0
    
    # Calculate throughput in GOPS (assuming 1GHz clock)
    throughput_gops = MACs / total_cycles / 1e9 if total_cycles > 0 else 0
    
    return LayerAnalysis(
        network=net_name,
        layer_name=layer_name,
        layer_type=layer_type,
        K=K,
        L=L,
        MACs=MACs,
        cycles=total_cycles,
        compute_cycles=compute_cycles,
        memory_cycles=memory_cycles,
        efficiency=efficiency,
        throughput_gops=throughput_gops
    )


def analyze_network(net_name: str, arch: PIMArchConfig, model: PIMCostModel) -> NetworkSummary:
    """Analyze all layers in a network."""
    try:
        network = import_network(net_name)
    except Exception as e:
        print(f"  Failed to load {net_name}: {e}")
        return None
    
    layers = []
    
    for layer_name in network:
        layer = network[layer_name]
        
        if isinstance(layer, FCLayer):
            K = layer.nifm * layer.hofm * layer.wofm
            L = layer.nofm
            layer_type = 'FC'
            output_positions = 1
        elif isinstance(layer, ConvLayer):
            K = layer.nifm * layer.hfil * layer.wfil
            L = layer.nofm
            layer_type = 'Conv'
            output_positions = layer.hofm * layer.wofm
        else:
            continue
        
        analysis = analyze_layer(net_name, layer_name, layer_type, K, L, arch, model,
                                 output_positions=output_positions)
        layers.append(analysis)
    
    if not layers:
        return None
    
    total_macs = sum(l.MACs for l in layers)
    total_cycles = sum(l.cycles for l in layers)
    avg_efficiency = sum(l.efficiency for l in layers) / len(layers) if layers else 0
    throughput_gops = total_macs / total_cycles / 1e9 if total_cycles > 0 else 0
    
    return NetworkSummary(
        name=net_name,
        total_layers=len(layers),
        total_MACs=total_macs,
        total_cycles=total_cycles,
        avg_efficiency=avg_efficiency,
        throughput_gops=throughput_gops,
        layers=layers
    )


def print_layer_table(layers: List[LayerAnalysis], max_rows: int = 20):
    """Print a table of layer analysis results."""
    print(f"\n{'Network':<15} {'Layer':<20} {'Type':<6} {'K':<8} {'L':<8} "
          f"{'MACs':<12} {'Cycles':<12} {'Eff%':<8}")
    print("-" * 100)
    
    for layer in layers[:max_rows]:
        print(f"{layer.network:<15} {layer.layer_name:<20} {layer.layer_type:<6} "
              f"{layer.K:<8} {layer.L:<8} {layer.MACs:<12,} {layer.cycles:<12,.0f} "
              f"{layer.efficiency*100:<8.1f}")


def main():
    print("=" * 80)
    print("Neural Network Analysis with PIM Cost Model")
    print("=" * 80)
    
    # Initialize architecture and model
    arch = PIMArchConfig()
    model = PIMCostModel(arch)
    
    print(f"\nArchitecture Configuration:")
    print(f"  Channels: {arch.num_channels}")
    print(f"  Banks per channel: {arch.num_bank_groups * arch.num_banks_per_group}")
    print(f"  PUs per channel: {arch.num_pu_per_channel}")
    print(f"  Total PUs: {arch.total_pus}")
    print(f"  Peak throughput: {arch.peak_throughput} MACs/cycle")
    
    # Analyze all networks
    all_summaries = []
    all_layers = []
    
    print("\n" + "-" * 80)
    print("Analyzing Networks:")
    print("-" * 80)
    
    for net_name in sorted(all_networks()):
        summary = analyze_network(net_name, arch, model)
        if summary:
            all_summaries.append(summary)
            all_layers.extend(summary.layers)
            print(f"  {net_name:<20}: {summary.total_layers:>3} layers, "
                  f"{summary.total_MACs/1e9:.2f}G MACs, "
                  f"{summary.total_cycles/1e6:.2f}M cycles, "
                  f"eff={summary.avg_efficiency*100:.1f}%")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("NETWORK SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Network':<20} {'Layers':<8} {'MACs (G)':<12} {'Cycles (M)':<12} "
          f"{'Eff (%)':<10} {'GOPS':<10}")
    print("-" * 80)
    
    total_macs = 0
    total_cycles = 0
    
    for summary in sorted(all_summaries, key=lambda x: -x.total_MACs):
        print(f"{summary.name:<20} {summary.total_layers:<8} "
              f"{summary.total_MACs/1e9:<12.2f} {summary.total_cycles/1e6:<12.2f} "
              f"{summary.avg_efficiency*100:<10.1f} {summary.throughput_gops:<10.4f}")
        total_macs += summary.total_MACs
        total_cycles += summary.total_cycles
    
    print("-" * 80)
    overall_eff = total_macs / (arch.peak_throughput * total_cycles) if total_cycles > 0 else 0
    print(f"{'TOTAL':<20} {len(all_layers):<8} "
          f"{total_macs/1e9:<12.2f} {total_cycles/1e6:<12.2f} "
          f"{overall_eff*100:<10.1f} {total_macs/total_cycles/1e9:<10.4f}")
    
    # Show largest layers
    print("\n" + "=" * 80)
    print("TOP 20 LAYERS BY MACs")
    print("=" * 80)
    
    largest_layers = sorted(all_layers, key=lambda x: -x.MACs)
    print_layer_table(largest_layers)
    
    # Show layers suitable for UniNDP validation (K >= 512, L >= 512)
    print("\n" + "=" * 80)
    print("LAYERS SUITABLE FOR UniNDP VALIDATION (K >= 512, L >= 512)")
    print("=" * 80)
    
    unindp_layers = [l for l in all_layers if l.K >= 512 and l.L >= 512]
    print(f"\nFound {len(unindp_layers)} layers suitable for UniNDP validation:")
    print_layer_table(sorted(unindp_layers, key=lambda x: -x.MACs))
    
    # Save results to JSON
    output_data = {
        'architecture': {
            'num_channels': arch.num_channels,
            'num_banks': arch.total_banks,
            'num_pus': arch.total_pus,
            'peak_throughput': arch.peak_throughput,
        },
        'networks': [
            {
                'name': s.name,
                'total_layers': s.total_layers,
                'total_MACs': s.total_MACs,
                'total_cycles': s.total_cycles,
                'avg_efficiency': s.avg_efficiency,
                'layers': [l.to_dict() for l in s.layers]
            }
            for s in all_summaries
        ],
        'summary': {
            'total_networks': len(all_summaries),
            'total_layers': len(all_layers),
            'total_MACs': total_macs,
            'total_cycles': total_cycles,
            'overall_efficiency': overall_eff,
            'unindp_suitable_layers': len(unindp_layers)
        }
    }
    
    output_file = PROJECT_ROOT / 'examples' / 'nn_analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")
    
    return all_summaries


if __name__ == '__main__':
    main()
