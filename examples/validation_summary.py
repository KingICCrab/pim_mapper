#!/usr/bin/env python3
"""
Generate a summary of nn_dataflow workload validation results.

This script:
1. Extracts all workloads from nn_dataflow networks
2. Reads existing UniNDP simulation results from log files
3. Compares with PIM Cost Model predictions
4. Generates a comprehensive validation report
"""

import sys
from pathlib import Path
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Import PIM Cost Model
sys.path.insert(0, str(SRC_DIR / 'golden_model'))
from pim_cost_model import PIMCostModel, PIMArchConfig, WorkloadSpec

# Import nn_dataflow
from nn_dataflow.nns import all_networks, import_network
from nn_dataflow.core import ConvLayer, FCLayer

UNINDP_PATH = PROJECT_ROOT / 'UniNDP'


def read_existing_unindp_results() -> Dict[Tuple[int, int], float]:
    """Read all existing UniNDP simulation results from log files."""
    results = {}
    log_dir = UNINDP_PATH / 'verify_result' / 'log'
    
    if not log_dir.exists():
        return results
    
    for log_file in log_dir.glob('*.log'):
        # Parse filename like "[4096, 4096].log"
        match = re.match(r'\[(\d+),\s*(\d+)\]\.log', log_file.name)
        if not match:
            continue
        
        K, L = int(match.group(1)), int(match.group(2))
        
        # Read log file and extract result
        with open(log_file, 'r') as f:
            content = f.read()
            result_match = re.search(r'result:\s*([\d.]+)', content)
            if result_match:
                cycles = float(result_match.group(1))
                results[(K, L)] = cycles
    
    return results


def extract_all_workloads() -> List[Dict]:
    """Extract all FC and Conv workloads from nn_dataflow networks."""
    workloads = []
    
    for net_name in sorted(all_networks()):
        try:
            network = import_network(net_name)
        except:
            continue
        
        for layer_name in network:
            layer = network[layer_name]
            
            if isinstance(layer, FCLayer):
                K = layer.nifm * layer.hofm * layer.wofm
                L = layer.nofm
                output_positions = 1
                layer_type = 'FC'
            elif isinstance(layer, ConvLayer):
                K = layer.nifm * layer.hfil * layer.wfil
                L = layer.nofm
                output_positions = layer.hofm * layer.wofm
                layer_type = 'Conv'
            else:
                continue
            
            total_MACs = K * L * output_positions
            
            workloads.append({
                'network': net_name,
                'layer': layer_name,
                'type': layer_type,
                'K': K,
                'L': L,
                'output_positions': output_positions,
                'total_MACs': total_MACs,
            })
    
    return workloads


def run_cost_model(K: int, L: int) -> float:
    """Run PIM Cost Model."""
    arch = PIMArchConfig()
    model = PIMCostModel(arch)
    workload = WorkloadSpec(M=1, N=L, K=K)
    result = model.estimate_gemm(workload)
    return result.total_cycles


def main():
    print("=" * 90)
    print("nn_dataflow Network Validation Summary")
    print("=" * 90)
    
    # Read existing UniNDP results
    unindp_results = read_existing_unindp_results()
    print(f"\nExisting UniNDP simulation results found: {len(unindp_results)}")
    for (K, L), cycles in sorted(unindp_results.items()):
        print(f"  [{K}, {L}]: {cycles:.2f} cycles")
    
    # Extract all workloads
    all_workloads = extract_all_workloads()
    print(f"\nTotal workloads in nn_dataflow: {len(all_workloads)}")
    
    # Match workloads with UniNDP results
    matched_workloads = []
    for w in all_workloads:
        key = (w['K'], w['L'])
        if key in unindp_results:
            w['unindp_cycles'] = unindp_results[key]
            w['cost_model_cycles'] = run_cost_model(w['K'], w['L'])
            w['error_pct'] = abs(w['unindp_cycles'] - w['cost_model_cycles']) / w['cost_model_cycles'] * 100
            matched_workloads.append(w)
    
    print(f"Workloads matching UniNDP results: {len(matched_workloads)}")
    
    if matched_workloads:
        print("\n" + "-" * 90)
        print("VALIDATED WORKLOADS (UniNDP vs PIM Cost Model)")
        print("-" * 90)
        
        print(f"\n{'Network':<15} {'Layer':<20} {'Type':<6} {'K':<8} {'L':<8} "
              f"{'UniNDP':<12} {'CostModel':<12} {'Error%':<8}")
        print("-" * 90)
        
        for w in sorted(matched_workloads, key=lambda x: -x['total_MACs']):
            print(f"{w['network']:<15} {w['layer']:<20} {w['type']:<6} {w['K']:<8} {w['L']:<8} "
                  f"{w['unindp_cycles']:<12,.1f} {w['cost_model_cycles']:<12,.1f} {w['error_pct']:<8.1f}")
        
        # Error statistics
        errors = [w['error_pct'] for w in matched_workloads]
        print(f"\nError Statistics:")
        print(f"  Average: {sum(errors)/len(errors):.2f}%")
        print(f"  Min: {min(errors):.2f}%")
        print(f"  Max: {max(errors):.2f}%")
    
    # Summary by network
    print("\n" + "=" * 90)
    print("NETWORK SUMMARY (PIM Cost Model Analysis)")
    print("=" * 90)
    
    # Initialize cost model
    arch = PIMArchConfig()
    model = PIMCostModel(arch)
    
    print(f"\nArchitecture:")
    print(f"  Total PUs: {arch.total_pus}")
    print(f"  Peak throughput: {arch.peak_throughput:,} MACs/cycle")
    
    network_stats = {}
    for w in all_workloads:
        net = w['network']
        if net not in network_stats:
            network_stats[net] = {
                'layers': 0,
                'total_MACs': 0,
                'total_cycles': 0,
            }
        
        # Get per-output-position cycles and scale by output_positions
        per_output_cycles = run_cost_model(w['K'], w['L'])
        total_cycles = per_output_cycles * w['output_positions']
        
        network_stats[net]['layers'] += 1
        network_stats[net]['total_MACs'] += w['total_MACs']
        network_stats[net]['total_cycles'] += total_cycles
    
    print(f"\n{'Network':<20} {'Layers':<8} {'MACs (G)':<12} {'Cycles (M)':<12} "
          f"{'Efficiency':<12} {'Throughput':<12}")
    print("-" * 90)
    
    total_macs = 0
    total_cycles = 0
    
    for net, stats in sorted(network_stats.items(), key=lambda x: -x[1]['total_MACs']):
        eff = stats['total_MACs'] / (arch.peak_throughput * stats['total_cycles']) * 100 if stats['total_cycles'] > 0 else 0
        throughput = stats['total_MACs'] / stats['total_cycles'] if stats['total_cycles'] > 0 else 0
        
        print(f"{net:<20} {stats['layers']:<8} {stats['total_MACs']/1e9:<12.2f} "
              f"{stats['total_cycles']/1e6:<12.2f} {eff:<12.1f}% {throughput/1e3:<12.2f}K/cyc")
        
        total_macs += stats['total_MACs']
        total_cycles += stats['total_cycles']
    
    print("-" * 90)
    overall_eff = total_macs / (arch.peak_throughput * total_cycles) * 100 if total_cycles > 0 else 0
    overall_throughput = total_macs / total_cycles if total_cycles > 0 else 0
    print(f"{'TOTAL':<20} {len(all_workloads):<8} {total_macs/1e9:<12.2f} "
          f"{total_cycles/1e6:<12.2f} {overall_eff:<12.1f}% {overall_throughput/1e3:<12.2f}K/cyc")
    
    # Save results
    output_data = {
        'architecture': {
            'total_pus': arch.total_pus,
            'peak_throughput': arch.peak_throughput,
        },
        'unindp_results': {f"[{k[0]}, {k[1]}]": v for k, v in unindp_results.items()},
        'validated_workloads': matched_workloads,
        'all_workloads': all_workloads,
        'network_summary': network_stats,
        'overall': {
            'total_networks': len(network_stats),
            'total_layers': len(all_workloads),
            'total_MACs': total_macs,
            'total_cycles': total_cycles,
            'efficiency': overall_eff,
        }
    }
    
    output_file = PROJECT_ROOT / 'examples' / 'validation_summary.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\n\nResults saved to: {output_file}")
    
    return output_data


if __name__ == '__main__':
    main()
