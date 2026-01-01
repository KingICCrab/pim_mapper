#!/usr/bin/env python3
"""
Extract all workloads from nn_dataflow networks for validation.
This script extracts workload specifications without requiring UniNDP.

Outputs a JSON file with all layer specifications that can be used for:
1. Analytical cost model validation
2. Manual UniNDP simulation
3. Comparison with other simulators
"""

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT))

from nn_dataflow.nns import all_networks, import_network
from nn_dataflow.core import ConvLayer, FCLayer, PoolingLayer


def extract_all_workloads():
    """Extract all compute workloads from all networks."""
    all_workloads = {}
    
    print('=' * 70)
    print('Extracting Workloads from nn_dataflow Networks')
    print('=' * 70)
    
    for net_name in sorted(all_networks()):
        try:
            network = import_network(net_name)
        except Exception as e:
            print(f'{net_name}: FAILED - {e}')
            continue
        
        workloads = []
        
        for layer_name in network:
            layer = network[layer_name]
            
            if isinstance(layer, FCLayer):
                K = layer.nifm * layer.hofm * layer.wofm
                L = layer.nofm
                MACs = K * L
                
                workloads.append({
                    'name': layer_name,
                    'type': 'FC',
                    'K': K,  # Input dimension
                    'L': L,  # Output dimension
                    'MACs': MACs,
                    'description': f'FC({K}→{L})'
                })
                
            elif isinstance(layer, ConvLayer):
                # For im2col conversion: K = C_in * kH * kW, L = C_out
                K = layer.nifm * layer.hfil * layer.wfil
                L = layer.nofm
                output_positions = layer.hofm * layer.wofm
                MACs = K * L * output_positions
                
                workloads.append({
                    'name': layer_name,
                    'type': 'Conv',
                    'K': K,  # Input channels * filter size
                    'L': L,  # Output channels
                    'C_in': layer.nifm,
                    'C_out': layer.nofm,
                    'H_in': layer.hifm,
                    'H_out': layer.hofm,
                    'filter_size': layer.hfil,
                    'stride': layer.htrd,
                    'output_positions': output_positions,
                    'MACs': MACs,
                    'description': f'Conv({layer.nifm}→{layer.nofm}, {layer.hfil}x{layer.hfil})'
                })
        
        all_workloads[net_name] = {
            'network_name': network.net_name,
            'total_layers': len(workloads),
            'total_MACs': sum(w['MACs'] for w in workloads),
            'layers': workloads
        }
        
        fc_count = sum(1 for w in workloads if w['type'] == 'FC')
        conv_count = sum(1 for w in workloads if w['type'] == 'Conv')
        total_macs = sum(w['MACs'] for w in workloads)
        print(f'{net_name}: {len(workloads)} layers ({conv_count} Conv, {fc_count} FC), {total_macs/1e9:.2f}G MACs')
    
    return all_workloads


def print_summary(all_workloads):
    """Print summary statistics."""
    print('\n' + '=' * 70)
    print('WORKLOAD SUMMARY')
    print('=' * 70)
    
    total_networks = len(all_workloads)
    total_layers = sum(w['total_layers'] for w in all_workloads.values())
    total_macs = sum(w['total_MACs'] for w in all_workloads.values())
    
    print(f'\nTotal networks: {total_networks}')
    print(f'Total compute layers: {total_layers}')
    print(f'Total MACs: {total_macs/1e12:.2f} TMACs')
    
    # Find representative workloads (large K and L)
    print('\n' + '-' * 70)
    print('Representative Large Workloads (K >= 1000, L >= 1000):')
    print('-' * 70)
    
    large_workloads = []
    for net_name, net_data in all_workloads.items():
        for layer in net_data['layers']:
            if layer['K'] >= 1000 and layer['L'] >= 1000:
                large_workloads.append({
                    'network': net_name,
                    **layer
                })
    
    if large_workloads:
        print(f"{'Network':<15} {'Layer':<25} {'Type':<6} {'K':<8} {'L':<8} {'MACs':<15}")
        print('-' * 80)
        for w in sorted(large_workloads, key=lambda x: -x['MACs'])[:20]:
            print(f"{w['network']:<15} {w['name']:<25} {w['type']:<6} {w['K']:<8} {w['L']:<8} {w['MACs']:,}")
    else:
        print('No workloads with K >= 1000 and L >= 1000 found.')
    
    # Workloads suitable for UniNDP (based on our testing)
    print('\n' + '-' * 70)
    print('Workloads Suitable for UniNDP Validation:')
    print('(K and L both >= 512, ideally divisible by 64)')
    print('-' * 70)
    
    unindp_suitable = []
    for net_name, net_data in all_workloads.items():
        for layer in net_data['layers']:
            if layer['K'] >= 512 and layer['L'] >= 512:
                unindp_suitable.append({
                    'network': net_name,
                    **layer
                })
    
    if unindp_suitable:
        print(f"Found {len(unindp_suitable)} suitable workloads:")
        for w in unindp_suitable[:15]:
            print(f"  {w['network']}/{w['name']}: {w['description']}")
    else:
        print('No workloads meet UniNDP size requirements.')


def main():
    all_workloads = extract_all_workloads()
    print_summary(all_workloads)
    
    # Save to JSON
    output_file = PROJECT_ROOT / 'examples' / 'nn_workloads.json'
    with open(output_file, 'w') as f:
        json.dump(all_workloads, f, indent=2)
    print(f'\n\nWorkload data saved to: {output_file}')
    
    return all_workloads


if __name__ == '__main__':
    main()
