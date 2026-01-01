#!/usr/bin/env python3
"""
Quick validation script for nn_dataflow networks.
Tests representative layers from all networks against UniNDP.
"""

import sys
from pathlib import Path
import time

# Setup paths - order matters!
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# First add project root for nn_dataflow
sys.path.insert(0, str(PROJECT_ROOT))
# Then add src for golden_model (this goes to position 0, pushing others down)
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Import from the correct location
from golden_model.unindp_bridge import UniNDPBridge, ilp_to_unindp_strategy, ILPMapping

# Import nn_dataflow components
from nn_dataflow.nns import all_networks, import_network
from nn_dataflow.core import ConvLayer, FCLayer


def main():
    bridge = UniNDPBridge()
    
    results = []
    print('=' * 70)
    print('Validation: nn_dataflow Networks → UniNDP Simulation')
    print('=' * 70)
    print('\nNote: UniNDP has constraints (l_block==4, requires larger workloads)')
    print('Testing with workload sizes that meet UniNDP requirements...\n')
    
    # UniNDP requires certain workload sizes to work correctly
    # The baseline strategy requires: L dimension divisible by 64*8 (channels*PUs)
    MIN_DIM = 512  # Minimum dimension for UniNDP to work well
    
    for net_name in sorted(all_networks()):
        try:
            network = import_network(net_name)
        except Exception as e:
            print(f'\n{net_name}: FAILED to load - {e}')
            continue
            
        print(f'\n{net_name} ({network.net_name}):')
        
        count = 0
        for layer_name in network:
            if count >= 3:  # Limit per network for speed
                break
            layer = network[layer_name]
            
            if isinstance(layer, FCLayer):
                K = layer.nifm * layer.hofm * layer.wofm
                L = layer.nofm
                
                # Skip small workloads that don't meet UniNDP requirements
                if K < MIN_DIM or L < MIN_DIM:
                    print(f'  - {layer_name}: FC({K}→{L}) skipped (too small for UniNDP)')
                    continue
                    
                ilp = ILPMapping(
                    K=K, L=L,
                    ch_parallel_dim='l',
                    pu_parallel_dim='l', 
                    num_channels=64,
                    num_pus_per_device=8
                )
                strategy = ilp_to_unindp_strategy(ilp)
                
                start = time.time()
                result = bridge.run_simulation(M=K, K=L, strategy=strategy)
                elapsed = time.time() - start
                
                if result.success:
                    print(f'  ✓ {layer_name}: FC({K}→{L}) = {result.cycles:.1f} cycles ({elapsed:.1f}s)')
                    results.append({
                        'network': net_name,
                        'layer': layer_name,
                        'type': 'FC',
                        'K': K,
                        'L': L,
                        'cycles': result.cycles
                    })
                else:
                    # Extract just the error type, not full traceback
                    err = result.error_message.split('\n')[-2] if result.error_message else 'Unknown'
                    print(f'  ✗ {layer_name}: FC({K}→{L}) - {err[:60]}')
                count += 1
                
            elif isinstance(layer, ConvLayer):
                K = layer.nifm * layer.hfil * layer.wfil
                L = layer.nofm
                
                # Skip small workloads
                if K < MIN_DIM or L < MIN_DIM:
                    print(f'  - {layer_name}: Conv(K={K}, L={L}) skipped (too small)')
                    continue
                    
                ilp = ILPMapping(
                    K=K, L=L,
                    ch_parallel_dim='l',
                    pu_parallel_dim='l',
                    num_channels=64,
                    num_pus_per_device=8
                )
                strategy = ilp_to_unindp_strategy(ilp)
                
                start = time.time()
                result = bridge.run_simulation(M=K, K=L, strategy=strategy)
                elapsed = time.time() - start
                
                if result.success:
                    print(f'  ✓ {layer_name}: Conv(K={K}, L={L}) = {result.cycles:.1f} cycles ({elapsed:.1f}s)')
                    results.append({
                        'network': net_name,
                        'layer': layer_name,
                        'type': 'Conv',
                        'K': K,
                        'L': L,
                        'cycles': result.cycles
                    })
                else:
                    err = result.error_message.split('\n')[-2] if result.error_message else 'Unknown'
                    print(f'  ✗ {layer_name}: Conv(K={K}, L={L}) - {err[:60]}')
                count += 1
    
    # Summary
    print('\n' + '=' * 70)
    print('VALIDATION SUMMARY')
    print('=' * 70)
    print(f'Total layers validated: {len(results)}')
    
    # Group by network
    by_net = {}
    for r in results:
        net = r['network']
        if net not in by_net:
            by_net[net] = []
        by_net[net].append(r)
    
    print(f'\nBy Network:')
    for net, layers in sorted(by_net.items()):
        avg_cycles = sum(l['cycles'] for l in layers) / len(layers)
        print(f'  {net}: {len(layers)} layers, avg {avg_cycles:,.0f} cycles')
    
    # Save results
    import json
    output_file = PROJECT_ROOT / 'examples' / 'validation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to: {output_file}')
    
    return results


if __name__ == '__main__':
    main()
