#!/usr/bin/env python3
"""
Comprehensive validation of nn_dataflow workloads against UniNDP.

This script focuses on finding workloads that satisfy UniNDP constraints
and performs detailed comparison with the PIM Cost Model.
"""

import sys
from pathlib import Path
import json
import subprocess
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

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


def is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def run_unindp(K: int, L: int, timeout: int = 120) -> Tuple[Optional[float], Optional[str]]:
    """Run UniNDP simulation and return (cycles, error_msg)."""
    cmd = [
        sys.executable,
        str(UNINDP_PATH / 'sim_verify.py'),
        '-S', str(K), str(L)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(UNINDP_PATH)
        )
        
        if result.returncode != 0:
            stderr = result.stderr
            if "AssertionError" in stderr:
                return None, "l_block assertion"
            return None, f"Error: {stderr[:100]}"
        
        # Read from log file
        log_file = UNINDP_PATH / 'verify_result' / 'log' / f'[{K}, {L}].log'
        if log_file.exists():
            with open(log_file, 'r') as f:
                content = f.read()
                match = re.search(r'result:\s*([\d.]+)', content)
                if match:
                    return float(match.group(1)), None
        
        return None, "No result in log"
        
    except subprocess.TimeoutExpired:
        return None, f"Timeout ({timeout}s)"
    except Exception as e:
        return None, str(e)[:50]


def run_cost_model(K: int, L: int) -> float:
    """Run PIM Cost Model."""
    arch = PIMArchConfig()
    model = PIMCostModel(arch)
    workload = WorkloadSpec(M=1, N=L, K=K)
    result = model.estimate_gemm(workload)
    return result.total_cycles


def extract_all_workloads() -> List[dict]:
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
                layer_type = 'FC'
            elif isinstance(layer, ConvLayer):
                K = layer.nifm * layer.hfil * layer.wfil
                L = layer.nofm
                layer_type = 'Conv'
            else:
                continue
            
            workloads.append({
                'network': net_name,
                'layer': layer_name,
                'type': layer_type,
                'K': K,
                'L': L,
                'MACs': K * L,
                'L_is_pow2': is_power_of_2(L),
                'K_is_pow2': is_power_of_2(K),
            })
    
    return workloads


def main():
    print("=" * 90)
    print("Comprehensive UniNDP Validation for nn_dataflow Networks")
    print("=" * 90)
    
    # Extract all workloads
    all_workloads = extract_all_workloads()
    print(f"\nTotal workloads extracted: {len(all_workloads)}")
    
    # Filter for workloads likely to work with UniNDP
    # UniNDP requires l_block == 4, which depends on L being divisible by certain factors
    # Based on testing, power-of-2 L values tend to work
    likely_compatible = [w for w in all_workloads 
                         if (w['L_is_pow2'] or w['L'] % 32 == 0) and w['K'] >= 64 and w['L'] >= 64]
    
    print(f"Workloads likely compatible with UniNDP: {len(likely_compatible)}")
    
    # Deduplicate by (K, L) to avoid redundant tests
    unique_dims = {}
    for w in likely_compatible:
        key = (w['K'], w['L'])
        if key not in unique_dims:
            unique_dims[key] = w
    
    workloads_to_test = list(unique_dims.values())
    workloads_to_test.sort(key=lambda x: -x['MACs'])  # Largest first
    
    print(f"Unique (K, L) pairs to test: {len(workloads_to_test)}")
    
    # Test each workload
    results = []
    successful = 0
    
    print("\n" + "-" * 90)
    print(f"{'Network':<15} {'Layer':<15} {'Type':<6} {'K':<8} {'L':<8} "
          f"{'Status':<10} {'UniNDP':<12} {'CostModel':<12} {'Error%':<8}")
    print("-" * 90)
    
    for i, w in enumerate(workloads_to_test[:50]):  # Test up to 50 unique workloads
        # Run UniNDP
        unindp_cycles, error = run_unindp(w['K'], w['L'], timeout=60)
        
        # Run cost model
        cm_cycles = run_cost_model(w['K'], w['L'])
        
        if unindp_cycles is not None:
            successful += 1
            status = "✓"
            error_pct = abs(unindp_cycles - cm_cycles) / cm_cycles * 100 if cm_cycles > 0 else 0
            print(f"{w['network']:<15} {w['layer']:<15} {w['type']:<6} {w['K']:<8} {w['L']:<8} "
                  f"{status:<10} {unindp_cycles:<12,.0f} {cm_cycles:<12,.0f} {error_pct:<8.1f}")
        else:
            status = "✗"
            error_pct = None
            print(f"{w['network']:<15} {w['layer']:<15} {w['type']:<6} {w['K']:<8} {w['L']:<8} "
                  f"{status:<10} {'N/A':<12} {cm_cycles:<12,.0f} {error[:20] if error else 'Unknown'}")
        
        results.append({
            **w,
            'unindp_cycles': unindp_cycles,
            'cost_model_cycles': cm_cycles,
            'error_pct': error_pct,
            'unindp_error': error,
            'success': unindp_cycles is not None
        })
    
    # Summary
    print("\n" + "=" * 90)
    print("VALIDATION SUMMARY")
    print("=" * 90)
    
    tested = len(results)
    print(f"Workloads tested: {tested}")
    print(f"Successful: {successful} ({successful/tested*100:.1f}%)")
    print(f"Failed: {tested - successful}")
    
    if successful > 0:
        successful_results = [r for r in results if r['success']]
        errors = [r['error_pct'] for r in successful_results if r['error_pct'] is not None]
        
        if errors:
            print(f"\nError Statistics (UniNDP vs Cost Model):")
            print(f"  Average: {sum(errors)/len(errors):.2f}%")
            print(f"  Min: {min(errors):.2f}%")
            print(f"  Max: {max(errors):.2f}%")
        
        # Show successful workloads by type
        print("\nSuccessful workloads by network:")
        by_network = {}
        for r in successful_results:
            net = r['network']
            if net not in by_network:
                by_network[net] = []
            by_network[net].append(r)
        
        for net, layers in sorted(by_network.items()):
            print(f"  {net}: {len(layers)} layers")
            for l in layers[:3]:
                print(f"    - {l['layer']}: K={l['K']}, L={l['L']}, error={l['error_pct']:.1f}%")
    
    # Save results
    output_file = PROJECT_ROOT / 'examples' / 'comprehensive_validation.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    main()
