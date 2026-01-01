#!/usr/bin/env python3
"""
Validate selected nn_dataflow workloads against UniNDP simulator.

This script attempts to run UniNDP validation on workloads that meet the
HBM-PIM constraints:
- l_block should be 4 (derived from SIMD width 32 / data size 8 = 4)
- Dimensions must be properly divisible

Strategy:
1. Try each workload with UniNDP
2. Record success/failure and cycle counts
3. Compare with PIM Cost Model predictions
"""

import sys
from pathlib import Path
import json
import subprocess
import traceback
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


@dataclass
class WorkloadTest:
    """A workload to test."""
    network: str
    layer: str
    layer_type: str
    K: int  # Input dimension
    L: int  # Output dimension
    MACs: int


def extract_suitable_workloads(min_dim: int = 128, max_workloads: int = 50) -> List[WorkloadTest]:
    """Extract workloads suitable for UniNDP testing."""
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
                MACs = K * L
                layer_type = 'FC'
            elif isinstance(layer, ConvLayer):
                K = layer.nifm * layer.hfil * layer.wfil
                L = layer.nofm
                MACs = K * L  # Per output position
                layer_type = 'Conv'
            else:
                continue
            
            # Filter by minimum dimensions
            if K >= min_dim and L >= min_dim:
                workloads.append(WorkloadTest(
                    network=net_name,
                    layer=layer_name,
                    layer_type=layer_type,
                    K=K,
                    L=L,
                    MACs=MACs
                ))
    
    # Sort by MACs (descending) and take top workloads
    workloads.sort(key=lambda w: -w.MACs)
    return workloads[:max_workloads]


def run_unindp_simulation(K: int, L: int, timeout: int = 60) -> Tuple[Optional[float], Optional[str]]:
    """
    Run UniNDP simulation for given workload dimensions.
    
    Returns:
        (cycles, error_msg): cycles if successful, error message if failed
    """
    # Use the sim_verify.py with default strategy
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
            # Check for known errors
            stderr = result.stderr
            if "AssertionError" in stderr:
                return None, "UniNDP assertion failed (likely l_block constraint)"
            return None, f"UniNDP error: {stderr[:200]}"
        
        # Results are written to log file, not stdout
        # Check the log file
        log_file = UNINDP_PATH / 'verify_result' / 'log' / f'[{K}, {L}].log'
        if log_file.exists():
            with open(log_file, 'r') as f:
                content = f.read()
                # Look for "result: <number>"
                import re
                match = re.search(r'result:\s*([\d.]+)', content)
                if match:
                    cycles = float(match.group(1))
                    return cycles, None
        
        # Also check CSV file
        csv_file = UNINDP_PATH / 'verify_result' / 'csv' / f'[{K}, {L}].csv'
        if csv_file.exists():
            with open(csv_file, 'r') as f:
                import csv as csv_module
                reader = csv_module.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        try:
                            cycles = float(row[2])
                            return cycles, None
                        except:
                            pass
        
        return None, f"Could not find result in log or csv files"
        
    except subprocess.TimeoutExpired:
        return None, f"Timeout after {timeout}s"
    except Exception as e:
        return None, f"Exception: {str(e)}"


def run_cost_model(K: int, L: int) -> float:
    """Run PIM Cost Model to get estimated cycles."""
    arch = PIMArchConfig()
    model = PIMCostModel(arch)
    workload = WorkloadSpec(M=1, N=L, K=K)
    result = model.estimate_gemm(workload)
    return result.total_cycles


def main():
    print("=" * 80)
    print("UniNDP Validation for nn_dataflow Workloads")
    print("=" * 80)
    
    # Extract suitable workloads
    workloads = extract_suitable_workloads(min_dim=128, max_workloads=30)
    print(f"\nExtracted {len(workloads)} workloads for testing:")
    
    results = []
    successful = 0
    failed = 0
    
    print("\n" + "-" * 80)
    print(f"{'Network':<15} {'Layer':<20} {'K':<8} {'L':<8} {'Status':<12} {'UniNDP':<15} {'CostModel':<15} {'Error%'}")
    print("-" * 80)
    
    for wl in workloads:
        # Get cost model prediction
        cm_cycles = run_cost_model(wl.K, wl.L)
        
        # Try UniNDP simulation
        unindp_cycles, error = run_unindp_simulation(wl.K, wl.L, timeout=30)
        
        if unindp_cycles is not None:
            successful += 1
            status = "✓"
            error_pct = abs(unindp_cycles - cm_cycles) / cm_cycles * 100 if cm_cycles > 0 else 0
            print(f"{wl.network:<15} {wl.layer:<20} {wl.K:<8} {wl.L:<8} {status:<12} {unindp_cycles:<15,.0f} {cm_cycles:<15,.0f} {error_pct:.1f}%")
        else:
            failed += 1
            status = "✗"
            print(f"{wl.network:<15} {wl.layer:<20} {wl.K:<8} {wl.L:<8} {status:<12} {'N/A':<15} {cm_cycles:<15,.0f} {error[:30] if error else 'Unknown'}")
        
        results.append({
            'network': wl.network,
            'layer': wl.layer,
            'layer_type': wl.layer_type,
            'K': wl.K,
            'L': wl.L,
            'MACs': wl.MACs,
            'unindp_cycles': unindp_cycles,
            'cost_model_cycles': cm_cycles,
            'error': error,
            'success': unindp_cycles is not None
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total workloads tested: {len(workloads)}")
    print(f"Successful UniNDP runs: {successful}")
    print(f"Failed UniNDP runs: {failed}")
    
    # If any successful, calculate error stats
    successful_results = [r for r in results if r['success']]
    if successful_results:
        errors = [abs(r['unindp_cycles'] - r['cost_model_cycles']) / r['cost_model_cycles'] * 100 
                  for r in successful_results if r['cost_model_cycles'] > 0]
        if errors:
            print(f"\nError Statistics (UniNDP vs Cost Model):")
            print(f"  Average error: {sum(errors)/len(errors):.2f}%")
            print(f"  Min error: {min(errors):.2f}%")
            print(f"  Max error: {max(errors):.2f}%")
    
    # Save results
    output_file = PROJECT_ROOT / 'examples' / 'unindp_validation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    main()
