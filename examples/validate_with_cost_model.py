#!/usr/bin/env python3
"""
Validate ILP optimizer against PIM Cost Model using nn_dataflow networks.

This script runs the ILP optimizer on real neural network workloads from
nn_dataflow and validates the results using the analytical PIM Cost Model.

The PIM Cost Model was previously validated against UniNDP cycle-accurate
simulation with ~3.5% error, so it can serve as a reliable golden model.
"""

import sys
from pathlib import Path
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT))

# Import PIM optimizer components
from pim_optimizer.optimizer import ILPOptimizer, OptimizationConfig
from pim_optimizer.mapping import ILPMapping

# Import PIM Cost Model
from golden_model.pim_cost_model import (
    PIMCostModel, PIMArchConfig, WorkloadSpec, TilingConfig, CostResult
)

# Import nn_dataflow networks
from nn_dataflow.nns import all_networks, import_network
from nn_dataflow.core import ConvLayer, FCLayer


@dataclass
class ValidationResult:
    """Result of validating one layer."""
    network: str
    layer: str
    layer_type: str
    K: int
    L: int
    ilp_cycles: Optional[float] = None
    cost_model_cycles: Optional[float] = None
    error_pct: Optional[float] = None
    ilp_time_ms: float = 0.0
    success: bool = False
    error_msg: str = ""
    
    def to_dict(self):
        return {
            'network': self.network,
            'layer': self.layer,
            'layer_type': self.layer_type,
            'K': self.K,
            'L': self.L,
            'ilp_cycles': self.ilp_cycles,
            'cost_model_cycles': self.cost_model_cycles,
            'error_pct': self.error_pct,
            'ilp_time_ms': self.ilp_time_ms,
            'success': self.success,
            'error_msg': self.error_msg
        }


def extract_workloads_from_network(net_name: str) -> List[Dict]:
    """Extract compute workloads from a network."""
    try:
        network = import_network(net_name)
    except Exception as e:
        print(f"  Failed to load network {net_name}: {e}")
        return []
    
    workloads = []
    for layer_name in network:
        layer = network[layer_name]
        
        if isinstance(layer, FCLayer):
            K = layer.nifm * layer.hofm * layer.wofm
            L = layer.nofm
            workloads.append({
                'name': layer_name,
                'type': 'FC',
                'K': K,
                'L': L,
            })
        elif isinstance(layer, ConvLayer):
            K = layer.nifm * layer.hfil * layer.wfil
            L = layer.nofm
            workloads.append({
                'name': layer_name,
                'type': 'Conv',
                'K': K,
                'L': L,
            })
    
    return workloads


def run_ilp_optimizer(K: int, L: int, timeout: float = 30.0) -> Tuple[Optional[ILPMapping], float]:
    """
    Run ILP optimizer for given workload dimensions.
    
    Returns:
        (mapping, time_ms): The optimal mapping and time taken in ms
    """
    config = OptimizationConfig(
        num_channels=64,
        num_banks_per_channel=16,
        simd_width=8,
        row_buffer_size=256,
        timeout_seconds=timeout,
        verbose=False
    )
    
    optimizer = ILPOptimizer(config)
    
    start_time = time.time()
    try:
        mapping = optimizer.optimize(K, L)
        elapsed_ms = (time.time() - start_time) * 1000
        return mapping, elapsed_ms
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        return None, elapsed_ms


def run_cost_model(K: int, L: int) -> CostResult:
    """
    Run PIM Cost Model for given workload dimensions.
    """
    arch = PIMArchConfig()
    model = PIMCostModel(arch)
    
    # For MVM (M=1), K is input dim, L (which we call N) is output dim
    workload = WorkloadSpec(M=1, N=L, K=K)
    result = model.estimate_gemm(workload)
    
    return result


def validate_layer(net_name: str, layer_name: str, layer_type: str, 
                   K: int, L: int) -> ValidationResult:
    """Validate one layer."""
    result = ValidationResult(
        network=net_name,
        layer=layer_name,
        layer_type=layer_type,
        K=K,
        L=L
    )
    
    # Run ILP optimizer
    mapping, ilp_time = run_ilp_optimizer(K, L)
    result.ilp_time_ms = ilp_time
    
    if mapping is None:
        result.error_msg = "ILP optimization failed"
        return result
    
    # Get ILP estimated cycles
    result.ilp_cycles = mapping.estimated_cycles
    
    # Run cost model
    try:
        cost_result = run_cost_model(K, L)
        result.cost_model_cycles = cost_result.total_cycles
    except Exception as e:
        result.error_msg = f"Cost model error: {e}"
        return result
    
    # Calculate error
    if result.cost_model_cycles > 0:
        result.error_pct = abs(result.ilp_cycles - result.cost_model_cycles) / result.cost_model_cycles * 100
    
    result.success = True
    return result


def validate_network(net_name: str, max_layers: int = 10) -> List[ValidationResult]:
    """Validate layers from one network."""
    workloads = extract_workloads_from_network(net_name)
    if not workloads:
        return []
    
    results = []
    for i, wl in enumerate(workloads[:max_layers]):
        result = validate_layer(
            net_name=net_name,
            layer_name=wl['name'],
            layer_type=wl['type'],
            K=wl['K'],
            L=wl['L']
        )
        results.append(result)
        
        # Print progress
        status = "✓" if result.success else "✗"
        if result.success:
            print(f"  [{status}] {wl['name']}: K={wl['K']}, L={wl['L']}, "
                  f"ILP={result.ilp_cycles:.0f}, CM={result.cost_model_cycles:.0f}, "
                  f"err={result.error_pct:.1f}%")
        else:
            print(f"  [{status}] {wl['name']}: {result.error_msg}")
    
    return results


def main():
    print("=" * 70)
    print("Neural Network Validation with ILP Optimizer & PIM Cost Model")
    print("=" * 70)
    
    all_results = []
    networks = sorted(all_networks())
    
    for net_name in networks:
        print(f"\n{net_name}:")
        results = validate_network(net_name, max_layers=5)  # Limit layers per network
        all_results.extend(results)
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    successful = [r for r in all_results if r.success]
    failed = [r for r in all_results if not r.success]
    
    print(f"\nTotal layers validated: {len(all_results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        errors = [r.error_pct for r in successful if r.error_pct is not None]
        if errors:
            avg_error = sum(errors) / len(errors)
            max_error = max(errors)
            min_error = min(errors)
            print(f"\nError Statistics (ILP vs Cost Model):")
            print(f"  Average error: {avg_error:.2f}%")
            print(f"  Min error: {min_error:.2f}%")
            print(f"  Max error: {max_error:.2f}%")
    
    if failed:
        print(f"\nFailed layers:")
        for r in failed[:10]:
            print(f"  {r.network}/{r.layer}: {r.error_msg}")
    
    # Save results to JSON
    output_file = PROJECT_ROOT / 'examples' / 'validation_results.json'
    with open(output_file, 'w') as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return all_results


if __name__ == '__main__':
    main()
