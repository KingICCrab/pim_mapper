"""
Validate ILP → UniNDP flow using all networks from nn_dataflow/nns.

This script loads all network definitions from nn_dataflow and extracts
the workloads (Conv, FC layers) to validate our ILP optimizer against
UniNDP simulation.
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import json
import time

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))  # Add project root to find nn_dataflow

from nn_dataflow.nns import all_networks, import_network
from nn_dataflow.core import Network, ConvLayer, FCLayer, PoolingLayer, EltwiseLayer


@dataclass
class WorkloadSpec:
    """Specification for a single workload (layer)."""
    network_name: str
    layer_name: str
    layer_type: str  # 'conv', 'fc', 'pool', 'eltwise'
    
    # For GEMM/MVM: K (input dim), L (output dim)
    K: int = 0
    L: int = 0
    
    # For Conv: additional parameters
    ifmap_size: int = 0
    ofmap_size: int = 0
    filter_size: int = 0
    stride: int = 1
    
    # Computed MACs
    macs: int = 0
    
    def __str__(self):
        if self.layer_type == 'fc':
            return f"{self.network_name}/{self.layer_name}: FC({self.K}→{self.L}), MACs={self.macs:,}"
        elif self.layer_type == 'conv':
            return f"{self.network_name}/{self.layer_name}: Conv(K={self.K}, L={self.L}, {self.filter_size}x{self.filter_size}), MACs={self.macs:,}"
        else:
            return f"{self.network_name}/{self.layer_name}: {self.layer_type}"


def extract_workloads_from_network(network: Network) -> List[WorkloadSpec]:
    """Extract all compute workloads from a network."""
    workloads = []
    
    for layer_name in network:
        layer = network[layer_name]
        
        if isinstance(layer, FCLayer):
            # FC layer: GEMM of (1, K) x (K, L) -> (1, L)
            # K = nifm * hofm * wofm (flattened input)
            # L = nofm (output features)
            K = layer.nifm * layer.hofm * layer.wofm
            L = layer.nofm
            macs = K * L
            
            workloads.append(WorkloadSpec(
                network_name=network.net_name,
                layer_name=layer_name,
                layer_type='fc',
                K=K,
                L=L,
                macs=macs,
            ))
            
        elif isinstance(layer, ConvLayer):
            # Conv layer: can be viewed as GEMM
            # K = nifm * filter_h * filter_w (input channels * filter size)
            # L = nofm (output channels)
            # Repeated for each output spatial position
            K = layer.nifm * layer.hfil * layer.wfil
            L = layer.nofm
            output_positions = layer.hofm * layer.wofm
            macs = K * L * output_positions
            
            workloads.append(WorkloadSpec(
                network_name=network.net_name,
                layer_name=layer_name,
                layer_type='conv',
                K=K,
                L=L,
                ifmap_size=layer.hifm,
                ofmap_size=layer.hofm,
                filter_size=layer.hfil,
                stride=layer.htrd,
                macs=macs,
            ))
            
        elif isinstance(layer, PoolingLayer):
            workloads.append(WorkloadSpec(
                network_name=network.net_name,
                layer_name=layer_name,
                layer_type='pool',
            ))
            
        elif isinstance(layer, EltwiseLayer):
            workloads.append(WorkloadSpec(
                network_name=network.net_name,
                layer_name=layer_name,
                layer_type='eltwise',
            ))
    
    return workloads


def load_all_workloads() -> Dict[str, List[WorkloadSpec]]:
    """Load workloads from all networks."""
    all_workloads = {}
    
    networks = all_networks()
    print(f"Found {len(networks)} networks: {networks}")
    
    for net_name in networks:
        try:
            network = import_network(net_name)
            workloads = extract_workloads_from_network(network)
            all_workloads[net_name] = workloads
            print(f"  {net_name}: {len(workloads)} layers")
        except Exception as e:
            print(f"  {net_name}: Failed to load - {e}")
    
    return all_workloads


def run_validation_experiments(all_workloads: Dict[str, List[WorkloadSpec]], 
                                max_workloads_per_network: int = 5,
                                skip_pool_eltwise: bool = True):
    """
    Run validation experiments using UniNDP.
    
    Args:
        all_workloads: Dictionary of network -> workloads
        max_workloads_per_network: Max layers to test per network (for speed)
        skip_pool_eltwise: Skip pooling and eltwise layers (not compute-intensive)
    """
    from golden_model.unindp_bridge import UniNDPBridge, UniNDPStrategy, ilp_to_unindp_strategy, ILPMapping
    
    bridge = UniNDPBridge()
    
    results = []
    total_tested = 0
    total_success = 0
    
    print("\n" + "=" * 80)
    print("Running Validation Experiments")
    print("=" * 80)
    
    for net_name, workloads in all_workloads.items():
        print(f"\n{'='*60}")
        print(f"Network: {net_name}")
        print(f"{'='*60}")
        
        # Filter and limit workloads
        compute_workloads = [w for w in workloads 
                           if w.layer_type in ('fc', 'conv') or not skip_pool_eltwise]
        
        if max_workloads_per_network > 0:
            compute_workloads = compute_workloads[:max_workloads_per_network]
        
        for workload in compute_workloads:
            if workload.layer_type not in ('fc', 'conv'):
                continue
                
            total_tested += 1
            
            # For validation, we use K and L dimensions
            K = workload.K
            L = workload.L
            
            # Skip very small workloads (not representative for PIM)
            if K < 64 or L < 64:
                print(f"  {workload.layer_name}: Skipped (too small: K={K}, L={L})")
                continue
            
            # Create ILP mapping
            ilp_mapping = ILPMapping(
                K=K, L=L,
                tile_k=8, tile_l=4,
                ch_parallel_dim='l',
                pu_parallel_dim='l',
                num_channels=64,
                num_pus_per_device=8,
            )
            
            # Convert to UniNDP strategy
            strategy = ilp_to_unindp_strategy(ilp_mapping)
            
            # Run simulation
            start_time = time.time()
            result = bridge.run_simulation(M=K, K=L, strategy=strategy)
            elapsed = time.time() - start_time
            
            if result.success:
                total_success += 1
                print(f"  ✓ {workload.layer_name}: K={K}, L={L} -> {result.cycles:.2f} cycles ({elapsed:.2f}s)")
                results.append({
                    'network': net_name,
                    'layer': workload.layer_name,
                    'type': workload.layer_type,
                    'K': K,
                    'L': L,
                    'macs': workload.macs,
                    'cycles': result.cycles,
                    'success': True,
                })
            else:
                print(f"  ✗ {workload.layer_name}: K={K}, L={L} -> Failed: {result.error_message}")
                results.append({
                    'network': net_name,
                    'layer': workload.layer_name,
                    'type': workload.layer_type,
                    'K': K,
                    'L': L,
                    'macs': workload.macs,
                    'success': False,
                    'error': result.error_message,
                })
    
    return results, total_tested, total_success


def print_summary(results: List[Dict], total_tested: int, total_success: int):
    """Print summary of validation results."""
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal layers tested: {total_tested}")
    print(f"Successful: {total_success}")
    print(f"Failed: {total_tested - total_success}")
    print(f"Success rate: {total_success/total_tested*100:.1f}%" if total_tested > 0 else "N/A")
    
    # Group by network
    by_network = {}
    for r in results:
        net = r['network']
        if net not in by_network:
            by_network[net] = {'success': 0, 'total': 0, 'cycles': []}
        by_network[net]['total'] += 1
        if r.get('success'):
            by_network[net]['success'] += 1
            by_network[net]['cycles'].append(r['cycles'])
    
    print(f"\n{'Network':<20} {'Success':<10} {'Avg Cycles':<15}")
    print("-" * 50)
    for net, stats in sorted(by_network.items()):
        avg_cycles = sum(stats['cycles']) / len(stats['cycles']) if stats['cycles'] else 0
        print(f"{net:<20} {stats['success']}/{stats['total']:<10} {avg_cycles:,.2f}")
    
    # Successful results table
    successful = [r for r in results if r.get('success')]
    if successful:
        print(f"\n{'='*80}")
        print("Detailed Results (Successful)")
        print(f"{'='*80}")
        print(f"{'Network':<15} {'Layer':<20} {'K':<8} {'L':<8} {'MACs':<15} {'Cycles':<12}")
        print("-" * 80)
        for r in successful:
            print(f"{r['network']:<15} {r['layer']:<20} {r['K']:<8} {r['L']:<8} {r['macs']:,<15} {r['cycles']:,.2f}")


def save_results(results: List[Dict], filename: str = "validation_results.json"):
    """Save results to JSON file."""
    output_path = PROJECT_ROOT / "examples" / filename
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    print("=" * 80)
    print("Neural Network Workload Validation for PIM Optimizer")
    print("=" * 80)
    
    # Load all workloads
    print("\nLoading networks from nn_dataflow/nns...")
    all_workloads = load_all_workloads()
    
    # Count total compute layers
    total_layers = sum(len(wl) for wl in all_workloads.values())
    compute_layers = sum(
        sum(1 for w in wl if w.layer_type in ('fc', 'conv'))
        for wl in all_workloads.values()
    )
    print(f"\nTotal layers: {total_layers}")
    print(f"Compute layers (FC/Conv): {compute_layers}")
    
    # Print workload summary
    print("\n" + "=" * 80)
    print("Workload Summary")
    print("=" * 80)
    for net_name, workloads in all_workloads.items():
        print(f"\n{net_name}:")
        for w in workloads:
            if w.layer_type in ('fc', 'conv'):
                print(f"  {w}")
    
    # Run validation
    print("\n" + "=" * 80)
    print("Starting Validation...")
    print("=" * 80)
    
    results, total_tested, total_success = run_validation_experiments(
        all_workloads,
        max_workloads_per_network=10,  # Test up to 10 layers per network
        skip_pool_eltwise=True,
    )
    
    # Print summary
    print_summary(results, total_tested, total_success)
    
    # Save results
    save_results(results)
    
    print("\n" + "=" * 80)
    print("Validation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
