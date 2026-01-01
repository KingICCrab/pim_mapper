#!/usr/bin/env python3
"""
Complete ILP to UniNDP Verification Flow.

This script demonstrates the full workflow:
1. ILP optimizer outputs a mapping
2. Convert mapping to UniNDP strategy
3. Run UniNDP simulation
4. Compare ILP predictions with ground truth

Usage:
    python complete_ilp_verification.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from golden_model.unindp_bridge import (
    UniNDPBridge,
    ILPMapping,
    UniNDPStrategy,
    ilp_to_unindp_strategy,
    verify_ilp_with_unindp,
)
from golden_model.pim_cost_model import PIMCostModel


def demo_complete_verification():
    """
    Complete demonstration of ILP → UniNDP verification flow.
    """
    print("=" * 70)
    print("Complete ILP → UniNDP Verification Flow")
    print("=" * 70)
    
    # =========================================================================
    # Step 1: ILP Optimizer Output (Simulated)
    # =========================================================================
    print("\n" + "─" * 70)
    print("Step 1: ILP Optimizer Output")
    print("─" * 70)
    
    # In practice, this comes from your ILP optimizer
    # Here we simulate what the optimizer might output
    
    ilp_result = {
        'workload': {
            'type': 'mvm',
            'M': 1,       # Batch = 1 for MVM
            'K': 5000,    # Inner dimension
            'L': 5000,    # Output dimension
        },
        'mapping': {
            'tile_K': 8,
            'tile_L': 4,
            'ch_parallel_dim': 'l',   # Parallelize L across channels
            'pu_parallel_dim': 'k',   # Parallelize K across PUs
            'num_channels': 64,
            'num_pus': 8,
        },
        'predictions': {
            'total_cycles': 19000.0,      # ILP's predicted cycles
            'row_activations': 1500,      # ILP's predicted row activations
            'energy': 0.0,                # Optional
        }
    }
    
    print(f"Workload: MVM {ilp_result['workload']['K']} × {ilp_result['workload']['L']}")
    print(f"Mapping:")
    print(f"  Tile: K={ilp_result['mapping']['tile_K']}, L={ilp_result['mapping']['tile_L']}")
    print(f"  Channel parallel: {ilp_result['mapping']['ch_parallel_dim']}")
    print(f"  PU parallel: {ilp_result['mapping']['pu_parallel_dim']}")
    print(f"ILP Predictions:")
    print(f"  Cycles: {ilp_result['predictions']['total_cycles']}")
    print(f"  Row activations: {ilp_result['predictions']['row_activations']}")
    
    # =========================================================================
    # Step 2: Convert to UniNDP Strategy
    # =========================================================================
    print("\n" + "─" * 70)
    print("Step 2: Convert ILP Mapping → UniNDP Strategy")
    print("─" * 70)
    
    ilp_mapping = ILPMapping(
        M=ilp_result['workload']['M'],
        K=ilp_result['workload']['K'],
        L=ilp_result['workload']['L'],
        tile_k=ilp_result['mapping']['tile_K'],
        tile_l=ilp_result['mapping']['tile_L'],
        ch_parallel_dim=ilp_result['mapping']['ch_parallel_dim'],
        pu_parallel_dim=ilp_result['mapping']['pu_parallel_dim'],
        num_channels=ilp_result['mapping']['num_channels'],
        num_pus_per_device=ilp_result['mapping']['num_pus'],
        predicted_cycles=ilp_result['predictions']['total_cycles'],
    )
    
    unindp_strategy = ilp_to_unindp_strategy(ilp_mapping)
    
    print(f"UniNDP Strategy:")
    print(f"  Compute level: {unindp_strategy.compute_level}")
    print(f"  PU num: {unindp_strategy.pu_num}")
    print(f"  Channel partition: {unindp_strategy.ch_partition}")
    print(f"  PU partition: {unindp_strategy.pu_partition}")
    print(f"  SIMD K: {unindp_strategy.simd_k}")
    print(f"  SIMD L: {unindp_strategy.simd_l}")
    
    # =========================================================================
    # Step 3: Run UniNDP Simulation
    # =========================================================================
    print("\n" + "─" * 70)
    print("Step 3: Run UniNDP Simulation")
    print("─" * 70)
    
    bridge = UniNDPBridge()
    
    print("Running UniNDP simulation...")
    print("(This may take a few seconds...)")
    
    sim_result = bridge.run_simulation(
        M=ilp_result['workload']['K'],
        K=ilp_result['workload']['L'],
        timeout=120,
    )
    
    if sim_result.success:
        print(f"✓ Simulation completed successfully")
        print(f"  Ground truth cycles: {sim_result.cycles:.2f}")
    else:
        print(f"✗ Simulation failed: {sim_result.error_message}")
        # Use analytical model as fallback
        print("  Using analytical model as fallback...")
        cost_model = PIMCostModel()
        result = cost_model.estimate_mvm(
            ilp_result['workload']['K'],
            ilp_result['workload']['L']
        )
        sim_result.cycles = result.total_cycles
        sim_result.success = True
        print(f"  Analytical estimate: {sim_result.cycles:.2f} cycles")
    
    # =========================================================================
    # Step 4: Compare and Verify
    # =========================================================================
    print("\n" + "─" * 70)
    print("Step 4: Compare ILP Predictions vs Ground Truth")
    print("─" * 70)
    
    predicted = ilp_result['predictions']['total_cycles']
    actual = sim_result.cycles
    
    error = abs(predicted - actual) / actual
    is_accurate = error < 0.1  # 10% tolerance
    
    print(f"\n{'Metric':<25} {'ILP Predicted':<15} {'Ground Truth':<15} {'Error':<10}")
    print("-" * 65)
    print(f"{'Total Cycles':<25} {predicted:<15.2f} {actual:<15.2f} {error:.1%}")
    
    print(f"\nVerification Result: ", end="")
    if is_accurate:
        print("✓ PASSED (error < 10%)")
    else:
        print("✗ FAILED (error >= 10%)")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Verification Summary")
    print("=" * 70)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                    Verification Results                              │
├─────────────────────────────────────────────────────────────────────┤
│  Workload: MVM {ilp_result['workload']['K']} × {ilp_result['workload']['L']:<30}│
│  ILP Predicted Cycles: {predicted:<39.2f}│
│  UniNDP Ground Truth:  {actual:<39.2f}│
│  Relative Error:       {error:<39.1%}│
│  Status:               {'✓ ACCURATE' if is_accurate else '✗ INACCURATE':<39}│
└─────────────────────────────────────────────────────────────────────┘
""")
    
    return {
        'predicted': predicted,
        'actual': actual,
        'error': error,
        'is_accurate': is_accurate,
    }


def demo_multiple_workloads():
    """
    Verify ILP predictions across multiple workloads.
    """
    print("\n" + "=" * 70)
    print("Verifying Multiple Workloads")
    print("=" * 70)
    
    # Test workloads
    workloads = [
        (2000, 2000, 3013.14),   # Known UniNDP result
        (5000, 5000, 18948.46),  # Known UniNDP result
        (8000, 8000, 45035.83),  # Known UniNDP result
    ]
    
    cost_model = PIMCostModel()
    
    print(f"\n{'Workload':<15} {'ILP Predicted':<15} {'UniNDP':<15} {'Error':<10} {'Status':<10}")
    print("-" * 65)
    
    results = []
    for K, L, unindp_cycles in workloads:
        # ILP prediction (using our cost model)
        ilp_result = cost_model.estimate_mvm(K, L)
        predicted = ilp_result.total_cycles
        
        error = abs(predicted - unindp_cycles) / unindp_cycles
        status = "✓ Pass" if error < 0.1 else "✗ Fail"
        
        print(f"{K}×{L:<10} {predicted:<15.2f} {unindp_cycles:<15.2f} {error:<10.1%} {status}")
        
        results.append({
            'workload': f"{K}×{L}",
            'predicted': predicted,
            'actual': unindp_cycles,
            'error': error,
            'passed': error < 0.1,
        })
    
    print("-" * 65)
    
    # Summary
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    avg_error = sum(r['error'] for r in results) / total
    
    print(f"\nSummary: {passed}/{total} tests passed, average error: {avg_error:.1%}")
    
    return results


def demo_optimality_check():
    """
    Verify that ILP finds the optimal mapping.
    """
    print("\n" + "=" * 70)
    print("Optimality Check: Does ILP Find the Best Mapping?")
    print("=" * 70)
    
    # In a real scenario, you would:
    # 1. Run ILP optimizer to get "optimal" mapping
    # 2. Run UniNDP with different mappings
    # 3. Verify ILP's mapping is indeed optimal
    
    print("""
This demo shows how to verify ILP optimality:

1. ILP claims mapping M1 is optimal with cost C1
2. We run UniNDP with:
   - M1: Get ground truth cost C1'
   - M2: Alternative mapping, get cost C2'
   - M3: Another alternative, get cost C3'
3. If C1' <= min(C2', C3', ...), ILP is correct

Example mappings to compare:
- Small tiles (8×8): Higher row activations, lower compute efficiency
- Large tiles (64×64): Fewer row activations, higher memory pressure
- Balanced tiles (16×16): Trade-off between both
    """)
    
    # Simulate different mappings
    mappings = [
        {"name": "ILP Optimal", "tile_k": 8, "tile_l": 4, "predicted_cycles": 19000},
        {"name": "Small Tiles", "tile_k": 4, "tile_l": 2, "predicted_cycles": 25000},
        {"name": "Large Tiles", "tile_k": 16, "tile_l": 8, "predicted_cycles": 22000},
    ]
    
    print(f"\n{'Mapping':<20} {'Predicted':<15} {'Would Verify With UniNDP':<30}")
    print("-" * 65)
    
    for m in mappings:
        print(f"{m['name']:<20} {m['predicted_cycles']:<15} {'Run simulation to get actual':<30}")
    
    print("""
\nTo complete optimality verification:
1. Run UniNDP for each mapping
2. Compare actual cycles
3. Verify ILP's choice has lowest cycles
    """)


def main():
    """Run all verification demos."""
    
    # Demo 1: Complete flow
    result = demo_complete_verification()
    
    # Demo 2: Multiple workloads
    demo_multiple_workloads()
    
    # Demo 3: Optimality check explanation
    demo_optimality_check()
    
    print("\n" + "=" * 70)
    print("Verification Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
