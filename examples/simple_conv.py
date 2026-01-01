#!/usr/bin/env python3
"""
Simple example demonstrating PIM optimizer usage.

This example shows:
1. Creating architecture and workload programmatically
2. Running optimization
3. Analyzing results
"""

import sys
sys.path.insert(0, "../src")

from pim_optimizer import (
    PIMOptimizer,
    PIMArchitecture,
    ConvWorkload,
)
from pim_optimizer.model.crossing import (
    compute_block_crossing_ratio_gcd,
    analyze_crossing_pattern,
)


def main():
    print("=" * 60)
    print("PIM Optimizer - Simple Example")
    print("=" * 60)
    
    # =========================================
    # Step 1: Create architecture
    # =========================================
    print("\n1. Creating architecture...")
    
    # Use default PIM architecture
    arch = PIMArchitecture()
    arch.print_info()
    
    # =========================================
    # Step 2: Create workloads
    # =========================================
    print("\n2. Creating workloads...")
    
    # Simple 3x3 convolution
    workload = ConvWorkload(
        name="conv3x3_64",
        R=3, S=3,           # 3x3 filter
        P=56, Q=56,         # 56x56 output
        C=64, K=128,        # 64->128 channels
        N=1,                # Batch size 1
        stride=(1, 1),
        dilation=(1, 1),
    )
    
    print(f"Workload: {workload.name}")
    print(f"  Dimensions: R={workload.R}, S={workload.S}, P={workload.P}, Q={workload.Q}")
    print(f"  Channels: C={workload.C}, K={workload.K}, N={workload.N}")
    print(f"  MACs: {workload.macs:,}")
    print(f"  Input size: H={workload.input_size['H']}, W={workload.input_size['W']}")
    
    # =========================================
    # Step 3: Analyze crossing ratios
    # =========================================
    print("\n3. Crossing ratio analysis...")
    
    # Example: tile height 7, block height 64, step = 4 (q_factor × stride)
    tile_h = 7
    block_h = 64
    step = 4
    
    crossing, g, period, cross_count = compute_block_crossing_ratio_gcd(
        block_h=block_h,
        tile_h=tile_h,
        step=step,
    )
    
    print(f"  Block height: {block_h}")
    print(f"  Tile height: {tile_h}")
    print(f"  Step (q_factor × stride): {step}")
    print(f"  GCD: {g}, Period: {period}")
    print(f"  Crossing ratio: {cross_count}/{period} = {crossing:.4f}")
    
    # =========================================
    # Step 4: Run optimization (if Gurobi available)
    # =========================================
    print("\n4. Running optimization...")
    
    try:
        optimizer = PIMOptimizer(
            arch=arch,
            verbose=False,
            time_limit=60.0,  # 1 minute time limit
            mip_gap=0.05,     # 5% gap tolerance
        )
        
        result = optimizer.optimize(
            workloads=[workload],
            objective="latency",
            enable_row_activation=True,
        )
        
        print(f"  Solver status: {result.solver_status}")
        print(f"  Solve time: {result.solve_time:.2f}s")
        
        if result.is_optimal or result.solver_status == "time_limit":
            print(f"  Total latency: {result.total_latency:.6e}")
            
            # Print mapping details
            print("\n  Mapping details:")
            mapping = result.mappings[0]
            print(mapping.pretty_print())
        else:
            print("  No solution found")
            
    except ImportError:
        print("  Gurobi not available. Skipping optimization.")
        print("  Install Gurobi and obtain a license to run optimization.")
    except Exception as e:
        print(f"  Error during optimization: {e}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
