#!/usr/bin/env python3
import sys
import os
import numpy as np
sys.path.insert(0, "src")

from pim_optimizer import PIMOptimizer, PIMArchitecture, ConvWorkload

def main():
    # 1. Create Architecture
    arch = PIMArchitecture()
    
    # 2. Create Workload (Tiny)
    # Tiny workload to fit in license
    workload = ConvWorkload(
        name="Tiny",
        R=3, S=3,
        P=4, Q=4,
        C=16, K=16,
        N=1,
        stride=(1, 1),
        dilation=(1, 1),
    )
    
    # 3. Run Optimization
    optimizer = PIMOptimizer(
        arch=arch,
        verbose=True,
        time_limit=60.0,
        mip_gap=0.05,
    )
    
    result = optimizer.optimize(
        workloads=[workload],
        objective="latency",
        enable_row_activation=True,
        optimize_bypass=True, # Allow bypass optimization
    )
    
    # 4. Analyze Output (t=2) at GlobalBuffer (m=1)
    mapping = result.mappings[0]
    
    print("\n" + "="*60)
    print("DEBUG ANALYSIS: Output Tensor @ GlobalBuffer")
    print("="*60)
    
    gb_idx = arch.mem_idx["GlobalBuffer"]
    rb_idx = arch.mem_idx["RowBuffer"]
    
    gb_level = arch.hierarchy[gb_idx]
    rb_level = arch.hierarchy[rb_idx]
    
    print(f"GlobalBuffer Capacity: {gb_level.entries} entries")
    print(f"RowBuffer Capacity:    {rb_level.entries} entries")
    
    # Check Bypass Variable xd
    if optimizer.vars and (0, gb_idx, 2) in optimizer.vars.xd:
        xd_var = optimizer.vars.xd[0, gb_idx, 2]
        print(f"\nBypass Variable xd[GlobalBuffer, Output]: {xd_var.X}")
        if xd_var.X < 0.5:
            print("  -> GlobalBuffer is BYPASSED for Output")
        else:
            print("  -> GlobalBuffer is USED for Output")
    else:
        print("\nBypass Variable xd not found")

    # Get tile sizes from mapping
    print("\nChosen Tile Sizes:")
    for level_name, dims in mapping.tile_sizes.items():
        print(f"  {level_name}: {dims}")
        
    # Calculate Output Tile Size at GlobalBuffer level
    # Tile at Level L is formed by loops at 0...L-1
    
    k_l0 = mapping.loop_factors.get("PELocalBuffer", {}).get("K", 1)
    p_l0 = mapping.loop_factors.get("PELocalBuffer", {}).get("P", 1)
    q_l0 = mapping.loop_factors.get("PELocalBuffer", {}).get("Q", 1)
    
    output_tile_size = k_l0 * p_l0 * q_l0
    print(f"\nOutput Tile Size (Elements) from L0 loops: {output_tile_size}")
    
    if output_tile_size > gb_level.entries:
        print(f"WARNING: Output Tile ({output_tile_size}) > GlobalBuffer Capacity ({gb_level.entries})")
    else:
        print(f"OK: Output Tile ({output_tile_size}) <= GlobalBuffer Capacity ({gb_level.entries})")

if __name__ == "__main__":
    main()
