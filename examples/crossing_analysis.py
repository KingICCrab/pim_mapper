#!/usr/bin/env python3
"""
Example demonstrating crossing ratio calculations.

The crossing ratio determines how often tiles cross memory block boundaries,
which affects row activation costs in DRAM-based PIM systems.

Key Formula (GCD Periodic Analysis):
    g = gcd(step, block_h)
    period = block_h // g
    safe_positions = ceil((block_h - tile_h + 1) / g)
    cross_count = period - safe_positions
    crossing_ratio = cross_count / period

IMPORTANT: For input tiles, step = q_factor × stride (NOT just stride!)
"""

import sys
sys.path.insert(0, "../src")

from pim_optimizer.model.crossing import (
    compute_block_crossing_ratio_gcd,
    compute_input_block_crossing,
    analyze_crossing_pattern,
)


def main():
    print("=" * 60)
    print("Crossing Ratio Analysis Examples")
    print("=" * 60)
    
    # =========================================
    # Example 1: Basic crossing ratio
    # =========================================
    print("\n1. Basic Crossing Ratio")
    print("-" * 40)
    
    block_h = 8
    tile_h = 3
    step = 2
    
    crossing, g, period, cross_count = compute_block_crossing_ratio_gcd(
        block_h, tile_h, step
    )
    
    print(f"Parameters:")
    print(f"  block_h = {block_h}")
    print(f"  tile_h = {tile_h}")
    print(f"  step = {step}")
    print()
    print(f"Analysis:")
    print(f"  g = gcd({step}, {block_h}) = {g}")
    print(f"  period = {block_h} / {g} = {period}")
    print(f"  cross_count = {cross_count}")
    print(f"  crossing_ratio = {cross_count}/{period} = {crossing:.4f}")
    
    # =========================================
    # Example 2: Detailed pattern analysis
    # =========================================
    print("\n2. Detailed Pattern Analysis")
    print("-" * 40)
    
    analysis = analyze_crossing_pattern(block_h, tile_h, step)
    
    print(f"Tile positions within one period:")
    for i, (pos, crosses) in enumerate(zip(analysis["positions"], analysis["crossings"])):
        end = pos + tile_h
        status = "CROSSES" if crosses else "safe"
        bar = "|" + "█" * pos + "▓" * tile_h + " " * (block_h - pos - tile_h) + "|"
        if end > block_h:
            bar = "|" + "█" * pos + "▓" * (block_h - pos) + "│" + "▓" * (end - block_h) + " " * (block_h - end + block_h) + "|"
        print(f"  iter {i}: start={pos}, end={end} -> {status}")
    
    # =========================================
    # Example 3: Input tile crossing (stride matters!)
    # =========================================
    print("\n3. Input Tile Crossing (with stride)")
    print("-" * 40)
    
    # Conv parameters
    tile_q = 4      # Output tile Q dimension
    tile_s = 3      # Filter S dimension
    stride_h = 2    # Vertical stride
    dilation_h = 1  # Vertical dilation
    
    # Memory block
    block_h = 64
    
    # Different outer Q factors
    for outer_q_factor in [1, 2, 4, 8]:
        # Input tile height = stride × Q + dilation × S - stride - dilation + 1
        input_tile_h = stride_h * tile_q + dilation_h * tile_s - stride_h - dilation_h + 1
        
        # CRITICAL: step = outer_q_factor × stride
        step = outer_q_factor * stride_h
        
        crossing, g, period, _ = compute_block_crossing_ratio_gcd(
            block_h, input_tile_h, step
        )
        
        print(f"outer_q_factor={outer_q_factor}:")
        print(f"  input_tile_h = {stride_h}×{tile_q} + {dilation_h}×{tile_s} - {stride_h} - {dilation_h} + 1 = {input_tile_h}")
        print(f"  step = {outer_q_factor} × {stride_h} = {step}")
        print(f"  crossing_ratio = {crossing:.4f}")
        print()
    
    # =========================================
    # Example 4: Compare different tile sizes
    # =========================================
    print("\n4. Crossing Ratio vs Tile Size")
    print("-" * 40)
    
    block_h = 64
    step = 4
    
    print(f"block_h = {block_h}, step = {step}")
    print()
    print(f"{'tile_h':<10} {'crossing':<10} {'formula':<20}")
    print("-" * 40)
    
    for tile_h in [1, 2, 4, 8, 16, 32, 48, 63, 64, 65]:
        crossing, g, period, cross_count = compute_block_crossing_ratio_gcd(
            block_h, tile_h, step
        )
        formula = f"{cross_count}/{period}"
        print(f"{tile_h:<10} {crossing:<10.4f} {formula:<20}")
    
    # =========================================
    # Example 5: Real-world scenario
    # =========================================
    print("\n5. Real-World Scenario: ResNet-50 Conv Layer")
    print("-" * 40)
    
    # ResNet-50 conv3x3 parameters
    print("Layer: ResNet-50 conv3_1 (56x56, 3x3, stride=1)")
    
    # DRAM parameters
    block_h = 64  # Typical DRAM row height in elements
    block_w = 256
    
    # Different mapping strategies
    strategies = [
        ("Large Q tile", {"tile_q": 56, "tile_s": 3, "outer_q": 1, "stride": 1}),
        ("Medium Q tile", {"tile_q": 14, "tile_s": 3, "outer_q": 4, "stride": 1}),
        ("Small Q tile", {"tile_q": 7, "tile_s": 3, "outer_q": 8, "stride": 1}),
        ("Tiny Q tile", {"tile_q": 2, "tile_s": 3, "outer_q": 28, "stride": 1}),
    ]
    
    for name, params in strategies:
        input_tile_h = params["stride"] * params["tile_q"] + params["tile_s"] - params["stride"]
        step = params["outer_q"] * params["stride"]
        
        crossing, _, _, _ = compute_block_crossing_ratio_gcd(block_h, input_tile_h, step)
        
        print(f"\n{name}:")
        print(f"  tile_q = {params['tile_q']}, outer_q = {params['outer_q']}")
        print(f"  input_tile_h = {input_tile_h}")
        print(f"  step = {step}")
        print(f"  crossing_ratio = {crossing:.4f}")
        print(f"  Extra row activations = +{crossing*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
