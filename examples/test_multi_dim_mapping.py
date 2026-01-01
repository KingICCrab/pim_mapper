#!/usr/bin/env python3
"""Test multi-dimension mapping analysis."""

import numpy as np

def analyze_multi_dim_mapping(h_dims, w_dims, pe_array_h=16, pe_array_w=16):
    """Analyze a multi-dimension mapping configuration."""
    h_total = np.prod(list(h_dims.values())) if h_dims else 1
    w_total = np.prod(list(w_dims.values())) if w_dims else 1
    
    # Dimension relevancy for convolution
    relevancy = {
        'R': {'Input': 1, 'Weight': 1, 'Output': 0},
        'S': {'Input': 1, 'Weight': 1, 'Output': 0},
        'P': {'Input': 1, 'Weight': 0, 'Output': 1},
        'Q': {'Input': 1, 'Weight': 0, 'Output': 1},
        'C': {'Input': 1, 'Weight': 1, 'Output': 0},
        'K': {'Input': 0, 'Weight': 1, 'Output': 1},
        'N': {'Input': 1, 'Weight': 0, 'Output': 1},
    }
    reduction_axes = {'R', 'S', 'C'}
    valid = h_total <= pe_array_h and w_total <= pe_array_w
    
    # Calculate bandwidth
    bandwidth = {}
    for dtype in ['Input', 'Weight', 'Output']:
        h_bw, w_bw = 1, 1
        for dim, par in h_dims.items():
            if relevancy.get(dim, {}).get(dtype, 0) == 1:
                h_bw *= par
        for dim, par in w_dims.items():
            if relevancy.get(dim, {}).get(dtype, 0) == 1:
                w_bw *= par
        bandwidth[dtype] = {'h': h_bw, 'w': w_bw, 'with_broadcast': h_bw}
    
    # Calculate reduction
    h_red, w_red = 1, 1
    for dim, par in h_dims.items():
        if dim in reduction_axes:
            h_red *= par
    for dim, par in w_dims.items():
        if dim in reduction_axes:
            w_red *= par
    
    return {
        'h_dims': h_dims, 'w_dims': w_dims,
        'h_total': int(h_total), 'w_total': int(w_total),
        'total_par': int(h_total * w_total), 'valid': valid,
        'bandwidth': bandwidth,
        'reduction': {
            'h': h_red, 'w': w_red, 'total': h_red * w_red,
            'depth': int(np.ceil(np.log2(max(h_red * w_red, 1)))),
            'is_2d': h_red > 1 and w_red > 1
        }
    }


def print_analysis(title, a):
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(f"H direction: {a['h_dims']} → total = {a['h_total']}")
    print(f"W direction: {a['w_dims']} → total = {a['w_total']}")
    print(f"Total Parallelism: {a['total_par']} ({a['h_total']}×{a['w_total']})")
    print(f"Valid for 16×16 PE array: {a['valid']}")
    
    print("\nBandwidth (with row broadcast):")
    for dt, bw in a['bandwidth'].items():
        print(f"  {dt:8s}: H-unicast={bw['h']:2d}, W-unicast={bw['w']:2d}, effective={bw['with_broadcast']}")
    
    r = a['reduction']
    print(f"\nReduction:")
    if r['total'] > 1:
        print(f"  H-reduction: {r['h']}×")
        print(f"  W-reduction: {r['w']}×")
        print(f"  Total: {r['total']}× → depth={r['depth']} stages")
        if r['is_2d']:
            print("  ⚠️  2D REDUCTION REQUIRED (complex)")
    else:
        print("  No reduction needed ✓")
    print()


if __name__ == "__main__":
    print("\n" + "🔷" * 30 + "\n")
    print("Multi-Dimension Mapping Analysis Examples")
    print("PE Array: 16×16 = 256 PEs")
    print("\n" + "🔷" * 30 + "\n")
    
    # Example 1: Classic Weight Stationary
    print_analysis(
        "Example 1: Weight Stationary (K in H, P×Q in W)",
        analyze_multi_dim_mapping({'K': 16}, {'P': 4, 'Q': 4})
    )
    
    # Example 2: Multi-dimension in H
    print_analysis(
        "Example 2: Multi-dim to H (K=4 AND C=4 both in H direction)",
        analyze_multi_dim_mapping({'K': 4, 'C': 4}, {'P': 4, 'Q': 4})
    )
    
    # Example 3: Split dimension (2D reduction)
    print_analysis(
        "Example 3: Split C across H and W (C_h=16, C_w=16)",
        analyze_multi_dim_mapping({'C': 16}, {'C': 16})
    )
    
    # Example 4: Output Stationary
    print_analysis(
        "Example 4: Output Stationary (P,N in H; Q,K in W)",
        analyze_multi_dim_mapping({'P': 4, 'N': 4}, {'Q': 4, 'K': 4})
    )
    
    # Example 5: Input Stationary (C in H, K in W)
    print_analysis(
        "Example 5: Input Stationary (C in H, K in W) - needs reduction",
        analyze_multi_dim_mapping({'C': 16}, {'K': 16})
    )
    
    # Example 6: Row Stationary style (R,S in H)
    print_analysis(
        "Example 6: Row Stationary (R×S in H, K×P in W) - needs reduction",
        analyze_multi_dim_mapping({'R': 3, 'S': 3}, {'K': 4, 'P': 4})
    )
