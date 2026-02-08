"""
Dataflow analysis utilities for PE array mappings.

This module provides tools to analyze mapping configurations,
identify dataflow patterns, and compute bandwidth/reduction requirements.
"""

import numpy as np
from typing import Optional


def get_reduction_axes(workload) -> list[int]:
    """
    Get the reduction axes for Output datatype.
    
    For convolution:
    - Output depends on: P, Q, K, N
    - Reduction axes (Output-irrelevant): R, S, C
    
    Args:
        workload: Workload definition with O (relevancy matrix)
        
    Returns:
        List of dimension indices that are reduction axes
    """
    O = workload.O
    output_datatype = 2  # Output
    
    reduction_axes = []
    for j in range(workload.num_dims):
        if O[j][output_datatype] == 0:  # Dimension j is NOT relevant to Output
            reduction_axes.append(j)
    
    return reduction_axes


def analyze_multi_dim_mapping(
    h_dims: dict[str, int],
    w_dims: dict[str, int],
    pe_array_h: int = 16,
    pe_array_w: int = 16,
) -> dict:
    """
    Analyze a multi-dimension mapping configuration.
    
    This is a utility function to understand the implications of
    mapping multiple dimensions to H and/or W directions.
    
    Args:
        h_dims: Dict of {dim_name: parallelism} for H direction
                e.g., {'K': 4, 'C': 4} means K=4 and C=4 both in H
        w_dims: Dict of {dim_name: parallelism} for W direction
        pe_array_h: PE array height
        pe_array_w: PE array width
        
    Returns:
        Analysis dict with bandwidth and reduction implications
        
    Examples:
        >>> # Example 1: Weight Stationary (K in H, P×Q in W)
        >>> analyze_multi_dim_mapping({'K': 16}, {'P': 4, 'Q': 4})
        
        >>> # Example 2: Mixed parallelism
        >>> analyze_multi_dim_mapping({'K': 4, 'C': 4}, {'P': 4, 'Q': 4})
        
        >>> # Example 3: Split dimension (C split across H and W)
        >>> analyze_multi_dim_mapping({'C': 16}, {'C': 16})
    """
    # Calculate total parallelism
    h_total = np.prod(list(h_dims.values())) if h_dims else 1
    w_total = np.prod(list(w_dims.values())) if w_dims else 1
    
    # Dimension relevancy for convolution
    # O[dim][datatype]: 1 if relevant, 0 if not
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
    
    # Check dimension fit
    valid = h_total <= pe_array_h and w_total <= pe_array_w
    
    # Calculate bandwidth for each datatype
    bandwidth = {}
    for dtype in ['Input', 'Weight', 'Output']:
        h_bw = 1
        w_bw = 1
        
        for dim, par in h_dims.items():
            if relevancy.get(dim, {}).get(dtype, 0) == 1:
                h_bw *= par
                
        for dim, par in w_dims.items():
            if relevancy.get(dim, {}).get(dtype, 0) == 1:
                w_bw *= par
        
        bandwidth[dtype] = {
            'h_direction': h_bw,
            'w_direction': w_bw,
            'total_unicast': h_bw * w_bw,
            'with_row_broadcast': h_bw,  # Feed from left, broadcast in W
            'with_col_broadcast': w_bw,  # Feed from top, broadcast in H
        }
    
    # Calculate reduction requirements
    h_reduction = 1
    w_reduction = 1
    
    for dim, par in h_dims.items():
        if dim in reduction_axes:
            h_reduction *= par
            
    for dim, par in w_dims.items():
        if dim in reduction_axes:
            w_reduction *= par
    
    total_reduction = h_reduction * w_reduction
    
    reduction = {
        'h_direction': h_reduction,
        'w_direction': w_reduction,
        'total': total_reduction,
        'needs_reduction': total_reduction > 1,
        'reduction_depth': int(np.ceil(np.log2(max(total_reduction, 1)))),
        'is_2d_reduction': h_reduction > 1 and w_reduction > 1,
    }
    
    # Identify dataflow pattern
    dataflow = identify_dataflow_pattern(h_dims, w_dims)
    
    return {
        'h_dims': h_dims,
        'w_dims': w_dims,
        'h_total': int(h_total),
        'w_total': int(w_total),
        'total_parallelism': int(h_total * w_total),
        'valid': valid,
        'pe_utilization': (h_total * w_total) / (pe_array_h * pe_array_w),
        'bandwidth': bandwidth,
        'reduction': reduction,
        'dataflow_pattern': dataflow,
    }


def identify_dataflow_pattern(h_dims: dict, w_dims: dict) -> str:
    """
    Identify the dataflow pattern from H/W dimension mapping.
    
    Args:
        h_dims: Dict of {dim_name: parallelism} for H direction
        w_dims: Dict of {dim_name: parallelism} for W direction
    
    Returns:
        String describing the dataflow pattern
    """
    h_set = set(h_dims.keys())
    w_set = set(w_dims.keys())
    
    # Check for classic patterns
    if h_set == {'K'} and w_set <= {'P', 'Q', 'N'}:
        return "Weight Stationary (K in H)"
    
    if w_set == {'K'} and h_set <= {'P', 'Q', 'N'}:
        return "Weight Stationary (K in W)"
    
    if h_set <= {'P', 'Q', 'N'} and w_set <= {'P', 'Q', 'N'}:
        return "Output Stationary (P,Q,N spatial)"
    
    if 'C' in h_set and 'K' in w_set:
        return "Input Stationary (C in H, K in W)"
    
    if 'C' in w_set and 'K' in h_set:
        return "Input Stationary (K in H, C in W)"
    
    if ('R' in h_set or 'S' in h_set) and 'K' in w_set:
        return "Row Stationary (R/S in H, K in W)"
    
    # Check for mixed patterns
    if len(h_set) > 1 or len(w_set) > 1:
        return f"Mixed Dataflow (H={h_set}, W={w_set})"
    
    return "Custom Dataflow"


def print_mapping_analysis(analysis: dict) -> None:
    """Pretty print the mapping analysis."""
    print("=" * 60)
    print("Multi-Dimension Mapping Analysis")
    print("=" * 60)
    
    print(f"\nH Direction: {analysis['h_dims']} → Total = {analysis['h_total']}")
    print(f"W Direction: {analysis['w_dims']} → Total = {analysis['w_total']}")
    print(f"Total Parallelism: {analysis['total_parallelism']}")
    print(f"PE Utilization: {analysis['pe_utilization']:.1%}")
    print(f"Valid: {analysis['valid']}")
    print(f"Dataflow Pattern: {analysis['dataflow_pattern']}")
    
    print("\n--- Bandwidth Requirements ---")
    for dtype, bw in analysis['bandwidth'].items():
        print(f"  {dtype}:")
        print(f"    H-direction unicast: {bw['h_direction']}")
        print(f"    W-direction unicast: {bw['w_direction']}")
        print(f"    With row broadcast:  {bw['with_row_broadcast']}")
    
    print("\n--- Reduction Requirements ---")
    red = analysis['reduction']
    if red['needs_reduction']:
        print(f"  Needs reduction: YES")
        print(f"  H-direction reduction: {red['h_direction']}x")
        print(f"  W-direction reduction: {red['w_direction']}x")
        print(f"  Total reduction: {red['total']}x")
        print(f"  Reduction depth: {red['reduction_depth']} stages")
        if red['is_2d_reduction']:
            print(f"  WARNING: 2D reduction required!")
    else:
        print(f"  Needs reduction: NO")
    
    print("=" * 60)
