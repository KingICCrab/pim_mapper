"""
Analysis utilities for mapping results.

These are post-optimization tools for understanding and visualizing mappings.
"""

from pim_optimizer.analysis.dataflow import (
    analyze_multi_dim_mapping,
    identify_dataflow_pattern,
    print_mapping_analysis,
    get_reduction_axes,
)

__all__ = [
    "analyze_multi_dim_mapping",
    "identify_dataflow_pattern", 
    "print_mapping_analysis",
    "get_reduction_axes",
]
