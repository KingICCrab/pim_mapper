"""
Exhaustive search module for golden model verification.

This module provides:
- Brute-force enumeration of all valid mappings
- Smart sampling for large search spaces
- Optimal solution finding by evaluation
"""

from .brute_force import (
    enumerate_tile_factors,
    enumerate_loop_orders,
    enumerate_all_mappings,
    find_optimal_exhaustive,
    count_mapping_space,
)

from .sampler import (
    sample_mappings_random,
    sample_mappings_latin_hypercube,
    sample_boundary_cases,
)

__all__ = [
    'enumerate_tile_factors',
    'enumerate_loop_orders', 
    'enumerate_all_mappings',
    'find_optimal_exhaustive',
    'count_mapping_space',
    'sample_mappings_random',
    'sample_mappings_latin_hypercube',
    'sample_boundary_cases',
]
