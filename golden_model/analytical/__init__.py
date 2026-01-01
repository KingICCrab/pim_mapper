"""
Analytical module for golden model verification.
"""

from .cost_formulas import (
    LoopBounds,
    TileFactors,
    compute_input_tile_size,
    compute_weight_tile_size,
    compute_output_tile_size,
    compute_input_access_count,
    compute_weight_access_count,
    compute_output_access_count,
    compute_analytical_memory_reads,
    compute_analytical_latency,
)

from .row_activation import (
    RowBufferConfig,
    compute_crossing_ratio_sequential,
    compute_crossing_ratio_sliding_window,
    compute_crossing_ratio_analytical,
    compute_row_activations_analytical,
    verify_row_activation_model,
    verify_crossing_ratio,
)

__all__ = [
    'LoopBounds',
    'TileFactors',
    'compute_input_tile_size',
    'compute_weight_tile_size',
    'compute_output_tile_size',
    'compute_input_access_count',
    'compute_weight_access_count',
    'compute_output_access_count',
    'compute_analytical_memory_reads',
    'compute_analytical_latency',
    'RowBufferConfig',
    'compute_crossing_ratio_sequential',
    'compute_crossing_ratio_sliding_window',
    'compute_crossing_ratio_analytical',
    'compute_row_activations_analytical',
    'verify_row_activation_model',
    'verify_crossing_ratio',
]
