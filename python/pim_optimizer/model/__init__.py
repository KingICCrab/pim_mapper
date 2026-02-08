"""
ILP Model components for PIM optimizer.
"""

from pim_optimizer.model.variables import create_decision_variables, VariableSet, SpatialDim
from pim_optimizer.model.constraints import (
    add_basic_constraints, 
    add_buffer_constraints,
    add_pe_parallelism_constraints,
    add_compute_unit_constraints,
)
from pim_optimizer.model.expressions import (
    build_memory_expressions, 
    build_tile_bytes_vars,
    TileInfo,
)
from pim_optimizer.model.objective import set_objective
from pim_optimizer.model.crossing import (
    compute_block_crossing_ratio_gcd,
    compute_input_block_crossing,
    analyze_crossing_pattern,
)
from pim_optimizer.model.row_activation import (
    build_row_activation_model,
    compute_input_block_crossing_count,
    precompute_input_block_crossing_table,
    build_input_block_crossing_expr,
)

__all__ = [
    "create_decision_variables",
    "VariableSet",
    "SpatialDim",
    "add_basic_constraints",
    "add_buffer_constraints",
    "add_pe_parallelism_constraints",
    "add_compute_unit_constraints",
    "build_memory_expressions",
    "build_tile_bytes_vars",
    "TileInfo",
    "set_objective",
    # Block Crossing ratio functions (layout block boundaries)
    "compute_block_crossing_ratio_gcd",
    "compute_input_block_crossing",
    "analyze_crossing_pattern",
    # Row activation
    "build_row_activation_model",
    # Input Block Crossing count
    "compute_input_block_crossing_count",
    "precompute_input_block_crossing_table",
    "build_input_block_crossing_expr",
]

