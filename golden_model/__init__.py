"""
Golden Model for PIM Optimizer Verification.

This package provides tools to verify:
1. Cost model correctness - analytical formulas based on Interstellar
2. ILP optimality - exhaustive search comparison
3. Row activation calculations - GCD-based crossing ratio

Reference projects analyzed:
- OptiPIM: Gurobi MILP with GCD LUT for Input access
- Interstellar: Access count = product of irrelevant loop factors
- nn_dataflow: Memory hierarchy cost model
- UniNDP: Instruction-level simulation
"""

# Core data structures
from .analytical.cost_formulas import (
    LoopBounds,
    TileFactors,
    compute_input_tile_size,
    compute_weight_tile_size,
    compute_output_tile_size,
    compute_analytical_memory_reads,
    compute_analytical_latency,
)

# Row activation model
from .analytical.row_activation import (
    RowBufferConfig,
    compute_crossing_ratio_sequential,
    compute_crossing_ratio_sliding_window,
    compute_crossing_ratio_analytical,
    compute_row_activations_analytical,
    verify_row_activation_model,
    verify_crossing_ratio,
)

# Exhaustive search
from .exhaustive.brute_force import (
    Mapping,
    enumerate_tile_factors,
    enumerate_loop_orders,
    enumerate_all_mappings,
    find_optimal_exhaustive,
    verify_ilp_optimality,
    count_mapping_space,
)

# Sampling
from .exhaustive.sampler import (
    sample_mappings_random,
    sample_mappings_latin_hypercube,
    sample_boundary_cases,
)

# Comparison and verification
from .comparison.compare import (
    CostModelVerification,
    OptimalityVerification,
    VerificationResult,
    verify_cost_model,
    verify_optimality,
    compare_with_ilp,
    run_verification_suite,
)

# Reporting
from .comparison.report import (
    generate_report,
    print_verification_summary,
    print_detailed_result,
    print_comparison_table,
    export_to_csv,
)

__all__ = [
    # Data structures
    'LoopBounds',
    'TileFactors',
    'Mapping',
    'RowBufferConfig',
    'CostModelVerification',
    'OptimalityVerification',
    'VerificationResult',
    
    # Cost formulas
    'compute_input_tile_size',
    'compute_weight_tile_size',
    'compute_output_tile_size',
    'compute_analytical_memory_reads',
    'compute_analytical_latency',
    
    # Row activation
    'compute_crossing_ratio_sequential',
    'compute_crossing_ratio_sliding_window',
    'compute_crossing_ratio_analytical',
    'compute_row_activations_analytical',
    'verify_row_activation_model',
    'verify_crossing_ratio',
    
    # Exhaustive search
    'enumerate_tile_factors',
    'enumerate_loop_orders',
    'enumerate_all_mappings',
    'find_optimal_exhaustive',
    'verify_ilp_optimality',
    'count_mapping_space',
    
    # Sampling
    'sample_mappings_random',
    'sample_mappings_latin_hypercube',
    'sample_boundary_cases',
    
    # Comparison
    'verify_cost_model',
    'verify_optimality',
    'compare_with_ilp',
    'run_verification_suite',
    
    # Reporting
    'generate_report',
    'print_verification_summary',
    'print_detailed_result',
    'print_comparison_table',
    'export_to_csv',
]
