#!/usr/bin/env python3
"""
Example: Verify PIM Optimizer ILP model with Golden Model.

This script demonstrates how to:
1. Verify cost model formulas are correct
2. Verify ILP finds optimal solutions
3. Verify row activation calculations
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from golden_model import (
    # Data structures
    LoopBounds, TileFactors, Mapping, RowBufferConfig,
    
    # Cost formulas
    compute_analytical_memory_reads,
    compute_analytical_latency,
    
    # Row activation
    compute_crossing_ratio_sequential,
    compute_crossing_ratio_sliding_window,
    verify_crossing_ratio,
    
    # Exhaustive search
    find_optimal_exhaustive,
    count_mapping_space,
    verify_ilp_optimality,
    
    # Verification
    verify_cost_model,
    compare_with_ilp,
    run_verification_suite,
    
    # Reporting
    print_verification_summary,
    print_detailed_result,
    print_comparison_table,
    generate_report,
    export_to_csv,
)


def example_1_cost_model_verification():
    """
    Example 1: Verify cost model formulas.
    
    This shows that your ILP cost model matches the analytical formulas
    from the reference projects.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Cost Model Verification")
    print("=" * 70)
    
    # Define problem
    bounds = LoopBounds(N=4, C=8, K=8, P=4, Q=4, R=3, S=3)
    factors = TileFactors(N=2, C=4, K=4, P=2, Q=2, R=1, S=1)
    loop_order = ['N', 'K', 'C', 'P', 'Q', 'R', 'S']
    
    print(f"\nProblem: Conv {bounds.N}×{bounds.C}×{bounds.K} with kernel {bounds.R}×{bounds.S}")
    print(f"Tiling:  {factors.N}×{factors.C}×{factors.K}×{factors.P}×{factors.Q}")
    
    # Compute analytical values
    mem_reads = compute_analytical_memory_reads(bounds, factors, loop_order)
    latency = compute_analytical_latency(bounds, factors, loop_order)
    
    print("\nAnalytical Results:")
    print(f"  Input:  tile_size={mem_reads['input_tile_size']:,}, "
          f"access_count={mem_reads['input_access_count']}, "
          f"total_reads={mem_reads['input_tile_size'] * mem_reads['input_access_count']:,}")
    print(f"  Weight: tile_size={mem_reads['weight_tile_size']:,}, "
          f"access_count={mem_reads['weight_access_count']}, "
          f"total_reads={mem_reads['weight_tile_size'] * mem_reads['weight_access_count']:,}")
    print(f"  Output: tile_size={mem_reads['output_tile_size']:,}, "
          f"access_count={mem_reads['output_access_count']}, "
          f"total_reads={mem_reads['output_tile_size'] * mem_reads['output_access_count']:,}")
    print(f"  Latency: {latency['latency']:,} cycles")
    
    # Simulate ILP results (here we use the analytical as "ILP" for demo)
    ilp_results = {
        'input_reads': mem_reads['input_tile_size'] * mem_reads['input_access_count'],
        'weight_reads': mem_reads['weight_tile_size'] * mem_reads['weight_access_count'],
        'output_reads': mem_reads['output_tile_size'] * mem_reads['output_access_count'],
        'latency': latency['latency'],
    }
    
    # Verify
    result = verify_cost_model(ilp_results, bounds, factors, loop_order)
    
    print(f"\nVerification: {'PASS' if result.passed else 'FAIL'}")
    for dtype in ['input', 'weight', 'output', 'latency']:
        d = result.details[dtype]
        print(f"  {dtype}: error={d['relative_error']:.2%} {'✓' if d['passed'] else '✗'}")


def example_2_optimality_verification():
    """
    Example 2: Verify ILP finds optimal solution by exhaustive search.
    
    For small problems, we can enumerate all mappings and verify the
    ILP solution is truly optimal.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Optimality Verification")
    print("=" * 70)
    
    # Small problem for exhaustive search
    bounds = LoopBounds(N=4, C=4, K=4, P=4, Q=4, R=3, S=3)
    
    # Count search space
    num_mappings = count_mapping_space(bounds, include_loop_order=False)
    print(f"\nSearch space size: {num_mappings:,} mappings")
    
    # Find optimal by exhaustive search
    print("\nRunning exhaustive search...")
    optimal, all_mappings = find_optimal_exhaustive(
        bounds, 
        objective='latency',
        enumerate_loop_order=False,
        verbose=True,
    )
    
    print(f"\nOptimal solution found:")
    print(f"  Factors: N={optimal.l1_factors.N}, C={optimal.l1_factors.C}, "
          f"K={optimal.l1_factors.K}, P={optimal.l1_factors.P}, Q={optimal.l1_factors.Q}")
    print(f"  Latency: {optimal.latency:,}")
    print(f"  Memory ops: {optimal.total_memory_ops:,}")
    
    # Test: Verify a "suboptimal" solution
    suboptimal = Mapping(
        l1_factors=TileFactors(N=1, C=1, K=1, P=1, Q=1, R=1, S=1),
        loop_order=['N', 'C', 'K', 'P', 'Q', 'R', 'S']
    )
    
    is_optimal, details = verify_ilp_optimality(
        suboptimal, bounds, 'latency',
        enumerate_loop_order=False,
        max_mappings=num_mappings,
    )
    
    print(f"\nSuboptimal solution test:")
    print(f"  Is optimal: {is_optimal}")
    print(f"  Rank: {details['ilp_rank']}/{details['total_unique_costs']}")
    print(f"  Gap: {details['relative_gap']:.2%}")


def example_3_crossing_ratio_verification():
    """
    Example 3: Verify row buffer crossing ratio calculations.
    
    The crossing ratio formula is critical for accurate DRAM
    latency estimation.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Crossing Ratio Verification")
    print("=" * 70)
    
    # Test cases for sequential access
    print("\nSequential Access Pattern:")
    test_cases = [
        (64, 256),   # tile << row
        (128, 256),  # tile = row/2
        (256, 256),  # tile = row
        (384, 256),  # tile > row
    ]
    
    for tile, row in test_cases:
        cr = compute_crossing_ratio_sequential(tile, row)
        print(f"  tile={tile:3d}, row={row:3d}: crossing_ratio={cr:.4f}")
    
    # Test sliding window pattern (for Input datatype)
    print("\nSliding Window Pattern (Input):")
    sliding_cases = [
        (16, 8, 1),   # block_h=16, tile_h=8, step=1
        (16, 8, 2),   # step=2
        (16, 16, 1),  # tile=block
        (16, 4, 4),   # step=tile
    ]
    
    for block_h, tile_h, step in sliding_cases:
        cr = compute_crossing_ratio_sliding_window(block_h, tile_h, step)
        print(f"  block_h={block_h:2d}, tile_h={tile_h:2d}, step={step}: "
              f"crossing_ratio={cr:.4f}")
    
    # Verify against ILP result
    print("\nVerification against ILP:")
    ilp_cr = 0.5  # Simulated ILP result
    passed, details = verify_crossing_ratio(ilp_cr, 128, 256, 'sequential')
    print(f"  ILP crossing_ratio: {ilp_cr}")
    print(f"  Analytical: {details['analytical_crossing_ratio']:.4f}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")


def example_4_full_verification():
    """
    Example 4: Complete verification suite.
    
    Run full verification including cost model, optimality,
    and generate reports.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Full Verification Suite")
    print("=" * 70)
    
    # Define test cases
    test_configs = [
        {"name": "small_4x4x4", "n": 4, "c": 4, "k": 4},
        {"name": "medium_4x8x8", "n": 4, "c": 8, "k": 8},
        {"name": "unbalanced_2x16x4", "n": 2, "c": 16, "k": 4},
    ]
    
    results = []
    
    for cfg in test_configs:
        bounds = LoopBounds(
            N=cfg['n'], C=cfg['c'], K=cfg['k'],
            P=4, Q=4, R=3, S=3
        )
        
        # Create a mapping (simulate ILP solution)
        factors = TileFactors(
            N=max(1, cfg['n']//2),
            C=max(1, cfg['c']//2),
            K=max(1, cfg['k']//2),
            P=2, Q=2, R=1, S=1
        )
        loop_order = ['N', 'K', 'C', 'P', 'Q', 'R', 'S']
        
        # Compute "ILP" results
        mem = compute_analytical_memory_reads(bounds, factors, loop_order)
        lat = compute_analytical_latency(bounds, factors, loop_order)
        
        ilp_results = {
            'input_reads': mem['input_tile_size'] * mem['input_access_count'],
            'weight_reads': mem['weight_tile_size'] * mem['weight_access_count'],
            'output_reads': mem['output_tile_size'] * mem['output_access_count'],
            'latency': lat['latency'],
        }
        
        # Run verification
        result = compare_with_ilp(
            test_name=cfg['name'],
            bounds=bounds,
            ilp_factors=factors,
            ilp_loop_order=loop_order,
            ilp_results=ilp_results,
            verbose=False,
        )
        results.append(result)
    
    # Print reports
    print_verification_summary(results)
    print_comparison_table(results)
    
    # Print detailed report for first case
    print("\nDetailed report for first test case:")
    print_detailed_result(results[0])
    
    # Generate markdown report
    print("\n--- Markdown Report ---")
    print(generate_report(results, 'markdown'))


def example_5_integration_with_pim_optimizer():
    """
    Example 5: How to integrate with actual pim_optimizer.
    
    This shows the pattern for extracting ILP results and
    running verification.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Integration with PIM Optimizer")
    print("=" * 70)
    
    print("""
To integrate with your pim_optimizer ILP model:

1. After solving the ILP, extract the solution:

    from pim_optimizer.optimizer import PIMOptimizer
    from golden_model import compare_with_ilp, LoopBounds, TileFactors
    
    # Solve ILP
    optimizer = PIMOptimizer(workload, arch)
    result = optimizer.solve()
    
    # Extract solution
    bounds = LoopBounds(
        N=workload.batch,
        C=workload.in_channels,
        K=workload.out_channels,
        P=workload.out_height,
        Q=workload.out_width,
        R=workload.kernel_h,
        S=workload.kernel_w,
    )
    
    factors = TileFactors(
        N=result.n_factor,
        C=result.c_factor,
        K=result.k_factor,
        P=result.p_factor,
        Q=result.q_factor,
        R=result.r_factor,
        S=result.s_factor,
    )
    
    ilp_results = {
        'input_reads': result.input_memory_reads,
        'weight_reads': result.weight_memory_reads,
        'output_reads': result.output_memory_reads,
        'latency': result.total_latency,
    }
    
    # Verify
    verification = compare_with_ilp(
        test_name="my_workload",
        bounds=bounds,
        ilp_factors=factors,
        ilp_loop_order=result.loop_order,
        ilp_results=ilp_results,
        verbose=True,
    )
    
    if verification.overall_passed:
        print("✓ ILP solution verified!")
    else:
        print("✗ Verification failed - check cost model")

2. For row activation verification:

    from golden_model import RowBufferConfig, verify_row_activation_model
    
    row_config = RowBufferConfig(
        row_size_bytes=1024,
        element_bytes=1,
        num_banks=arch.num_banks,
        activation_latency=arch.tRCD + arch.tRP,
    )
    
    passed, details = verify_row_activation_model(
        ilp_row_acts=result.input_row_activations,
        memory_reads=result.input_memory_reads,
        tile_info={'tile_h': result.input_tile_h, ...},
        row_config=row_config,
        datatype='input',
    )
""")


if __name__ == '__main__':
    print("=" * 70)
    print("GOLDEN MODEL VERIFICATION EXAMPLES")
    print("=" * 70)
    
    example_1_cost_model_verification()
    example_2_optimality_verification()
    example_3_crossing_ratio_verification()
    example_4_full_verification()
    example_5_integration_with_pim_optimizer()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
