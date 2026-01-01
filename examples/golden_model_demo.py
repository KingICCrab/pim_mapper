#!/usr/bin/env python3
"""
Golden Model Verification Example.

This script demonstrates how the cycle-accurate simulator provides
ground truth for verifying:
1. Cost model correctness
2. ILP solution optimality

The simulator models every memory access with precise DRAM timing,
unlike analytical formulas which are estimations.
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')

from golden_model import (
    Simulator, SimulatorConfig, DRAMTiming,
    AccessTrace, AccessPatternGenerator, analyze_row_crossing
)
from golden_model.simulator import RowCrossingAnalyzer, simulate_mapping


def example_1_basic_simulation():
    """
    Example 1: Basic DRAM simulation with trace.
    
    Shows how the simulator tracks row buffer hits/misses.
    """
    print("=" * 60)
    print("Example 1: Basic DRAM Simulation")
    print("=" * 60)
    
    # Create a simple access trace
    trace = AccessTrace()
    
    # Sequential accesses to same row -> row buffer hits
    for i in range(10):
        trace.add_read(bank_id=0, row_addr=100, tensor_name='A')
    
    # Access different row -> row buffer miss
    trace.add_read(bank_id=0, row_addr=200, tensor_name='A')
    
    # More accesses to new row -> hits
    for i in range(5):
        trace.add_read(bank_id=0, row_addr=200, tensor_name='A')
    
    # Back to first row -> miss
    trace.add_read(bank_id=0, row_addr=100, tensor_name='A')
    
    print(f"\nTrace has {len(trace)} accesses")
    
    # Simulate
    simulator = Simulator()
    result = simulator.simulate(trace, verbose=False)
    
    print(f"\nSimulation Results:")
    print(f"  Total cycles: {result.total_cycles}")
    print(f"  Row hits: {result.row_hits}")
    print(f"  Row misses: {result.row_misses}")
    print(f"  Row empty accesses: {result.row_empty_accesses}")
    print(f"  Row buffer hit rate: {result.row_buffer_hit_rate:.2%}")
    print(f"  Total row activations: {result.row_activations}")
    
    # Verify manually:
    # First access: empty (1 activation)
    # Next 9: hits
    # Access row 200: miss (1 activation)
    # Next 5: hits
    # Access row 100: miss (1 activation)
    # Expected: 3 activations, 14 hits
    assert result.row_activations == 3, f"Expected 3 activations, got {result.row_activations}"
    assert result.row_hits == 14, f"Expected 14 hits, got {result.row_hits}"
    print("\n✓ Manual verification passed!")


def example_2_tiled_access_pattern():
    """
    Example 2: Analyze tiled access patterns.
    
    This shows how tiling affects row buffer locality.
    """
    print("\n" + "=" * 60)
    print("Example 2: Tiled Access Pattern Analysis")
    print("=" * 60)
    
    analyzer = RowCrossingAnalyzer(row_size=8192, element_size=4)
    
    # Test case: 1024 elements with different tile sizes
    total_elements = 1024
    elements_per_row = 2048  # 8KB / 4B = 2048 elements
    
    print(f"\nTensor: {total_elements} elements")
    print(f"DRAM row: {elements_per_row} elements ({elements_per_row * 4} bytes)")
    
    for tile_size in [64, 256, 512, 1024]:
        analysis = analyzer.analyze_1d_tiling(
            total_elements=total_elements,
            tile_size=tile_size,
            num_iterations=1
        )
        print(f"\nTile size = {tile_size}:")
        print(f"  Num tiles: {analysis['num_tiles']}")
        print(f"  Total rows: {analysis['total_rows']}")
        print(f"  Row crossings: {analysis['row_crossings_per_iteration']}")
        print(f"  Crossing ratio: {analysis['crossing_ratio']:.4f}")


def example_3_gemm_simulation():
    """
    Example 3: Simulate GEMM with different mappings.
    
    Compare different tile sizes to find optimal mapping.
    """
    print("\n" + "=" * 60)
    print("Example 3: GEMM Simulation with Different Tile Sizes")
    print("=" * 60)
    
    workload = {
        'type': 'gemm',
        'M': 64,
        'N': 64,
        'K': 64,
    }
    
    print(f"\nWorkload: GEMM ({workload['M']}x{workload['K']}) @ ({workload['K']}x{workload['N']})")
    print(f"          = ({workload['M']}x{workload['N']})")
    
    # Test different tile sizes
    tile_configs = [
        {'tile_M': 8, 'tile_N': 8, 'tile_K': 64},    # Small tiles
        {'tile_M': 16, 'tile_N': 16, 'tile_K': 32},  # Medium tiles
        {'tile_M': 32, 'tile_N': 32, 'tile_K': 16},  # Larger tiles
        {'tile_M': 64, 'tile_N': 64, 'tile_K': 8},   # Very large tiles
    ]
    
    results = []
    for config in tile_configs:
        result = simulate_mapping(config, workload)
        results.append((config, result))
        
        print(f"\nTile ({config['tile_M']}x{config['tile_N']}x{config['tile_K']}):")
        print(f"  Total cycles: {result.total_cycles:,}")
        print(f"  Row activations: {result.row_activations}")
        print(f"  Row buffer hit rate: {result.row_buffer_hit_rate:.2%}")
    
    # Find best mapping
    best = min(results, key=lambda x: x[1].total_cycles)
    print(f"\n✓ Best mapping: Tile ({best[0]['tile_M']}x{best[0]['tile_N']}x{best[0]['tile_K']})")
    print(f"  with {best[1].total_cycles:,} cycles")


def example_4_verify_formula():
    """
    Example 4: Verify analytical formula against simulation.
    
    This is the key use case - checking if our cost model formulas
    are correct by comparing against simulation ground truth.
    """
    print("\n" + "=" * 60)
    print("Example 4: Verify Formula Against Simulation")
    print("=" * 60)
    
    # Simple scenario: accessing N elements with tile size T
    # Formula estimate vs simulation ground truth
    
    total_elements = 512
    tile_size = 128
    num_iterations = 4  # Reuse factor
    
    generator = AccessPatternGenerator(row_size=8192, col_size=64)
    
    # Generate trace for tiled access with reuse
    trace = AccessTrace()
    for iteration in range(num_iterations):
        tile_trace = generator.generate_1d_tile_accesses(
            tensor_name='input',
            base_bank=0,
            base_row=0,
            total_elements=total_elements,
            tile_size=tile_size,
        )
        for access in tile_trace:
            trace.add_access(access)
    
    print(f"\nScenario: {total_elements} elements, tile={tile_size}, {num_iterations} iterations")
    print(f"Total accesses: {len(trace)}")
    
    # Simulate
    simulator = Simulator()
    result = simulator.simulate(trace)
    
    # Analyze row crossing pattern
    crossing_stats = analyze_row_crossing(trace)
    
    print(f"\nSimulation (Ground Truth):")
    print(f"  Row activations: {result.row_activations}")
    print(f"  Row buffer hit rate: {result.row_buffer_hit_rate:.2%}")
    print(f"  Total cycles: {result.total_cycles}")
    
    # Compare with analytical formula (if we have one)
    # Example formula: row_activations = ceil(total_elements / elements_per_row) * num_iterations
    elements_per_row = 2048  # 8KB / 4B
    import math
    formula_activations = math.ceil(total_elements / elements_per_row) * num_iterations
    
    print(f"\nSimple Formula Estimate:")
    print(f"  Row activations: {formula_activations}")
    
    print(f"\nComparison:")
    print(f"  Simulation: {result.row_activations}")
    print(f"  Formula: {formula_activations}")
    print(f"  Difference: {abs(result.row_activations - formula_activations)}")
    
    if result.row_activations == formula_activations:
        print("\n✓ Formula matches simulation!")
    else:
        print(f"\n⚠ Formula differs from simulation by "
              f"{abs(result.row_activations - formula_activations)}")
        print("  This indicates the formula needs refinement.")


def example_5_compare_with_ilp():
    """
    Example 5: Compare ILP solution with simulation.
    
    This verifies that the ILP optimizer finds truly optimal solutions.
    """
    print("\n" + "=" * 60)
    print("Example 5: ILP Solution Verification Framework")
    print("=" * 60)
    
    print("""
    This example shows the framework for verifying ILP solutions:
    
    1. Run ILP optimizer to get optimal mapping
    2. Extract cost estimates from ILP (row activations, cycles, etc.)
    3. Run simulator with the ILP mapping
    4. Compare ILP estimates with simulation ground truth
    
    If ILP is correct:
    - Cost estimates should match simulation (or be conservative bounds)
    - No other mapping should have lower simulated cost
    
    If mismatch found:
    - Either cost model in ILP is wrong
    - Or ILP constraints are too loose/tight
    """)
    
    # Placeholder for actual ILP integration
    # In practice, this would:
    # 1. Call the ILP optimizer
    # 2. Get the optimal mapping and predicted costs
    # 3. Simulate the mapping
    # 4. Compare predicted vs actual
    
    print("\nTo integrate with your ILP optimizer:")
    print("  1. from golden_model import simulate_mapping")
    print("  2. result = simulate_mapping(ilp_mapping, workload)")
    print("  3. Compare result.row_activations with ILP prediction")
    print("  4. Compare result.total_cycles with ILP objective")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Golden Model: Cycle-Accurate DRAM Simulator")
    print("For Verifying Cost Models and ILP Optimality")
    print("=" * 60)
    
    example_1_basic_simulation()
    example_2_tiled_access_pattern()
    example_3_gemm_simulation()
    example_4_verify_formula()
    example_5_compare_with_ilp()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
