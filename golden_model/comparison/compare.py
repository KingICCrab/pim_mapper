"""
Comparison functions for verifying ILP model against golden model.

This module provides the core verification logic:
1. Cost model verification: Compare ILP cost formulas with analytical
2. Optimality verification: Compare ILP solution with exhaustive search
3. Row activation verification: Compare crossing ratios and row counts
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math

# Import from other modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytical.cost_formulas import (
    LoopBounds, TileFactors,
    compute_input_tile_size,
    compute_weight_tile_size,
    compute_output_tile_size,
    compute_analytical_memory_reads,
    compute_analytical_latency,
)

from analytical.row_activation import (
    RowBufferConfig,
    compute_crossing_ratio_analytical,
    compute_row_activations_analytical,
    verify_row_activation_model,
    verify_crossing_ratio,
)

from exhaustive.brute_force import (
    Mapping,
    find_optimal_exhaustive,
    verify_ilp_optimality,
    evaluate_mapping,
)


# =============================================================================
# Data Classes for Verification Results
# =============================================================================

@dataclass
class CostModelVerification:
    """Result of cost model verification."""
    passed: bool
    input_reads_match: bool
    weight_reads_match: bool
    output_reads_match: bool
    latency_match: bool
    details: Dict[str, Any]
    tolerance: float
    

@dataclass
class OptimalityVerification:
    """Result of optimality verification."""
    is_optimal: bool
    relative_gap: float
    ilp_rank: int
    total_solutions: int
    ilp_cost: float
    optimal_cost: float
    details: Dict[str, Any]


@dataclass
class VerificationResult:
    """Combined verification result."""
    test_name: str
    bounds: LoopBounds
    ilp_mapping: Mapping
    
    cost_model_result: CostModelVerification
    optimality_result: OptimalityVerification
    row_activation_result: Optional[Dict]
    
    overall_passed: bool
    summary: str


# =============================================================================
# Cost Model Verification
# =============================================================================

def verify_cost_model(
    ilp_results: Dict[str, Any],
    bounds: LoopBounds,
    factors: TileFactors,
    loop_order: List[str],
    tolerance: float = 0.01,
) -> CostModelVerification:
    """
    Verify ILP cost model results against analytical formulas.
    
    Args:
        ilp_results: Dictionary with ILP computed values
            - input_reads: Input memory reads
            - weight_reads: Weight memory reads  
            - output_reads: Output memory reads
            - latency: Total latency
        bounds: Loop bounds
        factors: Tile factors from ILP
        loop_order: Loop order from ILP
        tolerance: Acceptable relative error
        
    Returns:
        CostModelVerification with detailed results
    """
    # Compute analytical values
    analytical = compute_analytical_memory_reads(bounds, factors, loop_order)
    analytical_latency = compute_analytical_latency(bounds, factors, loop_order)
    
    analytical_input = analytical['input_tile_size'] * analytical['input_access_count']
    analytical_weight = analytical['weight_tile_size'] * analytical['weight_access_count']
    analytical_output = analytical['output_tile_size'] * analytical['output_access_count']
    
    # Compare with ILP results
    def relative_error(ilp_val, analytical_val):
        if analytical_val == 0:
            return 0.0 if ilp_val == 0 else float('inf')
        return abs(ilp_val - analytical_val) / analytical_val
    
    ilp_input = ilp_results.get('input_reads', 0)
    ilp_weight = ilp_results.get('weight_reads', 0)
    ilp_output = ilp_results.get('output_reads', 0)
    ilp_latency = ilp_results.get('latency', 0)
    
    input_err = relative_error(ilp_input, analytical_input)
    weight_err = relative_error(ilp_weight, analytical_weight)
    output_err = relative_error(ilp_output, analytical_output)
    latency_err = relative_error(ilp_latency, analytical_latency['latency'])
    
    input_match = input_err <= tolerance
    weight_match = weight_err <= tolerance
    output_match = output_err <= tolerance
    latency_match = latency_err <= tolerance
    
    passed = input_match and weight_match and output_match and latency_match
    
    details = {
        'input': {
            'ilp': ilp_input,
            'analytical': analytical_input,
            'relative_error': input_err,
            'passed': input_match,
        },
        'weight': {
            'ilp': ilp_weight,
            'analytical': analytical_weight,
            'relative_error': weight_err,
            'passed': weight_match,
        },
        'output': {
            'ilp': ilp_output,
            'analytical': analytical_output,
            'relative_error': output_err,
            'passed': output_match,
        },
        'latency': {
            'ilp': ilp_latency,
            'analytical': analytical_latency['latency'],
            'relative_error': latency_err,
            'passed': latency_match,
        },
        'analytical_breakdown': analytical,
        'analytical_latency_breakdown': analytical_latency,
    }
    
    return CostModelVerification(
        passed=passed,
        input_reads_match=input_match,
        weight_reads_match=weight_match,
        output_reads_match=output_match,
        latency_match=latency_match,
        details=details,
        tolerance=tolerance,
    )


# =============================================================================
# Optimality Verification
# =============================================================================

def verify_optimality(
    ilp_mapping: Mapping,
    bounds: LoopBounds,
    objective: str = 'latency',
    tolerance: float = 0.001,
    max_mappings: int = 10000,
    verbose: bool = False,
) -> OptimalityVerification:
    """
    Verify ILP solution optimality by exhaustive search.
    
    Args:
        ilp_mapping: Mapping from ILP solver
        bounds: Loop bounds
        objective: Optimization objective
        tolerance: Acceptable relative gap
        max_mappings: Max mappings to evaluate
        verbose: Print progress
        
    Returns:
        OptimalityVerification with detailed results
    """
    is_optimal, details = verify_ilp_optimality(
        ilp_mapping, bounds, objective, tolerance,
        enumerate_loop_order=False,
        max_mappings=max_mappings,
        verbose=verbose,
    )
    
    return OptimalityVerification(
        is_optimal=is_optimal,
        relative_gap=details['relative_gap'],
        ilp_rank=details['ilp_rank'],
        total_solutions=details['num_mappings_evaluated'],
        ilp_cost=details['ilp_cost'],
        optimal_cost=details['optimal_cost'],
        details=details,
    )


# =============================================================================
# Complete Verification
# =============================================================================

def compare_with_ilp(
    test_name: str,
    bounds: LoopBounds,
    ilp_factors: TileFactors,
    ilp_loop_order: List[str],
    ilp_results: Dict[str, Any],
    objective: str = 'latency',
    cost_tolerance: float = 0.01,
    opt_tolerance: float = 0.001,
    max_mappings: int = 10000,
    verify_row_activation: bool = False,
    row_config: Optional[RowBufferConfig] = None,
    verbose: bool = False,
) -> VerificationResult:
    """
    Complete verification of ILP solution.
    
    Args:
        test_name: Name for this test case
        bounds: Loop bounds
        ilp_factors: Tile factors from ILP
        ilp_loop_order: Loop order from ILP
        ilp_results: ILP computed results
        objective: Optimization objective
        cost_tolerance: Tolerance for cost model verification
        opt_tolerance: Tolerance for optimality verification
        max_mappings: Max mappings for exhaustive search
        verify_row_activation: Whether to verify row activation
        row_config: Row buffer configuration
        verbose: Print progress
        
    Returns:
        VerificationResult with all results
    """
    # Create mapping object
    ilp_mapping = Mapping(l1_factors=ilp_factors, loop_order=ilp_loop_order)
    
    # 1. Verify cost model
    if verbose:
        print(f"\n[{test_name}] Verifying cost model...")
    cost_result = verify_cost_model(
        ilp_results, bounds, ilp_factors, ilp_loop_order, cost_tolerance
    )
    if verbose:
        status = "✓ PASS" if cost_result.passed else "✗ FAIL"
        print(f"  Cost model: {status}")
    
    # 2. Verify optimality (only for small search spaces)
    if verbose:
        print(f"[{test_name}] Verifying optimality...")
    opt_result = verify_optimality(
        ilp_mapping, bounds, objective, opt_tolerance, max_mappings, verbose
    )
    if verbose:
        status = "✓ OPTIMAL" if opt_result.is_optimal else f"✗ SUBOPTIMAL (gap={opt_result.relative_gap:.4f})"
        print(f"  Optimality: {status}")
    
    # 3. Verify row activation (optional)
    row_result = None
    if verify_row_activation and row_config is not None:
        if verbose:
            print(f"[{test_name}] Verifying row activation...")
        
        row_result = {}
        for datatype in ['input', 'weight', 'output']:
            if f'{datatype}_row_acts' in ilp_results:
                passed, details = verify_row_activation_model(
                    ilp_results[f'{datatype}_row_acts'],
                    ilp_results[f'{datatype}_reads'],
                    ilp_results.get(f'{datatype}_tile_info', {}),
                    row_config,
                    datatype,
                )
                row_result[datatype] = details
        
        if verbose:
            for dt, res in row_result.items():
                status = "✓" if res['passed'] else "✗"
                print(f"    {dt}: {status} (err={res['relative_error']:.4f})")
    
    # Overall result
    overall = cost_result.passed and opt_result.is_optimal
    if row_result:
        overall = overall and all(r['passed'] for r in row_result.values())
    
    summary_parts = []
    summary_parts.append(f"Cost model: {'PASS' if cost_result.passed else 'FAIL'}")
    summary_parts.append(f"Optimality: {'OPTIMAL' if opt_result.is_optimal else f'GAP={opt_result.relative_gap:.2%}'}")
    if row_result:
        row_status = 'PASS' if all(r['passed'] for r in row_result.values()) else 'FAIL'
        summary_parts.append(f"Row activation: {row_status}")
    
    summary = " | ".join(summary_parts)
    
    return VerificationResult(
        test_name=test_name,
        bounds=bounds,
        ilp_mapping=ilp_mapping,
        cost_model_result=cost_result,
        optimality_result=opt_result,
        row_activation_result=row_result,
        overall_passed=overall,
        summary=summary,
    )


# =============================================================================
# Test Suite
# =============================================================================

def run_verification_suite(
    test_cases: List[Dict[str, Any]],
    verbose: bool = True,
) -> List[VerificationResult]:
    """
    Run verification on multiple test cases.
    
    Args:
        test_cases: List of test case dictionaries
        verbose: Print progress
        
    Returns:
        List of VerificationResults
    """
    results = []
    
    for tc in test_cases:
        name = tc.get('name', 'unnamed')
        bounds = tc['bounds']
        factors = tc['factors']
        loop_order = tc.get('loop_order', ['N', 'C', 'K', 'P', 'Q', 'R', 'S'])
        ilp_results = tc.get('ilp_results', {})
        objective = tc.get('objective', 'latency')
        
        result = compare_with_ilp(
            test_name=name,
            bounds=bounds,
            ilp_factors=factors,
            ilp_loop_order=loop_order,
            ilp_results=ilp_results,
            objective=objective,
            verbose=verbose,
        )
        results.append(result)
    
    if verbose:
        print("\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)
        passed = sum(1 for r in results if r.overall_passed)
        print(f"Passed: {passed}/{len(results)}")
        for r in results:
            status = "✓" if r.overall_passed else "✗"
            print(f"  {status} {r.test_name}: {r.summary}")
    
    return results


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    # Create test case
    bounds = LoopBounds(N=4, C=8, K=8, P=4, Q=4, R=3, S=3)
    factors = TileFactors(N=2, C=4, K=4, P=2, Q=2, R=1, S=1)
    loop_order = ['N', 'K', 'C', 'P', 'Q', 'R', 'S']
    
    # Simulate ILP results (use analytical to get "correct" values)
    analytical = compute_analytical_memory_reads(bounds, factors, loop_order)
    analytical_latency = compute_analytical_latency(bounds, factors, loop_order)
    
    ilp_results = {
        'input_reads': analytical['input_tile_size'] * analytical['input_access_count'],
        'weight_reads': analytical['weight_tile_size'] * analytical['weight_access_count'],
        'output_reads': analytical['output_tile_size'] * analytical['output_access_count'],
        'latency': analytical_latency['latency'],
    }
    
    # Run verification
    result = compare_with_ilp(
        test_name="test_4x8x8_conv",
        bounds=bounds,
        ilp_factors=factors,
        ilp_loop_order=loop_order,
        ilp_results=ilp_results,
        verbose=True,
    )
    
    print(f"\nOverall: {'PASS' if result.overall_passed else 'FAIL'}")
