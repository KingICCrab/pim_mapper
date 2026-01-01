"""
Report generation for golden model verification.

This module provides:
- Pretty printing of verification results
- Export to CSV for analysis
- Detailed breakdown reports
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime

# Import verification types
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comparison.compare import (
    VerificationResult,
    CostModelVerification,
    OptimalityVerification,
)
from analytical.cost_formulas import LoopBounds, TileFactors


# =============================================================================
# Report Data Structures  
# =============================================================================

@dataclass
class VerificationReport:
    """Complete verification report."""
    timestamp: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    results: List[VerificationResult]
    summary: str


# =============================================================================
# Pretty Printing
# =============================================================================

def print_verification_summary(results: List[VerificationResult]) -> None:
    """
    Print a summary of verification results.
    
    Args:
        results: List of verification results
    """
    print("\n" + "=" * 70)
    print("GOLDEN MODEL VERIFICATION REPORT")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total tests: {len(results)}")
    
    passed = sum(1 for r in results if r.overall_passed)
    failed = len(results) - passed
    
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("-" * 70)
    
    # Print each result
    for r in results:
        status = "✓ PASS" if r.overall_passed else "✗ FAIL"
        print(f"\n{status} {r.test_name}")
        print(f"  Bounds: N={r.bounds.N}, C={r.bounds.C}, K={r.bounds.K}, "
              f"P={r.bounds.P}, Q={r.bounds.Q}, R={r.bounds.R}, S={r.bounds.S}")
        print(f"  Factors: N={r.ilp_mapping.l1_factors.N}, C={r.ilp_mapping.l1_factors.C}, "
              f"K={r.ilp_mapping.l1_factors.K}, P={r.ilp_mapping.l1_factors.P}, "
              f"Q={r.ilp_mapping.l1_factors.Q}")
        print(f"  {r.summary}")
        
        # Cost model details
        if not r.cost_model_result.passed:
            print("  Cost Model Failures:")
            for dtype in ['input', 'weight', 'output', 'latency']:
                details = r.cost_model_result.details[dtype]
                if not details['passed']:
                    print(f"    - {dtype}: ILP={details['ilp']}, "
                          f"Analytical={details['analytical']}, "
                          f"Error={details['relative_error']:.2%}")
        
        # Optimality details
        if not r.optimality_result.is_optimal:
            print(f"  Optimality Gap: {r.optimality_result.relative_gap:.2%}")
            print(f"  ILP Rank: {r.optimality_result.ilp_rank}/{r.optimality_result.total_solutions}")
    
    print("\n" + "=" * 70)


def print_detailed_result(result: VerificationResult) -> None:
    """
    Print detailed breakdown of a single verification result.
    
    Args:
        result: Verification result to print
    """
    print("\n" + "=" * 70)
    print(f"DETAILED REPORT: {result.test_name}")
    print("=" * 70)
    
    # Problem specification
    print("\n1. PROBLEM SPECIFICATION")
    print("-" * 40)
    b = result.bounds
    print(f"  Loop Bounds:")
    print(f"    N (batch)      = {b.N}")
    print(f"    C (channels)   = {b.C}")
    print(f"    K (filters)    = {b.K}")
    print(f"    P (height)     = {b.P}")
    print(f"    Q (width)      = {b.Q}")
    print(f"    R (kernel H)   = {b.R}")
    print(f"    S (kernel W)   = {b.S}")
    
    # ILP solution
    print("\n2. ILP SOLUTION")
    print("-" * 40)
    f = result.ilp_mapping.l1_factors
    print(f"  Tile Factors:")
    print(f"    N_factor = {f.N}")
    print(f"    C_factor = {f.C}")
    print(f"    K_factor = {f.K}")
    print(f"    P_factor = {f.P}")
    print(f"    Q_factor = {f.Q}")
    print(f"    R_factor = {f.R}")
    print(f"    S_factor = {f.S}")
    print(f"  Loop Order: {result.ilp_mapping.loop_order}")
    
    # Cost model verification
    print("\n3. COST MODEL VERIFICATION")
    print("-" * 40)
    cm = result.cost_model_result
    status = "PASS" if cm.passed else "FAIL"
    print(f"  Overall: {status} (tolerance={cm.tolerance:.2%})")
    print()
    
    for dtype in ['input', 'weight', 'output', 'latency']:
        d = cm.details[dtype]
        s = "✓" if d['passed'] else "✗"
        print(f"  {s} {dtype.capitalize()}:")
        print(f"      ILP value:        {d['ilp']}")
        print(f"      Analytical value: {d['analytical']}")
        print(f"      Relative error:   {d['relative_error']:.4%}")
    
    # Analytical breakdown
    print("\n  Analytical Breakdown:")
    ab = cm.details['analytical_breakdown']
    print(f"    Input:  tile_size={ab['input_tile_size']}, access_count={ab['input_access_count']}")
    print(f"    Weight: tile_size={ab['weight_tile_size']}, access_count={ab['weight_access_count']}")
    print(f"    Output: tile_size={ab['output_tile_size']}, access_count={ab['output_access_count']}")
    
    # Optimality verification
    print("\n4. OPTIMALITY VERIFICATION")
    print("-" * 40)
    opt = result.optimality_result
    status = "OPTIMAL" if opt.is_optimal else "SUBOPTIMAL"
    print(f"  Result: {status}")
    print(f"  ILP Cost:     {opt.ilp_cost}")
    print(f"  Optimal Cost: {opt.optimal_cost}")
    print(f"  Relative Gap: {opt.relative_gap:.4%}")
    print(f"  ILP Rank:     {opt.ilp_rank} out of {opt.total_solutions}")
    
    # Row activation (if available)
    if result.row_activation_result:
        print("\n5. ROW ACTIVATION VERIFICATION")
        print("-" * 40)
        for dtype, details in result.row_activation_result.items():
            s = "✓" if details['passed'] else "✗"
            print(f"  {s} {dtype.capitalize()}:")
            print(f"      ILP row activations:        {details['ilp_row_acts']}")
            print(f"      Analytical row activations: {details['analytical_row_acts']}")
            print(f"      Relative error:             {details['relative_error']:.4%}")
    
    # Overall
    print("\n" + "=" * 70)
    status = "PASS" if result.overall_passed else "FAIL"
    print(f"OVERALL: {status}")
    print("=" * 70)


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(
    results: List[VerificationResult],
    output_format: str = 'text',
) -> str:
    """
    Generate verification report in specified format.
    
    Args:
        results: List of verification results
        output_format: 'text', 'json', or 'markdown'
        
    Returns:
        Formatted report string
    """
    passed = sum(1 for r in results if r.overall_passed)
    failed = len(results) - passed
    
    if output_format == 'json':
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(results),
            'passed': passed,
            'failed': failed,
            'results': []
        }
        
        for r in results:
            report_data['results'].append({
                'test_name': r.test_name,
                'passed': r.overall_passed,
                'summary': r.summary,
                'bounds': vars(r.bounds),
                'factors': vars(r.ilp_mapping.l1_factors),
                'cost_model_passed': r.cost_model_result.passed,
                'is_optimal': r.optimality_result.is_optimal,
                'optimality_gap': r.optimality_result.relative_gap,
            })
        
        return json.dumps(report_data, indent=2)
    
    elif output_format == 'markdown':
        lines = [
            "# Golden Model Verification Report",
            "",
            f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- Total tests: {len(results)}",
            f"- Passed: {passed}",
            f"- Failed: {failed}",
            "",
            "## Results",
            "",
            "| Test Name | Status | Cost Model | Optimality | Gap |",
            "|-----------|--------|------------|------------|-----|",
        ]
        
        for r in results:
            status = "✓" if r.overall_passed else "✗"
            cm_status = "✓" if r.cost_model_result.passed else "✗"
            opt_status = "✓" if r.optimality_result.is_optimal else "✗"
            gap = f"{r.optimality_result.relative_gap:.2%}"
            lines.append(f"| {r.test_name} | {status} | {cm_status} | {opt_status} | {gap} |")
        
        return "\n".join(lines)
    
    else:  # text format
        lines = [
            "=" * 60,
            "GOLDEN MODEL VERIFICATION REPORT",
            "=" * 60,
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total: {len(results)}, Passed: {passed}, Failed: {failed}",
            "-" * 60,
        ]
        
        for r in results:
            status = "PASS" if r.overall_passed else "FAIL"
            lines.append(f"[{status}] {r.test_name}: {r.summary}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# CSV Export
# =============================================================================

def export_to_csv(
    results: List[VerificationResult],
    filepath: str,
) -> None:
    """
    Export verification results to CSV file.
    
    Args:
        results: List of verification results
        filepath: Output CSV file path
    """
    import csv
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'test_name', 'overall_passed',
            'N', 'C', 'K', 'P', 'Q', 'R', 'S',
            'N_factor', 'C_factor', 'K_factor', 'P_factor', 'Q_factor', 'R_factor', 'S_factor',
            'cost_model_passed',
            'input_reads_ilp', 'input_reads_analytical', 'input_error',
            'weight_reads_ilp', 'weight_reads_analytical', 'weight_error',
            'output_reads_ilp', 'output_reads_analytical', 'output_error',
            'latency_ilp', 'latency_analytical', 'latency_error',
            'is_optimal', 'optimality_gap', 'ilp_rank', 'total_solutions',
        ])
        
        # Data rows
        for r in results:
            b = r.bounds
            f = r.ilp_mapping.l1_factors
            cm = r.cost_model_result
            opt = r.optimality_result
            
            row = [
                r.test_name, r.overall_passed,
                b.N, b.C, b.K, b.P, b.Q, b.R, b.S,
                f.N, f.C, f.K, f.P, f.Q, f.R, f.S,
                cm.passed,
                cm.details['input']['ilp'], cm.details['input']['analytical'], cm.details['input']['relative_error'],
                cm.details['weight']['ilp'], cm.details['weight']['analytical'], cm.details['weight']['relative_error'],
                cm.details['output']['ilp'], cm.details['output']['analytical'], cm.details['output']['relative_error'],
                cm.details['latency']['ilp'], cm.details['latency']['analytical'], cm.details['latency']['relative_error'],
                opt.is_optimal, opt.relative_gap, opt.ilp_rank, opt.total_solutions,
            ]
            writer.writerow(row)
    
    print(f"Results exported to {filepath}")


# =============================================================================
# Comparison Table
# =============================================================================

def print_comparison_table(results: List[VerificationResult]) -> None:
    """
    Print a compact comparison table.
    
    Args:
        results: List of verification results
    """
    # Header
    print()
    print("-" * 100)
    print(f"{'Test':<20} {'Status':<8} {'CostModel':<10} {'Optimal':<8} {'Gap':<10} {'Rank':<10}")
    print("-" * 100)
    
    for r in results:
        status = "PASS" if r.overall_passed else "FAIL"
        cm = "✓" if r.cost_model_result.passed else "✗"
        opt = "✓" if r.optimality_result.is_optimal else "✗"
        gap = f"{r.optimality_result.relative_gap:.2%}"
        rank = f"{r.optimality_result.ilp_rank}/{r.optimality_result.total_solutions}"
        
        print(f"{r.test_name:<20} {status:<8} {cm:<10} {opt:<8} {gap:<10} {rank:<10}")
    
    print("-" * 100)
    
    # Summary
    passed = sum(1 for r in results if r.overall_passed)
    print(f"Total: {len(results)}, Passed: {passed}, Failed: {len(results) - passed}")
    print()


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    from compare import compare_with_ilp
    from analytical.cost_formulas import compute_analytical_memory_reads, compute_analytical_latency
    
    # Create test cases
    test_cases = []
    
    for n, c, k in [(4, 8, 8), (2, 16, 16), (8, 4, 4)]:
        bounds = LoopBounds(N=n, C=c, K=k, P=4, Q=4, R=3, S=3)
        factors = TileFactors(N=n//2 or 1, C=c//2 or 1, K=k//2 or 1, P=2, Q=2, R=1, S=1)
        loop_order = ['N', 'K', 'C', 'P', 'Q', 'R', 'S']
        
        # Compute "ILP" results
        analytical = compute_analytical_memory_reads(bounds, factors, loop_order)
        analytical_latency = compute_analytical_latency(bounds, factors, loop_order)
        
        ilp_results = {
            'input_reads': analytical['input_tile_size'] * analytical['input_access_count'],
            'weight_reads': analytical['weight_tile_size'] * analytical['weight_access_count'],
            'output_reads': analytical['output_tile_size'] * analytical['output_access_count'],
            'latency': analytical_latency['latency'],
        }
        
        result = compare_with_ilp(
            test_name=f"conv_{n}x{c}x{k}",
            bounds=bounds,
            ilp_factors=factors,
            ilp_loop_order=loop_order,
            ilp_results=ilp_results,
            verbose=False,
        )
        test_cases.append(result)
    
    # Print reports
    print_verification_summary(test_cases)
    print_comparison_table(test_cases)
    
    # Print detailed for first case
    print_detailed_result(test_cases[0])
    
    # Generate markdown report
    print("\n" + generate_report(test_cases, 'markdown'))
