"""
ILP Optimizer Verification using Golden Model Simulator.

This module provides tools to verify:
1. Cost model correctness: Are the analytical formulas in ILP accurate?
2. Optimality verification: Does ILP find the truly optimal solution?

Key Concept:
- ILP uses analytical cost models (estimates)
- Golden Model Simulator provides ground truth
- Comparison reveals modeling errors or suboptimal solutions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import math

from .simulator import Simulator, SimulatorConfig, SimulationResult, simulate_mapping
from .access_trace import AccessTrace, AccessPatternGenerator


@dataclass
class VerificationResult:
    """
    Results from comparing ILP solution with simulation ground truth.
    """
    # ILP predictions
    ilp_row_activations: int = 0
    ilp_total_cycles: int = 0
    ilp_cost: float = 0.0
    
    # Simulation ground truth
    sim_row_activations: int = 0
    sim_total_cycles: int = 0
    sim_row_buffer_hit_rate: float = 0.0
    
    # Comparison metrics
    row_activation_error: float = 0.0
    cycle_error: float = 0.0
    is_cost_model_accurate: bool = False
    
    # Optimality check
    is_optimal: Optional[bool] = None  # None if not checked
    better_mapping_found: Optional[Dict] = None
    
    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "Verification Result",
            "=" * 60,
            "",
            "ILP Predictions:",
            f"  Row activations: {self.ilp_row_activations}",
            f"  Total cycles: {self.ilp_total_cycles}",
            f"  Cost objective: {self.ilp_cost:.2f}",
            "",
            "Simulation Ground Truth:",
            f"  Row activations: {self.sim_row_activations}",
            f"  Total cycles: {self.sim_total_cycles}",
            f"  Row buffer hit rate: {self.sim_row_buffer_hit_rate:.2%}",
            "",
            "Comparison:",
            f"  Row activation error: {self.row_activation_error:.2%}",
            f"  Cycle error: {self.cycle_error:.2%}",
            f"  Cost model accurate: {'✓' if self.is_cost_model_accurate else '✗'}",
        ]
        
        if self.is_optimal is not None:
            lines.append("")
            lines.append("Optimality Check:")
            lines.append(f"  Is optimal: {'✓' if self.is_optimal else '✗'}")
            if self.better_mapping_found:
                lines.append(f"  Better mapping: {self.better_mapping_found}")
        
        return "\n".join(lines)


class ILPVerifier:
    """
    Verifier for ILP optimizer solutions.
    
    Uses the Golden Model Simulator to verify:
    1. Cost model accuracy
    2. Solution optimality
    """
    
    def __init__(
        self,
        simulator_config: Optional[SimulatorConfig] = None,
        tolerance: float = 0.05,  # 5% tolerance for "accurate"
    ):
        """
        Initialize the verifier.
        
        Args:
            simulator_config: DRAM simulator configuration
            tolerance: Relative tolerance for considering predictions accurate
        """
        self.config = simulator_config or SimulatorConfig()
        self.tolerance = tolerance
        self.simulator = Simulator(self.config)
    
    def verify_mapping(
        self,
        mapping: Dict[str, Any],
        workload: Dict[str, Any],
        ilp_predictions: Dict[str, float],
    ) -> VerificationResult:
        """
        Verify a single mapping against ILP predictions.
        
        Args:
            mapping: The mapping configuration from ILP
            workload: The workload configuration
            ilp_predictions: Dictionary with ILP's predicted costs
                Expected keys: 'row_activations', 'total_cycles', 'cost'
                
        Returns:
            VerificationResult with comparison details
        """
        # Run simulation
        sim_result = simulate_mapping(mapping, workload, self.config)
        
        # Extract ILP predictions
        ilp_row_act = ilp_predictions.get('row_activations', 0)
        ilp_cycles = ilp_predictions.get('total_cycles', 0)
        ilp_cost = ilp_predictions.get('cost', 0.0)
        
        # Calculate errors
        row_act_error = abs(sim_result.row_activations - ilp_row_act) / max(ilp_row_act, 1)
        cycle_error = abs(sim_result.total_cycles - ilp_cycles) / max(ilp_cycles, 1)
        
        # Check if accurate within tolerance
        is_accurate = row_act_error <= self.tolerance and cycle_error <= self.tolerance
        
        return VerificationResult(
            ilp_row_activations=ilp_row_act,
            ilp_total_cycles=ilp_cycles,
            ilp_cost=ilp_cost,
            sim_row_activations=sim_result.row_activations,
            sim_total_cycles=sim_result.total_cycles,
            sim_row_buffer_hit_rate=sim_result.row_buffer_hit_rate,
            row_activation_error=row_act_error,
            cycle_error=cycle_error,
            is_cost_model_accurate=is_accurate,
        )
    
    def verify_optimality(
        self,
        ilp_mapping: Dict[str, Any],
        workload: Dict[str, Any],
        alternative_mappings: List[Dict[str, Any]],
    ) -> Tuple[VerificationResult, List[SimulationResult]]:
        """
        Verify that ILP mapping is optimal by comparing against alternatives.
        
        Args:
            ilp_mapping: The mapping found by ILP
            workload: The workload configuration
            alternative_mappings: List of alternative mappings to compare
            
        Returns:
            Tuple of (VerificationResult, list of all simulation results)
        """
        # Simulate ILP mapping
        ilp_result = simulate_mapping(ilp_mapping, workload, self.config)
        
        # Simulate all alternatives
        all_results = [(ilp_mapping, ilp_result)]
        better_mapping = None
        
        for alt_mapping in alternative_mappings:
            alt_result = simulate_mapping(alt_mapping, workload, self.config)
            all_results.append((alt_mapping, alt_result))
            
            # Check if alternative is better
            if alt_result.total_cycles < ilp_result.total_cycles:
                if better_mapping is None or alt_result.total_cycles < better_mapping[1].total_cycles:
                    better_mapping = (alt_mapping, alt_result)
        
        # Create verification result
        result = VerificationResult(
            sim_row_activations=ilp_result.row_activations,
            sim_total_cycles=ilp_result.total_cycles,
            sim_row_buffer_hit_rate=ilp_result.row_buffer_hit_rate,
            is_optimal=better_mapping is None,
            better_mapping_found=better_mapping[0] if better_mapping else None,
        )
        
        return result, [r for _, r in all_results]
    
    def exhaustive_search(
        self,
        workload: Dict[str, Any],
        tile_size_options: Dict[str, List[int]],
        max_configs: int = 1000,
    ) -> Tuple[Dict[str, Any], SimulationResult, List[Tuple[Dict, SimulationResult]]]:
        """
        Exhaustive search to find true optimal mapping.
        
        This is the ultimate ground truth - enumerate all possible
        mappings and simulate each one.
        
        Args:
            workload: The workload configuration
            tile_size_options: Dictionary mapping dimension names to possible tile sizes
            max_configs: Maximum configurations to test
            
        Returns:
            Tuple of (best_mapping, best_result, all_results)
        """
        import itertools
        
        # Generate all combinations
        dims = list(tile_size_options.keys())
        values = [tile_size_options[d] for d in dims]
        combinations = list(itertools.product(*values))
        
        if len(combinations) > max_configs:
            print(f"Warning: {len(combinations)} configs exceed max {max_configs}, sampling...")
            import random
            combinations = random.sample(combinations, max_configs)
        
        all_results = []
        best_mapping = None
        best_result = None
        
        for combo in combinations:
            mapping = {f'tile_{dims[i]}': combo[i] for i in range(len(dims))}
            
            try:
                result = simulate_mapping(mapping, workload, self.config)
                all_results.append((mapping, result))
                
                if best_result is None or result.total_cycles < best_result.total_cycles:
                    best_mapping = mapping
                    best_result = result
            except Exception as e:
                # Skip invalid configurations
                pass
        
        return best_mapping, best_result, all_results


def generate_report(
    verification_result: VerificationResult,
    detailed_stats: Optional[SimulationResult] = None,
) -> str:
    """
    Generate a detailed verification report.
    
    Args:
        verification_result: The verification result
        detailed_stats: Optional detailed simulation statistics
        
    Returns:
        Formatted report string
    """
    lines = [
        "=" * 70,
        "   PIM Optimizer Verification Report",
        "   Golden Model: Cycle-Accurate DRAM Simulator",
        "=" * 70,
        "",
        str(verification_result),
    ]
    
    if detailed_stats:
        lines.extend([
            "",
            "Detailed Simulation Statistics:",
            "-" * 40,
            f"  Total accesses: {detailed_stats.total_accesses}",
            f"  Read accesses: {detailed_stats.total_reads}",
            f"  Write accesses: {detailed_stats.total_writes}",
            "",
            "  Per-Bank Statistics:",
        ])
        
        for bank_id, stats in sorted(detailed_stats.bank_stats.items()):
            lines.append(
                f"    Bank {bank_id}: {stats['total']} accesses, "
                f"{stats['hit_rate']:.1%} hit rate"
            )
        
        if detailed_stats.tensor_stats:
            lines.extend([
                "",
                "  Per-Tensor Statistics:",
            ])
            for tensor, stats in sorted(detailed_stats.tensor_stats.items()):
                lines.append(
                    f"    {tensor}: {stats['total']} accesses, "
                    f"{stats['hit_rate']:.1%} hit rate"
                )
    
    lines.extend([
        "",
        "=" * 70,
        "Interpretation Guide:",
        "-" * 40,
        "• Row activation error > 5%: Cost model formula may be inaccurate",
        "• Cycle error > 5%: Timing model may need refinement",
        "• is_optimal = False: ILP may have suboptimal constraints",
        "",
        "Next Steps:",
        "• If cost model inaccurate: Review analytical formulas in ILP",
        "• If not optimal: Check ILP constraints and objective function",
        "• If both accurate: ILP is working correctly ✓",
        "=" * 70,
    ])
    
    return "\n".join(lines)


# Convenience function for quick verification
def quick_verify(
    mapping: Dict[str, Any],
    workload: Dict[str, Any],
    expected_row_activations: Optional[int] = None,
    expected_cycles: Optional[int] = None,
) -> bool:
    """
    Quick verification of a mapping.
    
    Args:
        mapping: The mapping configuration
        workload: The workload configuration
        expected_row_activations: Expected row activations (if known)
        expected_cycles: Expected cycles (if known)
        
    Returns:
        True if verification passes (within 5% tolerance)
    """
    result = simulate_mapping(mapping, workload)
    
    success = True
    
    if expected_row_activations is not None:
        error = abs(result.row_activations - expected_row_activations) / max(expected_row_activations, 1)
        if error > 0.05:
            print(f"Row activation mismatch: expected {expected_row_activations}, "
                  f"got {result.row_activations} (error: {error:.1%})")
            success = False
    
    if expected_cycles is not None:
        error = abs(result.total_cycles - expected_cycles) / max(expected_cycles, 1)
        if error > 0.05:
            print(f"Cycle mismatch: expected {expected_cycles}, "
                  f"got {result.total_cycles} (error: {error:.1%})")
            success = False
    
    if success:
        print(f"✓ Verification passed: {result.row_activations} row activations, "
              f"{result.total_cycles} cycles")
    
    return success
