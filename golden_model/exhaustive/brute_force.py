"""
Brute-force enumeration of all valid mappings.

This module implements exhaustive search over:
1. Tile factors for each dimension
2. Loop orderings (permutations)
3. Data layouts

For small problem sizes, we can enumerate all mappings and find
the globally optimal solution to verify the ILP model.
"""

import itertools
import math
from typing import List, Dict, Tuple, Optional, Generator, Callable
from dataclasses import dataclass

# Import cost formulas from analytical module
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


# =============================================================================
# Factor Enumeration
# =============================================================================

def get_divisors(n: int) -> List[int]:
    """Get all divisors of n."""
    divisors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted(divisors)


def enumerate_tile_factors(bounds: LoopBounds) -> Generator[TileFactors, None, None]:
    """
    Enumerate all valid tile factor combinations.
    
    For each dimension, the tile factor must divide the loop bound.
    
    Args:
        bounds: Loop bounds for each dimension
        
    Yields:
        TileFactors objects representing valid tiling
    """
    n_factors = get_divisors(bounds.N)
    c_factors = get_divisors(bounds.C)
    k_factors = get_divisors(bounds.K)
    p_factors = get_divisors(bounds.P)
    q_factors = get_divisors(bounds.Q)
    r_factors = get_divisors(bounds.R)
    s_factors = get_divisors(bounds.S)
    
    for n, c, k, p, q, r, s in itertools.product(
        n_factors, c_factors, k_factors,
        p_factors, q_factors, r_factors, s_factors
    ):
        yield TileFactors(N=n, C=c, K=k, P=p, Q=q, R=r, S=s)


def enumerate_two_level_tile_factors(
    bounds: LoopBounds,
    resource_constraint: Optional[Callable[[TileFactors, TileFactors], bool]] = None,
) -> Generator[Tuple[TileFactors, TileFactors], None, None]:
    """
    Enumerate all valid two-level (L1, L2) tile factor combinations.
    
    Constraint: L1_factor × L2_factor <= loop_bound
    
    Args:
        bounds: Loop bounds for each dimension
        resource_constraint: Optional function to check resource constraints
        
    Yields:
        Tuples of (L1_factors, L2_factors)
    """
    # Generate all factor pairs for each dimension
    def get_factor_pairs(bound: int) -> List[Tuple[int, int]]:
        """Get all (l1, l2) pairs where l1 * l2 <= bound."""
        pairs = []
        divisors = get_divisors(bound)
        for l1 in divisors:
            for l2 in divisors:
                if l1 * l2 <= bound:
                    pairs.append((l1, l2))
        return pairs
    
    n_pairs = get_factor_pairs(bounds.N)
    c_pairs = get_factor_pairs(bounds.C)
    k_pairs = get_factor_pairs(bounds.K)
    p_pairs = get_factor_pairs(bounds.P)
    q_pairs = get_factor_pairs(bounds.Q)
    r_pairs = get_factor_pairs(bounds.R)
    s_pairs = get_factor_pairs(bounds.S)
    
    for n, c, k, p, q, r, s in itertools.product(
        n_pairs, c_pairs, k_pairs, p_pairs, q_pairs, r_pairs, s_pairs
    ):
        l1_factors = TileFactors(
            N=n[0], C=c[0], K=k[0], P=p[0], Q=q[0], R=r[0], S=s[0]
        )
        l2_factors = TileFactors(
            N=n[1], C=c[1], K=k[1], P=p[1], Q=q[1], R=r[1], S=s[1]
        )
        
        if resource_constraint is None or resource_constraint(l1_factors, l2_factors):
            yield l1_factors, l2_factors


# =============================================================================
# Loop Order Enumeration
# =============================================================================

LOOP_DIMS = ['N', 'C', 'K', 'P', 'Q', 'R', 'S']


def enumerate_loop_orders(dims: List[str] = None) -> Generator[List[str], None, None]:
    """
    Enumerate all possible loop orderings.
    
    Args:
        dims: List of dimension names (default: LOOP_DIMS)
        
    Yields:
        List representing loop order (outer to inner)
    """
    if dims is None:
        dims = LOOP_DIMS
    
    for perm in itertools.permutations(dims):
        yield list(perm)


def enumerate_partial_loop_orders(
    dims: List[str] = None,
    num_outer: int = 3,
) -> Generator[Tuple[List[str], List[str]], None, None]:
    """
    Enumerate outer loop orders while keeping inner loops fixed.
    
    This reduces search space significantly.
    
    Args:
        dims: List of dimension names
        num_outer: Number of outer loops to permute
        
    Yields:
        Tuples of (outer_order, inner_order)
    """
    if dims is None:
        dims = LOOP_DIMS
    
    for outer_perm in itertools.permutations(dims, num_outer):
        outer = list(outer_perm)
        inner = [d for d in dims if d not in outer]
        yield outer, inner


# =============================================================================
# Full Mapping Enumeration
# =============================================================================

@dataclass
class Mapping:
    """Represents a complete mapping configuration."""
    l1_factors: TileFactors
    l2_factors: TileFactors = None  # Optional second level
    loop_order: List[str] = None
    
    # Computed costs
    input_reads: int = 0
    weight_reads: int = 0
    output_reads: int = 0
    output_writes: int = 0
    total_memory_ops: int = 0
    compute_cycles: int = 0
    memory_cycles: int = 0
    latency: int = 0
    
    def __hash__(self):
        return hash((
            tuple(vars(self.l1_factors).values()) if self.l1_factors else None,
            tuple(vars(self.l2_factors).values()) if self.l2_factors else None,
            tuple(self.loop_order) if self.loop_order else None,
        ))


def enumerate_all_mappings(
    bounds: LoopBounds,
    enumerate_loop_order: bool = True,
    max_mappings: int = 100000,
) -> Generator[Mapping, None, None]:
    """
    Enumerate all valid mappings for given loop bounds.
    
    Args:
        bounds: Loop bounds
        enumerate_loop_order: Whether to enumerate loop orders
        max_mappings: Maximum number of mappings to enumerate
        
    Yields:
        Mapping objects
    """
    count = 0
    
    for factors in enumerate_tile_factors(bounds):
        if enumerate_loop_order:
            for order in enumerate_loop_orders():
                if count >= max_mappings:
                    return
                yield Mapping(l1_factors=factors, loop_order=order)
                count += 1
        else:
            if count >= max_mappings:
                return
            yield Mapping(l1_factors=factors, loop_order=LOOP_DIMS)
            count += 1


def count_mapping_space(bounds: LoopBounds, include_loop_order: bool = True) -> int:
    """
    Count the total number of valid mappings.
    
    Args:
        bounds: Loop bounds
        include_loop_order: Whether to include loop order permutations
        
    Returns:
        Number of valid mappings
    """
    num_tile_combos = (
        len(get_divisors(bounds.N)) *
        len(get_divisors(bounds.C)) *
        len(get_divisors(bounds.K)) *
        len(get_divisors(bounds.P)) *
        len(get_divisors(bounds.Q)) *
        len(get_divisors(bounds.R)) *
        len(get_divisors(bounds.S))
    )
    
    if include_loop_order:
        num_loop_orders = math.factorial(7)  # 7! = 5040
        return num_tile_combos * num_loop_orders
    else:
        return num_tile_combos


# =============================================================================
# Optimal Solution Finding
# =============================================================================

def evaluate_mapping(
    mapping: Mapping,
    bounds: LoopBounds,
    stride: int = 1,
    dilation: int = 1,
    memory_bandwidth_gb_s: float = 25.6,
    compute_throughput_gops: float = 100.0,
) -> Mapping:
    """
    Evaluate cost for a mapping.
    
    Args:
        mapping: Mapping to evaluate
        bounds: Loop bounds
        stride: Convolution stride
        dilation: Convolution dilation
        memory_bandwidth_gb_s: Memory bandwidth
        compute_throughput_gops: Compute throughput
        
    Returns:
        Mapping with computed costs filled in
    """
    factors = mapping.l1_factors
    
    # Compute tile sizes
    input_tile = compute_input_tile_size(factors, stride, dilation)
    weight_tile = compute_weight_tile_size(factors)
    output_tile = compute_output_tile_size(factors)
    
    # Compute memory reads
    mem_reads = compute_analytical_memory_reads(
        bounds, factors, mapping.loop_order
    )
    
    mapping.input_reads = mem_reads['input_tile_size'] * mem_reads['input_access_count']
    mapping.weight_reads = mem_reads['weight_tile_size'] * mem_reads['weight_access_count']
    mapping.output_reads = mem_reads['output_tile_size'] * mem_reads['output_access_count']
    mapping.output_writes = mem_reads['output_tile_size'] * mem_reads['output_access_count']
    mapping.total_memory_ops = (
        mapping.input_reads + mapping.weight_reads + 
        mapping.output_reads + mapping.output_writes
    )
    
    # Compute latency
    latency_info = compute_analytical_latency(
        bounds, factors, mapping.loop_order,
        memory_bandwidth_gb_s, compute_throughput_gops
    )
    mapping.compute_cycles = latency_info['compute_cycles']
    mapping.memory_cycles = latency_info['memory_cycles']
    mapping.latency = latency_info['latency']
    
    return mapping


def find_optimal_exhaustive(
    bounds: LoopBounds,
    objective: str = 'latency',
    stride: int = 1,
    dilation: int = 1,
    enumerate_loop_order: bool = False,
    max_mappings: int = 100000,
    verbose: bool = False,
) -> Tuple[Mapping, List[Mapping]]:
    """
    Find optimal mapping by exhaustive search.
    
    Args:
        bounds: Loop bounds
        objective: Optimization objective ('latency', 'memory', 'compute')
        stride: Convolution stride
        dilation: Convolution dilation
        enumerate_loop_order: Whether to enumerate loop orders
        max_mappings: Maximum mappings to evaluate
        verbose: Print progress
        
    Returns:
        Tuple of (optimal_mapping, all_evaluated_mappings)
    """
    best_mapping = None
    best_cost = float('inf')
    all_mappings = []
    
    total = min(count_mapping_space(bounds, enumerate_loop_order), max_mappings)
    
    for i, mapping in enumerate(enumerate_all_mappings(bounds, enumerate_loop_order, max_mappings)):
        # Evaluate mapping
        mapping = evaluate_mapping(mapping, bounds, stride, dilation)
        all_mappings.append(mapping)
        
        # Get cost based on objective
        if objective == 'latency':
            cost = mapping.latency
        elif objective == 'memory':
            cost = mapping.total_memory_ops
        elif objective == 'compute':
            cost = mapping.compute_cycles
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Update best
        if cost < best_cost:
            best_cost = cost
            best_mapping = mapping
        
        # Progress
        if verbose and (i + 1) % 1000 == 0:
            print(f"Evaluated {i + 1}/{total} mappings, best {objective}={best_cost}")
    
    if verbose:
        print(f"Exhaustive search complete: {len(all_mappings)} mappings evaluated")
        print(f"Best {objective} = {best_cost}")
    
    return best_mapping, all_mappings


def verify_ilp_optimality(
    ilp_mapping: Mapping,
    bounds: LoopBounds,
    objective: str = 'latency',
    tolerance: float = 0.001,
    **kwargs,
) -> Tuple[bool, Dict]:
    """
    Verify that ILP solution is optimal by exhaustive search.
    
    Args:
        ilp_mapping: Mapping from ILP solver
        bounds: Loop bounds
        objective: Optimization objective
        tolerance: Acceptable relative error
        **kwargs: Additional arguments for find_optimal_exhaustive
        
    Returns:
        Tuple of (is_optimal, details)
    """
    # Find optimal by exhaustive search
    optimal_mapping, all_mappings = find_optimal_exhaustive(
        bounds, objective, **kwargs
    )
    
    # Evaluate ILP mapping
    ilp_evaluated = evaluate_mapping(ilp_mapping, bounds)
    
    # Get costs
    if objective == 'latency':
        ilp_cost = ilp_evaluated.latency
        optimal_cost = optimal_mapping.latency
    elif objective == 'memory':
        ilp_cost = ilp_evaluated.total_memory_ops
        optimal_cost = optimal_mapping.total_memory_ops
    elif objective == 'compute':
        ilp_cost = ilp_evaluated.compute_cycles
        optimal_cost = optimal_mapping.compute_cycles
    else:
        raise ValueError(f"Unknown objective: {objective}")
    
    # Check optimality
    if optimal_cost == 0:
        is_optimal = ilp_cost == 0
        relative_gap = 0.0
    else:
        relative_gap = (ilp_cost - optimal_cost) / optimal_cost
        is_optimal = relative_gap <= tolerance
    
    # Rank of ILP solution
    costs = []
    for m in all_mappings:
        if objective == 'latency':
            costs.append(m.latency)
        elif objective == 'memory':
            costs.append(m.total_memory_ops)
        else:
            costs.append(m.compute_cycles)
    
    sorted_costs = sorted(set(costs))
    ilp_rank = sorted_costs.index(ilp_cost) + 1 if ilp_cost in sorted_costs else -1
    
    details = {
        'objective': objective,
        'ilp_cost': ilp_cost,
        'optimal_cost': optimal_cost,
        'relative_gap': relative_gap,
        'is_optimal': is_optimal,
        'tolerance': tolerance,
        'num_mappings_evaluated': len(all_mappings),
        'ilp_rank': ilp_rank,
        'total_unique_costs': len(sorted_costs),
        'optimal_mapping': optimal_mapping,
        'ilp_mapping_evaluated': ilp_evaluated,
    }
    
    return is_optimal, details


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    # Small test case
    bounds = LoopBounds(N=4, C=4, K=4, P=4, Q=4, R=3, S=3)
    
    print("Counting mapping space...")
    num_mappings = count_mapping_space(bounds, include_loop_order=False)
    print(f"Total mappings (no loop order): {num_mappings}")
    
    print("\nFinding optimal mapping...")
    optimal, all_maps = find_optimal_exhaustive(
        bounds, 
        objective='latency',
        enumerate_loop_order=False,
        verbose=True,
    )
    
    print(f"\nOptimal mapping:")
    print(f"  Tile factors: N={optimal.l1_factors.N}, C={optimal.l1_factors.C}, "
          f"K={optimal.l1_factors.K}, P={optimal.l1_factors.P}, Q={optimal.l1_factors.Q}")
    print(f"  Total memory ops: {optimal.total_memory_ops}")
    print(f"  Latency: {optimal.latency}")
