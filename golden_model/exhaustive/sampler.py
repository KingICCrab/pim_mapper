"""
Sampling strategies for large mapping spaces.

When exhaustive search is infeasible, we use sampling to get
representative points from the mapping space.
"""

import random
import math
from typing import List, Dict, Generator, Optional
from dataclasses import dataclass

from .brute_force import (
    LoopBounds, TileFactors, Mapping,
    get_divisors, LOOP_DIMS, enumerate_loop_orders,
)


# =============================================================================
# Random Sampling
# =============================================================================

def sample_mappings_random(
    bounds: LoopBounds,
    num_samples: int = 1000,
    include_loop_order: bool = False,
    seed: Optional[int] = None,
) -> List[Mapping]:
    """
    Random sampling of mappings.
    
    Args:
        bounds: Loop bounds
        num_samples: Number of samples
        include_loop_order: Whether to sample loop orders
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled mappings
    """
    if seed is not None:
        random.seed(seed)
    
    # Pre-compute divisors
    n_divs = get_divisors(bounds.N)
    c_divs = get_divisors(bounds.C)
    k_divs = get_divisors(bounds.K)
    p_divs = get_divisors(bounds.P)
    q_divs = get_divisors(bounds.Q)
    r_divs = get_divisors(bounds.R)
    s_divs = get_divisors(bounds.S)
    
    mappings = []
    seen = set()
    
    attempts = 0
    max_attempts = num_samples * 10
    
    while len(mappings) < num_samples and attempts < max_attempts:
        attempts += 1
        
        # Sample tile factors
        factors = TileFactors(
            N=random.choice(n_divs),
            C=random.choice(c_divs),
            K=random.choice(k_divs),
            P=random.choice(p_divs),
            Q=random.choice(q_divs),
            R=random.choice(r_divs),
            S=random.choice(s_divs),
        )
        
        # Sample loop order
        if include_loop_order:
            loop_order = LOOP_DIMS.copy()
            random.shuffle(loop_order)
        else:
            loop_order = LOOP_DIMS.copy()
        
        # Create mapping
        mapping = Mapping(l1_factors=factors, loop_order=loop_order)
        
        # Deduplicate
        key = hash(mapping)
        if key not in seen:
            seen.add(key)
            mappings.append(mapping)
    
    return mappings


# =============================================================================
# Latin Hypercube Sampling
# =============================================================================

def sample_mappings_latin_hypercube(
    bounds: LoopBounds,
    num_samples: int = 100,
    seed: Optional[int] = None,
) -> List[Mapping]:
    """
    Latin Hypercube Sampling for better coverage.
    
    LHS ensures that each dimension is well-covered.
    
    Args:
        bounds: Loop bounds
        num_samples: Number of samples
        seed: Random seed
        
    Returns:
        List of sampled mappings
    """
    if seed is not None:
        random.seed(seed)
    
    # Get divisors for each dimension
    all_divs = {
        'N': get_divisors(bounds.N),
        'C': get_divisors(bounds.C),
        'K': get_divisors(bounds.K),
        'P': get_divisors(bounds.P),
        'Q': get_divisors(bounds.Q),
        'R': get_divisors(bounds.R),
        'S': get_divisors(bounds.S),
    }
    
    # Create intervals for LHS
    mappings = []
    
    for dim in all_divs:
        divs = all_divs[dim]
        n_divs = len(divs)
        
        # Create shuffled assignment
        if n_divs >= num_samples:
            indices = list(range(n_divs))
            random.shuffle(indices)
            all_divs[dim + '_indices'] = indices[:num_samples]
        else:
            # Repeat divisors to fill samples
            indices = [i % n_divs for i in range(num_samples)]
            random.shuffle(indices)
            all_divs[dim + '_indices'] = indices
    
    for i in range(num_samples):
        factors = TileFactors(
            N=all_divs['N'][all_divs['N_indices'][i]],
            C=all_divs['C'][all_divs['C_indices'][i]],
            K=all_divs['K'][all_divs['K_indices'][i]],
            P=all_divs['P'][all_divs['P_indices'][i]],
            Q=all_divs['Q'][all_divs['Q_indices'][i]],
            R=all_divs['R'][all_divs['R_indices'][i]],
            S=all_divs['S'][all_divs['S_indices'][i]],
        )
        mappings.append(Mapping(l1_factors=factors, loop_order=LOOP_DIMS.copy()))
    
    return mappings


# =============================================================================
# Boundary Case Sampling
# =============================================================================

def sample_boundary_cases(bounds: LoopBounds) -> List[Mapping]:
    """
    Sample boundary cases (extreme points of the search space).
    
    These include:
    - All 1s (smallest tiles)
    - All max (largest tiles = no tiling)
    - Single dimension tiled cases
    - Powers of 2 cases
    
    Args:
        bounds: Loop bounds
        
    Returns:
        List of boundary mappings
    """
    mappings = []
    
    # All 1s - minimum tile size
    mappings.append(Mapping(
        l1_factors=TileFactors(N=1, C=1, K=1, P=1, Q=1, R=1, S=1),
        loop_order=LOOP_DIMS.copy()
    ))
    
    # All max - no tiling
    mappings.append(Mapping(
        l1_factors=TileFactors(
            N=bounds.N, C=bounds.C, K=bounds.K,
            P=bounds.P, Q=bounds.Q, R=bounds.R, S=bounds.S
        ),
        loop_order=LOOP_DIMS.copy()
    ))
    
    # Single dimension tiled
    for dim in LOOP_DIMS:
        factors = TileFactors(N=1, C=1, K=1, P=1, Q=1, R=1, S=1)
        bound = getattr(bounds, dim)
        
        # Try power of 2 factor
        for p in [2, 4, 8, 16, 32]:
            if bound % p == 0:
                setattr(factors, dim, p)
                mappings.append(Mapping(
                    l1_factors=TileFactors(**vars(factors)),
                    loop_order=LOOP_DIMS.copy()
                ))
                break
    
    # Input stationary: maximize input reuse (large C, P, Q tiles)
    input_stat = TileFactors(
        N=1,
        C=bounds.C,
        K=1,
        P=bounds.P,
        Q=bounds.Q,
        R=bounds.R,
        S=bounds.S,
    )
    mappings.append(Mapping(l1_factors=input_stat, loop_order=LOOP_DIMS.copy()))
    
    # Weight stationary: maximize weight reuse (large C, K tiles)
    weight_stat = TileFactors(
        N=1,
        C=bounds.C,
        K=bounds.K,
        P=1,
        Q=1,
        R=bounds.R,
        S=bounds.S,
    )
    mappings.append(Mapping(l1_factors=weight_stat, loop_order=LOOP_DIMS.copy()))
    
    # Output stationary: maximize output reuse (large K, P, Q tiles)
    output_stat = TileFactors(
        N=1,
        C=1,
        K=bounds.K,
        P=bounds.P,
        Q=bounds.Q,
        R=1,
        S=1,
    )
    mappings.append(Mapping(l1_factors=output_stat, loop_order=LOOP_DIMS.copy()))
    
    return mappings


# =============================================================================
# Adaptive Sampling
# =============================================================================

def sample_around_point(
    center: TileFactors,
    bounds: LoopBounds,
    radius: int = 1,
) -> List[Mapping]:
    """
    Sample mappings in the neighborhood of a point.
    
    Useful for local refinement after finding a good solution.
    
    Args:
        center: Center point
        bounds: Loop bounds
        radius: Neighborhood radius (number of divisor steps)
        
    Returns:
        Neighboring mappings
    """
    all_divs = {
        'N': get_divisors(bounds.N),
        'C': get_divisors(bounds.C),
        'K': get_divisors(bounds.K),
        'P': get_divisors(bounds.P),
        'Q': get_divisors(bounds.Q),
        'R': get_divisors(bounds.R),
        'S': get_divisors(bounds.S),
    }
    
    def get_neighbors(val: int, divs: List[int], r: int) -> List[int]:
        idx = divs.index(val) if val in divs else 0
        neighbors = []
        for delta in range(-r, r + 1):
            new_idx = max(0, min(len(divs) - 1, idx + delta))
            neighbors.append(divs[new_idx])
        return list(set(neighbors))
    
    # Get neighbors for each dimension
    neighbors = {
        'N': get_neighbors(center.N, all_divs['N'], radius),
        'C': get_neighbors(center.C, all_divs['C'], radius),
        'K': get_neighbors(center.K, all_divs['K'], radius),
        'P': get_neighbors(center.P, all_divs['P'], radius),
        'Q': get_neighbors(center.Q, all_divs['Q'], radius),
        'R': get_neighbors(center.R, all_divs['R'], radius),
        'S': get_neighbors(center.S, all_divs['S'], radius),
    }
    
    # Generate all combinations
    mappings = []
    import itertools
    
    for n, c, k, p, q, r, s in itertools.product(
        neighbors['N'], neighbors['C'], neighbors['K'],
        neighbors['P'], neighbors['Q'], neighbors['R'], neighbors['S']
    ):
        factors = TileFactors(N=n, C=c, K=k, P=p, Q=q, R=r, S=s)
        mappings.append(Mapping(l1_factors=factors, loop_order=LOOP_DIMS.copy()))
    
    return mappings


# =============================================================================
# Stratified Sampling by Memory Footprint
# =============================================================================

def sample_by_memory_footprint(
    bounds: LoopBounds,
    num_samples: int = 100,
    num_strata: int = 10,
    seed: Optional[int] = None,
) -> List[Mapping]:
    """
    Sample mappings stratified by memory footprint.
    
    This ensures coverage across different memory utilization levels.
    
    Args:
        bounds: Loop bounds
        num_samples: Total samples
        num_strata: Number of memory footprint bins
        seed: Random seed
        
    Returns:
        Stratified samples
    """
    if seed is not None:
        random.seed(seed)
    
    # First, generate random samples
    candidates = sample_mappings_random(
        bounds, 
        num_samples=num_samples * 5,
        seed=seed
    )
    
    # Compute memory footprint for each
    def memory_footprint(factors: TileFactors) -> int:
        input_size = factors.N * factors.C * (factors.P + 2) * (factors.Q + 2)
        weight_size = factors.K * factors.C * factors.R * factors.S
        output_size = factors.N * factors.K * factors.P * factors.Q
        return input_size + weight_size + output_size
    
    # Sort by memory footprint
    candidates_with_mem = [(m, memory_footprint(m.l1_factors)) for m in candidates]
    candidates_with_mem.sort(key=lambda x: x[1])
    
    # Stratified sampling
    samples_per_stratum = num_samples // num_strata
    mappings = []
    
    chunk_size = len(candidates_with_mem) // num_strata
    
    for i in range(num_strata):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_strata - 1 else len(candidates_with_mem)
        stratum = candidates_with_mem[start:end]
        
        # Sample from this stratum
        num_to_sample = min(samples_per_stratum, len(stratum))
        sampled = random.sample(stratum, num_to_sample)
        mappings.extend([m for m, _ in sampled])
    
    return mappings


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    bounds = LoopBounds(N=16, C=64, K=128, P=32, Q=32, R=3, S=3)
    
    print("Random sampling...")
    random_samples = sample_mappings_random(bounds, num_samples=100, seed=42)
    print(f"  Generated {len(random_samples)} random samples")
    
    print("\nLatin Hypercube sampling...")
    lhs_samples = sample_mappings_latin_hypercube(bounds, num_samples=50, seed=42)
    print(f"  Generated {len(lhs_samples)} LHS samples")
    
    print("\nBoundary cases...")
    boundary = sample_boundary_cases(bounds)
    print(f"  Generated {len(boundary)} boundary cases")
    
    print("\nStratified by memory footprint...")
    stratified = sample_by_memory_footprint(bounds, num_samples=50, seed=42)
    print(f"  Generated {len(stratified)} stratified samples")
