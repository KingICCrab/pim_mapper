"""
Analytical cost formulas for memory access and compute cycles.

Based on established cost models from:
- Interstellar CNN Scheduler
- nn_dataflow
- OptiPIM

Key Concepts:
- For each data type, access count = product of "irrelevant" loop bounds
- Irrelevant loops are those that don't index into the data tensor

Simplified interface for single-level analysis.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class LoopBounds:
    """Loop bounds for a convolution workload."""
    N: int  # Batch size
    C: int  # Input channels
    K: int  # Output channels
    P: int  # Output height
    Q: int  # Output width
    R: int  # Kernel height
    S: int  # Kernel width
    
    @property
    def total_macs(self) -> int:
        return self.N * self.C * self.K * self.P * self.Q * self.R * self.S


@dataclass
class TileFactors:
    """
    Tile factors at a specific memory level.
    Factor means: how many of this dimension are processed per tile.
    """
    N: int = 1
    C: int = 1
    K: int = 1
    P: int = 1
    Q: int = 1
    R: int = 1
    S: int = 1
    
    def to_dict(self) -> Dict[str, int]:
        return {'N': self.N, 'C': self.C, 'K': self.K, 
                'P': self.P, 'Q': self.Q, 'R': self.R, 'S': self.S}


# =============================================================================
# Irrelevant Loop Definitions
# =============================================================================
# For Conv2D: Input[N, C, H, W], Weight[K, C, R, S], Output[N, K, P, Q]
#
# Input:  relevant dims = N, C, P+R, Q+S  -> irrelevant = K
# Weight: relevant dims = K, C, R, S      -> irrelevant = N, P, Q
# Output: relevant dims = N, K, P, Q      -> irrelevant = R, S, C

INPUT_IRRELEVANT_DIMS = {'K'}
WEIGHT_IRRELEVANT_DIMS = {'N', 'P', 'Q'}
OUTPUT_IRRELEVANT_DIMS = {'R', 'S', 'C'}


# =============================================================================
# Tile Size Computation
# =============================================================================

def compute_input_tile_size(
    factors: TileFactors,
    stride: int = 1,
    dilation: int = 1,
) -> int:
    """
    Compute input tile size for a conv tile.
    
    Input tile covers: N × C × H_in × W_in
    where H_in = (P - 1) * stride + (R - 1) * dilation + 1
    """
    h_in = (factors.P - 1) * stride + (factors.R - 1) * dilation + 1
    w_in = (factors.Q - 1) * stride + (factors.S - 1) * dilation + 1
    return factors.N * factors.C * h_in * w_in


def compute_weight_tile_size(factors: TileFactors) -> int:
    """Compute weight tile size: K × C × R × S."""
    return factors.K * factors.C * factors.R * factors.S


def compute_output_tile_size(factors: TileFactors) -> int:
    """Compute output tile size: N × K × P × Q."""
    return factors.N * factors.K * factors.P * factors.Q


# =============================================================================
# Access Count Computation
# =============================================================================

def compute_outer_factors(
    bounds: LoopBounds,
    tile_factors: TileFactors,
    dims: set,
) -> int:
    """
    Compute product of outer iteration counts for specified dimensions.
    
    Outer iterations = bound / tile_factor for each dimension.
    """
    product = 1
    for dim in dims:
        bound = getattr(bounds, dim)
        factor = getattr(tile_factors, dim)
        outer = bound // factor if factor > 0 else bound
        product *= outer
    return product


def compute_input_access_count(bounds: LoopBounds, factors: TileFactors) -> int:
    """
    Compute total input memory reads.
    
    Input is re-read when K iterates (K is irrelevant to input).
    
    正确公式: total_input_size × K_outer
    其中 total_input_size = N × C × H_in × W_in (整个输入的大小)
    K_outer = K / K_factor (因为 K 无关导致的重复读取)
    
    返回的是 **乘数因子**，用于 total_data_size × factor
    """
    return compute_outer_factors(bounds, factors, INPUT_IRRELEVANT_DIMS)


def compute_weight_access_count(bounds: LoopBounds, factors: TileFactors) -> int:
    """
    Compute total weight memory reads multiplier.
    
    Weight is re-read when N, P, Q iterate (irrelevant to weight).
    
    正确公式: total_weight_size × (N_outer × P_outer × Q_outer)
    """
    return compute_outer_factors(bounds, factors, WEIGHT_IRRELEVANT_DIMS)


def compute_output_access_count(bounds: LoopBounds, factors: TileFactors) -> int:
    """
    Compute total output memory reads multiplier.
    
    Output is re-read when R, S, C iterate (partial sum accumulation).
    
    正确公式: total_output_size × (R_outer × S_outer × C_outer)
    """
    return compute_outer_factors(bounds, factors, OUTPUT_IRRELEVANT_DIMS)


# =============================================================================
# Total Data Size Computation
# =============================================================================

def compute_total_input_size(bounds: LoopBounds, stride: int = 1, dilation: int = 1) -> int:
    """Compute total input tensor size: N × C × H_in × W_in."""
    h_in = (bounds.P - 1) * stride + (bounds.R - 1) * dilation + 1
    w_in = (bounds.Q - 1) * stride + (bounds.S - 1) * dilation + 1
    return bounds.N * bounds.C * h_in * w_in


def compute_total_weight_size(bounds: LoopBounds) -> int:
    """Compute total weight tensor size: K × C × R × S."""
    return bounds.K * bounds.C * bounds.R * bounds.S


def compute_total_output_size(bounds: LoopBounds) -> int:
    """Compute total output tensor size: N × K × P × Q."""
    return bounds.N * bounds.K * bounds.P * bounds.Q


# =============================================================================
# Total Memory Reads (CORRECTED FORMULA)
# =============================================================================

def compute_analytical_memory_reads(
    bounds: LoopBounds,
    factors: TileFactors,
    loop_order: List[str] = None,
    stride: int = 1,
    dilation: int = 1,
) -> Dict[str, int]:
    """
    Compute analytical memory reads for all data types.
    
    正确公式: total_reads = total_data_size × reuse_factor
    
    其中:
    - total_data_size = 整个 tensor 的大小
    - reuse_factor = 因为无关循环导致的重复访问次数
    
    For Input:  reuse_factor = K_outer (K 是无关维度)
    For Weight: reuse_factor = N_outer × P_outer × Q_outer  
    For Output: reuse_factor = C_outer × R_outer × S_outer (partial sums)
    
    Args:
        bounds: Loop bounds
        factors: Tile factors
        loop_order: Loop order (not used in simplified model)
        stride: Convolution stride
        dilation: Convolution dilation
        
    Returns:
        Dict with data sizes and reuse factors for each datatype
    """
    # Tile sizes (for reference, not used in total calculation)
    input_tile = compute_input_tile_size(factors, stride, dilation)
    weight_tile = compute_weight_tile_size(factors)
    output_tile = compute_output_tile_size(factors)
    
    # Total data sizes
    total_input = compute_total_input_size(bounds, stride, dilation)
    total_weight = compute_total_weight_size(bounds)
    total_output = compute_total_output_size(bounds)
    
    # Reuse factors (= outer loop iterations for irrelevant dimensions)
    input_reuse = compute_input_access_count(bounds, factors)    # K_outer
    weight_reuse = compute_weight_access_count(bounds, factors)  # N×P×Q outer
    output_reuse = compute_output_access_count(bounds, factors)  # C×R×S outer
    
    # Total reads = data_size × reuse_factor
    input_total = total_input * input_reuse
    weight_total = total_weight * weight_reuse
    output_total = total_output * output_reuse
    
    return {
        # Tile sizes (单个 tile 的大小)
        'input_tile_size': input_tile,
        'weight_tile_size': weight_tile,
        'output_tile_size': output_tile,
        
        # Total data sizes (整个 tensor 的大小)
        'input_total_size': total_input,
        'weight_total_size': total_weight,
        'output_total_size': total_output,
        
        # Reuse factors (重复访问因子)
        'input_access_count': input_reuse,
        'weight_access_count': weight_reuse,
        'output_access_count': output_reuse,
        
        # Total reads (正确的总访问次数)
        'input_total_reads': input_total,
        'weight_total_reads': weight_total,
        'output_total_reads': output_total,
        
        'total_memory_reads': input_total + weight_total + output_total,
    }


# =============================================================================
# Latency Computation
# =============================================================================

def compute_analytical_latency(
    bounds: LoopBounds,
    factors: TileFactors,
    loop_order: List[str] = None,
    memory_bandwidth_gb_s: float = 25.6,
    compute_throughput_gops: float = 100.0,
    element_bytes: int = 1,
) -> Dict[str, float]:
    """
    Compute analytical latency.
    
    Latency = max(compute_cycles, memory_cycles)
    
    Args:
        bounds: Loop bounds
        factors: Tile factors
        loop_order: Loop order (not used in simplified model)
        memory_bandwidth_gb_s: Memory bandwidth in GB/s
        compute_throughput_gops: Compute throughput in GOPS
        element_bytes: Element size in bytes
        
    Returns:
        Dict with compute cycles, memory cycles, and total latency
    """
    # Total MACs
    total_macs = bounds.total_macs
    
    # Compute cycles = MACs / throughput
    # Assuming 1 GHz clock, throughput is ops per cycle
    compute_cycles = math.ceil(total_macs / (compute_throughput_gops * 1e3))
    
    # Memory reads
    mem_reads = compute_analytical_memory_reads(bounds, factors)
    total_bytes = mem_reads['total_memory_reads'] * element_bytes
    
    # Memory cycles = bytes / bandwidth
    # Assuming 1 GHz clock
    bytes_per_cycle = memory_bandwidth_gb_s  # GB/s at 1GHz = bytes/cycle
    memory_cycles = math.ceil(total_bytes / bytes_per_cycle) if bytes_per_cycle > 0 else 0
    
    # Total latency = max(compute, memory)
    latency = max(compute_cycles, memory_cycles)
    
    return {
        'compute_cycles': compute_cycles,
        'memory_cycles': memory_cycles,
        'latency': latency,
        'total_macs': total_macs,
        'total_bytes': total_bytes,
    }


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    # Test case
    bounds = LoopBounds(N=4, C=8, K=8, P=4, Q=4, R=3, S=3)
    factors = TileFactors(N=2, C=4, K=4, P=2, Q=2, R=1, S=1)
    
    print("Loop Bounds:")
    print(f"  N={bounds.N}, C={bounds.C}, K={bounds.K}")
    print(f"  P={bounds.P}, Q={bounds.Q}, R={bounds.R}, S={bounds.S}")
    print(f"  Total MACs: {bounds.total_macs:,}")
    
    print("\nTile Factors:")
    print(f"  N={factors.N}, C={factors.C}, K={factors.K}")
    print(f"  P={factors.P}, Q={factors.Q}, R={factors.R}, S={factors.S}")
    
    # Compute memory reads
    reads = compute_analytical_memory_reads(bounds, factors)
    print("\nMemory Reads (CORRECTED FORMULA):")
    print(f"  Input:  total_size={reads['input_total_size']:,}, "
          f"reuse={reads['input_access_count']}, "
          f"total_reads={reads['input_total_reads']:,}")
    print(f"  Weight: total_size={reads['weight_total_size']:,}, "
          f"reuse={reads['weight_access_count']}, "
          f"total_reads={reads['weight_total_reads']:,}")
    print(f"  Output: total_size={reads['output_total_size']:,}, "
          f"reuse={reads['output_access_count']}, "
          f"total_reads={reads['output_total_reads']:,}")
    print(f"  Total:  {reads['total_memory_reads']:,}")
    
    # 手工验证
    print("\n手工验证:")
    H_in = (bounds.P - 1) * 1 + (bounds.R - 1) * 1 + 1  # stride=1, dilation=1
    W_in = (bounds.Q - 1) * 1 + (bounds.S - 1) * 1 + 1
    manual_input_size = bounds.N * bounds.C * H_in * W_in
    manual_input_reuse = bounds.K // factors.K
    manual_input_reads = manual_input_size * manual_input_reuse
    print(f"  Input: {manual_input_size} × {manual_input_reuse} = {manual_input_reads}")
    print(f"  Match: {manual_input_reads == reads['input_total_reads']}")
    
    # Compute latency
    latency = compute_analytical_latency(bounds, factors)
    print("\nLatency:")
    print(f"  Compute cycles: {latency['compute_cycles']:,}")
    print(f"  Memory cycles:  {latency['memory_cycles']:,}")
    print(f"  Total latency:  {latency['latency']:,}")
