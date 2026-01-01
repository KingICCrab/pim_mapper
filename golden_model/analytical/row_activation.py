"""
Analytical model for DRAM row activation and crossing ratio.

This module provides formulas to compute:
1. Row buffer crossing ratio for different data layouts
2. Row activation count based on access patterns
3. Verification against ILP model results
"""

import math
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


def gcd(a: int, b: int) -> int:
    """Compute greatest common divisor."""
    while b:
        a, b = b, a % b
    return a


@dataclass
class RowBufferConfig:
    """Configuration for DRAM row buffer."""
    row_size_bytes: int = 1024  # Row buffer size in bytes
    element_bytes: int = 1      # Element size in bytes
    num_banks: int = 1          # Number of DRAM banks
    activation_latency: float = 25.0  # tRCD + tRP cycles
    
    @property
    def row_size_elements(self) -> int:
        return self.row_size_bytes // self.element_bytes


# =============================================================================
# Crossing Ratio Formulas
# =============================================================================

def compute_crossing_ratio_sequential(
    tile_bytes: int,
    row_bytes: int,
) -> float:
    """
    Compute crossing ratio for sequential (row-major) data layout.
    
    Uses GCD-based periodic analysis:
    - period = row_bytes / gcd(tile_bytes, row_bytes)
    - crossing occurs when: start_position + tile_bytes > row_bytes
    
    Args:
        tile_bytes: Size of tile in bytes
        row_bytes: Size of row buffer in bytes
        
    Returns:
        Fraction of tiles that cross row boundaries (0.0 to 1.0)
    """
    if tile_bytes <= 0:
        return 0.0
    if tile_bytes > row_bytes:
        return 1.0
    if tile_bytes == row_bytes:
        return 0.0
    
    g = gcd(tile_bytes, row_bytes)
    period = row_bytes // g
    
    # Count crossings in one period
    threshold = row_bytes - tile_bytes + 1
    cross_count = period - math.ceil(threshold / g)
    cross_count = max(0, cross_count)
    
    return cross_count / period


def compute_crossing_ratio_sliding_window(
    block_h: int,
    tile_h: int,
    step: int,
    kernel_size: int = 1,
    dilation: int = 1,
) -> float:
    """
    Compute crossing ratio for sliding window access pattern (Input datatype).
    
    Sliding window pattern:
    - tile 0: rows [0, tile_h)
    - tile 1: rows [step, step + tile_h)
    - tile 2: rows [2*step, 2*step + tile_h)
    
    Crossing occurs when: (k * step) mod block_h + tile_h > block_h
    
    Args:
        block_h: Data layout block height (alignment unit)
        tile_h: Input tile height
        step: Step size between tiles (= Q_factor × stride)
        kernel_size: Kernel size for split handling
        dilation: Kernel dilation factor
        
    Returns:
        Crossing ratio (0.0 to 1.0)
    """
    if block_h <= 0 or tile_h <= 0 or step <= 0:
        return 0.0
    
    if tile_h > block_h:
        return 1.0
    
    g = gcd(step, block_h)
    period = block_h // g
    
    crossing_count = 0
    for k in range(period):
        pos_mod = (k * step) % block_h
        if pos_mod + tile_h > block_h:
            crossing_count += 1
    
    return crossing_count / period if period > 0 else 0.0


def compute_crossing_ratio_analytical(
    tile_size: int,
    row_size: int,
    access_pattern: str = 'sequential',
    **kwargs,
) -> float:
    """
    Compute crossing ratio for different access patterns.
    
    Args:
        tile_size: Tile size in elements or bytes
        row_size: Row buffer size in elements or bytes
        access_pattern: 'sequential' or 'sliding_window'
        **kwargs: Additional parameters for specific patterns
        
    Returns:
        Crossing ratio (0.0 to 1.0)
    """
    if access_pattern == 'sequential':
        return compute_crossing_ratio_sequential(tile_size, row_size)
    elif access_pattern == 'sliding_window':
        block_h = kwargs.get('block_h', row_size)
        tile_h = kwargs.get('tile_h', tile_size)
        step = kwargs.get('step', 1)
        return compute_crossing_ratio_sliding_window(block_h, tile_h, step)
    else:
        raise ValueError(f"Unknown access pattern: {access_pattern}")


# =============================================================================
# Row Activation Count
# =============================================================================

def compute_row_activations_sequential(
    memory_reads: int,
    tile_bytes: int,
    row_config: RowBufferConfig,
) -> int:
    """
    Compute row activations for sequential access pattern.
    
    Row activations = memory_reads × (1 + crossing_ratio) / tiles_per_row / banks
    
    Args:
        memory_reads: Number of memory read operations
        tile_bytes: Tile size in bytes
        row_config: Row buffer configuration
        
    Returns:
        Number of row activations
    """
    row_bytes = row_config.row_size_bytes
    
    # Compute crossing ratio
    cr = compute_crossing_ratio_sequential(tile_bytes, row_bytes)
    
    # Tiles that fit in one row
    tiles_per_row = max(1, row_bytes // tile_bytes) if tile_bytes > 0 else 1
    
    # Row activations
    crossing_factor = 1.0 + cr
    row_acts = memory_reads * crossing_factor / tiles_per_row / row_config.num_banks
    
    return math.ceil(row_acts)


def compute_row_activations_sliding_window(
    memory_reads: int,
    tile_h: int,
    block_h: int,
    step: int,
    row_config: RowBufferConfig,
) -> int:
    """
    Compute row activations for sliding window access pattern.
    
    For Input datatype in convolution.
    
    Args:
        memory_reads: Number of memory read operations
        tile_h: Input tile height
        block_h: Data layout block height
        step: Step size (Q_factor × stride)
        row_config: Row buffer configuration
        
    Returns:
        Number of row activations
    """
    # Compute crossing ratio for sliding window
    cr = compute_crossing_ratio_sliding_window(block_h, tile_h, step)
    
    # Row activations = base + crossing_extra
    # base = memory_reads / banks
    # crossing_extra = cr × base
    base_acts = memory_reads / row_config.num_banks
    crossing_extra = cr * base_acts
    
    row_acts = base_acts + crossing_extra
    
    return math.ceil(row_acts)


def compute_row_activations_analytical(
    memory_reads: int,
    tile_info: Dict,
    row_config: RowBufferConfig,
    datatype: str,
) -> int:
    """
    Compute row activations analytically for a specific datatype.
    
    Args:
        memory_reads: Number of memory read operations
        tile_info: Dictionary with tile size and access pattern info
        row_config: Row buffer configuration
        datatype: 'input', 'weight', or 'output'
        
    Returns:
        Number of row activations
    """
    if datatype == 'input':
        # Input uses sliding window pattern
        tile_h = tile_info.get('tile_h', 1)
        block_h = tile_info.get('block_h', row_config.row_size_bytes)
        step = tile_info.get('step', 1)
        return compute_row_activations_sliding_window(
            memory_reads, tile_h, block_h, step, row_config
        )
    else:
        # Weight and Output use sequential pattern
        tile_bytes = tile_info.get('tile_bytes', row_config.element_bytes)
        return compute_row_activations_sequential(
            memory_reads, tile_bytes, row_config
        )


def compute_row_activation_cycles(
    row_activations: int,
    row_config: RowBufferConfig,
) -> float:
    """
    Compute cycles spent on row activations.
    
    Args:
        row_activations: Number of row activations
        row_config: Row buffer configuration
        
    Returns:
        Cycles for row activations
    """
    return row_activations * row_config.activation_latency


# =============================================================================
# Verification
# =============================================================================

def verify_row_activation_model(
    ilp_row_acts: float,
    memory_reads: int,
    tile_info: Dict,
    row_config: RowBufferConfig,
    datatype: str,
    tolerance: float = 0.05,
) -> Tuple[bool, Dict]:
    """
    Verify ILP row activation result against analytical model.
    
    Args:
        ilp_row_acts: Row activations from ILP model
        memory_reads: Memory reads count
        tile_info: Tile information
        row_config: Row buffer config
        datatype: Data type
        tolerance: Acceptable relative error
        
    Returns:
        Tuple of (passed, details_dict)
    """
    analytical_row_acts = compute_row_activations_analytical(
        memory_reads, tile_info, row_config, datatype
    )
    
    if analytical_row_acts == 0:
        relative_error = 0.0 if ilp_row_acts == 0 else float('inf')
    else:
        relative_error = abs(ilp_row_acts - analytical_row_acts) / analytical_row_acts
    
    passed = relative_error <= tolerance
    
    details = {
        'datatype': datatype,
        'memory_reads': memory_reads,
        'ilp_row_acts': ilp_row_acts,
        'analytical_row_acts': analytical_row_acts,
        'relative_error': relative_error,
        'tolerance': tolerance,
        'passed': passed,
    }
    
    return passed, details


def verify_crossing_ratio(
    ilp_crossing_ratio: float,
    tile_size: int,
    row_size: int,
    access_pattern: str = 'sequential',
    tolerance: float = 0.01,
    **kwargs,
) -> Tuple[bool, Dict]:
    """
    Verify ILP crossing ratio against analytical formula.
    
    Args:
        ilp_crossing_ratio: Crossing ratio from ILP model
        tile_size: Tile size
        row_size: Row buffer size
        access_pattern: Access pattern type
        tolerance: Acceptable absolute error
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (passed, details_dict)
    """
    analytical_cr = compute_crossing_ratio_analytical(
        tile_size, row_size, access_pattern, **kwargs
    )
    
    absolute_error = abs(ilp_crossing_ratio - analytical_cr)
    passed = absolute_error <= tolerance
    
    details = {
        'tile_size': tile_size,
        'row_size': row_size,
        'access_pattern': access_pattern,
        'ilp_crossing_ratio': ilp_crossing_ratio,
        'analytical_crossing_ratio': analytical_cr,
        'absolute_error': absolute_error,
        'tolerance': tolerance,
        'passed': passed,
    }
    
    return passed, details


# =============================================================================
# Test Cases
# =============================================================================

def run_crossing_ratio_tests() -> List[Dict]:
    """Run test cases for crossing ratio formulas."""
    test_cases = [
        # Sequential pattern tests
        {'tile': 64, 'row': 256, 'pattern': 'sequential', 'expected_range': (0.0, 0.3)},
        {'tile': 128, 'row': 256, 'pattern': 'sequential', 'expected_range': (0.4, 0.6)},
        {'tile': 256, 'row': 256, 'pattern': 'sequential', 'expected_range': (0.0, 0.01)},
        {'tile': 512, 'row': 256, 'pattern': 'sequential', 'expected_range': (0.99, 1.0)},
        
        # Sliding window tests
        {'block_h': 16, 'tile_h': 8, 'step': 1, 'pattern': 'sliding_window', 'expected_range': (0.4, 0.5)},
        {'block_h': 16, 'tile_h': 16, 'step': 1, 'pattern': 'sliding_window', 'expected_range': (0.9, 1.0)},
        {'block_h': 16, 'tile_h': 16, 'step': 16, 'pattern': 'sliding_window', 'expected_range': (0.0, 0.01)},
    ]
    
    results = []
    for tc in test_cases:
        pattern = tc['pattern']
        if pattern == 'sequential':
            cr = compute_crossing_ratio_sequential(tc['tile'], tc['row'])
        else:
            cr = compute_crossing_ratio_sliding_window(
                tc['block_h'], tc['tile_h'], tc['step']
            )
        
        low, high = tc['expected_range']
        passed = low <= cr <= high
        
        results.append({
            'test_case': tc,
            'computed_cr': cr,
            'expected_range': tc['expected_range'],
            'passed': passed,
        })
    
    return results


if __name__ == '__main__':
    # Run tests
    results = run_crossing_ratio_tests()
    print("Crossing Ratio Test Results:")
    print("=" * 60)
    for r in results:
        status = "✓ PASS" if r['passed'] else "✗ FAIL"
        print(f"{status}: CR={r['computed_cr']:.4f}, expected={r['expected_range']}")
        print(f"       {r['test_case']}")
    print("=" * 60)
    passed = sum(1 for r in results if r['passed'])
    print(f"Passed: {passed}/{len(results)}")
