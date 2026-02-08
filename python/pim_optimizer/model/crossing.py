"""
Layout Block Crossing ratio calculation for row activation model.

This module implements the GCD-based periodic analysis for LAYOUT BLOCK crossing ratios.
Block Crossing = sliding window access crossing data layout block boundaries.
(NOT the same as DRAM Row Crossing which is about tile data crossing DRAM row buffer boundaries)

The key insight is that tile starting positions follow a periodic pattern within
memory blocks, allowing exact calculation of crossing probability.

Core Formula (GCD Periodic Analysis):
------------------------------------
Given:
    - block_h: Memory block height
    - tile_h: Tile height  
    - step: Tile stepping pattern = q_factor × stride (NOT just stride!)

The crossing ratio calculation:
    g = gcd(step, block_h)           # Period of tile start positions
    period = block_h // g            # Number of distinct positions
    safe_positions = ceil((block_h - tile_h + 1) / g)  # Positions that don't cross
    cross_count = period - safe_positions  # Positions that cross
    crossing_ratio = cross_count / period  # Final ratio

IMPORTANT: For inputs, step = q_factor × stride, NOT just stride!
           This is because tiles advance by multiple output rows.
"""

from math import gcd, ceil
from typing import Optional


def compute_block_crossing_ratio_gcd(
    block_h: int,
    tile_h: int,
    step: int,
    num_tiles: int = None,
    input_h: int = None,
) -> tuple[float, int, int, int]:
    """
    Compute LAYOUT BLOCK crossing ratio using GCD-based periodic analysis.
    
    This calculates how often tiles cross data layout block boundaries
    (NOT DRAM row boundaries - that's compute_dram_row_crossing_ratio).
    
    When input_h or num_tiles is provided, uses period decomposition for exact calculation:
    - Decomposes into complete periods + remainder
    - Counts actual crossings in remainder tiles
    
    When both are None, falls back to periodic approximation (assumes infinite tiles).
    
    Args:
        block_h: Memory block height
        tile_h: Tile height
        step: Tile stepping pattern (for inputs: q_factor × stride)
        num_tiles: Optional number of actual tiles. If provided, uses exact
                   calculation with period decomposition.
        input_h: Optional total input height. If provided (and num_tiles is None),
                 automatically calculates num_tiles = (input_h - tile_h) // step + 1.
                 This is more intuitive than specifying num_tiles directly.
        
    Returns:
        Tuple of:
        - crossing_ratio: Fraction of tiles that cross block boundary
        - g: GCD(step, block_h)
        - period: Number of distinct tile start positions
        - cross_count: Number of positions that result in crossing (per period)
        
    Example:
        >>> compute_block_crossing_ratio_gcd(block_h=8, tile_h=3, step=2)
        (0.25, 2, 4, 1)  # 1/4 tiles cross block boundary (periodic approximation)
        
        >>> compute_block_crossing_ratio_gcd(block_h=8, tile_h=3, step=2, input_h=14)
        (0.2857, 2, 4, 1)  # Exact calculation with 7 tiles
    """
    if block_h <= 0:
        return 0.0, 1, 1, 0
    
    if tile_h <= 0:
        return 0.0, 1, 1, 0
    
    # When tile_h > block_h, every tile crosses
    if tile_h > block_h:
        return 1.0, 1, 1, 1
    
    if step <= 0:
        step = 1
    
    # If input_h is provided but num_tiles is not, calculate num_tiles from input_h
    if num_tiles is None and input_h is not None and input_h > 0:
        if tile_h >= input_h:
            num_tiles = 1
        else:
            # Number of valid starting positions: 0, step, 2*step, ..., until start + tile_h > input_h
            # Last valid start: input_h - tile_h
            # num_tiles = floor((input_h - tile_h) / step) + 1
            num_tiles = (input_h - tile_h) // step + 1
    
    g = gcd(step, block_h)
    period = block_h // g
    
    # Number of safe positions (tile fits entirely within block)
    # These are positions where start + tile_h <= block_h (mod periodically)
    safe_zone = max(0, block_h - tile_h + 1)
    safe_positions = ceil(safe_zone / g) if safe_zone > 0 else 0
    
    # Crossing positions per period
    cross_count_per_period = max(0, period - safe_positions)
    
    # If num_tiles provided, use exact calculation with period decomposition
    if num_tiles is not None and num_tiles > 0:
        num_complete_periods = num_tiles // period
        remainder_tiles = num_tiles % period
        
        # Count crossings in remainder by simulating each tile position
        crossings_in_remainder = 0
        for k in range(remainder_tiles):
            # Position of tile k within the block (mod block_h)
            pos_mod = (k * step) % block_h
            if pos_mod + tile_h > block_h:
                crossings_in_remainder += 1
        
        total_crossings = num_complete_periods * cross_count_per_period + crossings_in_remainder
        crossing_ratio = total_crossings / num_tiles
        return crossing_ratio, g, period, cross_count_per_period
    
    # Fallback to periodic approximation
    crossing_ratio = cross_count_per_period / period if period > 0 else 0.0
    
    return crossing_ratio, g, period, cross_count_per_period


def compute_input_block_crossing(
    tile_p: int,
    tile_q: int,
    tile_r: int,
    tile_s: int,
    stride_h: int,
    stride_w: int,
    dilation_h: int,
    dilation_w: int,
    block_h: int,
    block_w: int,
    outer_q_factor: int = 1,
    outer_p_factor: int = 1,
    num_tiles_h: int = None,
    num_tiles_w: int = None,
    input_h: int = None,
    input_w: int = None,
) -> tuple[float, float]:
    """
    Compute LAYOUT BLOCK crossing ratios for input tiles in both dimensions.
    
    For input tiles:
    - Tile height = stride_h × tile_q + dilation_h × tile_s - stride_h - dilation_h + 1
    - Step = outer_q_factor × stride_h  (NOT just stride!)
    
    Args:
        tile_p, tile_q: Output tile dimensions
        tile_r, tile_s: Filter dimensions  
        stride_h, stride_w: Stride values
        dilation_h, dilation_w: Dilation values
        block_h, block_w: Memory block dimensions
        outer_q_factor: Outer loop Q factor (default 1)
        outer_p_factor: Outer loop P factor (default 1)
        num_tiles_h: Optional number of tiles in H direction for exact calculation
        num_tiles_w: Optional number of tiles in W direction for exact calculation
        input_h: Optional total input height (alternative to num_tiles_h)
        input_w: Optional total input width (alternative to num_tiles_w)
        
    Returns:
        Tuple of (crossing_ratio_h, crossing_ratio_w)
    """
    # Compute input tile dimensions
    input_tile_h = stride_h * tile_q + dilation_h * tile_s - stride_h - dilation_h + 1
    input_tile_w = stride_w * tile_p + dilation_w * tile_r - stride_w - dilation_w + 1
    
    # CRITICAL: Step is q_factor × stride, not just stride!
    step_h = outer_q_factor * stride_h
    step_w = outer_p_factor * stride_w
    
    # Compute block crossing ratios with optional exact calculation
    crossing_h, _, _, _ = compute_block_crossing_ratio_gcd(
        block_h, input_tile_h, step_h, num_tiles=num_tiles_h, input_h=input_h
    )
    crossing_w, _, _, _ = compute_block_crossing_ratio_gcd(
        block_w, input_tile_w, step_w, num_tiles=num_tiles_w, input_h=input_w
    )
    
    return crossing_h, crossing_w


def analyze_crossing_pattern(
    block_h: int,
    tile_h: int,
    step: int,
    num_iterations: int = 1,
    num_tiles: int = None,
    input_h: int = None,
) -> dict:
    """
    Analyze the crossing pattern for debugging and visualization.
    
    Args:
        block_h: Memory block height
        tile_h: Tile height
        step: Tile stepping pattern
        num_iterations: Number of iterations to simulate for visualization
        num_tiles: Optional actual number of tiles for exact calculation
        input_h: Optional total input height (alternative to num_tiles)
        
    Returns:
        Dictionary with analysis results
    """
    g = gcd(step, block_h)
    period = block_h // g
    
    # Simulate tile positions within one period
    positions = []
    crossings = []
    
    for i in range(min(num_iterations, period * 2)):
        pos = (i * step) % block_h
        crosses = (pos + tile_h) > block_h
        positions.append(pos)
        crossings.append(crosses)
    
    crossing_ratio, _, _, cross_count = compute_block_crossing_ratio_gcd(
        block_h, tile_h, step, num_tiles=num_tiles, input_h=input_h
    )

    result = {
        "block_h": block_h,
        "tile_h": tile_h,
        "step": step,
        "gcd": g,
        "period": period,
        "crossing_ratio": crossing_ratio,
        "cross_count": cross_count,
        "positions": positions[:period],
        "crossings": crossings[:period],
    }
    
    if num_tiles is not None:
        result["num_tiles"] = num_tiles
        result["formula"] = f"exact crossing ratio with {num_tiles} tiles = {crossing_ratio:.4f}"
    elif input_h is not None:
        actual_num_tiles = (input_h - tile_h) // step + 1 if tile_h < input_h else 1
        result["input_h"] = input_h
        result["num_tiles"] = actual_num_tiles
        result["formula"] = f"exact crossing ratio with input_h={input_h} ({actual_num_tiles} tiles) = {crossing_ratio:.4f}"
    else:
        result["formula"] = f"periodic crossing = {cross_count}/{period} = {crossing_ratio:.4f}"
    
    return result
