"""
Row activation ILP model constraints.

This module builds the ILP constraints for row activation modeling
in DRAM-based PIM systems.
"""

import math
import json
import os
from typing import Optional

import numpy as np
import gurobipy as gp

from pim_optimizer.model.crossing import compute_block_crossing_ratio_gcd


def precompute_tile_crossing_info(
    tile_entries_list: list,
    element_bytes: float,
    row_bytes: float,
    tensor_total_bytes: float,
) -> tuple[list[float], list[float]]:
    """
    Precompute precise activation counts for all tile size options.
    
    Args:
        tile_entries_list: List of possible tile sizes (in elements)
        element_bytes: Size of one element in bytes
        row_bytes: DRAM row buffer size in bytes
        tensor_total_bytes: Total tensor size in bytes
        
    Returns:
        (non_crossing_acts_list, crossing_counts_list):
        - non_crossing_acts_list[k]: Number of activations for non-crossing tiles (considering tiles_per_row)
        - crossing_counts_list[k]: Number of crossing tiles (each needs 2 activations)
    """
    non_crossing_acts_list = []
    crossing_counts_list = []
    
    for te in tile_entries_list:
        if te <= 0:
            raise ValueError(f"Invalid tile_entries value: {te}. Must be positive.")
        tile_bytes = te * element_bytes
        num_tiles = max(1, int(tensor_total_bytes / tile_bytes))
        crossing_count = compute_dram_row_crossing_count(tile_bytes, row_bytes, num_tiles)
        
        # tiles_per_row = floor(row_bytes / tile_bytes)
        tiles_per_row = max(1, int(row_bytes / tile_bytes))
        
        # non_crossing tiles can share rows
        non_crossing_count = num_tiles - crossing_count
        non_crossing_acts = math.ceil(non_crossing_count / tiles_per_row) if non_crossing_count > 0 else 0
        
        non_crossing_acts_list.append(non_crossing_acts)
        crossing_counts_list.append(crossing_count)
    
    return non_crossing_acts_list, crossing_counts_list


def precompute_input_block_crossing_table(
    block_options: list[int],
    spatial_factor_options: list[int],
    kernel_factor_options: list[int],
    stride: int,
    dilation: int,
    total_spatial: int,
    total_kernel: int,
    input_size: int,
) -> dict[tuple[int, int, int], int | None]:
    """
    Precompute Input Block Crossing counts for one direction (H or W).
    
    For H direction: block_h, (P, R), stride[0], dilation[0], input W_in
    For W direction: block_w, (Q, S), stride[1], dilation[1], input H_in
    
    Args:
        block_options: List of block size candidates (block_h or block_w)
        spatial_factor_options: List of spatial dimension FACTORS (P or Q)
        kernel_factor_options: List of kernel dimension FACTORS (R or S)
        stride: Convolution stride for this direction
        dilation: Convolution dilation for this direction
        total_spatial: Total spatial dimension (P or Q)
        total_kernel: Total kernel dimension (R or S)
        input_size: Total input size for this direction (W_in or H_in)
        
    Returns:
        Dictionary mapping (i, j, k) -> crossing_count or None for invalid combinations
    """
    crossing_table = {}
    
    for i, block_size in enumerate(block_options):
        for j, spatial_factor in enumerate(spatial_factor_options):
            # Convert factor to tile size: tile_size = total / factor
            spatial_tile = total_spatial // spatial_factor if spatial_factor > 0 else total_spatial
            
            for k, kernel_factor in enumerate(kernel_factor_options):
                # Convert factor to tile size: tile_size = total / factor
                kernel_tile = total_kernel // kernel_factor if kernel_factor > 0 else total_kernel
                
                # Calculate tile size in input space
                tile_size = stride * spatial_tile + dilation * kernel_tile - stride - dilation + 1
                
                # Calculate step: distance between adjacent tiles in input space
                step = spatial_tile * stride
                
                # Compute crossing count using spatial_factor as num_tiles
                # (spatial_factor = number of DRAM tiles in this dimension)
                crossing_count, _ = compute_input_block_crossing_count(
                    block_h=block_size,
                    tile_h=tile_size,
                    step=step,
                    tile_s=kernel_tile,
                    total_S=total_kernel,
                    dilation=dilation,
                    num_tiles=spatial_factor,  # Use DRAM loop count, not derived from input_size
                )
                crossing_table[i, j, k] = crossing_count
    
    return crossing_table


def build_input_block_crossing_expr(
    model: gp.Model,
    vars,
    workload,
    w: int,
    dram_level: int,
    s_temporal: int,
    direction: str = "H",
) -> tuple[gp.LinExpr, int]:
    """
    Build ILP expression for Input Block Crossing count for one direction.
    
    Args:
        model: Gurobi model
        vars: Variable set
        workload: Workload definition
        w: Workload index
        dram_level: DRAM memory level index
        s_temporal: Temporal dimension index for xb variables
        direction: "H" or "W"
        
    Returns:
        (crossing_expr, max_crossing): ILP expression and upper bound
    """
    # Direction-specific parameters
    if direction == "H":
        # H direction: (P, R), stride[0], dilation[0]
        block_options = getattr(workload, 'hw_divisors', {}).get('H', [1])
        block_var_dict = vars.rowbuf_input_block_h
        spatial_dim = 2  # P
        kernel_dim = 0   # R
        stride_idx = 0   # Wstride
        dilation_idx = 0 # Wdilation
        var_prefix = "z_ibc_h"
    else:  # W direction
        # W direction: (Q, S), stride[1], dilation[1]
        block_options = getattr(workload, 'hw_divisors', {}).get('W', [1])
        block_var_dict = vars.rowbuf_input_block_w
        spatial_dim = 3  # Q
        kernel_dim = 1   # S
        stride_idx = 1   # Hstride
        dilation_idx = 1 # Hdilation
        var_prefix = "z_ibc_w"
    
    if not block_options:
        block_options = [1]
    
    # Get factor options
    spatial_factor_options = workload.divisors[spatial_dim]
    kernel_factor_options = workload.divisors[kernel_dim]
    
    # Get stride/dilation
    stride = workload.stride[stride_idx] if hasattr(workload, 'stride') else 1
    dilation = workload.dilation[dilation_idx] if hasattr(workload, 'dilation') else 1
    
    total_spatial = workload.bounds[spatial_dim]
    total_kernel = workload.bounds[kernel_dim]
    
    # Calculate input size for this direction
    input_size = stride * (total_spatial - 1) + dilation * (total_kernel - 1) + 1
    
    # Precompute crossing table
    crossing_table = precompute_input_block_crossing_table(
        block_options=block_options,
        spatial_factor_options=spatial_factor_options,
        kernel_factor_options=kernel_factor_options,
        stride=stride,
        dilation=dilation,
        total_spatial=total_spatial,
        total_kernel=total_kernel,
        input_size=input_size,
    )
    
    # Build linearized expression
    crossing_expr = gp.LinExpr(0)
    max_crossing = 0
    
    for i, block_size in enumerate(block_options):
        for j, spatial_factor in enumerate(spatial_factor_options):
            for k, kernel_factor in enumerate(kernel_factor_options):
                crossing_count = crossing_table.get((i, j, k))
                
                if crossing_count is None or crossing_count == 0:
                    continue
                
                max_crossing = max(max_crossing, crossing_count)
                
                # Get the three one-hot variables
                block_var = block_var_dict.get((w, i))
                spatial_var = vars.xb.get((w, dram_level, s_temporal, spatial_dim, j))
                kernel_var = vars.xb.get((w, dram_level, s_temporal, kernel_dim, k))
                
                if block_var is None or spatial_var is None or kernel_var is None:
                    continue
                
                # Create auxiliary variable z = block_var × spatial_var × kernel_var
                z = model.addVar(vtype=gp.GRB.BINARY, name=f"{var_prefix}_({w},{i},{j},{k})")
                
                # Linearization constraints: z = 1 iff all three = 1
                model.addConstr(z <= block_var, name=f"C_{var_prefix}_b_({w},{i},{j},{k})")
                model.addConstr(z <= spatial_var, name=f"C_{var_prefix}_s_({w},{i},{j},{k})")
                model.addConstr(z <= kernel_var, name=f"C_{var_prefix}_k_({w},{i},{j},{k})")
                model.addConstr(z >= block_var + spatial_var + kernel_var - 2,
                               name=f"C_{var_prefix}_lb_({w},{i},{j},{k})")
                
                crossing_expr += z * crossing_count
    
    return crossing_expr, max_crossing


def compute_dram_row_crossing_count(tile_bytes: float, row_bytes: float, num_tiles: int) -> int:
    """
    Calculate precise crossing count for DRAM row buffer.
    
    Args:
        tile_bytes: Tile size in bytes
        row_bytes: Row buffer size in bytes
        num_tiles: Number of tiles to access
        
    Returns:
        Exact number of tiles that cross row boundaries
    """
    if tile_bytes <= 0 or num_tiles <= 0:
        return 0
    if tile_bytes > row_bytes:
        return num_tiles  # All tiles cross
    if tile_bytes == row_bytes:
        return 0  # No tiles cross
    
    tile_bytes_int = int(tile_bytes)
    row_bytes_int = int(row_bytes)
    
    g = math.gcd(tile_bytes_int, row_bytes_int)
    period = row_bytes_int // g
    
    # Calculate crossings in one period
    threshold = row_bytes_int - tile_bytes_int + 1
    cross_count_per_period = period - math.ceil(threshold / g)
    cross_count_per_period = max(0, cross_count_per_period)
    
    # Period decomposition for exact count
    num_complete_periods = num_tiles // period
    remainder_tiles = num_tiles % period
    
    # Count crossings in remainder
    crossings_in_remainder = 0
    for i in range(remainder_tiles):
        start_offset = i * tile_bytes_int
        start_row = start_offset // row_bytes_int
        end_row = (start_offset + tile_bytes_int - 1) // row_bytes_int
        if end_row > start_row:
            crossings_in_remainder += 1
            
    return num_complete_periods * cross_count_per_period + crossings_in_remainder


def compute_input_block_crossing_count(
    block_h: int, 
    tile_h: int, 
    step: int, 
    tile_s: int, 
    total_S: int, 
    dilation: int,
    input_h: int = 0,
    num_tiles: int = 0,
) -> tuple[int, int]:
    """
    Calculate precise LAYOUT BLOCK crossing count for Input datatype based on sliding window access.
    (Used for row_aligned layout mode)
    
    Block Crossing = sliding window access crossing data layout block boundaries.
    (NOT the same as DRAM Row Crossing)
    Input tiles are accessed with a sliding window pattern:
    - tile 0: rows [0, tile_h)
    - tile 1: rows [step, step + tile_h)
    - tile 2: rows [2*step, 2*step + tile_h)
    
    A crossing occurs when: (k * step) mod block_h + tile_h > block_h
    
    Args:
        block_h: Data layout block height (H_rb factor)
        tile_h: Input tile height (access size)
        step: Stride between adjacent tiles = spatial_factor × stride
        tile_s: RowBuffer level kernel factor (kernel split factor)
        total_S: Full kernel dimension size
        dilation: Kernel dilation
        input_h: Total input dimension size (used when num_tiles not provided)
        num_tiles: Number of DRAM tiles (spatial_factor). If provided, overrides input_h calculation.
        
    Returns:
        (crossing_count, total_accesses): Exact crossing count and total number of accesses
    """
    # Edge cases
    if block_h <= 0 or tile_h <= 0 or step <= 0:
        return (0, 0)
    
    # Calculate number of tiles: prefer explicit num_tiles, else derive from input_h
    if num_tiles > 0:
        # Use explicitly provided num_tiles (= spatial_factor = DRAM loop count)
        pass
    elif input_h > 0:
        if tile_h >= input_h:
            num_tiles = 1
        else:
            num_tiles = (input_h - tile_h) // step + 1
    else:
        return (0, 0)
    
    # When tile_h > block_h, every tile crosses
    if tile_h > block_h:
        num_kernel_groups = total_S // tile_s if tile_s < total_S else 1
        total_accesses = num_tiles * num_kernel_groups
        return (total_accesses, total_accesses)
    
    g = math.gcd(int(step), int(block_h))
    period = int(block_h) // g
    
    if tile_s >= total_S:
        # Case 1: Full kernel in RowBuffer - single contiguous access
        
        # Compute crossing positions within one period
        crossing_positions = set()
        for k in range(period):
            pos_mod = (k * step) % block_h
            if pos_mod + tile_h > block_h:
                crossing_positions.add(k)
        
        # Decompose into complete periods + remainder
        num_complete_periods = num_tiles // period
        remainder_tiles = num_tiles % period
        
        # Count crossings in the remainder
        crossing_in_remainder = sum(1 for k in range(remainder_tiles) if k in crossing_positions)
        
        total_crossing = num_complete_periods * len(crossing_positions) + crossing_in_remainder
        return (total_crossing, num_tiles)
    
    else:
        # Case 2: Kernel split - multiple accesses with different offsets
        # NOTE: tile_s is required to be a factor of total_S
        num_kernel_groups = total_S // tile_s
        total_crossing_count = 0
        total_access_count = 0
        
        for group_idx in range(num_kernel_groups):
            base_kernel_row = group_idx * tile_s
            offset = base_kernel_row * dilation
            
            # Compute crossing positions within one period for this offset
            crossing_positions = set()
            for k in range(period):
                pos_mod = (k * step + offset) % block_h
                if pos_mod + tile_h > block_h:
                    crossing_positions.add(k)
            
            # Decompose into complete periods + remainder
            num_complete_periods = num_tiles // period
            remainder_tiles = num_tiles % period
            crossing_in_remainder = sum(1 for k in range(remainder_tiles) if k in crossing_positions)
            
            group_crossing = num_complete_periods * len(crossing_positions) + crossing_in_remainder
            total_crossing_count += group_crossing
            total_access_count += num_tiles
        
        return (total_crossing_count, total_access_count)


# =============================================================================
# Modular helper functions for row activation model
# =============================================================================

def _build_log_product_expr(
    model: gp.Model,
    vars,
    workload,
    w: int,
    t_id: int,
    dims: list[int],
    shared_rowbuf_idx: int,
    dram_level: int,
    s_temporal: int,
    var_prefix: str,
    target_levels: list[int] = None,
    inclusion_vars: dict = None,
    ignore_xj: bool = False,
) -> tuple[gp.LinExpr, float]:
    """
    Build log-space expression for Π_{j ∈ dims} bound_j^{xj}.
    
    Uses binary×binary linearization (no Big-M, tighter LP relaxation):
        xj × log_bound_j = Σ z[j,i] × log(div_i)
    where z[j,i] = xj × xb[j,i]
    
    Args:
        model: Gurobi model
        vars: Variable set
        workload: Workload definition
        w: Workload index
        t_id: Tensor ID (0=input, 1=weight, 2=output)
        dims: List of dimension indices to include
        shared_rowbuf_idx: RowBuffer memory level index
        dram_level: DRAM memory level index
        s_temporal: Temporal dimension index for xb variables
        var_prefix: Prefix for variable names (e.g., "rp" or "aligned")
        target_levels: List of memory levels to collect xb factors from. 
                       If None, defaults to [dram_level].
        inclusion_vars: Optional dict {dim_idx: binary_var} to use instead of vars.xj
        ignore_xj: If True, assume xj=1 (always include if xb=1).
        
    Returns:
        (log_expr, max_product): Log-space linear expression and maximum possible product
    """
    log_expr = gp.LinExpr(0)
    max_product = 1
    
    if target_levels is None:
        target_levels = [dram_level]
    
    # Deduplicate levels just in case
    unique_levels = sorted(list(set(target_levels)))
    
    for j in dims:
        if j >= len(workload.divisors):
            continue
        divs = workload.divisors[j]
        max_bound_j = workload.bounds[j]
        max_product *= max_bound_j
        
        if ignore_xj:
            xj = 1
        elif inclusion_vars is not None:
            xj = inclusion_vars.get(j)
        else:
            xj = vars.xj.get((w, t_id, shared_rowbuf_idx, dram_level, j))
        
        if xj is not None:
            for i, div in enumerate(divs):
                # Collect xb variables from specified levels
                xb_vars = []
                
                for level in unique_levels:
                    xb = vars.xb.get((w, level, s_temporal, j, i))
                    if xb is not None:
                        xb_vars.append((xb, f"L{level}"))

                for xb, level_suffix in xb_vars:
                    log_div = np.log(div)
                    if log_div > 1e-9:
                        z = model.addVar(
                            vtype=gp.GRB.BINARY,
                            name=f"z_{var_prefix}_{level_suffix}_({w},{t_id},{j},{i})"
                        )
                        if isinstance(xj, int) and xj == 1:
                            # Simplified constraints if xj is constant 1
                            model.addConstr(z <= xb, name=f"C_z_{var_prefix}_{level_suffix}_xb_({w},{t_id},{j},{i})")
                            model.addConstr(z >= xb, name=f"C_z_{var_prefix}_{level_suffix}_lb_({w},{t_id},{j},{i})")
                        else:
                            model.addConstr(z <= xj, name=f"C_z_{var_prefix}_{level_suffix}_xj_({w},{t_id},{j},{i})")
                            model.addConstr(z <= xb, name=f"C_z_{var_prefix}_{level_suffix}_xb_({w},{t_id},{j},{i})")
                            model.addConstr(z >= xj + xb - 1, name=f"C_z_{var_prefix}_{level_suffix}_lb_({w},{t_id},{j},{i})")
                        log_expr += z * log_div
    
    return log_expr, max_product


def _build_exp_var_from_log(
    model: gp.Model,
    log_expr: gp.LinExpr,
    max_value: float,
    var_name: str,
    pwl_opts: str = "FuncPieces=-2 FuncPieceError=0.002",
) -> gp.Var:
    """
    Create exp(log_expr) variable with PWL approximation.
    
    Args:
        model: Gurobi model
        log_expr: Log-space linear expression
        max_value: Maximum possible value of the result
        var_name: Base name for variables
        pwl_opts: PWL approximation options
        
    Returns:
        Variable representing exp(log_expr)
    """
    log_var = model.addVar(
        lb=0, ub=np.log(max(1, max_value)),
        name=f"log_{var_name}"
    )
    model.addConstr(log_var == log_expr, name=f"C_log_{var_name}")
    
    exp_var = model.addVar(
        lb=1, ub=float(max_value),
        name=var_name
    )
    model.addGenConstrExp(log_var, exp_var, options=pwl_opts, name=f"C_exp_{var_name}")
    
    return exp_var


def _build_outer_irrelevant_product(
    model: gp.Model,
    vars,
    workload,
    w: int,
    t_id: int,
    shared_rowbuf_idx: int,
    dram_level: int,
    s_temporal: int,
    outer_vars: dict = None,
) -> tuple[gp.Var, float]:
    """
    Build outer_irrelevant_product = Π_{j: O[j][t]==0 且 xj==1} bound_j
    
    This counts how many times the entire tensor is traversed due to 
    irrelevant dimensions that have relevant inner loops.
    
    Uses xj variable: xj[w, t, m, m_, j] = 1 means dimension j has 
    a relevant dimension inside it (inner loop).
    
    Args:
        model: Gurobi model
        vars: Variable set
        workload: Workload definition  
        w: Workload index
        t_id: Tensor ID (0=input, 1=weight, 2=output)
        shared_rowbuf_idx: RowBuffer memory level index
        dram_level: DRAM memory level index
        s_temporal: Temporal dimension index for xb variables
        outer_vars: Optional dict {dim_idx: binary_var} for outer irrelevant dims
        
    Returns:
        (outer_irr_product, max_product): Variable and maximum possible value
    """
    # This is exactly the same calculation as reuse_penalty!
    # Π_{j ∈ irrelevant} bound_j^{xj}
    # where xj=1 means j has inner relevant loop
    irrelevant_dims = [j for j in range(len(workload.O)) if workload.O[j][t_id] == 0]
    
    # Include all levels from RowBuffer+1 to DRAM
    target_levels = list(range(shared_rowbuf_idx + 1, dram_level + 1))
    # Also include RowBuffer level itself?
    # If an irrelevant loop is at RowBuffer level and is Outer (xj=1), it repeats the tensor.
    # Yes, it should be included.
    target_levels = list(range(shared_rowbuf_idx, dram_level + 1))
    
    log_expr, max_product = _build_log_product_expr(
        model, vars, workload, w, t_id, irrelevant_dims,
        shared_rowbuf_idx, dram_level, s_temporal, "outer_irr",
        target_levels=target_levels,
        inclusion_vars=outer_vars
    )
    
    PWL_OPTS = "FuncPieces=1 FuncPieceError=0.1"
    outer_irr_product = _build_exp_var_from_log(
        model, log_expr, max_product, f"outer_irr_prod_({w},{t_id})", PWL_OPTS
    )
    
    return outer_irr_product, max_product


def _build_sequential_dram_crossing(
    model: gp.Model,
    tile_entries_list: tuple,
    xu_vars: list,
    element_bytes: float,
    row_buffer_size_bytes: float,
    tensor_bytes: float,
    reuse_penalty: gp.Var,
    max_reuse_penalty: float,
    outer_irr_product: gp.Var,
    max_outer_irr: float,
    w: int,
    t_id: int,
    is_thrashing: gp.Var = None,
    is_reused: gp.Var = None,
) -> tuple[gp.Var, float]:
    """
    Build Sequential mode DRAM Row Crossing expression.
    
    Formula:
        row_acts = (non_crossing_acts + Multiplier * crossing_count * reuse_penalty) * outer_irr_product
        
    Multiplier Logic:
        - If is_thrashing=1 (Innermost loop is irrelevant): Multiplier = 2 (Thrashing)
        - If is_thrashing=0 (Innermost loop is relevant):
            - If Small Block & Unaligned & Reuse > 1: Multiplier = 2 (Thrashing)
            - Else: Multiplier = 1 (Streaming)
    
    Where:
        - non_crossing_acts: number of unique rows touched by non-crossing tiles
        - crossing_count: number of tiles that cross row boundaries
        - reuse_penalty: repeated access to same crossing tile (due to innermost irrelevant loops)
        - outer_irr_product: number of times entire tensor is traversed (due to outer irrelevant loops)
    
    Args:
        model: Gurobi model
        tile_entries_list: List of possible tile sizes
        xu_vars: One-hot tile selection variables
        element_bytes: Element size in bytes
        row_buffer_size_bytes: Row buffer size in bytes
        tensor_bytes: Total tensor size in bytes
        reuse_penalty: Reuse penalty variable (for crossing tiles)
        max_reuse_penalty: Maximum reuse penalty value
        outer_irr_product: Outer irrelevant product variable
        max_outer_irr: Maximum outer irrelevant product value
        w: Workload index
        t_id: Tensor ID
        is_thrashing: Binary variable indicating if innermost loop is irrelevant
        is_reused: Binary variable indicating if reuse_penalty > 1
        
    Returns:
        (row_acts_var, row_acts_ub): Row activation variable and upper bound
    """
    non_crossing_acts_list, crossing_counts_list = precompute_tile_crossing_info(
        tile_entries_list, element_bytes, row_buffer_size_bytes, tensor_bytes
    )
    
    max_non_crossing_acts = max(non_crossing_acts_list) if non_crossing_acts_list else 1
    max_crossing_count = max(crossing_counts_list) if crossing_counts_list else 0
    
    # =========================================================
    # New Hybrid Cost Model (Streaming vs Thrashing)
    # =========================================================
    # 1. Streaming Cost (is_thrashing=0):
    #    We assume sequential access sweeps through the memory.
    #    Cost = Total Size / Row Size.
    #    This is constant regardless of tile size (mostly).
    
    streaming_cost = max(1.0, tensor_bytes / row_buffer_size_bytes)
    
    # 2. Thrashing Cost (is_thrashing=1):
    #    We assume we pay for every tile access, plus extra for crossings.
    #    Cost = (Num Tiles + Crossing Count) * Reuse.
    #    (Crossing tiles cost 2*Reuse, Non-crossing cost 1*Reuse).
    
    # RE-APPLYING FIX: Use aux_reuse_k for Large Blocks.
    # And also for Small Blocks that are not aligned/resident.
    
    base_expr = gp.LinExpr(0)
    for k in range(len(xu_vars)):
        # 1. Linearize reuse_penalty * xu_vars[k]
        aux_reuse_k = model.addVar(lb=0, ub=max_reuse_penalty, name=f"aux_reuse_k_({w},{t_id},{k})")
        model.addConstr(aux_reuse_k <= max_reuse_penalty * xu_vars[k], name=f"C_aux_reuse_ub1_({w},{t_id},{k})")
        model.addConstr(aux_reuse_k <= reuse_penalty, name=f"C_aux_reuse_ub2_({w},{t_id},{k})")
        model.addConstr(aux_reuse_k >= reuse_penalty - max_reuse_penalty * (1 - xu_vars[k]), name=f"C_aux_reuse_lb_({w},{t_id},{k})")
        
        te = tile_entries_list[k]
        tile_bytes = te * element_bytes
        is_small_block = tile_bytes < row_buffer_size_bytes
        
        if is_small_block:
            remainder = row_buffer_size_bytes % tile_bytes
            is_aligned = remainder < 1e-6 or abs(remainder - tile_bytes) < 1e-6
            
            # FIX: If the entire tensor fits in the row buffer, it's always a hit.
            if tensor_bytes <= row_buffer_size_bytes:
                base_expr += streaming_cost * xu_vars[k]
            elif is_aligned:
                # Aligned Small Block: Fits in Row Buffer.
                # If we reuse it, it's a Row Hit. We only pay for the first access.
                # So Cost = StreamingCost * 1 (xu_vars[k])
                base_expr += streaming_cost * xu_vars[k]
            else:
                # Unaligned Small Block: Might thrash.
                # Conservative: Pay for Reuse.
                base_expr += streaming_cost * aux_reuse_k
        else:
            # Large Block: Does NOT fit in Row Buffer.
            # Must be streamed every time it is reused.
            # Cost = StreamingCost * Reuse
            base_expr += streaming_cost * aux_reuse_k
            
    # Max base for bounds
    # Max Thrashing = (Max Num Tiles + Max Crossing) * Max Reuse
    # Max Num Tiles = Tensor / Min Tile Size
    min_tile_entries = min(tile_entries_list) if tile_entries_list else 1
    max_num_tiles = tensor_bytes / (min_tile_entries * element_bytes)
    max_thrashing_cost = (max_num_tiles + max_crossing_count) * max_reuse_penalty
    
    # Max Streaming = Streaming Cost * Max Reuse
    max_streaming_cost = streaming_cost * max_reuse_penalty
    
    max_base = max(max_streaming_cost, max_thrashing_cost)
    
    base_row_acts = model.addVar(lb=1, ub=max_base, name=f"base_row_acts_({w},{t_id})")
    model.addConstr(base_row_acts == base_expr, name=f"C_base_row_acts_({w},{t_id})")
    
    # Final: row_acts = base_row_acts × outer_irr_product
    # Use log space: log(row_acts) = log(base_row_acts) + log(outer_irr_product)
    row_acts_ub = max_base * max_outer_irr
    
    # log(base_row_acts) via PWL
    log_base = model.addVar(lb=0, ub=np.log(max(1, max_base)), name=f"log_base_row_acts_({w},{t_id})")
    model.addGenConstrLog(base_row_acts, log_base, options="FuncPieces=1 FuncPieceError=0.1", 
                          name=f"C_log_base_({w},{t_id})")
    
    # log(outer_irr_product) already exists as log_outer_irr_prod_(w,t_id)
    # We need to get the log variable from outer_irr_product
    log_outer_irr = model.addVar(lb=0, ub=np.log(max(1, max_outer_irr)), name=f"log_outer_irr_for_mult_({w},{t_id})")
    model.addGenConstrLog(outer_irr_product, log_outer_irr, options="FuncPieces=1 FuncPieceError=0.1",
                          name=f"C_log_outer_irr_({w},{t_id})")
    
    # log(row_acts) = log(base_row_acts) + log(outer_irr_product)
    log_row_acts = model.addVar(lb=0, ub=np.log(max(1, row_acts_ub)), name=f"log_row_acts_({w},{t_id})")
    
    # FIX: If tensor fits in Row Buffer, Outer Loops do NOT cause thrashing (assuming separate banks).
    # In this case, row_acts = base_row_acts (which is streaming_cost = 1).
    if tensor_bytes <= row_buffer_size_bytes:
        model.addConstr(log_row_acts == log_base, name=f"C_log_row_acts_sum_({w},{t_id})")
    else:
        model.addConstr(log_row_acts == log_base + log_outer_irr, name=f"C_log_row_acts_sum_({w},{t_id})")
    
    # exp back to get row_acts
    row_acts = model.addVar(lb=1, ub=row_acts_ub, name=f"row_acts_({w},{t_id})")
    model.addGenConstrExp(log_row_acts, row_acts, options="FuncPieces=1 FuncPieceError=0.1",
                          name=f"C_exp_row_acts_({w},{t_id})")
    
    return row_acts, row_acts_ub


def _build_input_block_crossing_acts(
    model: gp.Model,
    vars,
    workload,
    w: int,
    dram_level: int,
    s_temporal: int,
    reuse_penalty: gp.Var,
    max_reuse_penalty: float,
    direction: str = "H",
) -> tuple[gp.Var | int, int]:
    """
    Build Input Block Crossing activations expression for one direction.
    
    Args:
        model: Gurobi model
        vars: Variable set
        workload: Workload definition
        w: Workload index
        dram_level: DRAM memory level index
        s_temporal: Temporal dimension index
        reuse_penalty: Reuse penalty variable
        max_reuse_penalty: Maximum reuse penalty value
        direction: "H" or "W"
        
    Returns:
        (crossing_acts_var, max_crossing): Crossing activations variable/constant and max crossing count
    """
    suffix = "" if direction == "H" else "_w"
    
    try:
        crossing_expr, max_crossing = build_input_block_crossing_expr(
            model=model,
            vars=vars,
            workload=workload,
            w=w,
            dram_level=dram_level,
            s_temporal=s_temporal,
            direction=direction,
        )
        
        if max_crossing <= 0:
            return 0, 0
        
        crossing_acts = model.addVar(
            lb=0,
            ub=max_crossing * 2 * max_reuse_penalty,
            name=f"input_block_crossing_acts{suffix}_({w})"
        )
        
        selected_count = model.addVar(lb=0, ub=max_crossing, name=f"selected_ibc_count{suffix}_({w})")
        model.addConstr(selected_count == crossing_expr, name=f"C_selected_ibc_count{suffix}_({w})")
        
        # Use log space for tighter approximation: aux = selected_count × reuse_penalty
        M_ibc = max_crossing * max_reuse_penalty
        aux = model.addVar(lb=0, ub=M_ibc, name=f"aux_ibc_rp{suffix}_({w})")
        
        # Binary indicator: is_nonzero = 1 if selected_count > 0
        is_nonzero = model.addVar(vtype=gp.GRB.BINARY, name=f"ibc_nonzero{suffix}_({w})")
        
        # Force is_nonzero = 0 when selected_count = 0
        model.addConstr(selected_count <= max_crossing * is_nonzero, name=f"C_ibc_nz_ub{suffix}_({w})")
        epsilon = 0.01
        model.addConstr(selected_count >= epsilon * is_nonzero, name=f"C_ibc_nz_lb{suffix}_({w})")
        
        # Case 1: selected_count = 0 → aux = 0
        model.addConstr(aux <= M_ibc * is_nonzero, name=f"C_aux_zero_case{suffix}_({w})")
        
        # Case 2: selected_count > 0 → use log space
        sc_pos = model.addVar(lb=epsilon, ub=max_crossing, name=f"sc_pos{suffix}_({w})")
        model.addGenConstrIndicator(is_nonzero, True, sc_pos == selected_count, 
                                    name=f"C_sc_pos_nz{suffix}_({w})")
        model.addGenConstrIndicator(is_nonzero, False, sc_pos == epsilon,
                                    name=f"C_sc_pos_zero{suffix}_({w})")
        
        log_sc = model.addVar(lb=np.log(epsilon), ub=np.log(max(epsilon, max_crossing)),
                             name=f"log_sc_ibc{suffix}_({w})")
        model.addGenConstrLog(sc_pos, log_sc, options="FuncPieces=1 FuncPieceError=0.1",
                             name=f"C_log_sc_ibc{suffix}_({w})")
        
        log_rp = model.addVar(lb=0, ub=np.log(max(1, max_reuse_penalty)),
                             name=f"log_rp_ibc{suffix}_({w})")
        model.addGenConstrLog(reuse_penalty, log_rp, options="FuncPieces=1 FuncPieceError=0.1",
                             name=f"C_log_rp_ibc{suffix}_({w})")
        
        log_aux = model.addVar(lb=np.log(epsilon), ub=np.log(max(1, M_ibc)),
                              name=f"log_aux_ibc{suffix}_({w})")
        model.addConstr(log_aux == log_sc + log_rp, name=f"C_log_aux_sum{suffix}_({w})")
        
        aux_pos = model.addVar(lb=epsilon, ub=M_ibc, name=f"aux_pos{suffix}_({w})")
        model.addGenConstrExp(log_aux, aux_pos, options="FuncPieces=1 FuncPieceError=0.1",
                             name=f"C_exp_aux_ibc{suffix}_({w})")
        
        model.addGenConstrIndicator(is_nonzero, True, aux == aux_pos,
                                    name=f"C_aux_link_nz{suffix}_({w})")
        model.addGenConstrIndicator(is_nonzero, False, aux == 0,
                                    name=f"C_aux_link_zero{suffix}_({w})")
        
        model.addConstr(crossing_acts == 2 * aux, name=f"C_input_block_crossing_acts{suffix}_({w})")
        
        return crossing_acts, max_crossing
        
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to build input block crossing ({direction}) for workload {w}: {e}")
        return 0, 0


def _build_layout_conditional_acts(
    model: gp.Model,
    row_aligned_var: gp.Var | None,
    row_acts_seq: gp.Var,
    row_acts_seq_ub: float,
    row_acts_aligned: gp.Var,
    row_acts_aligned_ub: float,
    block_crossing_acts: gp.Var | int,
    block_crossing_ub: float,
    w: int,
    t_id: int,
) -> tuple[gp.Var, float]:
    """
    Build layout-conditional total row activations.
    
    Formula:
        total = (1 - row_aligned) × seq_acts + row_aligned × aligned_acts + block_crossing
    
    Args:
        model: Gurobi model
        row_aligned_var: Layout choice variable (None = sequential only)
        row_acts_seq: Sequential mode row activations
        row_acts_seq_ub: Upper bound for sequential activations
        row_acts_aligned: Row-aligned mode row activations
        row_acts_aligned_ub: Upper bound for aligned activations
        block_crossing_acts: Block crossing activations (for Input)
        block_crossing_ub: Upper bound for block crossing
        w: Workload index
        t_id: Tensor ID
        
    Returns:
        (total_acts_var, total_acts_ub): Total activations variable and upper bound
    """
    has_block_crossing = isinstance(block_crossing_acts, gp.Var)
    
    if row_aligned_var is not None:
        # Linearize: (1 - row_aligned) × row_acts_seq
        # Use Indicator Constraints for better numerical stability than Big-M
        seq_part = model.addVar(lb=0, ub=row_acts_seq_ub, name=f"dram_crossing_seq_({w},{t_id})")
        
        # If row_aligned = 0 (Sequential), seq_part = row_acts_seq
        model.addGenConstrIndicator(row_aligned_var, 0, seq_part == row_acts_seq, name=f"C_ind_seq_active_({w},{t_id})")
        # If row_aligned = 1 (Aligned), seq_part = 0
        model.addGenConstrIndicator(row_aligned_var, 1, seq_part == 0, name=f"C_ind_seq_inactive_({w},{t_id})")
        
        # Linearize: row_aligned × row_acts_aligned
        aligned_part = model.addVar(lb=0, ub=row_acts_aligned_ub, name=f"row_aligned_acts_({w},{t_id})")
        
        # If row_aligned = 1 (Aligned), aligned_part = row_acts_aligned
        model.addGenConstrIndicator(row_aligned_var, 1, aligned_part == row_acts_aligned, name=f"C_ind_aligned_active_({w},{t_id})")
        # If row_aligned = 0 (Sequential), aligned_part = 0
        model.addGenConstrIndicator(row_aligned_var, 0, aligned_part == 0, name=f"C_ind_aligned_inactive_({w},{t_id})")
        
        max_base_ub = max(row_acts_seq_ub, row_acts_aligned_ub)
        
        if has_block_crossing:
            total_ub = max_base_ub + block_crossing_ub
            total_acts = model.addVar(lb=0, ub=total_ub, name=f"total_row_acts_({w},{t_id})")
            model.addConstr(total_acts == seq_part + aligned_part + block_crossing_acts, name=f"C_total_row_acts_({w},{t_id})")
        else:
            total_ub = max_base_ub
            total_acts = model.addVar(lb=0, ub=total_ub, name=f"total_row_acts_({w},{t_id})")
            model.addConstr(total_acts == seq_part + aligned_part, name=f"C_total_row_acts_({w},{t_id})")
    else:
        # Sequential layout only
        if has_block_crossing:
            total_ub = row_acts_seq_ub + block_crossing_ub
            total_acts = model.addVar(lb=1, ub=total_ub, name=f"total_row_acts_({w},{t_id})")
            model.addConstr(total_acts == row_acts_seq + block_crossing_acts, name=f"C_total_row_acts_({w},{t_id})")
        else:
            total_acts = row_acts_seq
            total_ub = row_acts_seq_ub
    
    return total_acts, total_ub


def _compute_tensor_bytes(workload, t_id: int, element_bytes: float) -> float:
    """Compute total tensor size in bytes."""
    if t_id == 0:  # Input
        elements = workload.bounds[4] * workload.bounds[3] * workload.bounds[2]
    elif t_id == 1:  # Weight
        elements = workload.bounds[5] * workload.bounds[4] * workload.bounds[1] * workload.bounds[0]
    else:  # Output
        elements = workload.bounds[6] * workload.bounds[5] * workload.bounds[3] * workload.bounds[2]
    return elements * element_bytes


# =============================================================================
# Main entry point
# =============================================================================

def _build_lookup_table_constraints(
    model: gp.Model,
    vars,
    workload,
    w: int,
    t_id: int,
    dram_level: int,
    s_temporal: int,
    reuse_penalty: gp.Var,
    max_rp: float,
    table_path: str = "row_activation_cost_table.json"
) -> tuple[Optional[gp.Var], float]:
    """
    Build constraints using the precomputed lookup table for Input Tensor.
    """
    if not os.path.exists(table_path):
        return None, 0.0

    try:
        with open(table_path, 'r') as f:
            data = json.load(f)
    except Exception:
        return None, 0.0
    
    # Group by (P, Q, C) and find min cost
    min_costs = {}
    for entry in data:
        # Ensure keys exist
        if 'P' not in entry or 'Q' not in entry or 'C' not in entry or 'row_acts' not in entry:
            continue
        key = (entry['P'], entry['Q'], entry['C'])
        cost = entry['row_acts']
        if key not in min_costs or cost < min_costs[key]:
            min_costs[key] = cost
            
    if not min_costs:
        return None, 0.0

    # Map dimension names to indices (P=2, Q=3, C=4)
    dim_map = {'P': 2, 'Q': 3, 'C': 4}
    
    # -------------------------------------------------------------------------
    # Pruning Step: Pareto Frontier Filter (Minimize Cost, Minimize Size)
    # -------------------------------------------------------------------------
    # 1. Collect all valid candidates first
    candidates = []
    
    for (p_tile, q_tile, c_tile), avg_cost in min_costs.items():
        # Check validity
        divs = {}
        valid = True
        for dim_name, tile_size in [('P', p_tile), ('Q', q_tile), ('C', c_tile)]:
            dim_idx = dim_map[dim_name]
            total_size = workload.bounds[dim_idx]
            if total_size % tile_size != 0:
                valid = False; break
            div = total_size // tile_size
            if div not in workload.divisors[dim_idx]:
                valid = False; break
            div_idx = workload.divisors[dim_idx].index(div)
            divs[dim_idx] = div_idx
            
        if not valid:
            continue
            
        # Calculate metrics
        # Total Cost = Avg_Cost * Num_Tiles
        p_tiles = workload.bounds[dim_map['P']] // p_tile
        q_tiles = workload.bounds[dim_map['Q']] // q_tile
        c_tiles = workload.bounds[dim_map['C']] // c_tile
        total_cost = avg_cost * p_tiles * q_tiles * c_tiles
        
        # Size = p * q * c (Logical size, proportional to buffer usage)
        size = p_tile * q_tile * c_tile
        
        candidates.append({
            'p': p_tile, 'q': q_tile, 'c': c_tile,
            'divs': divs,
            'cost': total_cost,
            'size': size,
            'avg_cost': avg_cost,
            'num_tiles': p_tiles * q_tiles * c_tiles
        })
        
    # 2. Sort by Cost (Ascending)
    candidates.sort(key=lambda x: x['cost'])
    
    # 3. Pareto Filter
    # Keep candidate if it has smaller size than all previous (lower cost) candidates
    kept_candidates = []
    min_size_seen = float('inf')
    
    for cand in candidates:
        if cand['size'] < min_size_seen:
            kept_candidates.append(cand)
            min_size_seen = cand['size']
            
    # Optional: Hard limit if Pareto set is still huge (e.g. > 50)
    if len(kept_candidates) > 10:
        kept_candidates = kept_candidates[:10]
        
    # -------------------------------------------------------------------------
    # Build Constraints for Kept Candidates
    # -------------------------------------------------------------------------
    
    terms = []
    max_total_cost = 0.0
    
    for cand in kept_candidates:
        p_tile, q_tile, c_tile = cand['p'], cand['q'], cand['c']
        divs = cand['divs']
        total_cost_coeff = cand['cost']
        
        # 2. Create binary variable for this configuration
        # z_entry = 1 iff this tile configuration is selected
        z_entry = model.addVar(vtype=gp.GRB.BINARY, name=f"z_lut_{w}_{p_tile}_{q_tile}_{c_tile}")
        
        # Store mapping for later constraint generation
        entry_info = {
            'z': z_entry,
            'divs': divs,
            'coeff': total_cost_coeff,
            'num_tiles': cand['num_tiles'],
            'suffix': f"{p_tile}_{q_tile}_{c_tile}"
        }
        terms.append(entry_info)
        max_total_cost = max(max_total_cost, total_cost_coeff)

    if not terms:
        return None, 0.0
        
    # -------------------------------------------------------------------------
    # Fallback Mechanism
    # -------------------------------------------------------------------------
    # Allow optimizer to pick a configuration NOT in the table.
    # If z_fallback = 1, we use a conservative cost estimate.
    
    z_fallback = model.addVar(vtype=gp.GRB.BINARY, name=f"z_lut_fallback_{w}")
    
    # Conservative estimate: assume worst case (e.g. max possible cost)
    # Or use the analytical model?
    # For simplicity, let's use a very high cost to discourage fallback unless necessary.
    # But it must be feasible.
    # Let's use 2x the max cost seen in the table.
    fallback_cost = max_total_cost * 2.0 if max_total_cost > 0 else 1e6
    
    # 3. Link z variables to xb variables (Optimized with Fallback)
    # xb[dim, div_idx] >= sum(z for z where z.divs[dim] == div_idx)
    # If z_fallback=1, xb is unconstrained by the table (can be anything).
    # If z_fallback=0, xb must match the selected z.
    
    for dim_idx in dim_map.values():
        # Group z variables by their divisor index for this dimension
        z_by_div_idx = {}
        for info in terms:
            div_idx = info['divs'][dim_idx]
            if div_idx not in z_by_div_idx:
                z_by_div_idx[div_idx] = []
            z_by_div_idx[div_idx].append(info['z'])
            
        # Add constraint for each divisor index present in the table
        for div_idx, z_list in z_by_div_idx.items():
            xb = vars.xb[w, dram_level, s_temporal, dim_idx, div_idx]
            
            # If z_fallback=0: xb == sum(z_list)
            # If z_fallback=1: xb can be anything (0 or 1)
            # So: xb >= sum(z_list) is always true (since sum(z) is 0 or 1)
            # And: xb <= sum(z_list) + z_fallback
            
            model.addConstr(xb >= gp.quicksum(z_list), name=f"C_link_lut_xb_lb_{w}_{dim_idx}_{div_idx}")
            model.addConstr(xb <= gp.quicksum(z_list) + z_fallback, name=f"C_link_lut_xb_ub_{w}_{dim_idx}_{div_idx}")

    # 4. Build the final cost variable
    # row_acts_seq = sum(z_entry * coeff * reuse_penalty) + z_fallback * fallback_cost * reuse_penalty
    
    row_acts_seq = model.addVar(lb=0, ub=fallback_cost * max_rp, name=f"row_acts_lut_{w}")
    
    # Constraint: sum(z_entry) + z_fallback == 1
    z_sum = gp.LinExpr()
    for info in terms:
        z_sum += info['z']
    z_sum += z_fallback
    model.addConstr(z_sum == 1, name=f"C_lut_selection_{w}")
    
    # Now sum the terms
    total_expr = gp.LinExpr()
    for info in terms:
        z = info['z']
        coeff = info['coeff']
        suffix = info['suffix']
        
        # Create variable for z * reuse_penalty
        term_var = model.addVar(lb=0, ub=max_rp, name=f"term_lut_{w}_{suffix}")
        
        # Indicator constraints
        model.addGenConstrIndicator(z, True, term_var == reuse_penalty)
        model.addGenConstrIndicator(z, False, term_var == 0)
        
        # New Cost Formula: Cost = NumTiles + (AvgCost - 1) * NumTiles * ReusePenalty
        # coeff = AvgCost * NumTiles
        # So: Cost = NumTiles + (coeff - NumTiles) * ReusePenalty
        # term_var represents (z * ReusePenalty)
        # We need: z * NumTiles + (coeff - NumTiles) * term_var
        
        num_tiles = info['num_tiles']
        total_expr += z * num_tiles + (coeff - num_tiles) * term_var
        
    # Add fallback term
    term_fallback = model.addVar(lb=0, ub=max_rp, name=f"term_lut_fallback_{w}")
    model.addGenConstrIndicator(z_fallback, True, term_fallback == reuse_penalty)
    model.addGenConstrIndicator(z_fallback, False, term_fallback == 0)
    total_expr += fallback_cost * term_fallback
        
    model.addConstr(row_acts_seq == total_expr, name=f"C_row_acts_lut_{w}")
    
    return row_acts_seq, fallback_cost * max_rp
    



def build_row_activation_model(
    model: gp.Model,
    vars,
    arch,
    workloads: list,
    tile_info,
    buf_util_var: dict,
    mem_reads_inst: dict,
    macs_scale_factors: list[float],
    macs_scale_factor_logs: np.ndarray,
) -> dict[tuple[int, int], gp.Var]:
    """
    Build row activation model for all datatypes.
    
    Returns:
        Dict mapping (w, t_id) -> row_activation_cycles variable for each workload/datatype
    
    ==========================================================================
    两种 Crossing 类型与布局模式的关系
    ==========================================================================
    
    | 布局模式       | DRAM Row Crossing        | Input Block Crossing       |
    |----------------|--------------------------|----------------------------|
    | Sequential     | ✅ 存在 (tile vs row)    | ✅ 仅 Input 存在           |
    | Row-aligned    | ❌ = 0 (block 对齐到行)  | ✅ 仅 Input 存在           |
    
    ==========================================================================
    完整 ILP 公式
    ==========================================================================
    
    对于 Input:
      total = (1 - row_aligned) × DRAM_Row_Crossing_seq 
            + row_aligned × row_acts_row_aligned 
            + Input_Block_Crossing
    
    对于 Weight/Output:
      total = (1 - row_aligned) × DRAM_Row_Crossing_seq 
            + row_aligned × row_acts_row_aligned
    
    ==========================================================================
    注意: 本模块只计算 Row Activation Cycles (row_acts × activation_latency)
    Data Transfer Cycles 在 objective.py 的 build_latency_objective 中计算:
        mem_cycles[m] = Σ mem_reads[m,t] / read_bandwidth[m]
    ==========================================================================
    """
    TENSOR_NAMES = {0: "input", 1: "weight", 2: "output"}
    PWL_OPTS = "FuncPieces=1 FuncPieceError=0.1"
    
    # Get memory level indices
    shared_rowbuf_idx = arch.mem_idx.get("RowBuffer") if hasattr(arch, "mem_idx") else None
    dram_level = arch.mem_idx.get("LocalDRAM") if hasattr(arch, "mem_idx") else None
    
    if dram_level is None or shared_rowbuf_idx is None:
        return {}
    
    # Get DRAM parameters
    row_buffer_bytes = 1024.0
    if hasattr(arch, "mem_row_buffer_size"):
        rb_size = arch.mem_row_buffer_size[shared_rowbuf_idx]
        if rb_size not in (None, 0):
            row_buffer_bytes = float(rb_size)
    
    activation_latency = getattr(arch, "dram_activation_latency", 25.0)
    allowed_dtypes = getattr(arch, "row_activation_dtypes", ["input", "weight", "output"]) or []
    s_temporal = 1 if dram_level > 0 else 3
    
    # Dict to store per-datatype row activation cycles
    row_act_cycles_dict = {}
    
    for w, workload in enumerate(workloads):
        for t_id in range(3):
            t_name = TENSOR_NAMES[t_id]
            if t_name not in allowed_dtypes:
                continue
            if not arch.mem_stores_datatype[dram_level][t_id]:
                continue
            
            # -----------------------------------------------------------------
            # Step 1: Compute reuse_penalty and outer_irr_product
            # -----------------------------------------------------------------
            irrelevant_dims = [j for j in range(len(workload.O)) if workload.O[j][t_id] == 0]
            
            # xj[j] = 1 means dimension j is "Outer" (Interleaved/Thrashing)
            # xj[j] = 0 means dimension j is "Inner" (Stationary/Reuse)
            
            # Formula: RowActs = (NC + 2 * C * ReusePenalty) * OuterPenalty
            # ReusePenalty = Product of Inner Irrelevant Dims (xj=0) -> Causes thrashing on Crossing Tiles
            # OuterPenalty = Product of Outer Irrelevant Dims (xj=1) -> Repeats the whole process
            
            # Construct inclusion_vars for xj=0 (1-xj) for ReusePenalty
            inner_vars = {}
            for j in irrelevant_dims:
                xj = vars.xj.get((w, t_id, shared_rowbuf_idx, dram_level, j))
                if xj is not None:
                    inner_vars[j] = 1 - xj

            # 1. ReusePenalty Components
            
            # 1a. L3 Inner (xj=0) -> reuse_penalty_dram
            log_rp_dram, max_rp_dram = _build_log_product_expr(
                model, vars, workload, w, t_id, irrelevant_dims,
                shared_rowbuf_idx, dram_level, s_temporal, "rp_dram",
                target_levels=[dram_level],
                inclusion_vars=inner_vars
            )
            # reuse_penalty_dram = _build_exp_var_from_log(...) # Optional if needed separately
            
            # 1b. L2 Inner (xj=0) -> reuse_penalty_rowbuf
            # Note: Previously this was labeled "L2 Outer", but reuse_penalty is for Inner loops.
            # We must use inner_vars (1-xj) here too.
            rb_levels = [shared_rowbuf_idx]
            if rb_levels:
                log_rp_rb, max_rp_rb = _build_log_product_expr(
                    model, vars, workload, w, t_id, irrelevant_dims,
                    shared_rowbuf_idx, dram_level, s_temporal, "rp_rb",
                    target_levels=rb_levels,
                    inclusion_vars=inner_vars # Use 1-xj for Inner
                )
                
                # Total Reuse Penalty = L3_Inner * L2_Inner
                log_rp_total = log_rp_dram + log_rp_rb
                max_rp_total = max_rp_dram * max_rp_rb
            else:
                log_rp_total = log_rp_dram
                max_rp_total = max_rp_dram

            total_reuse_penalty = _build_exp_var_from_log(
                model, log_rp_total, max_rp_total, f"total_reuse_penalty_({w},{t_id})", PWL_OPTS
            )
            
            # -----------------------------------------------------------------
            # Step 2-3: Sequential mode DRAM Row Crossing
            # -----------------------------------------------------------------
            tile_entries = tile_info.tile_entry_options.get((w, t_id), ())
            xu_vars = tile_info.tile_xu_vars.get((w, t_id), [])
            
            elem_bits_map = getattr(arch, "element_bits_per_dtype", None)
            element_bits = elem_bits_map.get(t_name, 8) if isinstance(elem_bits_map, dict) else 8
            element_bytes = max(1.0, math.ceil(float(element_bits) / 8.0))
            tensor_bytes = _compute_tensor_bytes(workload, t_id, element_bytes)
            
            # outer_irr_product = Product of Outer Irrelevant Dims (xj=1)
            # We use default xj (which is 1 for Outer)
            
            outer_irr_product, max_outer_irr = _build_outer_irrelevant_product(
                model, vars, workload, w, t_id,
                shared_rowbuf_idx, dram_level, s_temporal,
                outer_vars=None # Use default xj
            )
            
            # Determine if the innermost loop is irrelevant (Thrashing)
            # Innermost position index is 0 (based on debug output Position 1..6, assuming 0-based internal)
            # We check if the dimension at Position 0 is irrelevant.
            
            is_thrashing = model.addVar(vtype=gp.GRB.BINARY, name=f"is_thrashing_({w},{t_id})")
            thrashing_expr = gp.LinExpr()
            
            # Innermost position index (assuming 0 is innermost)
            innermost_pos = 0
            
            for dim in irrelevant_dims:
                # Check if vars.xp has this key
                if (w, dram_level, innermost_pos, dim) in vars.xp:
                     thrashing_expr += vars.xp[w, dram_level, innermost_pos, dim]
            
            model.addConstr(is_thrashing == thrashing_expr, name=f"C_is_thrashing_({w},{t_id})")
            
            # Create is_reused binary variable
            # is_reused = 1 if log_rp_total > 0 (Reuse > 1)
            # is_reused = 0 if log_rp_total == 0 (Reuse = 1)
            is_reused = model.addVar(vtype=gp.GRB.BINARY, name=f"is_reused_({w},{t_id})")
            # Big-M constraint: log_rp_total <= M * is_reused
            # M should be large enough. log(1e9) ~ 20. 100 is safe.
            model.addConstr(log_rp_total <= 100 * is_reused, name=f"C_is_reused_({w},{t_id})")
            
            row_acts_seq = None
            row_acts_seq_ub = 0
            used_lookup = False
            
            if t_id == 0:
                 # Try lookup table
                 la_seq, la_ub = _build_lookup_table_constraints(
                     model, vars, workload, w, t_id, dram_level, s_temporal,
                     total_reuse_penalty, max_rp_total
                 )
                 if la_seq is not None:
                     row_acts_seq = la_seq
                     row_acts_seq_ub = la_ub
                     used_lookup = True

            if not used_lookup:
                if tile_entries and xu_vars:
                    row_acts_seq, row_acts_seq_ub = _build_sequential_dram_crossing(
                        model, tile_entries, xu_vars, element_bytes, row_buffer_bytes,
                        tensor_bytes, total_reuse_penalty, max_rp_total,
                        outer_irr_product, max_outer_irr, w, t_id,
                        is_thrashing=is_thrashing,
                        is_reused=is_reused
                    )
                else:
                    # Fallback: conservative estimate
                    row_acts_seq_ub = tensor_bytes / element_bytes * 2 * max_rp_total * max_outer_irr
                    row_acts_seq = model.addVar(lb=1, ub=row_acts_seq_ub, name=f"row_acts_({w},{t_id})")
                    model.addConstr(row_acts_seq == tensor_bytes / element_bytes, name=f"C_row_acts_fallback_({w},{t_id})")
            
            # -----------------------------------------------------------------
            # Step 1.5: Compute row_acts_row_aligned
            # -----------------------------------------------------------------
            # Revised Logic:
            # row_acts = Product(Relevant Dims) * Product(Outer Irrelevant Dims)
            # We assume Inner Irrelevant Dims are "free" (Cost=1) because row_aligned
            # implies we are optimizing for the row buffer, so small inner loops fit.
            
            relevant_dims = [j for j in range(len(workload.O)) if workload.O[j][t_id] == 1]
            target_levels = list(range(shared_rowbuf_idx + 1, dram_level + 1))
            
            log_relevant_expr, max_relevant = _build_log_product_expr(
                model, vars, workload, w, t_id, relevant_dims,
                shared_rowbuf_idx, dram_level, s_temporal, "relevant",
                target_levels=target_levels
            )
            
            # Combine with outer_irr_product
            # log(row_acts) = log(relevant) + log(outer_irr)
            
            # We need log_outer_irr. It was created inside _build_outer_irrelevant_product but not returned.
            # We can recreate it or use the helper again.
            # Actually, _build_outer_irrelevant_product returns the exp var.
            # We can get the log var from it if we named it consistently, or just recompute log sum.
            
            # Recompute log_outer_irr for summation
            log_outer_irr, _ = _build_log_product_expr(
                model, vars, workload, w, t_id, irrelevant_dims,
                shared_rowbuf_idx, dram_level, s_temporal, "outer_irr_recalc",
                target_levels=[dram_level],
                inclusion_vars=None # Use default xj=1
            )
            
            # FIX: If tensor fits in Row Buffer, Outer Loops do NOT cause thrashing.
            if tensor_bytes <= row_buffer_bytes:
                log_aligned_total = log_relevant_expr
                max_aligned = max_relevant
            else:
                log_aligned_total = log_relevant_expr + log_outer_irr
                max_aligned = max_relevant * max_outer_irr
            
            row_acts_aligned = _build_exp_var_from_log(
                model, log_aligned_total, max_aligned, f"row_acts_row_aligned_({w},{t_id})", PWL_OPTS
            )
            
            # -----------------------------------------------------------------
            # Step 3.5: Input Block Crossing (Input only) - H and W directions
            # -----------------------------------------------------------------
            if t_id == 0 and not used_lookup:
                # H direction block crossing (P, R)
                block_crossing_acts_h, max_block_crossing_h = _build_input_block_crossing_acts(
                    model, vars, workload, w, dram_level,
                    s_temporal, total_reuse_penalty, max_rp_total, direction="H"
                )
                
                # W direction block crossing (Q, S)
                block_crossing_acts_w, max_block_crossing_w = _build_input_block_crossing_acts(
                    model, vars, workload, w, dram_level,
                    s_temporal, total_reuse_penalty, max_rp_total, direction="W"
                )
                
                # Combine H and W direction block crossings
                max_block_crossing = max_block_crossing_h + max_block_crossing_w
                block_crossing_ub = max_block_crossing * 2 * max_rp_total
                
                if isinstance(block_crossing_acts_h, gp.Var) and isinstance(block_crossing_acts_w, gp.Var):
                    # Both H and W have crossings
                    block_crossing_acts = model.addVar(
                        lb=0, ub=block_crossing_ub,
                        name=f"input_block_crossing_acts_total_({w})"
                    )
                    model.addConstr(
                        block_crossing_acts == block_crossing_acts_h + block_crossing_acts_w,
                        name=f"C_input_block_crossing_total_({w})"
                    )
                elif isinstance(block_crossing_acts_h, gp.Var):
                    # Only H has crossings
                    block_crossing_acts = block_crossing_acts_h
                elif isinstance(block_crossing_acts_w, gp.Var):
                    # Only W has crossings
                    block_crossing_acts = block_crossing_acts_w
                else:
                    # Neither direction has crossings
                    block_crossing_acts = 0
                    block_crossing_ub = 0
            else:
                block_crossing_acts, block_crossing_ub = 0, 0
            
            # -----------------------------------------------------------------
            # Step 4: Combine based on layout mode
            # -----------------------------------------------------------------
            row_aligned_var = vars.layout_choice.get((w, t_id, "row_aligned"))
            
            total_acts, total_acts_ub = _build_layout_conditional_acts(
                model, row_aligned_var,
                row_acts_seq, row_acts_seq_ub,
                row_acts_aligned, max_aligned,
                block_crossing_acts, block_crossing_ub,
                w, t_id
            )
            
            # -----------------------------------------------------------------
            # Step 5: Scale and convert to cycles
            # -----------------------------------------------------------------
            scale = macs_scale_factors[w]
            
            scaled_acts = model.addVar(lb=0, ub=total_acts_ub * scale, name=f"row_acts_scaled_({w},{t_id})")
            model.addConstr(scaled_acts == total_acts * scale, name=f"C_row_acts_scaled_({w},{t_id})")
            
            cycles = model.addVar(lb=0, ub=total_acts_ub * scale * activation_latency, name=f"row_acts_cycles_({w},{t_id})")
            model.addConstr(cycles == scaled_acts * activation_latency, name=f"C_row_acts_cycles_({w},{t_id})")
            
            row_act_cycles_dict[w, t_id] = cycles
    
    return row_act_cycles_dict
