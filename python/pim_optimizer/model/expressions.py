"""
Expression builders for memory reads/writes.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import gurobipy as gp

from pim_optimizer.model.variables import VariableSet, SpatialDim


@dataclass
class TileInfo:
    """Container for tile information used in row activation model."""
    tile_entry_options: dict = field(default_factory=dict)  # (w, t) -> tuple of tile sizes
    tile_xu_vars: dict = field(default_factory=dict)        # (w, t) -> list of binary selection vars
    tile_bytes_var: dict = field(default_factory=dict)      # (w, t) -> gp.Var
    buf_util_var: dict = field(default_factory=dict)        # (w, m, t) -> gp.Var
    input_tile_info: dict = field(default_factory=dict)     # w -> {yh, xh, yw, xw, ...}


def compute_unique_input_size(stride: int, dilation: int, output_tile: int, kernel_tile: int) -> int:
    """
    Compute the input tile size (span) for conv2d.
    
    Formula: H_in = (P-1)*stride + (R-1)*dilation + 1
    
    Args:
        stride: Convolution stride
        dilation: Convolution dilation
        output_tile: Output dimension tile size (P or Q)
        kernel_tile: Kernel dimension tile size (R or S)
    
    Returns:
        Input tile size (span)
    """
    if output_tile <= 0 or kernel_tile <= 0:
        return 0
    
    return stride * (output_tile - 1) + dilation * (kernel_tile - 1) + 1


def _get_s_range(m: int) -> range:
    """Get the s index range for a given memory level."""
    if m == 0:
        return range(SpatialDim.num_dims_pe())
    else:
        return range(SpatialDim.num_dims_other())


def _get_temporal_s(m: int) -> int:
    """Get the temporal s index for a given memory level."""
    return SpatialDim.TEMPORAL if m == 0 else 1


def _get_spatial_s_range(m: int) -> list:
    """Get the spatial s indices for a given memory level."""
    if m == 0:
        return [SpatialDim.H, SpatialDim.W, SpatialDim.INTERNAL]
    else:
        return [0]  # Spatial placeholder (should always be factor 1)


def build_buffer_utilization(
    model: gp.Model,
    vars: VariableSet,
    arch,
    workloads: list,
    optimize_bypass: bool = False,
) -> tuple[dict, dict, TileInfo]:
    """
    Build buffer utilization expressions.
    
    Args:
        model: Gurobi model
        vars: Variable set
        arch: Architecture definition
        workloads: List of workloads
        optimize_bypass: Whether bypass is being optimized
        
    Returns:
        Tuple of (buf_util, buf_util_log, tile_info) 
    """
    num_mems = arch.num_mems
    num_datatypes = 3
    
    buf_util_log = {}
    buf_util = {}
    tile_info = TileInfo()
    
    # Auxiliary storage for tile selection
    yu = {}  # yu[m, t] = list of possible tile sizes
    
    # Initialize to 0
    for w in range(len(workloads)):
        for m in range(num_mems):
            for t in range(num_datatypes):
                buf_util_log[w, m, t] = 0
                buf_util[w, m, t] = 0
    
    # Get key memory level indices
    shared_rowbuf_idx = arch.mem_idx.get("RowBuffer") if hasattr(arch, "mem_idx") else None
    dram_level = arch.mem_idx.get("LocalDRAM") if hasattr(arch, "mem_idx") else None
    
    for w, workload in enumerate(workloads):
        factors = {}
        per_pe_factors = {}  # For PELocalBuffer: only temporal factors (per-PE tile size)
        
        for m in range(num_mems - 1):
            optimize_bypass_m = optimize_bypass and not arch.mem_bypass_defined[m]
            
            # Compute factors for each dimension up to this memory level
            for j, divs in enumerate(workload.divisors):
                factors[m, j] = gp.LinExpr(0)
                for m_ in range(m + 1):
                    for s in _get_s_range(m_):
                        for i, div in enumerate(divs):
                            factors[m, j] += vars.xb[w, m_, s, j, i] * np.log(div)
            
            # For PELocalBuffer (m=0): compute per-PE factors (temporal only)
            # Each PE stores only the temporal portion; spatial is distributed across PEs
            if m == 0:
                for j, divs in enumerate(workload.divisors):
                    per_pe_factors[j] = gp.LinExpr(0)
                    # Only include Temporal dimension (s=3), not H/W/Internal
                    for i, div in enumerate(divs):
                        per_pe_factors[j] += vars.xb[w, 0, SpatialDim.TEMPORAL, j, i] * np.log(div)
            
            # =====================
            # Inputs (t=0)
            # =====================
            t = 0
            if arch.mem_stores_datatype[m][t] or optimize_bypass_m:
                # Input tile size: use gcd formula for accurate unique address count
                # Old formula: stride*P + dilation*R - stride - dilation + 1 (overestimates when gcd > 1)
                # New formula: min((stride*(P-1) + dilation*(R-1)) / gcd(stride, dilation) + 1, P * R)
                
                # Width dimension selection
                xw, xwr, xwp, yw, w_log = [], [], [], [], gp.LinExpr(0)
                yw_p_list, yw_r_list = [], []  # Save P, R factors for crossing calculation
                for dp, p_ in enumerate(workload.divisors[2]):
                    for dr, r_ in enumerate(workload.divisors[0]):
                        xwr.append(np.log(r_))
                        xwp.append(np.log(p_))
                        xw.append(model.addVar(vtype=gp.GRB.BINARY, name=f"XW({w},{m},{dp},{dr})"))
                        # Use gcd formula for accurate unique input count
                        unique_w = compute_unique_input_size(
                            workload.stride[0], workload.dilation[0], p_, r_
                        )
                        yw.append(unique_w)
                        yw_p_list.append(p_)
                        yw_r_list.append(r_)
                        w_log += xw[-1] * np.log(max(unique_w, 1))  # Avoid log(0)
                
                model.addConstr(gp.quicksum(xw) == 1, name=f"C_xw_({w},{m})")
                model.addConstr(gp.LinExpr(xwr, xw) == factors[m, 0], name=f"C_xwr_({w},{m})")
                model.addConstr(gp.LinExpr(xwp, xw) == factors[m, 2], name=f"C_xwp_({w},{m})")
                
                # Height dimension selection
                xh, xhs, xhq, yh, h_log = [], [], [], [], gp.LinExpr(0)
                yh_q_list, yh_s_list = [], []  # Save Q, S factors for crossing calculation
                for dq, q_ in enumerate(workload.divisors[3]):
                    for ds, s_ in enumerate(workload.divisors[1]):
                        xhs.append(np.log(s_))
                        xhq.append(np.log(q_))
                        xh.append(model.addVar(vtype=gp.GRB.BINARY, name=f"XH({w},{m},{dq},{ds})"))
                        # Use gcd formula for accurate unique input count
                        unique_h = compute_unique_input_size(
                            workload.stride[1], workload.dilation[1], q_, s_
                        )
                        yh.append(unique_h)
                        yh_q_list.append(q_)
                        yh_s_list.append(s_)
                        h_log += xh[-1] * np.log(max(unique_h, 1))  # Avoid log(0)
                
                model.addConstr(gp.quicksum(xh) == 1, name=f"C_xh_({w},{m})")
                model.addConstr(gp.LinExpr(xhs, xh) == factors[m, 1], name=f"C_xhs_({w},{m})")
                model.addConstr(gp.LinExpr(xhq, xh) == factors[m, 3], name=f"C_xhq_({w},{m})")
                
                # For PELocalBuffer (m=0): use per-PE factors (temporal only)
                # Each PE stores only its portion; spatial dimensions are distributed across PEs
                if m == 0:
                    # Build per-PE input tile size using temporal-only P, Q, R, S
                    xw_pe, xwr_pe, xwp_pe, w_log_pe = [], [], [], gp.LinExpr(0)
                    for dp, p_ in enumerate(workload.divisors[2]):
                        for dr, r_ in enumerate(workload.divisors[0]):
                            xwr_pe.append(np.log(r_))
                            xwp_pe.append(np.log(p_))
                            xw_pe.append(model.addVar(vtype=gp.GRB.BINARY, name=f"XW_PE({w},{dp},{dr})"))
                            unique_w = compute_unique_input_size(
                                workload.stride[0], workload.dilation[0], p_, r_
                            )
                            w_log_pe += xw_pe[-1] * np.log(max(unique_w, 1))
                    
                    model.addConstr(gp.quicksum(xw_pe) == 1, name=f"C_xw_pe_({w})")
                    model.addConstr(gp.LinExpr(xwr_pe, xw_pe) == per_pe_factors[0], name=f"C_xwr_pe_({w})")
                    model.addConstr(gp.LinExpr(xwp_pe, xw_pe) == per_pe_factors[2], name=f"C_xwp_pe_({w})")
                    
                    xh_pe, xhs_pe, xhq_pe, h_log_pe = [], [], [], gp.LinExpr(0)
                    for dq, q_ in enumerate(workload.divisors[3]):
                        for ds, s_ in enumerate(workload.divisors[1]):
                            xhs_pe.append(np.log(s_))
                            xhq_pe.append(np.log(q_))
                            xh_pe.append(model.addVar(vtype=gp.GRB.BINARY, name=f"XH_PE({w},{dq},{ds})"))
                            unique_h = compute_unique_input_size(
                                workload.stride[1], workload.dilation[1], q_, s_
                            )
                            h_log_pe += xh_pe[-1] * np.log(max(unique_h, 1))
                    
                    model.addConstr(gp.quicksum(xh_pe) == 1, name=f"C_xh_pe_({w})")
                    model.addConstr(gp.LinExpr(xhs_pe, xh_pe) == per_pe_factors[1], name=f"C_xhs_pe_({w})")
                    model.addConstr(gp.LinExpr(xhq_pe, xh_pe) == per_pe_factors[3], name=f"C_xhq_pe_({w})")
                    
                    buf_util_log[w, m, t] = w_log_pe + h_log_pe + per_pe_factors[4] + per_pe_factors[6]
                else:
                    buf_util_log[w, m, t] = w_log + h_log + factors[m, 4] + factors[m, 6]
                
                # Build enumerated buffer utilization and save input tile info for RowBuffer level
                if arch.mem_stores_multiple_datatypes[m] or optimize_bypass_m:
                    xu = []
                    yu[m, t] = []
                    min_tile = None
                    for dw, w_ in enumerate(yw):
                        for dh, h_ in enumerate(yh):
                            for dc, c_ in enumerate(workload.divisors[4]):
                                for dn, n_ in enumerate(workload.divisors[6]):
                                    val = w_ * h_ * c_ * n_
                                    if min_tile is None or val < min_tile:
                                        min_tile = val
                                    if val <= arch.mem_entries[m]:
                                        xu.append(model.addVar(
                                            vtype=gp.GRB.BINARY,
                                            name=f"XU({w},{m},{t},{dw},{dh},{dc},{dn})"
                                        ))
                                        yu[m, t].append(val)
                                        buf_util[w, m, t] += xu[-1] * val
                    
                    if xu:
                        model.addConstr(gp.quicksum(xu) == 1, name=f"C_xu_({w},{m},{t})")
                    
                    # Save tile entry options for RowBuffer level
                    if shared_rowbuf_idx is not None and m == shared_rowbuf_idx:
                        tile_info.tile_entry_options[w, t] = tuple(int(v) for v in yu[m, t])
                
                # Save input tile info for RowBuffer level (for crossing calculation)
                if shared_rowbuf_idx is not None and m == shared_rowbuf_idx:
                    tile_info.input_tile_info[w] = {
                        'yh': yh,
                        'xh': xh,
                        'yh_q': yh_q_list,
                        'yh_s': yh_s_list,
                        'yw': yw,
                        'xw': xw,
                        'yw_p': yw_p_list,
                        'yw_r': yw_r_list,
                        'stride_h': workload.stride[1],
                        'stride_w': workload.stride[0],
                        'dilation_h': workload.dilation[1],
                        'dilation_w': workload.dilation[0],
                        'total_Q': workload.bounds[3],
                        'total_P': workload.bounds[2],
                        'total_S': workload.bounds[1],
                        'total_R': workload.bounds[0],
                    }
                    
                    # =========================================
                    # Add block >= tile constraints for Input
                    # =========================================
                    # For each (block, tile) combination, if block < tile, forbid it.
                    # This ensures no crossing due to tile larger than block.
                    h_divisors = getattr(workload, 'hw_divisors', {}).get('H', [1])
                    w_divisors = getattr(workload, 'hw_divisors', {}).get('W', [1])
                    
                    # H dimension: block_h >= tile_h
                    for i, block_h in enumerate(h_divisors):
                        for j, tile_h in enumerate(yh):
                            if block_h < tile_h:
                                # block_h_var[i] + tile_h_var[j] <= 1
                                # (cannot select both small block AND large tile)
                                model.addConstr(
                                    vars.rowbuf_input_block_h[w, i] + xh[j] <= 1,
                                    name=f"C_block_ge_tile_h_({w},{i},{j})"
                                )
                    
                    # W dimension: block_w >= tile_w  
                    for i, block_w in enumerate(w_divisors):
                        for j, tile_w in enumerate(yw):
                            if block_w < tile_w:
                                # block_w_var[i] + tile_w_var[j] <= 1
                                model.addConstr(
                                    vars.rowbuf_input_block_w[w, i] + xw[j] <= 1,
                                    name=f"C_block_ge_tile_w_({w},{i},{j})"
                                )
                    
                    # =========================================
                    # CRITICAL: block_h × block_w × C_tile × N_tile <= row_buffer_size
                    # =========================================
                    # For row_aligned layout, the data block must fit in a DRAM row.
                    # This is a joint constraint on block size and C/N tiling.
                    #
                    # In log space: log(block_h) + log(block_w) + log(C) + log(N) <= log(row_size)
                    #
                    # We use the layout_choice variable to only enforce when row_aligned.
                    row_buffer_size = 1024  # Default
                    if hasattr(arch, 'dram_bank_row_buffer_size'):
                        row_buffer_size = arch.dram_bank_row_buffer_size
                    elif hasattr(arch, 'mem_row_buffer_size') and dram_level is not None:
                        rb_size = arch.mem_row_buffer_size[dram_level]
                        if rb_size is not None:
                            row_buffer_size = rb_size
                    
                    # log(block_h) + log(block_w) is from the block selection variables
                    block_h_log = vars.rowbuf_input_block_h_log[w]
                    block_w_log = vars.rowbuf_input_block_w_log[w]
                    
                    # log(C_tile) + log(N_tile) at RowBuffer level
                    c_tile_log = factors[m, 4]  # C dimension (index 4)
                    n_tile_log = factors[m, 6]  # N dimension (index 6)
                    
                    # Total log size = log(block_h) + log(block_w) + log(C) + log(N)
                    total_block_log = block_h_log + block_w_log + c_tile_log + n_tile_log
                    
                    # For row_aligned layout: total <= log(row_size)
                    # We use indicator constraint: if layout_choice[row_aligned]=1, then total <= log(row_size)
                    model.addGenConstrIndicator(
                        vars.layout_choice[w, 0, "row_aligned"], True,
                        total_block_log <= np.log(row_buffer_size),
                        name=f"C_input_block_fits_row_({w})"
                    )
            
            # =====================
            # Weights (t=1)
            # =====================
            t = 1
            if arch.mem_stores_datatype[m][t] or optimize_bypass_m:
                # For PELocalBuffer (m=0): use per-PE factors (temporal only)
                if m == 0:
                    buf_util_log[w, m, t] = per_pe_factors[0] + per_pe_factors[1] + per_pe_factors[4] + per_pe_factors[5]
                else:
                    buf_util_log[w, m, t] = factors[m, 0] + factors[m, 1] + factors[m, 4] + factors[m, 5]
                
                # Build enumerated buffer utilization
                if arch.mem_stores_multiple_datatypes[m] or optimize_bypass_m:
                    # Re-implementing the loop to build log expressions
                    xu_r_log = gp.LinExpr(0)
                    xu_s_log = gp.LinExpr(0)
                    xu_c_log = gp.LinExpr(0)
                    xu_k_log = gp.LinExpr(0)
                    
                    # Clear xu and rebuild to capture the log sums
                    xu = []
                    yu[m, t] = []
                    buf_util[w, m, t] = 0 # Reset
                    min_tile = None
                    
                    for dr, r_ in enumerate(workload.divisors[0]):
                        for ds, s_ in enumerate(workload.divisors[1]):
                            for dc, c_ in enumerate(workload.divisors[4]):
                                for dk, k_ in enumerate(workload.divisors[5]):
                                    val = r_ * s_ * c_ * k_
                                    if min_tile is None or val < min_tile:
                                        min_tile = val
                                    if val <= arch.mem_entries[m]:
                                        xu_var = model.addVar(
                                            vtype=gp.GRB.BINARY,
                                            name=f"XU({w},{m},{t},{dr},{ds},{dc},{dk})"
                                        )
                                        xu.append(xu_var)
                                        yu[m, t].append(val)
                                        buf_util[w, m, t] += xu[-1] * val
                                        
                                        xu_r_log += xu_var * np.log(r_)
                                        xu_s_log += xu_var * np.log(s_)
                                        xu_c_log += xu_var * np.log(c_)
                                        xu_k_log += xu_var * np.log(k_)
                    
                    if xu:
                        model.addConstr(gp.quicksum(xu) == 1, name=f"C_xu_({w},{m},{t})")
                        
                        # Consistency constraints
                        model.addConstr(xu_r_log == factors[m, 0], name=f"C_xu_xb_consistent_r_({w},{m},{t})")
                        model.addConstr(xu_s_log == factors[m, 1], name=f"C_xu_xb_consistent_s_({w},{m},{t})")
                        model.addConstr(xu_c_log == factors[m, 4], name=f"C_xu_xb_consistent_c_({w},{m},{t})")
                        model.addConstr(xu_k_log == factors[m, 5], name=f"C_xu_xb_consistent_k_({w},{m},{t})")
                    
                    # Save tile entry options and selection vars for RowBuffer level
                    if shared_rowbuf_idx is not None and m == shared_rowbuf_idx:
                        tile_info.tile_entry_options[w, t] = tuple(int(v) for v in yu[m, t])
                        tile_info.tile_xu_vars[w, t] = xu  # Save selection variables
            
            # =====================
            # Outputs (t=2)
            # =====================
            t = 2
            if arch.mem_stores_datatype[m][t] or optimize_bypass_m:
                # For PELocalBuffer (m=0): use per-PE factors (temporal only)
                if m == 0:
                    buf_util_log[w, m, t] = per_pe_factors[2] + per_pe_factors[3] + per_pe_factors[5] + per_pe_factors[6]
                else:
                    buf_util_log[w, m, t] = factors[m, 2] + factors[m, 3] + factors[m, 5] + factors[m, 6]
                
                # Build enumerated buffer utilization
                if arch.mem_stores_multiple_datatypes[m] or optimize_bypass_m:
                    # Re-implementing the loop to build log expressions
                    xu_p_log = gp.LinExpr(0)
                    xu_q_log = gp.LinExpr(0)
                    xu_k_log = gp.LinExpr(0)
                    xu_n_log = gp.LinExpr(0)
                    
                    # Clear xu and rebuild to capture the log sums
                    xu = []
                    yu[m, t] = []
                    buf_util[w, m, t] = 0 # Reset
                    min_tile = None
                    
                    for dp, p_ in enumerate(workload.divisors[2]):
                        for dq, q_ in enumerate(workload.divisors[3]):
                            for dk, k_ in enumerate(workload.divisors[5]):
                                for dn, n_ in enumerate(workload.divisors[6]):
                                    val = p_ * q_ * k_ * n_
                                    if min_tile is None or val < min_tile:
                                        min_tile = val
                                    if val <= arch.mem_entries[m]:
                                        xu_var = model.addVar(
                                            vtype=gp.GRB.BINARY,
                                            name=f"XU({w},{m},{t},{dp},{dq},{dk},{dn})"
                                        )
                                        xu.append(xu_var)
                                        yu[m, t].append(val)
                                        buf_util[w, m, t] += xu[-1] * val
                                        
                                        xu_p_log += xu_var * np.log(p_)
                                        xu_q_log += xu_var * np.log(q_)
                                        xu_k_log += xu_var * np.log(k_)
                                        xu_n_log += xu_var * np.log(n_)
                    
                    if xu:
                        model.addConstr(gp.quicksum(xu) == 1, name=f"C_xu_({w},{m},{t})")
                        
                        # Consistency constraints
                        model.addConstr(xu_p_log == factors[m, 2], name=f"C_xu_xb_consistent_p_({w},{m},{t})")
                        model.addConstr(xu_q_log == factors[m, 3], name=f"C_xu_xb_consistent_q_({w},{m},{t})")
                        model.addConstr(xu_k_log == factors[m, 5], name=f"C_xu_xb_consistent_k_({w},{m},{t})")
                        model.addConstr(xu_n_log == factors[m, 6], name=f"C_xu_xb_consistent_n_({w},{m},{t})")
                    
                    # Save tile entry options and selection vars for RowBuffer level
                    if shared_rowbuf_idx is not None and m == shared_rowbuf_idx:
                        tile_info.tile_entry_options[w, t] = tuple(int(v) for v in yu[m, t])
                        tile_info.tile_xu_vars[w, t] = xu  # Save selection variables
    
    return buf_util, buf_util_log, tile_info


def build_memory_expressions(
    model: gp.Model,
    vars: VariableSet,
    arch,
    workloads: list,
    buf_util_log: dict,
    optimize_bypass: bool = False,
) -> tuple[dict, dict, dict, dict]:
    """
    Build memory read and write expressions.
    
    Args:
        model: Gurobi model
        vars: Variable set
        arch: Architecture definition
        workloads: List of workloads
        buf_util_log: Log of buffer utilization expressions
        optimize_bypass: Whether bypass is being optimized
        
    Returns:
        Tuple of (mem_writes_inst, mem_writes, mem_reads_inst, mem_reads)
    """
    num_mems = arch.num_mems
    num_datatypes = 3
    
    mem_writes_inst = {}
    mem_writes = {}
    mem_reads_inst = {}
    mem_reads = {}
    
    for w, workload in enumerate(workloads):
        macs = np.log(workload.macs)
        
        # Memory writes
        for m in range(num_mems - 1):
            for t in range(num_datatypes):
                mem_writes_inst[w, m, t] = gp.LinExpr(0)
                mem_writes[w, m, t] = gp.LinExpr(0)
                
                if not arch.mem_stores_datatype[m][t] and not (optimize_bypass and not arch.mem_bypass_defined[m]):
                    continue
                
                # Buffer utilization multiplied by outer loops
                mem_writes_inst[w, m, t] = gp.LinExpr(buf_util_log[w, m, t])
                
                for m_ in range(m + 1, num_mems):
                    for j, divs in enumerate(workload.divisors):
                        for i, div in enumerate(divs):
                            # xb_xj = xb[...] AND xj[...]
                            xb_xj = model.addVar(
                                vtype=gp.GRB.BINARY,
                                name=f"V_and_xb_xj_({w},{t},{m},{m_},{j},{i})"
                            )
                            xj_var = vars.xj.get((w, t, m, m_, j))
                            temporal_s = _get_temporal_s(m_)
                            if xj_var is not None:
                                model.addGenConstrAnd(
                                    xb_xj,
                                    [vars.xb[w, m_, temporal_s, j, i], xj_var],
                                    name=f"C_and_xb_xj_({w},{t},{m},{m_},{j},{i})"
                                )
                            mem_writes_inst[w, m, t] += xb_xj * np.log(div)
                
                # Multiply by spatial instances
                mem_writes[w, m, t] = gp.LinExpr(mem_writes_inst[w, m, t])
                for m_ in range(m + 1, num_mems):
                    for j, divs in enumerate(workload.divisors):
                        for i, div in enumerate(divs):
                            # Sum over all spatial directions
                            for s in _get_spatial_s_range(m_):
                                mem_writes[w, m, t] += vars.xb[w, m_, s, j, i] * np.log(div)
        
        # Memory reads
        for t in range(num_datatypes):
            m_ = -1
            for m in range(num_mems):
                if arch.mem_stores_datatype[m][t]:
                    if m_ == -1:
                        # First level storing this datatype
                        mem_reads_inst[w, m, t] = gp.LinExpr(macs)
                        mem_reads[w, m, t] = gp.LinExpr(macs)
                        
                        # Subtract spatial loops (all dimensions)
                        for m__ in range(m + 1, num_mems):
                            for j, divs in enumerate(workload.divisors):
                                for i, div in enumerate(divs):
                                    for s in _get_spatial_s_range(m__):
                                        mem_reads_inst[w, m, t] -= vars.xb[w, m__, s, j, i] * np.log(div)
                        
                        # Subtract irrelevant spatial loops (within buffer levels)
                        for m__ in range(0, m + 1):
                            for j, divs in enumerate(workload.divisors):
                                for i, div in enumerate(divs):
                                    for s in _get_spatial_s_range(m__):
                                        isl = vars.xb[w, m__, s, j, i] * (1 - workload.O[j][t]) * np.log(div)
                                        mem_reads_inst[w, m, t] -= isl
                                        mem_reads[w, m, t] -= isl
                        
                        # Subtract irrelevant temporal loops (outer memory levels)
                        # Only subtract when xj=0 (no inner relevant loop).
                        # If xj=1, even though dimension is irrelevant, inner relevant loops
                        # require different data, so re-read is needed.
                        for m__ in range(m + 1, num_mems):
                            temporal_s = _get_temporal_s(m__)
                            for j, divs in enumerate(workload.divisors):
                                if workload.O[j][t] == 0:  # Irrelevant dimension
                                    for i, div in enumerate(divs):
                                        # Subtract xb AND (NOT xj): only when no inner relevant loop
                                        xj_var = vars.xj.get((w, t, m, m__, j))
                                        xb_var = vars.xb[w, m__, temporal_s, j, i]
                                        
                                        if xj_var is not None:
                                            # xb_not_xj = xb AND (1 - xj)
                                            xb_not_xj = model.addVar(
                                                vtype=gp.GRB.BINARY,
                                                name=f"V_xb_not_xj_read_({w},{t},{m},{m__},{j},{i})"
                                            )
                                            # xb_not_xj = 1 iff xb=1 AND xj=0
                                            model.addConstr(
                                                xb_not_xj <= xb_var,
                                                name=f"C_xb_not_xj_ub1_({w},{t},{m},{m__},{j},{i})"
                                            )
                                            model.addConstr(
                                                xb_not_xj <= 1 - xj_var,
                                                name=f"C_xb_not_xj_ub2_({w},{t},{m},{m__},{j},{i})"
                                            )
                                            model.addConstr(
                                                xb_not_xj >= xb_var - xj_var,
                                                name=f"C_xb_not_xj_lb_({w},{t},{m},{m__},{j},{i})"
                                            )
                                            itl = xb_not_xj * np.log(div)
                                        else:
                                            # No xj variable means no relevant inner loop tracking
                                            itl = xb_var * np.log(div)
                                        
                                        mem_reads_inst[w, m, t] -= itl
                                        mem_reads[w, m, t] -= itl
                    else:
                        # Subsequent levels
                        mem_reads_inst[w, m, t] = gp.LinExpr(mem_writes_inst[w, m_, t])
                        
                        # Multiply by spatial loops between levels
                        for m__ in range(m_ + 1, m + 1):
                            for j, divs in enumerate(workload.divisors):
                                for i, div in enumerate(divs):
                                    for s in _get_spatial_s_range(m__):
                                        mem_reads_inst[w, m, t] += vars.xb[w, m__, s, j, i] * workload.O[j][t] * np.log(div)
                        
                        mem_reads[w, m, t] = gp.LinExpr(mem_reads_inst[w, m, t])
                        for m__ in range(m + 1, num_mems):
                            for j, divs in enumerate(workload.divisors):
                                for i, div in enumerate(divs):
                                    for s in _get_spatial_s_range(m__):
                                        mem_reads[w, m, t] += vars.xb[w, m__, s, j, i] * np.log(div)
                    
                    m_ = m
                else:
                    mem_reads[w, m, t] = gp.LinExpr(0)
                    mem_reads_inst[w, m, t] = gp.LinExpr(0)
    
    return mem_writes_inst, mem_writes, mem_reads_inst, mem_reads


def build_tile_bytes_vars(
    model: gp.Model,
    vars: VariableSet,
    arch,
    workloads: list,
    tile_info: TileInfo,
    buf_util_var: dict,
) -> None:
    """
    Build tile_bytes variables for row activation model.
    
    Args:
        model: Gurobi model
        vars: Variable set
        arch: Architecture definition
        workloads: List of workloads
        tile_info: TileInfo object to populate
        buf_util_var: Buffer utilization variables from constraint building
    """
    num_datatypes = 3
    tensor_idx_to_name = {0: "input", 1: "weight", 2: "output"}
    
    shared_rowbuf_idx = arch.mem_idx.get("RowBuffer") if hasattr(arch, "mem_idx") else None
    dram_level = arch.mem_idx.get("LocalDRAM") if hasattr(arch, "mem_idx") else None
    
    if shared_rowbuf_idx is None or dram_level is None:
        return
    
    row_buffer_size_bytes = None
    if hasattr(arch, "mem_row_buffer_size"):
        rb_size = arch.mem_row_buffer_size[shared_rowbuf_idx]
        if rb_size not in (None, 0):
            row_buffer_size_bytes = float(rb_size)
        else:
            rb_entries = arch.mem_entries[shared_rowbuf_idx]
            if rb_entries not in (None, 0, -1):
                row_buffer_size_bytes = float(rb_entries)
    
    for w in range(len(workloads)):
        for t in range(num_datatypes):
            buf_key = (w, shared_rowbuf_idx, t)
            if buf_key not in buf_util_var:
                continue
            
            # Get element size
            elem_bits_map = getattr(arch, "element_bits_per_dtype", None)
            if isinstance(elem_bits_map, dict):
                dtype_name = tensor_idx_to_name.get(t, str(t))
                element_bits = elem_bits_map.get(dtype_name, getattr(arch, "default_element_bits", 8))
            else:
                element_bits = getattr(arch, "default_element_bits", 8)
            try:
                element_bits = float(element_bits)
            except Exception:
                element_bits = float(getattr(arch, "default_element_bits", 8))
            element_bytes = max(1.0, math.ceil(element_bits / 8.0))
            
            # Get tile entries upper bound
            tile_entries_candidates = tile_info.tile_entry_options.get((w, t))
            tile_entries_ub = arch.mem_entries[shared_rowbuf_idx]
            if tile_entries_ub in (None, -1):
                tile_entries_ub = 0
            if tile_entries_candidates:
                tile_entries_ub = max(tile_entries_ub, max(tile_entries_candidates))
            
            tile_bytes_ub = float(element_bytes) * float(tile_entries_ub)
            
            # Create tile_bytes variable
            tile_bytes = model.addVar(
                ub=tile_bytes_ub if tile_bytes_ub > 0 else gp.GRB.INFINITY,
                name=f"V_tile_bytes_({w},{t})"
            )
            tile_info.tile_bytes_var[w, t] = tile_bytes
            
            # Link to buffer utilization
            util_src = buf_util_var[buf_key]
            model.addConstr(
                tile_bytes == float(element_bytes) * util_src,
                name=f"C_tile_bytes_link_({w},{t})"
            )
            
            # Row-aligned layout must fit in single row buffer
            if row_buffer_size_bytes and (w, t, "row_aligned") in vars.layout_choice:
                model.addGenConstrIndicator(
                    vars.layout_choice[w, t, "row_aligned"],
                    True,
                    tile_bytes,
                    gp.GRB.LESS_EQUAL,
                    row_buffer_size_bytes,
                    name=f"C_row_aligned_tile_fit_({w},{t})"
                )
