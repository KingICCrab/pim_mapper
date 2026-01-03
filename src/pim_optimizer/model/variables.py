"""
Decision variables for the ILP model.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import gurobipy as gp


# Spatial dimension indices for PE layer (m=0)
# For m>0, only TEMPORAL is used (no spatial parallelism above PE)
class SpatialDim:
    """Constants for spatial dimension indices in xb[w, m, s, j, i]."""
    H = 0           # PE array H direction (rows)
    W = 1           # PE array W direction (columns)
    INTERNAL = 2    # PE internal parallelism (e.g., Tensor Core)
    TEMPORAL = 3    # Temporal loop (no spatial parallelism)
    
    # For backward compatibility and m>0 levels
    SPATIAL = 0     # Legacy: combined spatial (only for m>0 compatibility)
    
    @classmethod
    def spatial_dims(cls) -> list:
        """Return list of spatial dimension indices (H, W, Internal)."""
        return [cls.H, cls.W, cls.INTERNAL]
    
    @classmethod
    def all_dims(cls) -> list:
        """Return all dimension indices for PE layer."""
        return [cls.H, cls.W, cls.INTERNAL, cls.TEMPORAL]
    
    @classmethod
    def num_dims_pe(cls) -> int:
        """Number of dimensions for PE layer (m=0)."""
        return 4
    
    @classmethod
    def num_dims_other(cls) -> int:
        """Number of dimensions for non-PE layers (m>0)."""
        return 2  # Just spatial(dummy) and temporal


@dataclass
class VariableSet:
    """
    Container for all decision variables in the ILP model.
    
    Attributes:
        xb: Loop bound selection (xb[w, m, s, j, i])
            w: workload index
            m: memory level
            s: spatial direction index
               - For m=0 (PE layer): {0=H, 1=W, 2=Internal, 3=Temporal}
               - For m>0 (other layers): {0=spatial(must be 1), 1=temporal}
            j: dimension index (0=R, 1=S, 2=P, 3=Q, 4=C, 5=K, 6=N)
            i: divisor index
            
        xp: Temporal loop permutation (xp[w, m, p, j])
            p: permutation level
            j: dimension index
            
        xd: Memory datatype bypass (xd[w, m, t])
            t: datatype (0=input, 1=weight, 2=output)
            
        layout_choice: Data layout mode selection
        
        xr: Inner relevant loop tracking
        xj: Dimension inner loop tracking
    """
    xb: gp.tupledict = field(default_factory=gp.tupledict)
    xp: gp.tupledict = field(default_factory=gp.tupledict)
    xd: gp.tupledict = field(default_factory=gp.tupledict)
    layout_choice: gp.tupledict = field(default_factory=gp.tupledict)
    xr: gp.tupledict = field(default_factory=gp.tupledict)
    xj: dict = field(default_factory=dict)
    is_tiled: gp.tupledict = field(default_factory=gp.tupledict)
    
    # RowBuffer Input block selection
    rowbuf_input_block_h: gp.tupledict = field(default_factory=gp.tupledict)
    rowbuf_input_block_w: gp.tupledict = field(default_factory=gp.tupledict)
    rowbuf_input_block_h_log: dict = field(default_factory=dict)
    rowbuf_input_block_w_log: dict = field(default_factory=dict)
    rowbuf_input_block_c_log: dict = field(default_factory=dict)


def create_decision_variables(
    model: gp.Model,
    arch,
    workloads: list,
    fix_permutations: bool = False,
    fix_bypass: bool = False,
    optimize_bypass: bool = False,
) -> VariableSet:
    """
    Create all decision variables for the ILP model.
    
    Args:
        model: Gurobi model
        arch: Architecture definition
        workloads: List of workload definitions
        fix_permutations: Whether to fix permutations across workloads
        fix_bypass: Whether to fix bypass across workloads
        optimize_bypass: Whether to optimize bypass decisions
        
    Returns:
        VariableSet containing all decision variables
    """
    vars = VariableSet()
    num_mems = arch.num_mems
    num_datatypes = 3
    
    # =========================================
    # Loop Bound Variables (xb)
    # For PE layer (m=0): s ∈ {H, W, Internal, Temporal}
    # For other layers (m>0): s ∈ {spatial(dummy), temporal}
    # =========================================
    for w, workload in enumerate(workloads):
        for m in range(num_mems):
            if m == 0:
                # PE layer: 4 spatial dimensions (H, W, Internal, Temporal)
                s_range = SpatialDim.num_dims_pe()
            else:
                # Other layers: only 2 (spatial placeholder, temporal)
                s_range = SpatialDim.num_dims_other()
            
            for s in range(s_range):
                for j, divs in enumerate(workload.divisors):
                    for i, div in enumerate(divs):
                        vname = f"XB_({w},{m},{s},{j},{i})"
                        vars.xb[w, m, s, j, i] = model.addVar(
                            vtype=gp.GRB.BINARY, name=vname
                        )
    
    # =========================================
    # Temporal Loop Permutation Variables (xp)
    # =========================================
    for w, workload in enumerate(workloads):
        for m in range(num_mems):
            for p, _ in enumerate(workload.bounds):
                for j, dim in enumerate(workload.bounds):
                    vname = f"XP_({w},{m},{p},{j})"
                    vars.xp[w, m, p, j] = model.addVar(
                        vtype=gp.GRB.BINARY, name=vname
                    )
    
    # =========================================
    # Memory Datatype Bypass Variables (xd)
    # =========================================
    if fix_bypass:
        # Same bypass for all workloads
        for m in range(num_mems):
            for t in range(num_datatypes):
                vname = f"XD_f_({m},{t})"
                xd_mt = model.addVar(vtype=gp.GRB.BINARY, name=vname)
                for w, workload in enumerate(workloads):
                    vars.xd[w, m, t] = xd_mt
    else:
        for w, workload in enumerate(workloads):
            for m in range(num_mems):
                for t in range(num_datatypes):
                    vname = f"XD_({w},{m},{t})"
                    vars.xd[w, m, t] = model.addVar(
                        vtype=gp.GRB.BINARY, name=vname
                    )
    
    # =========================================
    # Layout Choice Variables
    # =========================================
    dram_level = arch.mem_idx.get("LocalDRAM")
    if dram_level is not None:
        layout_modes = ("sequential", "row_aligned")
        for w in range(len(workloads)):
            for t in range(num_datatypes):
                for mode in layout_modes:
                    vname = f"X_LAYOUT_({w},{t},{mode})"
                    vars.layout_choice[w, t, mode] = model.addVar(
                        vtype=gp.GRB.BINARY, name=vname
                    )
    
    # =========================================
    # RowBuffer Input Block Selection Variables
    # =========================================
    shared_rowbuf_idx = arch.mem_idx.get("RowBuffer")
    if shared_rowbuf_idx is not None:
        for w, workload in enumerate(workloads):
            h_divisors = getattr(workload, 'hw_divisors', {}).get('H', [1])
            w_divisors = getattr(workload, 'hw_divisors', {}).get('W', [1])
            
            if not h_divisors:
                h_divisors = [1]
            if not w_divisors:
                w_divisors = [1]
            
            # Block H selection
            h_log_expr = gp.LinExpr(0)
            for i, h_div in enumerate(h_divisors):
                vname = f"X_ROWBUF_INPUT_BLOCK_H_({w},{i})"
                vars.rowbuf_input_block_h[w, i] = model.addVar(
                    vtype=gp.GRB.BINARY, name=vname
                )
                h_log_expr += vars.rowbuf_input_block_h[w, i] * float(np.log(h_div))
            vars.rowbuf_input_block_h_log[w] = h_log_expr
            
            # Block W selection
            w_log_expr = gp.LinExpr(0)
            for j, w_div in enumerate(w_divisors):
                vname = f"X_ROWBUF_INPUT_BLOCK_W_({w},{j})"
                vars.rowbuf_input_block_w[w, j] = model.addVar(
                    vtype=gp.GRB.BINARY, name=vname
                )
                w_log_expr += vars.rowbuf_input_block_w[w, j] * float(np.log(w_div))
            vars.rowbuf_input_block_w_log[w] = w_log_expr
    
    return vars


def create_auxiliary_variables(
    model: gp.Model,
    vars: VariableSet,
    arch,
    workloads: list,
    fix_permutations: bool = False,
) -> None:
    """
    Create auxiliary variables for data reuse tracking.
    
    This creates xr and xj variables that track data reuse patterns.
    
    Args:
        model: Gurobi model
        vars: Variable set to populate
        arch: Architecture definition
        workloads: List of workloads
        fix_permutations: Whether permutations are fixed
    """
    num_mems = arch.num_mems
    num_datatypes = 3
    
    # xr[w, t, m, m_, p] = 1 if there is a relevant inner loop at p' <= p
    for w, workload in enumerate(workloads):
        for t in range(num_datatypes):
            for m in range(num_mems - 1):
                for m_ in range(m + 1, num_mems):
                    for p, _ in enumerate(workload.bounds):
                        vname = f"XR_({w},{t},{m},{m_},{p})"
                        vars.xr[w, t, m, m_, p] = model.addVar(
                            vtype=gp.GRB.BINARY, name=vname
                        )
    
    # xj[w, t, m, m_, j] = 1 if dimension j has inner relevant loop
    for w, workload in enumerate(workloads):
        for t in range(num_datatypes):
            for m in range(num_mems - 1):
                for m_ in range(m + 1, num_mems):
                    for j, _ in enumerate(workload.bounds):
                        vname = f"XJ_({w},{t},{m},{m_},{j})"
                        vars.xj[w, t, m, m_, j] = model.addVar(
                            vtype=gp.GRB.BINARY, name=vname
                        )

    # is_tiled[w, m, j] = 1 if dimension j has tiling factor > 1 at level m
    for w, workload in enumerate(workloads):
        for m in range(num_mems):
            for j, _ in enumerate(workload.bounds):
                vname = f"IS_TILED_({w},{m},{j})"
                vars.is_tiled[w, m, j] = model.addVar(
                    vtype=gp.GRB.BINARY, name=vname
                )
