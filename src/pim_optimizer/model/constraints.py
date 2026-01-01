"""
Constraint builders for the ILP model.
"""

import numpy as np
import gurobipy as gp

from pim_optimizer.model.variables import VariableSet, SpatialDim


def add_basic_constraints(
    model: gp.Model,
    vars: VariableSet,
    arch,
    workloads: list,
    optimize_bypass: bool = False,
) -> None:
    """
    Add basic constraints to the ILP model.
    
    These include:
    - Dimension factorization constraints
    - One factor per loop constraints
    - Permutation constraints
    - Bypass constraints
    - PE layer spatial constraints (H/W/Internal mutual exclusion)
    - Non-PE layer no-spatial constraints
    
    Args:
        model: Gurobi model
        vars: Variable set
        arch: Architecture definition
        workloads: List of workloads
        optimize_bypass: Whether bypass is being optimized
    """
    num_mems = arch.num_mems
    num_datatypes = 3
    
    # =========================================
    # Dimension Factorization Constraints
    # Product of loop bounds equals problem dimension
    # For PE layer (m=0): sum over H, W, Internal, Temporal
    # For other layers (m>0): sum over spatial(dummy), temporal
    # =========================================
    for w, workload in enumerate(workloads):
        for j, divs in enumerate(workload.divisors):
            dim = workload.bounds[j]
            log_bound_sum = gp.LinExpr(0)
            
            for m in range(num_mems):
                if m == 0:
                    # PE layer: sum over H, W, Internal, Temporal
                    for s in range(SpatialDim.num_dims_pe()):
                        for i, div in enumerate(divs):
                            log_bound_sum += vars.xb[w, m, s, j, i] * np.log(div)
                else:
                    # Other layers: sum over spatial(dummy), temporal
                    for s in range(SpatialDim.num_dims_other()):
                        for i, div in enumerate(divs):
                            log_bound_sum += vars.xb[w, m, s, j, i] * np.log(div)
            
            model.addConstr(
                log_bound_sum == np.log(dim),
                name=f"C_dim_factorization_({w},{j})"
            )
    
    # =========================================
    # One Factor Per Loop Constraints
    # Each loop selects exactly one divisor
    # =========================================
    for w, workload in enumerate(workloads):
        for m in range(num_mems):
            if m == 0:
                s_range = SpatialDim.num_dims_pe()
            else:
                s_range = SpatialDim.num_dims_other()
            
            for s in range(s_range):
                for j, divs in enumerate(workload.divisors):
                    loop_sum = gp.quicksum(
                        vars.xb[w, m, s, j, i] for i in range(len(divs))
                    )
                    model.addConstr(
                        loop_sum == 1,
                        name=f"C_one_bound_per_loop_({w},{m},{s},{j})"
                    )
    
    # =========================================
    # Non-PE Layers: No Spatial Parallelism
    # For m > 0, spatial (s=0) must select factor 1
    # =========================================
    for w, workload in enumerate(workloads):
        for m in range(1, num_mems):  # Skip PE layer (m=0)
            for j, divs in enumerate(workload.divisors):
                # Find index of factor 1
                factor_1_idx = 0
                for idx, d in enumerate(divs):
                    if d == 1:
                        factor_1_idx = idx
                        break
                
                # Spatial must be factor 1 (no parallelism above PE)
                model.addConstr(
                    vars.xb[w, m, 0, j, factor_1_idx] == 1,
                    name=f"C_no_spatial_above_PE_({w},{m},{j})"
                )
    
    # =========================================
    # PE Layer: Dimension-Direction Mutual Exclusion
    # Each dimension can only be mapped to one spatial direction
    # (at most one of H, W, Internal can have factor > 1)
    # =========================================
    for w, workload in enumerate(workloads):
        m = 0  # PE layer
        for j, divs in enumerate(workload.divisors):
            # Find index of factor 1
            factor_1_idx = 0
            for idx, d in enumerate(divs):
                if d == 1:
                    factor_1_idx = idx
                    break
            
            # h_active = 1 if H uses non-1 factor, 0 otherwise
            # h_active = 1 - xb[w,0,H,j,factor_1_idx]
            # Similarly for W and Internal
            h_not_1 = 1 - vars.xb[w, m, SpatialDim.H, j, factor_1_idx]
            w_not_1 = 1 - vars.xb[w, m, SpatialDim.W, j, factor_1_idx]
            int_not_1 = 1 - vars.xb[w, m, SpatialDim.INTERNAL, j, factor_1_idx]
            
            # At most one direction can have non-1 factor
            model.addConstr(
                h_not_1 + w_not_1 + int_not_1 <= 1,
                name=f"C_dim_direction_mutex_({w},{j})"
            )
    
    # =========================================
    # Permutation Constraints
    # Only apply to temporal loops
    # For PE layer: s=TEMPORAL, for others: s=1
    # =========================================
    for w, workload in enumerate(workloads):
        for m in range(num_mems):
            # Determine temporal index
            temporal_s = SpatialDim.TEMPORAL if m == 0 else 1
            
            # One perm level per loop (if bound != 1)
            for j, divs in enumerate(workload.divisors):
                # Find factor 1 index
                factor_1_idx = 0
                for idx, d in enumerate(divs):
                    if d == 1:
                        factor_1_idx = idx
                        break
                
                loop_sum_b = 1 - vars.xb[w, m, temporal_s, j, factor_1_idx]  # 1 if bound != 1
                loop_sum_p = gp.quicksum(
                    vars.xp[w, m, p, j] for p in range(len(workload.bounds))
                )
                model.addConstr(
                    loop_sum_p == loop_sum_b,
                    name=f"C_one_perm_level_per_loop_if_bound_neq_1_({w},{m},{j})"
                )
            
            # Max one loop per perm level
            for p in range(len(workload.divisors)):
                perm_level_sum = gp.quicksum(
                    vars.xp[w, m, p, j] for j in range(len(workload.bounds))
                )
                model.addConstr(
                    perm_level_sum <= 1,
                    name=f"C_max_one_loop_per_perm_level_({w},{m},{p})"
                )
    
    # =========================================
    # Bypass Constraints
    # =========================================
    for w, workload in enumerate(workloads):
        for m in range(num_mems):
            for t in range(num_datatypes):
                if arch.mem_bypass_defined[m] or not optimize_bypass:
                    model.addConstr(
                        vars.xd[w, m, t] == int(arch.mem_stores_datatype[m][t]),
                        name=f"C_bypass_fixed_({w},{m},{t})"
                    )
    
    # =========================================
    # Layout Choice Constraints
    # =========================================
    dram_level = arch.mem_idx.get("LocalDRAM")
    if dram_level is not None:
        for w in range(len(workloads)):
            for t in range(num_datatypes):
                layout_sum = (
                    vars.layout_choice[w, t, "sequential"] +
                    vars.layout_choice[w, t, "row_aligned"]
                )
                model.addConstr(
                    layout_sum == 1,
                    name=f"C_layout_mode_select_({w},{t})"
                )
                
                # Layout requires storage
                if optimize_bypass and not arch.mem_bypass_defined[dram_level]:
                    model.addConstr(
                        vars.layout_choice[w, t, "row_aligned"] <= vars.xd[w, dram_level, t],
                        name=f"C_layout_requires_storage_({w},{t})"
                    )
                elif not arch.mem_stores_datatype[dram_level][t]:
                    model.addConstr(
                        vars.layout_choice[w, t, "row_aligned"] == 0,
                        name=f"C_layout_not_stored_({w},{t})"
                    )
    
    # =========================================
    # RowBuffer Input Block Constraints
    # =========================================
    shared_rowbuf_idx = arch.mem_idx.get("RowBuffer")
    dram_level = arch.mem_idx.get("LocalDRAM")
    
    if shared_rowbuf_idx is not None:
        for w, workload in enumerate(workloads):
            # -----------------------------------------------------------------
            # Constraint: R and S factors at DRAM and RowBuffer must be 1
            # This simplifies crossing analysis by avoiding kernel splitting
            # -----------------------------------------------------------------
            R_IDX = 0
            S_IDX = 1
            
            # For RowBuffer (m=shared_rowbuf_idx) and DRAM (m=dram_level)
            # We want the tiling factor (xb) to be 1.
            # Note: xb[w, m, s, j, i] is 1 if divisor i is selected.
            # We need to force selection of divisor=1.
            
            target_levels = [shared_rowbuf_idx]
            if dram_level is not None:
                target_levels.append(dram_level)
                
            for m in target_levels:
                # Determine spatial range
                if m == 0: # PE level (unlikely for RowBuffer/DRAM but safe to check)
                    s_range = SpatialDim.num_dims_pe()
                else:
                    s_range = SpatialDim.num_dims_other()
                    
                for s in range(s_range):
                    for dim_idx in [R_IDX, S_IDX]:
                        # Find index of factor 1
                        factor_1_idx = -1
                        for idx, d in enumerate(workload.divisors[dim_idx]):
                            if d == 1:
                                factor_1_idx = idx
                                break
                        
                        if factor_1_idx != -1:
                            model.addConstr(
                                vars.xb[w, m, s, dim_idx, factor_1_idx] == 1,
                                name=f"C_RS_factor_1_({w},{m},{s},{dim_idx})"
                            )

            h_divisors = getattr(workload, 'hw_divisors', {}).get('H', [1])
            w_divisors = getattr(workload, 'hw_divisors', {}).get('W', [1])
            
            if h_divisors:
                model.addConstr(
                    gp.quicksum(vars.rowbuf_input_block_h[w, i] for i in range(len(h_divisors))) == 1,
                    name=f"C_rowbuf_input_block_h_onehot_({w})"
                )
            
            if w_divisors:
                model.addConstr(
                    gp.quicksum(vars.rowbuf_input_block_w[w, j] for j in range(len(w_divisors))) == 1,
                    name=f"C_rowbuf_input_block_w_onehot_({w})"
                )
            
            # NOTE: block_h × block_w is NOT the Input tile size in RowBuffer!
            # - block_h, block_w: Data layout block size for Input tensor
            # - Input tile size in RowBuffer: determined by P, Q, R, S tiling factors
            # The constraint for Input tile <= RowBuffer capacity is in add_buffer_constraints()
            # via: buf_util_log[w, m, t] <= np.log(entries_limit)


def add_buffer_constraints(
    model: gp.Model,
    vars: VariableSet,
    arch,
    workloads: list,
    buf_util: dict,
    buf_util_log: dict,
    optimize_bypass: bool = False,
) -> dict:
    """
    Add buffer capacity constraints.
    
    Args:
        model: Gurobi model
        vars: Variable set
        arch: Architecture definition
        workloads: List of workloads
        buf_util: Buffer utilization expressions
        buf_util_log: Log of buffer utilization expressions
        optimize_bypass: Whether bypass is being optimized
        
    Returns:
        Dictionary of buffer utilization variables
    """
    num_mems = arch.num_mems
    num_datatypes = 3
    
    # Scaling factors for numerical stability
    MAX_BOUND = 1e4
    entries_list = [arch.mem_entries[m] for m in range(num_mems) if arch.mem_entries[m] > 0]
    max_entries = max(entries_list) if entries_list else 1.0
    util_scale_factors = [
        MAX_BOUND / (1.02 * arch.mem_entries[m]) 
        for m in range(num_mems - 1) 
        if arch.mem_entries[m] > 0
    ]
    
    buf_util_var = {}
    shared_rowbuf_idx = arch.mem_idx.get("RowBuffer")
    dram_level = arch.mem_idx.get("LocalDRAM")
    
    for w in range(len(workloads)):
        for m in range(num_mems - 1):
            entries_limit = arch.mem_entries[m]
            if entries_limit in (None, -1):
                entries_limit = 0
            
            if arch.mem_stores_multiple_datatypes[m] or (optimize_bypass and not arch.mem_bypass_defined[m]):
                for t in range(num_datatypes):
                    # RowBuffer也设上界：超过容量会导致更多row activation，性能更差
                    buf_ub = entries_limit if entries_limit > 0 else gp.GRB.INFINITY
                    
                    buf_util_var[w, m, t] = model.addVar(
                        ub=buf_ub, name=f"V_buffer_util_({w},{m},{t})"
                    )
                    model.addConstr(buf_util_var[w, m, t] == buf_util[w, m, t])
                    
                    # ===== 修复：使用 buf_util_log 约束（正确关联到 xb 变量）=====
                    # 每个 datatype 的 tile size 不能超过 buffer 容量
                    if arch.mem_bypass_defined[m] or not optimize_bypass:
                        if arch.mem_stores_datatype[m][t] and entries_limit > 0:
                            model.addConstr(
                                buf_util_log[w, m, t] <= np.log(entries_limit),
                                name=f"C_buffer_entries_log_({w},{m},{t})"
                            )
                            
                            # =========================================================
                            # Consistency Constraint: xu (Cost Model) >= xb (Execution)
                            # =========================================================
                            # buf_util_var represents the tile size selected by xu (for cost calculation)
                            # buf_util_log represents the tile size defined by xb (for execution/trace)
                            # We must ensure xu >= xb to prevent ILP from underestimating costs.
                            
                            # 1. Create variable for required size: min_req = exp(buf_util_log)
                            # Note: buf_util_log is a LinExpr, so we need an auxiliary variable for it
                            log_size_var = model.addVar(lb=-gp.GRB.INFINITY, ub=np.log(entries_limit), name=f"log_size_var_({w},{m},{t})")
                            model.addConstr(log_size_var == buf_util_log[w, m, t], name=f"C_log_size_eq_({w},{m},{t})")
                            
                            min_req_size = model.addVar(lb=0, ub=entries_limit, name=f"min_req_size_({w},{m},{t})")
                            model.addGenConstrExp(log_size_var, min_req_size, options="FuncPieces=-1 FuncPieceError=0.01", name=f"C_exp_req_size_({w},{m},{t})")
                            
                            # 2. Enforce buf_util_var >= min_req_size
                            # Note: We allow a small tolerance (0.99) to handle floating point issues
                            model.addConstr(buf_util_var[w, m, t] >= 0.99 * min_req_size, name=f"C_xu_ge_xb_({w},{m},{t})")

                    else:
                        # 当 bypass 可优化时，只有存储该 datatype 时才约束
                        if entries_limit > 0:
                            # xd[w,m,t] = 1 表示存储，此时需要满足容量约束
                            # 使用 indicator constraint: 如果 xd=1，则 buf_util_log <= log(limit)
                            model.addGenConstrIndicator(
                                vars.xd[w, m, t], True,
                                buf_util_log[w, m, t] <= np.log(entries_limit),
                                name=f"C_buffer_entries_log_opt_({w},{m},{t})"
                            )
                            
                            # Consistency Constraint with Indicator
                            log_size_var = model.addVar(lb=-gp.GRB.INFINITY, ub=np.log(entries_limit), name=f"log_size_var_({w},{m},{t})")
                            # Indicator constraint for equality? Or just equality?
                            # If xd=0, buf_util_log is unconstrained (or irrelevant).
                            # But we can just define log_size_var == buf_util_log always.
                            model.addConstr(log_size_var == buf_util_log[w, m, t], name=f"C_log_size_eq_opt_({w},{m},{t})")
                            
                            min_req_size = model.addVar(lb=0, ub=entries_limit, name=f"min_req_size_({w},{m},{t})")
                            model.addGenConstrExp(log_size_var, min_req_size, options="FuncPieces=-1 FuncPieceError=0.01", name=f"C_exp_req_size_({w},{m},{t})")
                            
                            model.addGenConstrIndicator(
                                vars.xd[w, m, t], True,
                                buf_util_var[w, m, t] >= 0.99 * min_req_size,
                                name=f"C_xu_ge_xb_opt_({w},{m},{t})"
                            )
            
            elif len(arch.mem_stored_datatypes[m]) == 1:
                t = arch.mem_stored_datatypes[m][0]
                if entries_limit > 0:
                    model.addConstr(
                        buf_util_log[w, m, t] <= np.log(entries_limit),
                        name=f"C_buffer_entries_({w},{m})_log"
                    )
    
    return buf_util_var


def add_reuse_tracking_constraints(
    model: gp.Model,
    vars: VariableSet,
    arch,
    workloads: list,
    fix_permutations: bool = False,
) -> None:
    """
    Add constraints for tracking data reuse (xr and xj variables).
    
    These variables are used for accurate row activation modeling.
    
    Args:
        model: Gurobi model
        vars: Variable set
        arch: Architecture definition
        workloads: List of workloads
        fix_permutations: Whether permutations are fixed
    """
    num_mems = arch.num_mems
    num_datatypes = 3
    
    # xr constraints: track relevant inner loops
    for w, workload in enumerate(workloads):
        for t in range(num_datatypes):
            for m in range(num_mems - 1):
                for m_ in range(m + 1, num_mems):
                    for p, _ in enumerate(workload.bounds):
                        perm_level_sum = gp.LinExpr(0)
                        for j, divs in enumerate(workload.divisors):
                            perm_level_sum += vars.xp[w, m_, p, j] * workload.O[j][t]
                        
                        if m_ == m + 1 and p == 0:
                            model.addConstr(
                                vars.xr[w, t, m, m_, p] == perm_level_sum,
                                name=f"C_xr_pls_({w},{t},{m},{m_},{p})"
                            )
                        elif m_ > m + 1 and p == 0:
                            model.addConstr(
                                vars.xr[w, t, m, m_, p] >= vars.xr[w, t, m, m_ - 1, len(workload.bounds) - 1],
                                name=f"C_xr_gt_({w},{t},{m},{m_},{p})"
                            )
                            model.addConstr(
                                vars.xr[w, t, m, m_, p] >= perm_level_sum,
                                name=f"C_xr_pls2_({w},{t},{m},{m_},{p})"
                            )
                        elif p > 0:
                            model.addConstr(
                                vars.xr[w, t, m, m_, p] >= vars.xr[w, t, m, m_, p - 1],
                                name=f"C_xr_inherit_({w},{t},{m},{m_},{p})"
                            )
                            model.addConstr(
                                vars.xr[w, t, m, m_, p] >= perm_level_sum,
                                name=f"C_xr_pls3_({w},{t},{m},{m_},{p})"
                            )
    
    # xj constraints: track dimension inner loops
    # xj[w,t,m,m_,j] indicates whether dimension j has an inner loop at position p
    # where xr[w,t,m,m_,p]=1 (i.e., there's a relevant inner loop for tensor t)
    # 
    # NOTE: xj does NOT depend on O[j][t] (relevancy)!
    # Even if dimension j is irrelevant to tensor t, if j has an inner loop,
    # it still causes repeated access to tensor t (reuse penalty).
    # Example: for Q (irrelevant to Weight): for S: access Weight
    #          Weight is accessed Q times even though Q is irrelevant.
    for w, workload in enumerate(workloads):
        for t in range(num_datatypes):
            for m in range(num_mems - 1):
                for m_ in range(m + 1, num_mems):
                    for j, _ in enumerate(workload.bounds):
                        # xj = sum_p (xp[w,m_,p,j] AND xr[w,t,m,m_,p])
                        # This counts if dimension j appears at any position p
                        # that has a relevant inner loop for tensor t
                        prod = gp.LinExpr(0)
                        for p, _ in enumerate(workload.bounds):
                            # xp_yp = xp[w,m_,p,j] AND xr[w,t,m,m_,p]
                            xp_yp = model.addVar(
                                vtype=gp.GRB.BINARY,
                                name=f"AUX_XP_YP({w},{t},{m},{m_},{j},{p})"
                            )
                            model.addGenConstrAnd(
                                xp_yp,
                                [vars.xp[w, m_, p, j], vars.xr[w, t, m, m_, p]],
                                name=f"AND_XP_YP_({w},{t},{m},{m_},{j},{p})"
                            )
                            prod += xp_yp
                        
                        model.addConstr(
                            vars.xj[w, t, m, m_, j] == prod,
                            name=f"C_xj_({w},{t},{m},{m_},{j})"
                        )


def add_pe_parallelism_constraints(
    model: gp.Model,
    vars: VariableSet,
    arch,
    workloads: list,
) -> dict:
    """
    Add PE layer parallelism constraints.
    
    Constraints:
    - H direction parallelism <= PE array H size
    - W direction parallelism <= PE array W size
    - Internal parallelism <= compute unit internal size
    
    Args:
        model: Gurobi model
        vars: Variable set
        arch: Architecture definition (must have pe_array attribute)
        workloads: List of workloads
        
    Returns:
        Dictionary of log parallelism expressions for each direction
    """
    pe_h = arch.pe_array.dim.h
    pe_w = arch.pe_array.dim.w
    internal_size = arch.pe_array.compute_unit.num_macs
    
    log_parallel = {}
    
    for w, workload in enumerate(workloads):
        # H direction parallelism
        log_h = gp.LinExpr(0)
        for j, divs in enumerate(workload.divisors):
            for i, div in enumerate(divs):
                log_h += vars.xb[w, 0, SpatialDim.H, j, i] * np.log(div)
        
        model.addConstr(
            log_h <= np.log(pe_h),
            name=f"C_pe_parallel_h_({w})"
        )
        log_parallel[w, 'H'] = log_h
        
        # W direction parallelism
        log_w = gp.LinExpr(0)
        for j, divs in enumerate(workload.divisors):
            for i, div in enumerate(divs):
                log_w += vars.xb[w, 0, SpatialDim.W, j, i] * np.log(div)
        
        model.addConstr(
            log_w <= np.log(pe_w),
            name=f"C_pe_parallel_w_({w})"
        )
        log_parallel[w, 'W'] = log_w
        
        # Internal parallelism
        log_int = gp.LinExpr(0)
        for j, divs in enumerate(workload.divisors):
            for i, div in enumerate(divs):
                log_int += vars.xb[w, 0, SpatialDim.INTERNAL, j, i] * np.log(div)
        
        if internal_size > 1:
            model.addConstr(
                log_int <= np.log(internal_size),
                name=f"C_pe_parallel_int_({w})"
            )
        else:
            # Scalar PE: Internal must all be factor 1
            model.addConstr(
                log_int == 0,
                name=f"C_pe_parallel_int_scalar_({w})"
            )
        log_parallel[w, 'Internal'] = log_int
    
    return log_parallel


def add_compute_unit_constraints(
    model: gp.Model,
    vars: VariableSet,
    arch,
    workloads: list,
) -> None:
    """
    Add constraints based on compute unit type.
    
    Supported types:
    - scalar: No internal parallelism (Internal must be 1)
    - simd: Output dims can be in Internal, reduction dims cannot
    - tensor_core: Specific reduction dim in Internal, output dims in H/W
    - reduction_tree: Any reduction dims can be in Internal, output dims cannot
    - systolic: Fixed mapping (K->H, C->W)
    
    Args:
        model: Gurobi model
        vars: Variable set
        arch: Architecture definition
        workloads: List of workloads
    """
    compute_unit = arch.pe_array.compute_unit
    unit_type = getattr(compute_unit, 'unit_type', 'scalar')
    
    # Dimension indices for conv: [R, S, P, Q, C, K, N]
    REDUCTION_DIMS = [0, 1, 4]  # R, S, C
    OUTPUT_DIMS = [2, 3, 5, 6]  # P, Q, K, N
    
    for w, workload in enumerate(workloads):
        if unit_type == 'scalar':
            # Scalar PE: Internal parallelism must be 1
            for j, divs in enumerate(workload.divisors):
                factor_1_idx = 0
                for idx, d in enumerate(divs):
                    if d == 1:
                        factor_1_idx = idx
                        break
                
                model.addConstr(
                    vars.xb[w, 0, SpatialDim.INTERNAL, j, factor_1_idx] == 1,
                    name=f"C_scalar_no_internal_({w},{j})"
                )
        
        elif unit_type == 'tensor_core':
            # Tensor Core: Reduction dims must be in Internal
            # Output dims must be in H or W (not Internal)
            internal_dim = getattr(compute_unit, 'internal_dim', 4)  # Default: C
            
            for j, divs in enumerate(workload.divisors):
                factor_1_idx = 0
                for idx, d in enumerate(divs):
                    if d == 1:
                        factor_1_idx = idx
                        break
                
                if j == internal_dim:
                    # This reduction dim must be in Internal (not factor 1)
                    # We don't force non-1, but H and W must be 1
                    model.addConstr(
                        vars.xb[w, 0, SpatialDim.H, j, factor_1_idx] == 1,
                        name=f"C_tc_reduction_not_h_({w},{j})"
                    )
                    model.addConstr(
                        vars.xb[w, 0, SpatialDim.W, j, factor_1_idx] == 1,
                        name=f"C_tc_reduction_not_w_({w},{j})"
                    )
                elif j in REDUCTION_DIMS:
                    # Other reduction dims: can be Internal or Temporal, but not H/W
                    model.addConstr(
                        vars.xb[w, 0, SpatialDim.H, j, factor_1_idx] == 1,
                        name=f"C_tc_other_red_not_h_({w},{j})"
                    )
                    model.addConstr(
                        vars.xb[w, 0, SpatialDim.W, j, factor_1_idx] == 1,
                        name=f"C_tc_other_red_not_w_({w},{j})"
                    )
                else:
                    # Output dims: can be H or W, but not Internal
                    model.addConstr(
                        vars.xb[w, 0, SpatialDim.INTERNAL, j, factor_1_idx] == 1,
                        name=f"C_tc_output_not_internal_({w},{j})"
                    )
        
        elif unit_type == 'simd':
            # SIMD: Multiple independent MACs, no internal reduction
            # Output dims (P,Q,K,N) CAN be mapped to Internal (independent outputs)
            # Reduction dims (R,S,C) CANNOT be in Internal (need accumulation)
            for j, divs in enumerate(workload.divisors):
                factor_1_idx = 0
                for idx, d in enumerate(divs):
                    if d == 1:
                        factor_1_idx = idx
                        break
                
                if j in REDUCTION_DIMS:
                    # Reduction dims must NOT be in Internal (SIMD doesn't reduce)
                    model.addConstr(
                        vars.xb[w, 0, SpatialDim.INTERNAL, j, factor_1_idx] == 1,
                        name=f"C_simd_reduction_not_internal_({w},{j})"
                    )
                # Output dims: no constraint, can be in Internal
        
        elif unit_type == 'reduction_tree':
            # Reduction Tree: Adder tree for internal reduction
            # Reduction dims (R,S,C) CAN be mapped to Internal (any combination)
            # Output dims (P,Q,K,N) CANNOT be in Internal (only 1 output per tree)
            for j, divs in enumerate(workload.divisors):
                factor_1_idx = 0
                for idx, d in enumerate(divs):
                    if d == 1:
                        factor_1_idx = idx
                        break
                
                if j in OUTPUT_DIMS:
                    # Output dims must NOT be in Internal (tree produces single output)
                    model.addConstr(
                        vars.xb[w, 0, SpatialDim.INTERNAL, j, factor_1_idx] == 1,
                        name=f"C_reduction_tree_output_not_internal_({w},{j})"
                    )
                # Reduction dims: no constraint, can be in Internal
        
        elif unit_type == 'systolic':
            # Systolic Array: K and C must be in different spatial directions
            # - K and C must be mutually exclusive (one in H, one in W)
            # - Other dims (R,S,P,Q,N) can use H or W to fill the PE array
            # - No Internal parallelism (systolic has no internal reduction)
            # 
            # This allows mappings like:
            #   K4*P2 -> H, C8*Q2 -> W  (K in H, C in W)
            #   C4*P2 -> H, K8*Q2 -> W  (C in H, K in W)
            K_IDX = 5
            C_IDX = 4
            
            # Find factor_1_idx for K and C
            k_factor_1_idx = 0
            c_factor_1_idx = 0
            for idx, d in enumerate(workload.divisors[K_IDX]):
                if d == 1:
                    k_factor_1_idx = idx
                    break
            for idx, d in enumerate(workload.divisors[C_IDX]):
                if d == 1:
                    c_factor_1_idx = idx
                    break
            
            # K and C cannot both be in H: at least one must have H=1
            # k_in_h = 1 - xb[K,H,1], c_in_h = 1 - xb[C,H,1]
            # k_in_h + c_in_h <= 1  =>  (1 - xb[K,H,1]) + (1 - xb[C,H,1]) <= 1
            # => xb[K,H,1] + xb[C,H,1] >= 1
            model.addConstr(
                vars.xb[w, 0, SpatialDim.H, K_IDX, k_factor_1_idx] +
                vars.xb[w, 0, SpatialDim.H, C_IDX, c_factor_1_idx] >= 1,
                name=f"C_systolic_k_c_not_both_h_({w})"
            )
            
            # K and C cannot both be in W: at least one must have W=1
            model.addConstr(
                vars.xb[w, 0, SpatialDim.W, K_IDX, k_factor_1_idx] +
                vars.xb[w, 0, SpatialDim.W, C_IDX, c_factor_1_idx] >= 1,
                name=f"C_systolic_k_c_not_both_w_({w})"
            )
            
            # No Internal parallelism for any dimension
            for j, divs in enumerate(workload.divisors):
                factor_1_idx = 0
                for idx, d in enumerate(divs):
                    if d == 1:
                        factor_1_idx = idx
                        break
                
                model.addConstr(
                    vars.xb[w, 0, SpatialDim.INTERNAL, j, factor_1_idx] == 1,
                    name=f"C_systolic_no_internal_({w},{j})"
                )
