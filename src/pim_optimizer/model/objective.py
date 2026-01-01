"""
Objective function setup for the ILP model.
"""

import numpy as np
import gurobipy as gp

from pim_optimizer.model.variables import VariableSet, SpatialDim


def _get_temporal_s(m: int) -> int:
    """Get the temporal s index for a given memory level."""
    return SpatialDim.TEMPORAL if m == 0 else 1


def _build_dram_latency_cycles(
    model: gp.Model,
    arch,
    workload,
    w: int,
    mem_reads_vars: dict,
    macs_scale_factor: float,
    enable_row_activation: bool,
    row_act_cycles_dict: dict | None,
) -> tuple[gp.Var | None, dict[int, gp.Var]]:
    """
    Build DRAM latency cycles for each datatype.
    
    DRAM latency (per datatype) = RowBuffer Data Transfer + Row Activation
    Total DRAM latency = max(input_latency, weight_latency, output_latency)
    
    RowBuffer 是 DRAM 的一部分，所以这两个延迟应该相加：
    - RowBuffer Data Transfer: mem_reads[t] / rowbuffer_bandwidth
    - Row Activation: row_acts × activation_latency (来自 row_activation.py)
    
    Args:
        model: Gurobi model
        arch: Architecture definition
        workload: Workload definition
        w: Workload index
        mem_reads_vars: Memory reads variables dict
        macs_scale_factor: Scaling factor for numerical stability
        enable_row_activation: Whether to include row activation model
        row_act_cycles_dict: Dict of (w, t_id) -> row_activation_cycles
        
    Returns:
        Tuple of (total_dram_cycles, per_datatype_dict)
        - total_dram_cycles: max(input, weight, output) DRAM latency
        - per_datatype_dict: {t_id: dram_latency_var} for each datatype
    """
    TENSOR_NAMES = {0: "input", 1: "weight", 2: "output"}
    num_datatypes = 3
    
    dram_level = arch.mem_idx.get("LocalDRAM")
    rowbuffer_level = arch.mem_idx.get("RowBuffer")
    
    if dram_level is None:
        return None, {}
    
    activation_latency = getattr(arch, "dram_activation_latency", 25.0)
    
    # Get RowBuffer bandwidth
    rowbuffer_bw = None
    if rowbuffer_level is not None:
        bw = arch.read_bandwidth[rowbuffer_level]
        if bw is not None and np.isfinite(bw) and bw > 0:
            rowbuffer_bw = bw
    
    # =========================================================================
    # Step 1: Compute per-datatype DRAM latency
    # =========================================================================
    per_datatype_latency = {}
    per_datatype_cycles = []
    
    for t_id in range(num_datatypes):
        t_name = TENSOR_NAMES[t_id]
        
        # RowBuffer data transfer for this datatype
        data_transfer = gp.LinExpr(0)
        if rowbuffer_bw is not None:
            reads_var = mem_reads_vars.get((w, rowbuffer_level, t_id))
            if reads_var is not None:
                data_transfer = reads_var * (1.0 / rowbuffer_bw)
        
        # Row activation cycles for this datatype
        row_act = gp.LinExpr(0)
        if enable_row_activation and row_act_cycles_dict is not None:
            row_act_var = row_act_cycles_dict.get((w, t_id))
            if row_act_var is not None:
                row_act = row_act_var
        
        # Total for this datatype
        dtype_total = data_transfer + row_act
        
        # Create variable for this datatype's DRAM latency
        mem_cycles_ub = 1.02 * workload.macs * macs_scale_factor * max(1, activation_latency)
        
        dtype_latency = model.addVar(
            ub=mem_cycles_ub,
            name=f"V_dram_latency_{t_name}_({w})"
        )
        model.addConstr(
            dtype_latency == dtype_total,
            name=f"C_dram_latency_{t_name}_({w})"
        )
        
        per_datatype_latency[t_id] = dtype_latency
        per_datatype_cycles.append(dtype_latency)
    
    # =========================================================================
    # Step 2: Total DRAM latency = max(input, weight, output)
    # =========================================================================
    mem_cycles_ub = 1.02 * workload.macs * macs_scale_factor * max(1, activation_latency)
    
    dram_cycles = model.addVar(
        ub=mem_cycles_ub,
        name=f"V_dram_latency_({w})"
    )
    model.addGenConstrMax(
        dram_cycles,
        per_datatype_cycles,
        name=f"C_dram_latency_max_({w})"
    )
    
    return dram_cycles, per_datatype_latency


def set_objective(
    model: gp.Model,
    total_latency: gp.LinExpr,
    total_energy: gp.LinExpr = None,
    objective_type: str = "latency",
    latency_weight: float = 1.0,
    energy_weight: float = 0.0,
) -> None:
    """
    Set the optimization objective.
    
    Args:
        model: Gurobi model
        total_latency: Total latency expression
        total_energy: Total energy expression (optional)
        objective_type: Type of objective ("latency", "energy", "blended")
        latency_weight: Weight for latency in blended objective
        energy_weight: Weight for energy in blended objective
    """
    model.ModelSense = gp.GRB.MINIMIZE
    
    if objective_type == "latency":
        model.setObjective(total_latency, gp.GRB.MINIMIZE)
        print("Setting objective: latency only")
    
    elif objective_type == "energy" and total_energy is not None:
        model.setObjective(total_energy, gp.GRB.MINIMIZE)
        print("Setting objective: energy only")
    
    elif objective_type == "blended" and total_energy is not None:
        objective = latency_weight * total_latency + energy_weight * total_energy
        model.setObjective(objective, gp.GRB.MINIMIZE)
        print(f"Setting objective: blended (latency={latency_weight}, energy={energy_weight})")
    
    else:
        # Default to latency
        model.setObjective(total_latency, gp.GRB.MINIMIZE)
        print("Setting objective: latency (default)")


def build_compute_cycles(
    model: gp.Model,
    vars: VariableSet,
    arch,
    workloads: list,
    macs_scale_factors: list[float],
    macs_scale_factor_logs: np.ndarray,
) -> tuple[dict, gp.LinExpr]:
    """
    Build compute cycles expressions.
    
    Args:
        model: Gurobi model
        vars: Variable set
        arch: Architecture definition
        workloads: List of workloads
        macs_scale_factors: Scaling factors for numerical stability
        macs_scale_factor_logs: Log of scaling factors
        
    Returns:
        Tuple of (compute_cycles dict, total_compute LinExpr)
    """
    num_mems = arch.num_mems
    MAX_BOUND = 1e4
    max_macs = max(w.macs for w in workloads)
    max_macs_scale_factor = MAX_BOUND / (1.02 * max_macs)
    
    pwl_opts = "FuncPieces=-2 FuncPieceError=0.002"
    
    total_compute = gp.LinExpr(0)
    compute_cycles = {}
    
    for w, workload in enumerate(workloads):
        log_compute_cycles_expr = gp.LinExpr(0)
        
        # Sum temporal loop bounds across all memory levels
        for m in range(num_mems):
            temporal_s = _get_temporal_s(m)
            for j, divs in enumerate(workload.divisors):
                for i, div in enumerate(divs):
                    log_compute_cycles_expr += vars.xb[w, m, temporal_s, j, i] * np.log(div)
        
        ub = 1.02 * workload.macs * macs_scale_factors[w]
        
        log_compute_cycles = model.addVar(
            lb=macs_scale_factor_logs[w],
            ub=np.log(ub),
            name=f"LOG_COMPUTE_CYCLES_{w}"
        )
        model.addConstr(
            log_compute_cycles == log_compute_cycles_expr + macs_scale_factor_logs[w],
            name=f"C_log_compute_cycles_{w}"
        )
        
        compute_cycles[w] = model.addVar(ub=ub, name=f"COMPUTE_CYCLES_{w}")
        model.addGenConstrExp(
            log_compute_cycles,
            compute_cycles[w],
            options=pwl_opts,
            name=f"C_exp_compute_cycles_{w}"
        )
        
        total_compute += workload.weight * (compute_cycles[w] / macs_scale_factors[w]) * max_macs_scale_factor
    
    return compute_cycles, total_compute


def build_latency_objective(
    model: gp.Model,
    vars: VariableSet,
    arch,
    workloads: list,
    compute_cycles: dict,
    mem_reads_vars: dict,
    macs_scale_factors: list[float],
    enable_row_activation: bool = True,
    activation_cycles: gp.LinExpr = None,
) -> tuple[dict, gp.LinExpr]:
    """
    Build latency objective including memory and compute bottlenecks.
    
    Args:
        model: Gurobi model
        vars: Variable set
        arch: Architecture definition
        workloads: List of workloads
        compute_cycles: Compute cycles dict
        mem_reads_vars: Memory reads variables
        macs_scale_factors: Scaling factors
        enable_row_activation: Whether to include row activation model
        activation_cycles: Row activation cycles (from build_row_activation_model)
        
    Returns:
        Tuple of (latency dict, total_latency LinExpr)
    """
    num_mems = arch.num_mems
    num_datatypes = 3
    
    MAX_BOUND = 1e4
    max_macs = max(w.macs for w in workloads)
    max_macs_scale_factor = MAX_BOUND / (1.02 * max_macs)
    
    dram_level = arch.mem_idx.get("LocalDRAM")
    rowbuffer_level = arch.mem_idx.get("RowBuffer")
    
    # Get activation latency for scaling upper bounds
    activation_latency = getattr(arch, "dram_activation_latency", 25.0)
    
    total_latency = gp.LinExpr(0)
    latency = {}
    memory_cycles = {}
    
    for w, workload in enumerate(workloads):
        bottleneck_candidates = [compute_cycles[w]]
        
        # =====================================================================
        # DRAM latency (RowBuffer data transfer + Row Activation)
        # Per-datatype: input, weight, output
        # Total = max(input, weight, output)
        # =====================================================================
        dram_cycles, per_dtype_dram = _build_dram_latency_cycles(
            model=model,
            arch=arch,
            workload=workload,
            w=w,
            mem_reads_vars=mem_reads_vars,
            macs_scale_factor=macs_scale_factors[w],
            enable_row_activation=enable_row_activation,
            row_act_cycles_dict=activation_cycles,
        )
        if dram_cycles is not None:
            memory_cycles[w, dram_level] = dram_cycles
            # Store per-datatype latencies for debugging
            for t_id, dtype_lat in per_dtype_dram.items():
                memory_cycles[w, dram_level, t_id] = dtype_lat
            bottleneck_candidates.append(dram_cycles)
        
        # =====================================================================
        # Other memory levels (e.g., GlobalBuffer, PELocalBuffer)
        # =====================================================================
        for m in range(1, num_mems):
            # Skip DRAM and RowBuffer - already handled above
            if m == dram_level or m == rowbuffer_level:
                continue
            
            mem_cycles_ub = 1.02 * workload.macs * macs_scale_factors[w]
            
            memory_cycles[w, m] = model.addVar(
                ub=mem_cycles_ub,
                name=f"V_mem_cycles_({w},{m})"
            )
            
            base_cycles = gp.LinExpr(0)
            for t_id in range(num_datatypes):
                reads_var = mem_reads_vars.get((w, m, t_id))
                if reads_var is None:
                    continue
                bw = arch.read_bandwidth[m]
                if bw is not None and np.isfinite(bw) and bw > 0:
                    base_cycles += reads_var * (1.0 / bw)
            
            model.addConstr(
                memory_cycles[w, m] == base_cycles,
                name=f"C_mem_cycles_({w},{m})"
            )
            
            bottleneck_candidates.append(memory_cycles[w, m])
        
        # Latency = max of all bottlenecks
        # Use larger upper bound to account for potential row activation dominance
        latency_ub = 1.02 * workload.macs * macs_scale_factors[w]
        if enable_row_activation:
            latency_ub *= max(1, activation_latency)
        latency[w] = model.addVar(
            ub=latency_ub,
            name=f"LATENCY_{w}"
        )
        model.addGenConstrMax(
            latency[w],
            bottleneck_candidates,
            name=f"max_latency_{w}"
        )
        
        total_latency += workload.weight * (latency[w] / macs_scale_factors[w]) * max_macs_scale_factor
    
    return latency, total_latency
