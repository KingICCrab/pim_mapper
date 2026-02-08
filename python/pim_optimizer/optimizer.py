"""
Main optimizer entry point.
"""

from typing import Optional

import numpy as np
import gurobipy as gp

from pim_optimizer.arch import PIMArchitecture
from pim_optimizer.workload import ConvWorkload
from pim_optimizer.mapping import Mapping, OptimizationResult
from pim_optimizer.model import (
    create_decision_variables,
    add_basic_constraints,
    add_buffer_constraints,
)
from pim_optimizer.model.variables import create_auxiliary_variables
from pim_optimizer.model.constraints import add_reuse_tracking_constraints, add_pe_parallelism_constraints, add_compute_unit_constraints
from pim_optimizer.model.expressions import (
    build_buffer_utilization,
    build_memory_expressions,
    build_tile_bytes_vars,
    TileInfo,
)
from pim_optimizer.model.objective import (
    set_objective,
    build_compute_cycles,
    build_latency_objective,
)
from pim_optimizer.model.row_activation import (
    build_row_activation_model,
)
from pim_optimizer.utils import create_gurobi_model, Timer


class PIMOptimizer:
    """
    Main PIM optimizer class.
    
    This class orchestrates the complete optimization process:
    1. Load architecture and workloads
    2. Build ILP model (variables, constraints, objective)
    3. Solve the model
    4. Extract and return mapping results
    
    Usage:
        optimizer = PIMOptimizer(arch_file="arch.yaml")
        result = optimizer.optimize(workloads)
    """
    
    def __init__(
        self,
        arch: PIMArchitecture = None,
        arch_file: str = None,
        verbose: bool = False,
        time_limit: float = 300.0,
        mip_gap: float = 0.01,
    ):
        """
        Initialize the PIM optimizer.
        
        Args:
            arch: PIMArchitecture object
            arch_file: Path to architecture YAML file
            verbose: Enable verbose output
            time_limit: Solver time limit in seconds
            mip_gap: MIP optimality gap tolerance
        """
        if arch is not None:
            self.arch = arch
        elif arch_file is not None:
            self.arch = PIMArchitecture.from_yaml(arch_file)
        else:
            self.arch = PIMArchitecture()  # Default architecture
        
        self.verbose = verbose
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        
        self.timer = Timer()
        self.model = None
        self.vars = None
    
    def optimize(
        self,
        workloads: list[ConvWorkload],
        objective: str = "latency",
        fix_permutations: bool = False,
        fix_bypass: bool = False,
        optimize_bypass: bool = False,
        enable_row_activation: bool = True,
        gurobi_params: dict = None,
    ) -> OptimizationResult:
        """
        Run optimization for the given workloads.
        
        Args:
            workloads: List of ConvWorkload objects
            objective: Objective type ("latency", "energy", "blended")
            fix_permutations: Fix permutations across workloads
            fix_bypass: Fix bypass across workloads
            optimize_bypass: Optimize bypass decisions
            enable_row_activation: Enable row activation modeling
            
        Returns:
            OptimizationResult containing mappings and summary
        """
        self.timer.start("total")
        
        # =========================================
        # Step 1: Create Gurobi model
        # =========================================
        self.timer.start("model_creation")
        self.model = create_gurobi_model(
            name="PIMOptimizer",
            verbose=self.verbose,
            time_limit=self.time_limit,
            mip_gap=self.mip_gap,
        )
        
        # =========================================
        # Step 2: Create decision variables
        # =========================================
        self.vars = create_decision_variables(
            self.model,
            self.arch,
            workloads,
            fix_permutations=fix_permutations,
            fix_bypass=fix_bypass,
            optimize_bypass=optimize_bypass,
        )
        
        create_auxiliary_variables(
            self.model,
            self.vars,
            self.arch,
            workloads,
            fix_permutations=fix_permutations,
        )
        self.timer.stop("model_creation")
        
        # =========================================
        # Step 3: Add constraints
        # =========================================
        self.timer.start("constraints")
        add_basic_constraints(
            self.model,
            self.vars,
            self.arch,
            workloads,
            optimize_bypass=optimize_bypass,
        )
        
        add_reuse_tracking_constraints(
            self.model,
            self.vars,
            self.arch,
            workloads,
            fix_permutations=fix_permutations,
        )
        
        # PE array parallelism constraints
        add_pe_parallelism_constraints(
            self.model,
            self.vars,
            self.arch,
            workloads,
        )
        
        # Compute unit type constraints (scalar, simd, tensor_core, etc.)
        add_compute_unit_constraints(
            self.model,
            self.vars,
            self.arch,
            workloads,
        )
        self.timer.stop("constraints")
        
        # =========================================
        # Step 4: Build expressions
        # =========================================
        self.timer.start("expressions")
        buf_util, buf_util_log, tile_info = build_buffer_utilization(
            self.model,
            self.vars,
            self.arch,
            workloads,
            optimize_bypass=optimize_bypass,
        )
        
        buf_util_var = add_buffer_constraints(
            self.model,
            self.vars,
            self.arch,
            workloads,
            buf_util,
            buf_util_log,
            optimize_bypass=optimize_bypass,
        )
        
        mem_writes_inst, mem_writes, mem_reads_inst, mem_reads = build_memory_expressions(
            self.model,
            self.vars,
            self.arch,
            workloads,
            buf_util_log,
            optimize_bypass=optimize_bypass,
        )
        
        # Build tile_bytes_var for row activation model
        if enable_row_activation:
            build_tile_bytes_vars(
                self.model,
                self.vars,
                self.arch,
                workloads,
                tile_info,
                buf_util_var,
            )
        self.timer.stop("expressions")
        
        # =========================================
        # Step 5: Build objective function
        # =========================================
        self.timer.start("objective")
        
        # Scaling factors for numerical stability
        MAX_BOUND = 1e4
        max_macs = max(w.macs for w in workloads)
        macs_scale_factors = [MAX_BOUND / (1.02 * w.macs) for w in workloads]
        macs_scale_factor_logs = np.array([np.log(sf) for sf in macs_scale_factors])
        
        compute_cycles, total_compute = build_compute_cycles(
            self.model,
            self.vars,
            self.arch,
            workloads,
            macs_scale_factors,
            macs_scale_factor_logs,
        )
        
        # Memory reads variables for latency calculation
        pwl_opts = "FuncPieces=-2 FuncPieceError=0.002"
        mem_reads_vars = {}
        
        for w, workload in enumerate(workloads):
            for m in range(1, self.arch.num_mems):
                for t in range(3):
                    if mem_reads.get((w, m, t)) is not None:
                        ub = 1.02 * workload.macs * macs_scale_factors[w]
                        log_var = self.model.addVar(
                            lb=macs_scale_factor_logs[w],
                            ub=np.log(ub),
                            name=f"LOG_MEM_READS_{w}_{m}_{t}"
                        )
                        self.model.addConstr(
                            log_var == mem_reads[w, m, t] + macs_scale_factor_logs[w],
                            name=f"C_log_mem_reads_{w}_{m}_{t}"
                        )
                        
                        mem_reads_vars[w, m, t] = self.model.addVar(
                            ub=ub,
                            name=f"MEM_READS_{w}_{m}_{t}"
                        )
                        self.model.addGenConstrExp(
                            log_var,
                            mem_reads_vars[w, m, t],
                            options=pwl_opts,
                            name=f"C_exp_mem_reads_{w}_{m}_{t}"
                        )
        
        # Build row activation model
        activation_cycles = None
        if enable_row_activation:
            activation_cycles = build_row_activation_model(
                self.model,
                self.vars,
                self.arch,
                workloads,
                tile_info,
                buf_util_var,
                mem_reads_inst,
                macs_scale_factors,
                macs_scale_factor_logs,
            )
        
        latency, total_latency = build_latency_objective(
            self.model,
            self.vars,
            self.arch,
            workloads,
            compute_cycles,
            mem_reads_vars,
            macs_scale_factors,
            enable_row_activation=enable_row_activation,
            activation_cycles=activation_cycles,
        )
        
        set_objective(
            self.model,
            total_latency,
            objective_type=objective,
        )
        self.timer.stop("objective")
        
        # =========================================
        # Step 6: Solve the model
        # =========================================
        if gurobi_params:
            for k, v in gurobi_params.items():
                self.model.setParam(k, v)

        self.timer.start("solve")
        self.model.optimize()
        self.timer.stop("solve")
        
        # =========================================
        # Step 8: Extract results
        # =========================================
        self.timer.start("extract")
        
        # Store for external access
        self.compute_cycles = compute_cycles
        self.latency_vars = latency
        self.activation_cycles = activation_cycles
        self.macs_scale_factors = macs_scale_factors
        
        result = self._extract_results(workloads, compute_cycles, latency, activation_cycles, macs_scale_factors)
        self.timer.stop("extract")
        
        self.timer.stop("total")
        
        if self.verbose:
            print(self.timer.report())
        
        return result
    
    def _extract_results(
        self,
        workloads: list[ConvWorkload],
        compute_cycles: dict,
        latency: dict,
        activation_cycles: dict = None,
        macs_scale_factors: list = None,
        var_attr: str = "X",
    ) -> OptimizationResult:
        """Extract optimization results from solved model."""
        
        if self.model.Status not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT):
            return OptimizationResult(
                solver_status=self._get_status_string(),
                solve_time=self.model.Runtime,
            )
        
        mappings = []
        
        for w, workload in enumerate(workloads):
            mapping = self._extract_mapping(w, workload, var_attr=var_attr)
            
            # Add metrics
            mapping.metrics["latency"] = getattr(latency[w], var_attr) if hasattr(latency[w], var_attr) else 0.0
            mapping.metrics["compute_cycles"] = (
                getattr(compute_cycles[w], var_attr) if hasattr(compute_cycles[w], var_attr) else 0.0
            )
            
            # Extract row activation metrics
            if activation_cycles is not None:
                total_row_acts = 0.0
                activation_latency = getattr(self.arch, "dram_activation_latency", 25.0)
                # Get scale factor (cycles = row_acts × scale × activation_latency)
                scale = macs_scale_factors[w] if macs_scale_factors else 1.0
                
                for t_id, t_name in [(0, "input"), (1, "weight"), (2, "output")]:
                    key = (w, t_id)
                    if key in activation_cycles:
                        val = getattr(activation_cycles[key], var_attr) if hasattr(activation_cycles[key], var_attr) else 0.0
                        # Convert cycles back to activations: row_acts = cycles / scale / activation_latency
                        row_acts = val / scale / activation_latency if (scale > 0 and activation_latency > 0) else val
                        mapping.metrics[f"row_activations_{t_name}"] = row_acts
                        total_row_acts += row_acts
                    else:
                        mapping.metrics[f"row_activations_{t_name}"] = 0.0
                mapping.metrics["row_activations"] = total_row_acts
            
            mappings.append(mapping)
        
        # Determine objective value based on var_attr
        obj_val = None
        if self.model.SolCount > 0:
            if var_attr == "Xn":
                obj_val = self.model.PoolObjVal
            else:
                obj_val = self.model.ObjVal

        result = OptimizationResult(
            mappings=mappings,
            solver_status=self._get_status_string(),
            solve_time=self.model.Runtime,
        )
        
        result.summary = {
            "total_latency": result.total_latency,
            "num_workloads": len(workloads),
            "objective_value": obj_val,
        }
        
        return result
    
    def _extract_mapping(self, w: int, workload: ConvWorkload, var_attr: str = "X") -> Mapping:
        """Extract mapping for a single workload."""
        mapping = Mapping()
        mapping.workload_name = workload.name
        mapping.workload_bounds = list(workload.bounds)
        
        num_mems = self.arch.num_mems
        
        # Import SpatialDim for PE layer
        from pim_optimizer.model.variables import SpatialDim
        
        # Extract loop bounds
        for m in range(num_mems):
            if m == 0:
                # PE layer: H, W, Internal, Temporal
                mapping.loop_bounds[m] = {
                    "H": {},
                    "W": {},
                    "Internal": {},
                    "temporal": {},
                }
                s_names = {
                    SpatialDim.H: "H",
                    SpatialDim.W: "W",
                    SpatialDim.INTERNAL: "Internal",
                    SpatialDim.TEMPORAL: "temporal",
                }
                s_range = SpatialDim.num_dims_pe()
            else:
                # Other layers: spatial (dummy), temporal
                mapping.loop_bounds[m] = {"spatial": {}, "temporal": {}}
                s_names = {0: "spatial", 1: "temporal"}
                s_range = SpatialDim.num_dims_other()
            
            for j, divs in enumerate(workload.divisors):
                for s in range(s_range):
                    s_name = s_names[s]
                    for i, div in enumerate(divs):
                        if getattr(self.vars.xb[w, m, s, j, i], var_attr) > 0.5:
                            mapping.loop_bounds[m][s_name][j] = div
        
        # Extract permutation
        for m in range(num_mems):
            mapping.permutation[m] = {}
            
            for p in range(len(workload.bounds)):
                for j in range(len(workload.bounds)):
                    if getattr(self.vars.xp[w, m, p, j], var_attr) > 0.5:
                        mapping.permutation[m][p] = j
        
        # Extract bypass
        for m in range(num_mems):
            mapping.bypass[m] = {}
            for t in range(3):
                mapping.bypass[m][t] = getattr(self.vars.xd[w, m, t], var_attr) > 0.5
        
        # Extract layout
        dram_level = self.arch.mem_idx.get("LocalDRAM")
        if dram_level is not None:
            for t in range(3):
                if (w, t, "row_aligned") in self.vars.layout_choice:
                    if getattr(self.vars.layout_choice[w, t, "row_aligned"], var_attr) > 0.5:
                        mapping.layout[t] = "row_aligned"
                    else:
                        mapping.layout[t] = "sequential"
        
        # Extract input block_h and block_w for data layout
        h_divisors = getattr(workload, 'hw_divisors', {}).get('H', [1])
        w_divisors = getattr(workload, 'hw_divisors', {}).get('W', [1])
        
        block_h = 1
        for i, h_div in enumerate(h_divisors):
            if (w, i) in self.vars.rowbuf_input_block_h:
                if getattr(self.vars.rowbuf_input_block_h[w, i], var_attr) > 0.5:
                    block_h = h_div
                    break
        
        block_w = 1
        for j, w_div in enumerate(w_divisors):
            if (w, j) in self.vars.rowbuf_input_block_w:
                if getattr(self.vars.rowbuf_input_block_w[w, j], var_attr) > 0.5:
                    block_w = w_div
                    break
        
        mapping.tile_info['block_h'] = block_h
        mapping.tile_info['block_w'] = block_w
        
        return mapping
    
    def _get_status_string(self) -> str:
        """Convert Gurobi status to string."""
        status_map = {
            gp.GRB.OPTIMAL: "optimal",
            gp.GRB.INFEASIBLE: "infeasible",
            gp.GRB.INF_OR_UNBD: "infeasible_or_unbounded",
            gp.GRB.UNBOUNDED: "unbounded",
            gp.GRB.TIME_LIMIT: "time_limit",
            gp.GRB.SUBOPTIMAL: "suboptimal",
        }
        return status_map.get(self.model.Status, f"unknown_{self.model.Status}")


def run_optimization(
    arch_file: str,
    workload_files: list[str],
    objective: str = "latency",
    verbose: bool = False,
    time_limit: float = 300.0,
    **kwargs,
) -> OptimizationResult:
    """
    Convenience function to run optimization.
    
    Args:
        arch_file: Path to architecture YAML file
        workload_files: List of paths to workload YAML files
        objective: Optimization objective
        verbose: Enable verbose output
        time_limit: Solver time limit
        **kwargs: Additional optimizer arguments
        
    Returns:
        OptimizationResult
    """
    # Load architecture
    arch = PIMArchitecture.from_yaml(arch_file)
    
    # Load workloads
    workloads = []
    for wf in workload_files:
        workloads.append(ConvWorkload.from_yaml(wf))
    
    # Create optimizer and run
    optimizer = PIMOptimizer(
        arch=arch,
        verbose=verbose,
        time_limit=time_limit,
    )
    
    return optimizer.optimize(
        workloads,
        objective=objective,
        **kwargs,
    )
