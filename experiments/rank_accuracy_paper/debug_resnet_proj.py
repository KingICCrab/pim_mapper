#!/usr/bin/env python3
import sys
import os
import numpy as np
import gurobipy as gp
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from pim_optimizer.optimizer import PIMOptimizer
from experiments.rank_accuracy_paper.workloads import get_paper_workloads
from experiments.rank_accuracy_paper.fast_trace_generator import FastTraceGenerator
from validation.dram.trace_generator import DRAMConfig

def main():
    print("Debugging VGG_Conv1_1 Weight Correlation...")
    
    workloads = get_paper_workloads()
    target_workload = next(w for w in workloads if w.name == "VGG_Conv1_1")
    
    print(f"Workload: {target_workload.name}")
    print(f"Dimensions: P={target_workload.P}, Q={target_workload.Q}, C={target_workload.C}, K={target_workload.K}, R={target_workload.R}, S={target_workload.S}")
    
    # Setup Optimizer
    optimizer = PIMOptimizer(verbose=False, time_limit=30.0, mip_gap=0.05)
    
    # Setup Trace Generator
    dram_config = DRAMConfig(
        row_buffer_bytes=1024, # 1KB
        num_banks=8
    )
    trace_gen = FastTraceGenerator(dram_config)
    
    # Optimize
    gurobi_params = {
        "PoolSearchMode": 2,
        "PoolSolutions": 20,
        "PoolGap": 0.5
    }
    
    result = optimizer.optimize(
        workloads=[target_workload],
        objective="latency",
        enable_row_activation=True,
        gurobi_params=gurobi_params
    )
    
    num_solutions = optimizer.model.SolCount
    print(f"Found {num_solutions} solutions.")
    
    print(f"{'Sol':<3} | {'ILP Weight':<10} | {'Trace Weight':<12} | {'Tile Config (P,Q,K,C)':<30} | {'Loop Order (Outer->Inner)':<40}")
    print("-" * 110)
    
    ilp_vals = []
    trace_vals = []
    
    for i in range(num_solutions):
        optimizer.model.params.SolutionNumber = i
        
        sol_result = optimizer._extract_results(
            [target_workload],
            optimizer.compute_cycles,
            optimizer.latency_vars,
            optimizer.activation_cycles,
            optimizer.macs_scale_factors,
            var_attr="Xn"
        )
        
        if not sol_result.mappings:
            continue
            
        mapping = sol_result.mappings[0]
        
        # ILP Cost
        ilp_w = mapping.metrics.get("row_activations_weight", 0)
        
        # Trace Cost
        trace_stats = trace_gen.generate_trace(mapping, target_workload)
        trace_w = trace_stats[1]
        
        ilp_vals.append(ilp_w)
        trace_vals.append(trace_w)
        
        # Extract Tile Config
        l2_tiles = mapping.loop_bounds.get(2, {})
        tile_str = []
        for dim_name in ['P', 'Q', 'K', 'C']:
            dim_idx = -1
            if dim_name == 'P': dim_idx = 2
            elif dim_name == 'Q': dim_idx = 3
            elif dim_name == 'K': dim_idx = 5
            elif dim_name == 'C': dim_idx = 4
            
            val = 1
            if 2 in mapping.loop_bounds:
                for type_ in ['spatial', 'temporal']:
                    if type_ in mapping.loop_bounds[2]:
                        val *= mapping.loop_bounds[2][type_].get(dim_idx, 1)
            tile_str.append(f"{dim_name}={val}")
            
        tile_config = ", ".join(tile_str)
        
        # Extract Loop Order (DRAM Level = 3)
        perm = mapping.permutation.get(3, {})
        sorted_perm = sorted(perm.items(), key=lambda x: x[0], reverse=True)
        dim_names = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
        loop_order = " -> ".join([dim_names[d] for _, d in sorted_perm])
        
        print(f"{i:<3} | {ilp_w:<10.4f} | {trace_w:<12} | {tile_config:<30} | {loop_order:<40}")

    # Calculate stats
    ilp_std = np.std(ilp_vals)
    trace_std = np.std(trace_vals)
    print("-" * 110)
    print(f"ILP Std: {ilp_std:.4f}, Trace Std: {trace_std:.4f}")
    
    if ilp_std < 1e-6 and trace_std < 1e-6:
        print("Both constant. Correlation is 0.0 (by definition in test script) but effectively consistent.")
    else:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(ilp_vals, trace_vals)
        print(f"Correlation: {corr:.4f}")

if __name__ == "__main__":
    main()
