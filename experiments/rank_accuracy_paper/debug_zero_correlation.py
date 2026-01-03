#!/usr/bin/env python3
"""
分析 Output 相关性为 0 的 Workload，找出根本原因。
"""
import sys
import os
import numpy as np
from scipy.stats import spearmanr
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.arch import PIMArchitecture
from experiments.rank_accuracy_paper.workloads import get_paper_workloads
from experiments.rank_accuracy_paper.fast_trace_generator import FastTraceGenerator
from validation.dram.trace_generator import DRAMConfig

def analyze_workload(workload, arch):
    """详细分析单个 Workload 的 Output 开销预测。"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {workload.name}")
    print(f"{'='*80}")
    print(f"Workload Shape: P={workload.P}, Q={workload.Q}, C={workload.C}, K={workload.K}, R={workload.R}, S={workload.S}")
    
    # Output Tensor Size
    output_size_elements = workload.P * workload.Q * workload.K
    output_size_bytes = output_size_elements * 2  # 2 bytes per element
    row_buffer_bytes = 1024
    print(f"Output Tensor: {output_size_elements} elements = {output_size_bytes} bytes")
    print(f"Row Buffer: {row_buffer_bytes} bytes")
    print(f"Ratio: {output_size_bytes / row_buffer_bytes:.2f}x Row Buffer")
    
    # Optimize
    optimizer = PIMOptimizer(verbose=False, time_limit=30.0, mip_gap=0.05)
    gurobi_params = {
        "PoolSearchMode": 2,
        "PoolSolutions": 20,
        "PoolGap": 0.5
    }
    
    result = optimizer.optimize(
        workloads=[workload],
        objective="latency",
        enable_row_activation=True,
        gurobi_params=gurobi_params
    )
    
    if not result.mappings:
        print("  ERROR: No solution found")
        return
    
    # Setup Trace Generator
    dram_config = DRAMConfig(row_buffer_bytes=1024, num_banks=8)
    trace_gen = FastTraceGenerator(dram_config)
    
    # Collect data
    num_solutions = optimizer.model.SolCount
    print(f"\nFound {num_solutions} solutions. Analyzing Output Row Activations...\n")
    
    ilp_costs = []
    trace_costs = []
    details = []
    
    for i in range(min(num_solutions, 20)):
        optimizer.model.params.SolutionNumber = i
        
        sol_result = optimizer._extract_results(
            [workload],
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
        ilp_cost = mapping.metrics.get("row_activations_output", 0)
        
        # Trace Cost
        trace_stats = trace_gen.generate_trace(mapping, workload)
        trace_cost = trace_stats[2]  # Output
        
        ilp_costs.append(ilp_cost)
        trace_costs.append(trace_cost)
        
        # Extract mapping details
        # Get DRAM level tile sizes
        dram_level = arch.mem_idx.get("LocalDRAM", 2)
        tile_p = mapping.loop_bounds.get(dram_level, {}).get("spatial", {}).get("P", 1)
        tile_q = mapping.loop_bounds.get(dram_level, {}).get("spatial", {}).get("Q", 1)
        tile_k = mapping.loop_bounds.get(dram_level, {}).get("spatial", {}).get("K", 1)
        
        tile_output_elements = tile_p * tile_q * tile_k
        tile_output_bytes = tile_output_elements * 2
        
        # Get reuse (temporal loops at DRAM level)
        reuse_p = mapping.loop_bounds.get(dram_level, {}).get("temporal", {}).get("P", 1)
        reuse_q = mapping.loop_bounds.get(dram_level, {}).get("temporal", {}).get("Q", 1)
        reuse_k = mapping.loop_bounds.get(dram_level, {}).get("temporal", {}).get("K", 1)
        total_reuse = reuse_p * reuse_q * reuse_k
        
        details.append({
            "Sol": i,
            "ILP": ilp_cost,
            "Trace": trace_cost,
            "Tile_P": tile_p,
            "Tile_Q": tile_q,
            "Tile_K": tile_k,
            "Tile_Bytes": tile_output_bytes,
            "Reuse_P": reuse_p,
            "Reuse_Q": reuse_q,
            "Reuse_K": reuse_k,
            "Total_Reuse": total_reuse,
        })
    
    # Statistics
    print(f"ILP Predictions:   Mean={np.mean(ilp_costs):.2f}, Std={np.std(ilp_costs):.4f}, Range=[{np.min(ilp_costs):.2f}, {np.max(ilp_costs):.2f}]")
    print(f"Trace Validation:  Mean={np.mean(trace_costs):.2f}, Std={np.std(trace_costs):.4f}, Range=[{np.min(trace_costs):.2f}, {np.max(trace_costs):.2f}]")
    
    # Correlation
    if np.std(ilp_costs) > 1e-6 and np.std(trace_costs) > 1e-6:
        corr, _ = spearmanr(ilp_costs, trace_costs)
        print(f"\nSpearman Correlation: {corr:.4f}")
    else:
        print(f"\nSpearman Correlation: N/A (one or both are constant)")
        if np.std(ilp_costs) < 1e-6:
            print(f"  → ILP is constant at {ilp_costs[0]:.2f}")
        if np.std(trace_costs) < 1e-6:
            print(f"  → Trace is constant at {trace_costs[0]:.2f}")
    
    # Show details
    df = pd.DataFrame(details)
    print(f"\n{'='*80}")
    print("Detailed Mapping Analysis (first 10 solutions):")
    print(f"{'='*80}")
    print(df.head(10).to_string(index=False))
    
    # Analyze why Trace is constant
    if np.std(trace_costs) < 1e-6:
        print(f"\n{'='*80}")
        print("ROOT CAUSE ANALYSIS: Why is Trace constant?")
        print(f"{'='*80}")
        
        unique_tiles = df[['Tile_P', 'Tile_Q', 'Tile_K', 'Tile_Bytes']].drop_duplicates()
        print(f"\nUnique Tile Configurations: {len(unique_tiles)}")
        print(unique_tiles.to_string(index=False))
        
        # Check if all tiles are small enough to fit in row buffer
        all_fit = (df['Tile_Bytes'] <= row_buffer_bytes).all()
        if all_fit:
            print(f"\n✓ All tiles fit in Row Buffer ({row_buffer_bytes} bytes)")
            print(f"  → Trace Generator correctly predicts constant cost (likely 1 row activation)")
        else:
            print(f"\n✗ Some tiles exceed Row Buffer ({row_buffer_bytes} bytes)")
            print(f"  → But Trace is still constant - this might be a Trace Generator issue")
        
        # Check if total tensor fits in row buffer
        if output_size_bytes <= row_buffer_bytes:
            print(f"\n✓ Entire Output Tensor fits in Row Buffer")
            print(f"  → All mappings should have minimal cost (1 row activation)")
        
    # Analyze why ILP varies
    if np.std(ilp_costs) > 1e-6 and np.std(trace_costs) < 1e-6:
        print(f"\n{'='*80}")
        print("ROOT CAUSE ANALYSIS: Why does ILP vary but Trace doesn't?")
        print(f"{'='*80}")
        
        print(f"\nILP sees {len(df['ILP'].unique())} different cost values: {sorted(df['ILP'].unique())}")
        print(f"Trace sees only: {df['Trace'].unique()}")
        
        # Group by ILP cost
        for ilp_val in sorted(df['ILP'].unique()):
            subset = df[df['ILP'] == ilp_val]
            print(f"\n--- ILP predicts {ilp_val:.2f} (Trace says {subset['Trace'].iloc[0]:.0f}) ---")
            print(f"  Solutions: {subset['Sol'].tolist()}")
            print(f"  Tile Sizes: {subset[['Tile_P', 'Tile_Q', 'Tile_K', 'Tile_Bytes']].drop_duplicates().to_dict('records')}")
            print(f"  Reuse: {subset[['Reuse_P', 'Reuse_Q', 'Reuse_K', 'Total_Reuse']].drop_duplicates().to_dict('records')}")

def main():
    workloads = get_paper_workloads()
    arch = PIMArchitecture()
    
    # Focus on workloads with Output correlation = 0
    problem_workloads = ["VGG_Conv5_1", "ResNet_L1", "ResNet_1x1_Proj", "ResNet_1x1_Red"]
    
    for wl in workloads:
        if wl.name in problem_workloads:
            analyze_workload(wl, arch)

if __name__ == "__main__":
    main()
