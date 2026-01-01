#!/usr/bin/env python3
"""
Verify Rank Accuracy of ILP Model vs Trace Simulation.

This script generates multiple suboptimal mappings using Gurobi's Solution Pool,
evaluates them with both the ILP model and the Trace Simulator, and calculates
the Spearman Rank Correlation Coefficient to verify if the ILP model correctly
ranks mappings despite absolute value discrepancies.

Focus: Input and Weight Row Activations.
"""

import sys
import os
import gurobipy as gp
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload import ConvWorkload
from validation.dram.trace_generator import TraceGenerator, DRAMConfig

def analyze_trace(trace, dram_config):
    """Parse trace and count row activations per bank."""
    row_size = dram_config.row_buffer_bytes
    bank_size = row_size * dram_config.num_rows
    
    # Bank 0: Input, Bank 1: Weight, Bank 2: Output
    bank_acts = {0: 0, 1: 0, 2: 0}
    current_rows = {0: None, 1: None, 2: None}
    
    for line in trace:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        addr = int(parts[1], 16)
        
        # Simple bank mapping (assuming linear mapping for validation)
        # This matches debug_small_detail.py logic
        bank = addr // bank_size
        
        if bank in bank_acts:
            row = (addr % bank_size) // row_size
            if row != current_rows[bank]:
                bank_acts[bank] += 1
                current_rows[bank] = row
                
    return bank_acts

def verify_rank_accuracy():
    print("========================================================")
    print("      ILP vs Trace Rank Accuracy Verification")
    print("      (Focus: Input and Weight Row Activations)")
    print("========================================================")

    # 1. Setup Optimizer and Architecture
    # Use default architecture
    optimizer = PIMOptimizer(verbose=False)
    
    # Create DRAM Config for Trace Generator (matching debug_small_detail.py)
    dram_config = DRAMConfig(
        num_channels=32,
        num_banks=16,
        row_buffer_bytes=2048,
        element_size=1
    )
    
    # 2. Setup Workload
    # Using VGG-L1 (Conv1_1)
    workload = ConvWorkload(
        name="VGG-L1",
        P=224, Q=224, C=3, K=64, R=3, S=3, N=1,
        stride=(1, 1), dilation=(1, 1)
    )
    print(f"Workload: {workload.name} (P={workload.P}, Q={workload.Q}, C={workload.C}, K={workload.K})")
    
    # 3. Optimize with Solution Pool
    print("\n[1/3] Generating mappings with Gurobi Solution Pool...")
    gurobi_params = {
        "PoolSearchMode": 2,  # Find n best solutions
        "PoolSolutions": 50,  # Number of solutions to keep
        "PoolGap": 1.0,       # Allow solutions within 100% of optimal
    }
    
    # Run optimization
    optimizer.optimize(
        [workload], 
        objective="latency", 
        gurobi_params=gurobi_params,
        enable_row_activation=True
    )
    
    # Test run without pool
    # optimizer.optimize(
    #     [workload], 
    #     objective="latency", 
    #     gurobi_params=gurobi_params,
    #     enable_row_activation=False
    # )
    
    model = optimizer.model
    num_solutions = model.SolCount
    print(f"Found {num_solutions} solutions in the pool.")
    
    if num_solutions < 2:
        print("Error: Not enough solutions found to calculate correlation.")
        return

    # 4. Evaluate Solutions
    print("\n[2/3] Evaluating solutions with Trace Simulator...")
    
    # Data storage
    data = {
        "input": {"ilp": [], "trace": []},
        "weight": {"ilp": [], "trace": []}
    }
    
    print(f"{'ID':<4} | {'Input (ILP)':<12} {'Input (Trace)':<14} | {'Weight (ILP)':<12} {'Weight (Trace)':<14}")
    print("-" * 70)
    
    for i in range(num_solutions):
        # Set solution number to retrieve from pool
        model.params.SolutionNumber = i
        
        # Extract result
        result = optimizer._extract_results(
            [workload],
            optimizer.compute_cycles,
            optimizer.latency_vars,
            optimizer.activation_cycles,
            optimizer.macs_scale_factors,
            var_attr="Xn"
        )
        
        mapping = result.mappings[0]
        
        # Get ILP metrics
        ilp_input = mapping.metrics.get("row_activations_input", 0)
        ilp_weight = mapping.metrics.get("row_activations_weight", 0)
        
        # Run Trace Simulation
        trace_gen = TraceGenerator(dram_config)
        trace = trace_gen.generate_trace(mapping, workload)
        trace_acts = analyze_trace(trace, dram_config)
        
        trace_input = trace_acts[0]
        trace_weight = trace_acts[1]
        
        # Store data
        data["input"]["ilp"].append(ilp_input)
        data["input"]["trace"].append(trace_input)
        data["weight"]["ilp"].append(ilp_weight)
        data["weight"]["trace"].append(trace_weight)
        
        print(f"{i:<4} | {ilp_input:<12.0f} {trace_input:<14.0f} | {ilp_weight:<12.0f} {trace_weight:<14.0f}")

    # 5. Calculate Correlation
    print("\n[3/3] Calculating Rank Correlation...")
    
    for tensor in ["input", "weight"]:
        ilp_vals = data[tensor]["ilp"]
        trace_vals = data[tensor]["trace"]
        
        # Check for constant values (correlation undefined)
        if len(set(ilp_vals)) <= 1 or len(set(trace_vals)) <= 1:
            print(f"\n{tensor.upper()}: Cannot calculate correlation (constant values).")
            continue
            
        corr, p_value = spearmanr(ilp_vals, trace_vals)
        
        print(f"\n{tensor.upper()} Correlation:")
        print(f"  Spearman r: {corr:.4f}")
        print(f"  P-value:    {p_value:.4e}")
        
        if corr > 0.8:
            print("  -> Strong positive correlation.")
        elif corr > 0.5:
            print("  -> Moderate positive correlation.")
        else:
            print("  -> Weak or no correlation.")

if __name__ == "__main__":
    verify_rank_accuracy()
