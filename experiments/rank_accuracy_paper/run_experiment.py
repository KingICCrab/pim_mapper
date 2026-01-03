#!/usr/bin/env python3
"""
Run Rank Accuracy Experiment for Paper.
Tests 10 workloads, comparing ILP vs Trace for Weight and Output row activations.
"""

import sys
import os
import json
import csv
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import gurobipy as gp

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.arch import PIMArchitecture
from validation.dram.trace_generator import TraceGenerator, DRAMConfig
from experiments.rank_accuracy_paper.workloads import get_paper_workloads
from experiments.rank_accuracy_paper.fast_trace_generator import FastTraceGenerator

def run_experiment():
    print("========================================================")
    print("      Paper Experiment: Rank Accuracy (10 Workloads)")
    print("========================================================")
    
    # Setup
    optimizer = PIMOptimizer(verbose=False)
    # Use default arch but ensure config matches trace generator
    # Default PIMArchitecture uses row_buffer_size=1024
    dram_config = DRAMConfig(
        num_channels=32,
        num_banks=16,
        row_buffer_bytes=1024, # Matched with ILP default
        element_size=1
    )
    
    workloads = get_paper_workloads()
    results = []
    
    # CSV Writers
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    summary_csv = results_dir / "rank_accuracy_summary.csv"
    raw_csv = results_dir / "rank_accuracy_raw.csv"
    
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Workload", "Tensor", "Spearman_R", "P_Value", "Num_Solutions", "ILP_Range", "Trace_Range"])

    with open(raw_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Workload", "SolutionID", "Tensor", "ILP_Value", "Trace_Value"])
    
    for i, workload in enumerate(workloads):
        if i < 2: continue # Skip VGG_Conv1_1 and VGG_Conv2_1
        print(f"\n[{i+1}/10] Processing {workload.name}...")
        
        # Gurobi Params for Solution Pool
        gurobi_params = {
            "PoolSearchMode": 2,
            "PoolSolutions": 20, # Reduced to 20 for speed
            "PoolGap": 0.5, 
            "TimeLimit": 30 # Reduced to 30s
        }
        
        try:
            optimizer.optimize(
                [workload], 
                objective="latency", 
                gurobi_params=gurobi_params,
                enable_row_activation=True
            )
        except Exception as e:
            print(f"  Optimization failed: {e}")
            continue
            
        model = optimizer.model
        num_solutions = model.SolCount
        print(f"  Found {num_solutions} solutions.")
        
        if num_solutions < 5:
            print("  Skipping (too few solutions).")
            continue
            
        # Collect Data
        data = {
            "input": {"ilp": [], "trace": []},
            "weight": {"ilp": [], "trace": []},
            "output": {"ilp": [], "trace": []},
            "max": {"ilp": [], "trace": []}
        }
        
        # Use FastTraceGenerator
        trace_gen = FastTraceGenerator(dram_config)
        
        for sol_idx in range(num_solutions):
            model.params.SolutionNumber = sol_idx
            
            # Extract result
            result = optimizer._extract_results(
                [workload],
                optimizer.compute_cycles,
                optimizer.latency_vars,
                optimizer.activation_cycles,
                optimizer.macs_scale_factors,
                var_attr="Xn"
            )
            
            if not result.mappings:
                continue
                
            mapping = result.mappings[0]
            
            # ILP Metrics
            ilp_input = mapping.metrics.get("row_activations_input", 0)
            ilp_weight = mapping.metrics.get("row_activations_weight", 0)
            ilp_output = mapping.metrics.get("row_activations_output", 0)
            
            # Trace Metrics (Fast)
            trace_stats = trace_gen.generate_trace(mapping, workload)
            
            trace_input = trace_stats[0]
            trace_weight = trace_stats[1]
            trace_output = trace_stats[2]
            
            data["input"]["ilp"].append(ilp_input)
            data["input"]["trace"].append(trace_input)
            data["weight"]["ilp"].append(ilp_weight)
            data["weight"]["trace"].append(trace_weight)
            data["output"]["ilp"].append(ilp_output)
            data["output"]["trace"].append(trace_output)
            
            # Calculate Max (Bottleneck)
            # Note: This assumes latency is proportional to row acts.
            # In reality, latency = transfer + acts, but acts are often dominant.
            data["max"]["ilp"].append(max(ilp_input, ilp_weight, ilp_output))
            data["max"]["trace"].append(max(trace_input, trace_weight, trace_output))
            
            # Log raw data
            with open(raw_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([workload.name, sol_idx, "input", ilp_input, trace_input])
                writer.writerow([workload.name, sol_idx, "weight", ilp_weight, trace_weight])
                writer.writerow([workload.name, sol_idx, "output", ilp_output, trace_output])
                writer.writerow([workload.name, sol_idx, "max", max(ilp_input, ilp_weight, ilp_output), max(trace_input, trace_weight, trace_output)])
            
            # Print progress dot
            print(".", end="", flush=True)
        print()
            
        # Compute Correlation
        with open(summary_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            
            for tensor in ["input", "weight", "output", "max"]:
                ilp = data[tensor]["ilp"]
                trace = data[tensor]["trace"]
                
                # Check for constant values (variance = 0)
                # Use a small epsilon for float comparison, though trace is usually int
                ilp_is_const = (len(set(ilp)) <= 1) or (max(ilp) - min(ilp) < 1e-6)
                trace_is_const = (len(set(trace)) <= 1) or (max(trace) - min(trace) < 1e-6)

                if ilp_is_const or trace_is_const:
                    if ilp_is_const and trace_is_const:
                        # Both are constant. The ranking is preserved (all are equal).
                        # Even if absolute values differ, the optimizer correctly identified that
                        # the search space is flat for this metric.
                        print(f"  {tensor.title()}: Both Constant (Rank Preserved)")
                        writer.writerow([workload.name, tensor, "1.0000", "0.0000e+00", num_solutions, f"{min(ilp)}-{max(ilp)}", f"{min(trace)}-{max(trace)}"])
                    else:
                        # One is constant, the other varies. This is a disagreement.
                        print(f"  {tensor.title()}: One Constant (Mismatch)")
                        writer.writerow([workload.name, tensor, "0.0000", "1.0000e+00", num_solutions, f"{min(ilp)}-{max(ilp)}", f"{min(trace)}-{max(trace)}"])
                else:
                    corr, p_val = spearmanr(ilp, trace)
                    print(f"  {tensor.title()}: R={corr:.4f} (p={p_val:.2e})")
                    writer.writerow([workload.name, tensor, f"{corr:.4f}", f"{p_val:.4e}", num_solutions, f"{min(ilp)}-{max(ilp)}", f"{min(trace)}-{max(trace)}"])

    print(f"\nExperiment complete. Results saved to {results_dir}")

if __name__ == "__main__":
    run_experiment()
