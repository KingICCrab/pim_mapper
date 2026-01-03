#!/usr/bin/env python3
"""
Run Rank Accuracy Experiment for Multiple Workloads.
Focus: Weight and Output Row Activations.
"""

import sys
import os
import json
import csv
import time
import gurobipy as gp
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload import ConvWorkload
from validation.dram.trace_generator import TraceGenerator, DRAMConfig
from experiments.rank_accuracy.workloads import get_experiment_workloads

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
        
        bank = addr // bank_size
        
        if bank in bank_acts:
            row = (addr % bank_size) // row_size
            if row != current_rows[bank]:
                bank_acts[bank] += 1
                current_rows[bank] = row
                
    return bank_acts

def run_experiment():
    print("========================================================")
    print("      Rank Accuracy Experiment (Weight & Output)")
    print("========================================================")
    
    # Setup
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    optimizer = PIMOptimizer(verbose=False)
    dram_config = DRAMConfig(
        num_channels=32,
        num_banks=16,
        row_buffer_bytes=2048,
        element_size=1
    )
    
    workloads = get_experiment_workloads()
    results_summary = []
    
    # CSV Writer for summary
    csv_file = output_dir / "correlation_summary.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Workload", "Tensor", "Spearman_R", "P_Value", "Num_Solutions", "Status"])

    for workload in workloads:
        print(f"\nProcessing {workload.name}...")
        
        # Optimize
        gurobi_params = {
            "PoolSearchMode": 2,
            "PoolSolutions": 50,
            "PoolGap": 1.0,
        }
        
        try:
            optimizer.optimize(
                [workload], 
                objective="latency", 
                gurobi_params=gurobi_params,
                enable_row_activation=True
            )
        except Exception as e:
            print(f"Optimization failed for {workload.name}: {e}")
            continue
            
        model = optimizer.model
        num_solutions = model.SolCount
        print(f"  Found {num_solutions} solutions.")
        
        if num_solutions < 5:
            print("  Skipping: Too few solutions.")
            continue
            
        # Evaluate
        data = {
            "weight": {"ilp": [], "trace": []},
            "output": {"ilp": [], "trace": []}
        }
        
        for i in range(num_solutions):
            model.params.SolutionNumber = i
            
            result = optimizer._extract_results(
                [workload],
                optimizer.compute_cycles,
                optimizer.latency_vars,
                optimizer.activation_cycles,
                optimizer.macs_scale_factors,
                var_attr="Xn"
            )
            mapping = result.mappings[0]
            
            # ILP Metrics
            ilp_weight = mapping.metrics.get("row_activations_weight", 0)
            ilp_output = mapping.metrics.get("row_activations_output", 0)
            
            # Trace Simulation
            trace_gen = TraceGenerator(dram_config)
            trace = trace_gen.generate_trace(mapping, workload)
            trace_acts = analyze_trace(trace, dram_config)
            
            trace_weight = trace_acts[1]
            trace_output = trace_acts[2]
            
            data["weight"]["ilp"].append(ilp_weight)
            data["weight"]["trace"].append(trace_weight)
            data["output"]["ilp"].append(ilp_output)
            data["output"]["trace"].append(trace_output)
            
        # Calculate Correlation & Save
        workload_results = {"name": workload.name, "solutions": num_solutions}
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            for tensor in ["weight", "output"]:
                ilp_vals = data[tensor]["ilp"]
                trace_vals = data[tensor]["trace"]
                
                # Save raw data for plotting
                raw_data_file = output_dir / f"{workload.name}_{tensor}_raw.json"
                with open(raw_data_file, 'w') as rf:
                    json.dump({"ilp": ilp_vals, "trace": trace_vals}, rf)
                
                if len(set(ilp_vals)) <= 1 or len(set(trace_vals)) <= 1:
                    print(f"  {tensor.upper()}: Constant values (No correlation)")
                    writer.writerow([workload.name, tensor, "N/A", "N/A", num_solutions, "Constant"])
                    workload_results[f"{tensor}_corr"] = None
                else:
                    corr, p_val = spearmanr(ilp_vals, trace_vals)
                    print(f"  {tensor.upper()}: R={corr:.4f} (p={p_val:.2e})")
                    writer.writerow([workload.name, tensor, f"{corr:.4f}", f"{p_val:.4e}", num_solutions, "Success"])
                    workload_results[f"{tensor}_corr"] = corr
                    workload_results[f"{tensor}_pval"] = p_val
        
        results_summary.append(workload_results)

    print("\nExperiment Complete. Results saved to experiments/rank_accuracy/results/")

if __name__ == "__main__":
    run_experiment()
