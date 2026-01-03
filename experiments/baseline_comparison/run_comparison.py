#!/usr/bin/env python3
"""
Run Baseline Comparison Experiment.
Compares ILP Optimizer against Heuristic Baselines (Weight Stationary, Output Stationary).
"""

import sys
import os
import csv
import time
from pathlib import Path
import gurobipy as gp

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.baselines import HeuristicOptimizer
from pim_optimizer.arch import PIMArchitecture
from validation.dram.trace_generator import DRAMConfig
from experiments.rank_accuracy_paper.workloads import get_paper_workloads
from experiments.rank_accuracy_paper.fast_trace_generator import FastTraceGenerator

def run_comparison():
    print("========================================================")
    print("      Baseline Comparison Experiment")
    print("      ILP vs Weight Stationary vs Output Stationary")
    print("========================================================")
    
    # Setup
    # Use default arch
    arch = PIMArchitecture()
    dram_config = DRAMConfig(
        num_channels=32,
        num_banks=16,
        row_buffer_bytes=1024,
        element_size=1
    )
    
    # Optimizers
    ilp_opt = PIMOptimizer(verbose=False, time_limit=60) # 60s limit for ILP
    ws_opt = HeuristicOptimizer(arch, strategy="weight_stationary")
    os_opt = HeuristicOptimizer(arch, strategy="output_stationary")
    
    trace_gen = FastTraceGenerator(dram_config)
    
    workloads = get_paper_workloads()
    
    # CSV Writer
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "baseline_comparison.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            "Workload", 
            "Method", 
            "Input_Acts", "Weight_Acts", "Output_Acts", "Max_Acts", "Total_Acts",
            "Improvement_vs_WS(%)", "Improvement_vs_OS(%)"
        ]
        writer.writerow(header)
        
    for i, workload in enumerate(workloads):
        # if i < 2: continue # Skip small VGG layers if needed
        print(f"\n[{i+1}/{len(workloads)}] Processing {workload.name}...")
        
        results = {}
        
        # 1. Run Heuristic: Weight Stationary
        print("  Running Weight Stationary...", end="", flush=True)
        ws_res = ws_opt.optimize([workload])
        ws_mapping = ws_res.mappings[0]
        ws_stats = trace_gen.generate_trace(ws_mapping, workload)
        ws_max = max(ws_stats.values())
        ws_total = sum(ws_stats.values())
        results["WS"] = {
            "stats": ws_stats, 
            "max": ws_max, 
            "total": ws_total
        }
        print(" Done.")
        
        # 2. Run Heuristic: Output Stationary
        print("  Running Output Stationary...", end="", flush=True)
        os_res = os_opt.optimize([workload])
        os_mapping = os_res.mappings[0]
        os_stats = trace_gen.generate_trace(os_mapping, workload)
        os_max = max(os_stats.values())
        os_total = sum(os_stats.values())
        results["OS"] = {
            "stats": os_stats, 
            "max": os_max, 
            "total": os_total
        }
        print(" Done.")
        
        # 3. Run ILP
        print("  Running ILP Optimizer...", end="", flush=True)
        try:
            # Use latency objective which minimizes max(input, weight, output)
            ilp_res = ilp_opt.optimize(
                [workload], 
                objective="latency",
                enable_row_activation=True
            )
            if ilp_res.mappings:
                ilp_mapping = ilp_res.mappings[0]
                ilp_stats = trace_gen.generate_trace(ilp_mapping, workload)
                ilp_max = max(ilp_stats.values())
                ilp_total = sum(ilp_stats.values())
                results["ILP"] = {
                    "stats": ilp_stats, 
                    "max": ilp_max, 
                    "total": ilp_total
                }
                print(" Done.")
            else:
                print(" Failed (No Solution).")
                results["ILP"] = None
        except Exception as e:
            print(f" Failed ({e}).")
            results["ILP"] = None
            
        # Log Results
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Helper to write row
            def write_method_row(method_name, res_dict):
                if res_dict is None:
                    writer.writerow([workload.name, method_name] + ["N/A"]*7)
                    return
                
                stats = res_dict["stats"]
                mx = res_dict["max"]
                tot = res_dict["total"]
                
                # Calculate improvement vs Baselines (based on Max Acts)
                imp_ws = 0.0
                if results["WS"] and results["WS"]["max"] > 0:
                    imp_ws = (results["WS"]["max"] - mx) / results["WS"]["max"] * 100
                    
                imp_os = 0.0
                if results["OS"] and results["OS"]["max"] > 0:
                    imp_os = (results["OS"]["max"] - mx) / results["OS"]["max"] * 100
                    
                writer.writerow([
                    workload.name, 
                    method_name,
                    stats[0], stats[1], stats[2], mx, tot,
                    f"{imp_ws:.2f}", f"{imp_os:.2f}"
                ])
                
                print(f"    {method_name}: Max={mx}, Total={tot}")
                if method_name == "ILP":
                    print(f"      vs WS: {imp_ws:+.2f}%")
                    print(f"      vs OS: {imp_os:+.2f}%")

            write_method_row("Weight Stationary", results["WS"])
            write_method_row("Output Stationary", results["OS"])
            write_method_row("ILP", results["ILP"])

    print(f"\nComparison complete. Results saved to {csv_path}")

if __name__ == "__main__":
    run_comparison()
