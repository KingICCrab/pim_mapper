#!/usr/bin/env python3
import sys
import os
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
import gurobipy as gp
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.arch import PIMArchitecture
from experiments.rank_accuracy_paper.workloads import get_paper_workloads
from experiments.rank_accuracy_paper.fast_trace_generator import FastTraceGenerator
from validation.dram.trace_generator import TraceGenerator, DRAMConfig

def main():
    print("Running Correlation Experiment on 10 Workloads...")
    
    workloads = get_paper_workloads()
    
    # Setup Optimizer
    optimizer = PIMOptimizer(verbose=False, time_limit=60.0, mip_gap=0.05)
    
    # Setup Trace Generator
    dram_config = DRAMConfig(
        row_buffer_bytes=1024, # 1KB
        num_banks=8,
        # row_size=1024, # Not an init arg
        # burst_size=32  # Not an init arg
    )
    trace_gen = FastTraceGenerator(dram_config)
    
    results_data = []
    
    # Save to CSV for plotting
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    for w_idx, workload in enumerate(workloads):
        print(f"\nProcessing Workload {w_idx+1}/{len(workloads)}: {workload.name}")
        
        # Optimize
        # We want multiple solutions to compute correlation per workload
        gurobi_params = {
            "PoolSearchMode": 2,
            "PoolSolutions": 20,
            "PoolGap": 0.5
        }
        
        try:
            result = optimizer.optimize(
                workloads=[workload],
                objective="latency",
                enable_row_activation=True,
                gurobi_params=gurobi_params
            )
            
            if not result.mappings:
                print(f"  No solution found for {workload.name}")
                continue
                
            num_solutions = optimizer.model.SolCount
            print(f"  Found {num_solutions} solutions.")
            
            ilp_weights = []
            trace_weights = []
            ilp_outputs = []
            trace_outputs = []
            
            for i in range(num_solutions):
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
                
                # ILP Costs
                ilp_w = mapping.metrics.get("row_activations_weight", 0)
                ilp_o = mapping.metrics.get("row_activations_output", 0)
                
                # Trace Costs
                # generate_trace returns (input, weight, output)
                trace_stats = trace_gen.generate_trace(mapping, workload)
                trace_w = trace_stats[1]
                trace_o = trace_stats[2]
                
                ilp_weights.append(ilp_w)
                trace_weights.append(trace_w)
                ilp_outputs.append(ilp_o)
                trace_outputs.append(trace_o)
            
            # Compute Correlation
            if len(ilp_weights) > 1:
                # Check if both are constant
                ilp_w_const = np.std(ilp_weights) < 1e-6
                trace_w_const = np.std(trace_weights) < 1e-6
                
                if ilp_w_const and trace_w_const:
                    # Both constant: check if they're equal
                    corr_w = 1.0 if np.allclose(ilp_weights[0], trace_weights[0]) else 0.0
                    p_w = 1.0
                elif ilp_w_const or trace_w_const:
                    # One constant, one varies: no correlation
                    corr_w = 0.0
                    p_w = 1.0
                else:
                    # Both vary: compute Spearman
                    corr_w, p_w = spearmanr(ilp_weights, trace_weights)
                    if np.isnan(corr_w):
                        corr_w = 0.0
            else:
                corr_w, p_w = 1.0, 1.0
            
            if len(ilp_outputs) > 1:
                # Check if both are constant
                ilp_o_const = np.std(ilp_outputs) < 1e-6
                trace_o_const = np.std(trace_outputs) < 1e-6
                
                if ilp_o_const and trace_o_const:
                    # Both constant: check if they're equal
                    corr_o = 1.0 if np.allclose(ilp_outputs[0], trace_outputs[0]) else 0.0
                    p_o = 1.0
                elif ilp_o_const or trace_o_const:
                    # One constant, one varies: no correlation
                    corr_o = 0.0
                    p_o = 1.0
                else:
                    # Both vary: compute Spearman
                    corr_o, p_o = spearmanr(ilp_outputs, trace_outputs)
                    if np.isnan(corr_o):
                        corr_o = 0.0
            else:
                corr_o, p_o = 1.0, 1.0
            
            # Debug output
            if workload.name in ["VGG_Conv1_1", "VGG_Conv5_1", "ResNet_L1", "YOLO_Tiny"]:
                print(f"  Weight: ILP_std={np.std(ilp_weights):.4f}, Trace_std={np.std(trace_weights):.4f}, Corr={corr_w:.4f}")
                print(f"  Output: ILP_std={np.std(ilp_outputs):.4f}, Trace_std={np.std(trace_outputs):.4f}, Corr={corr_o:.4f}")
            
            print(f"  Weight Correlation: {corr_w:.4f}")
            print(f"  Output Correlation: {corr_o:.4f}")
            
            results_data.append({
                "Workload": workload.name,
                "Weight_Corr": corr_w,
                "Output_Corr": corr_o,
                "Num_Sols": num_solutions
            })
            
        except Exception as e:
            print(f"  Error processing {workload.name}: {e}")
            import traceback
            traceback.print_exc()
            
    # Summary
    print("\n=== Summary Results ===")
    df = pd.DataFrame(results_data)
    print(df)
    if not df.empty:
        print(f"\nAverage Weight Correlation: {df['Weight_Corr'].mean():.4f}")
        print(f"Average Output Correlation: {df['Output_Corr'].mean():.4f}")        
        # Save results
        df.to_csv(results_dir / "correlation_summary.csv", index=False)
        print(f"\nResults saved to {results_dir / 'correlation_summary.csv'}")
        
        # Generate plot
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 11
        
        # Prepare data for grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df['Weight_Corr'], width, label='weight', color='steelblue')
        bars2 = ax.bar(x + width/2, df['Output_Corr'], width, label='output', color='darkorange')
        
        ax.set_xlabel('Workload', fontsize=12)
        ax.set_ylabel('Spearman_R', fontsize=12)
        ax.set_title('Spearman Rank Correlation by Workload', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(df['Workload'], rotation=45, ha='right')
        ax.legend(title='Tensor')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = Path(__file__).parent / "figures" / "rank_accuracy_correlation_bar.png"
        fig_path.parent.mkdir(exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {fig_path}")
        plt.close()
if __name__ == "__main__":
    main()
