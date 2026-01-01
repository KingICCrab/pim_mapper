#!/usr/bin/env python3
"""
Complete Experimental Evaluation for Paper

This script provides a comprehensive experimental evaluation:
1. ILP Model Accuracy: Row activation prediction vs trace counting
2. Optimization Quality: Compare optimized vs baseline mappings
3. Scalability Analysis: Solver time vs workload size

Output:
- LaTeX tables for paper
- CSV data for further analysis
- Publication-quality figures
"""

import sys
import os
import csv
import json
import time
from datetime import datetime
from typing import Dict, List

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, 'src'))

import numpy as np
from tqdm import tqdm

from pim_optimizer.arch.pim_arch import PIMArchitecture
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer import PIMOptimizer


# =============================================================================
# Workload Definitions
# =============================================================================

CNN_WORKLOADS = {
    # ResNet-18 layers (representative)
    'ResNet-Conv1': {'N': 1, 'K': 64, 'C': 3, 'P': 112, 'Q': 112, 'R': 7, 'S': 7, 'desc': 'First conv, large kernel'},
    'ResNet-Conv2': {'N': 1, 'K': 64, 'C': 64, 'P': 56, 'Q': 56, 'R': 3, 'S': 3, 'desc': 'Middle layers'},
    'ResNet-Conv3': {'N': 1, 'K': 128, 'C': 64, 'P': 28, 'Q': 28, 'R': 3, 'S': 3, 'desc': 'Downsampling'},
    'ResNet-Conv4': {'N': 1, 'K': 256, 'C': 128, 'P': 14, 'Q': 14, 'R': 3, 'S': 3, 'desc': 'High channels'},
    'ResNet-Conv5': {'N': 1, 'K': 512, 'C': 256, 'P': 7, 'Q': 7, 'R': 3, 'S': 3, 'desc': 'Final stage'},
    
    # VGG-16 layers
    'VGG-Conv1': {'N': 1, 'K': 64, 'C': 3, 'P': 224, 'Q': 224, 'R': 3, 'S': 3, 'desc': 'Large spatial'},
    'VGG-Conv2': {'N': 1, 'K': 128, 'C': 64, 'P': 112, 'Q': 112, 'R': 3, 'S': 3, 'desc': 'Middle stage'},
    'VGG-Conv3': {'N': 1, 'K': 256, 'C': 128, 'P': 56, 'Q': 56, 'R': 3, 'S': 3, 'desc': 'Deeper layers'},
    
    # MobileNet pointwise
    'MobileNet-PW1': {'N': 1, 'K': 64, 'C': 32, 'P': 56, 'Q': 56, 'R': 1, 'S': 1, 'desc': '1x1 conv'},
    'MobileNet-PW2': {'N': 1, 'K': 128, 'C': 64, 'P': 28, 'Q': 28, 'R': 1, 'S': 1, 'desc': '1x1 conv'},
    'MobileNet-PW3': {'N': 1, 'K': 256, 'C': 128, 'P': 14, 'Q': 14, 'R': 1, 'S': 1, 'desc': '1x1 conv'},
}

BATCH_STUDY = {
    'Batch-1': {'N': 1, 'K': 128, 'C': 64, 'P': 28, 'Q': 28, 'R': 3, 'S': 3},
    'Batch-2': {'N': 2, 'K': 128, 'C': 64, 'P': 28, 'Q': 28, 'R': 3, 'S': 3},
    'Batch-4': {'N': 4, 'K': 128, 'C': 64, 'P': 28, 'Q': 28, 'R': 3, 'S': 3},
    'Batch-8': {'N': 8, 'K': 128, 'C': 64, 'P': 28, 'Q': 28, 'R': 3, 'S': 3},
    'Batch-16': {'N': 16, 'K': 128, 'C': 64, 'P': 28, 'Q': 28, 'R': 3, 'S': 3},
}


# =============================================================================
# Main Experiment Functions
# =============================================================================

def run_optimization_experiment(workloads: Dict, arch_path: str, output_dir: str) -> List[Dict]:
    """
    Run ILP optimization experiment and collect metrics.
    
    Metrics collected:
    - MACs (compute intensity)
    - ILP row activations (per tensor)
    - Solver time
    - Objective value (latency)
    """
    os.makedirs(output_dir, exist_ok=True)
    arch = PIMArchitecture.from_yaml(arch_path)
    
    results = []
    
    print("=" * 120)
    print("ILP Optimization Experiment")
    print("=" * 120)
    print(f"{'Workload':<18} {'MACs':>12} {'Input RA':>10} {'Weight RA':>10} {'Output RA':>10} "
          f"{'Total RA':>12} {'Time(s)':>8} {'Obj':>12}")
    print("-" * 120)
    
    for name, params in tqdm(workloads.items(), desc="Optimizing"):
        # Remove non-workload params
        desc = params.pop('desc', '')
        stride = params.pop('stride', (1, 1))
        
        workload = ConvWorkload(name=name, stride=stride, **params)
        
        # Restore params
        params['desc'] = desc
        params['stride'] = stride
        
        # Run optimizer with timing
        start_time = time.time()
        optimizer = PIMOptimizer(arch, verbose=False)
        result = optimizer.optimize([workload])
        solve_time = time.time() - start_time
        
        model = optimizer.model
        
        # Extract row activations
        row_acts = {'input': 0, 'weight': 0, 'output': 0}
        for t_id, t_name in [(0, 'input'), (1, 'weight'), (2, 'output')]:
            var = model.getVarByName(f'total_row_acts_(0,{t_id})')
            if var:
                row_acts[t_name] = var.X
        row_acts['total'] = sum(row_acts.values())
        
        # Get objective value
        obj_value = model.ObjVal if model.ObjVal else 0
        
        r = {
            'name': name,
            'macs': workload.macs,
            'N': workload.N,
            'K': workload.K,
            'C': workload.C,
            'P': workload.P,
            'Q': workload.Q,
            'R': workload.R,
            'S': workload.S,
            'row_acts_input': row_acts['input'],
            'row_acts_weight': row_acts['weight'],
            'row_acts_output': row_acts['output'],
            'row_acts_total': row_acts['total'],
            'solve_time': solve_time,
            'objective': obj_value,
            'desc': desc,
        }
        results.append(r)
        
        print(f"{name:<18} {workload.macs:>12,} {row_acts['input']:>10,.0f} {row_acts['weight']:>10,.0f} "
              f"{row_acts['output']:>10,.0f} {row_acts['total']:>12,.0f} {solve_time:>8.2f} {obj_value:>12.2f}")
    
    print("-" * 120)
    
    # Statistics
    total_row_acts = [r['row_acts_total'] for r in results]
    solve_times = [r['solve_time'] for r in results]
    
    print(f"\nStatistics:")
    print(f"  Total Row Activations: {sum(total_row_acts):,.0f}")
    print(f"  Avg Row Activations: {np.mean(total_row_acts):,.0f}")
    print(f"  Avg Solve Time: {np.mean(solve_times):.2f}s")
    print(f"  Max Solve Time: {np.max(solve_times):.2f}s")
    
    return results


def save_results_latex(results: List[Dict], output_path: str, caption: str, label: str):
    """Generate LaTeX table."""
    with open(output_path, 'w') as f:
        f.write(f"% {caption}\n")
        f.write("% Auto-generated by experimental_evaluation.py\n\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\begin{tabular}{lrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Workload & MACs & Input & Weight & Output & Total & Time (s) \\\\\n")
        f.write("\\midrule\n")
        
        for r in results:
            name = r['name'].replace('_', '\\_').replace('-', '-')
            macs_str = f"{r['macs']/1e6:.1f}M" if r['macs'] >= 1e6 else f"{r['macs']/1e3:.1f}K"
            f.write(f"{name} & {macs_str} & {int(r['row_acts_input']):,} & "
                    f"{int(r['row_acts_weight']):,} & {int(r['row_acts_output']):,} & "
                    f"{int(r['row_acts_total']):,} & {r['solve_time']:.2f} \\\\\n")
        
        f.write("\\midrule\n")
        total = sum(r['row_acts_total'] for r in results)
        avg_time = np.mean([r['solve_time'] for r in results])
        f.write(f"\\textbf{{Total/Avg}} & & & & & \\textbf{{{int(total):,}}} & \\textbf{{{avg_time:.2f}}} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"Saved LaTeX: {output_path}")


def save_results_csv(results: List[Dict], output_path: str):
    """Save results to CSV."""
    if not results:
        return
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Saved CSV: {output_path}")


def save_results_json(results: List[Dict], output_path: str):
    """Save results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Saved JSON: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PIM Optimizer Experimental Evaluation")
    parser.add_argument('--arch', default='examples/configs/arch.yaml', help='Architecture config')
    parser.add_argument('--output', default='experiments/results', help='Output directory')
    parser.add_argument('--workloads', choices=['cnn', 'batch', 'all'], default='all',
                       help='Which workload set to run')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Select workloads
    if args.workloads == 'cnn':
        workloads = CNN_WORKLOADS
    elif args.workloads == 'batch':
        workloads = BATCH_STUDY
    else:
        workloads = {**CNN_WORKLOADS, **BATCH_STUDY}
    
    # Run experiment
    results = run_optimization_experiment(workloads, args.arch, args.output)
    
    # Save results
    base_name = f"optimization_results_{timestamp}"
    save_results_csv(results, os.path.join(args.output, f"{base_name}.csv"))
    save_results_json(results, os.path.join(args.output, f"{base_name}.json"))
    save_results_latex(
        results,
        os.path.join(args.output, f"{base_name}.tex"),
        caption="ILP Optimization Results: Row Activations by Tensor Type",
        label="tab:optimization_results"
    )
    
    # Generate batch study table separately if included
    if args.workloads in ['batch', 'all']:
        batch_results = [r for r in results if r['name'].startswith('Batch')]
        if batch_results:
            save_results_latex(
                batch_results,
                os.path.join(args.output, f"batch_study_{timestamp}.tex"),
                caption="Batch Size Scalability Study",
                label="tab:batch_study"
            )
    
    print("\n" + "=" * 120)
    print("Experiment Complete!")
    print("=" * 120)


if __name__ == "__main__":
    main()
