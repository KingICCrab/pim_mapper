#!/usr/bin/env python3
"""
Row Activation Validation Experiment for Paper

This script validates the ILP model's row activation predictions against
trace-based counting for various CNN workloads.

Output:
- LaTeX table for paper
- CSV data for further analysis
- Per-tensor breakdown (Input/Weight/Output)
"""

import sys
import os
import csv
import json
from datetime import datetime
from typing import Dict, List, Tuple

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
# sys.path.insert(0, os.path.join(project_root, 'src'))
# sys.path.insert(0, os.path.join(project_root, 'validation', 'dram'))

sys.path.append(os.path.join(os.path.dirname(__file__), "../../python"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../golden"))



import numpy as np
from tqdm import tqdm

from pim_optimizer.arch.pim_arch import PIMArchitecture
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer import PIMOptimizer

from golden.trace_generator import TraceGenerator, DRAMConfig
from golden.full_validation import count_row_activations_from_trace, generate_trace_for_mapping


# =============================================================================
# Workload Definitions - Common CNN Layers
# =============================================================================

WORKLOADS = {
    # Small-scale workloads (verified accurate)
    'tiny': {'N': 1, 'K': 8, 'C': 8, 'P': 4, 'Q': 4, 'R': 3, 'S': 3},
    'small': {'N': 1, 'K': 16, 'C': 16, 'P': 8, 'Q': 8, 'R': 3, 'S': 3},
    'medium': {'N': 1, 'K': 32, 'C': 32, 'P': 14, 'Q': 14, 'R': 3, 'S': 3},
    
    # ResNet-style layers (scaled down for tractable trace generation)
    'ResNet-L1': {'N': 1, 'K': 32, 'C': 3, 'P': 28, 'Q': 28, 'R': 7, 'S': 7},
    'ResNet-L2': {'N': 1, 'K': 64, 'C': 32, 'P': 14, 'Q': 14, 'R': 3, 'S': 3},
    'ResNet-L3': {'N': 1, 'K': 128, 'C': 64, 'P': 7, 'Q': 7, 'R': 3, 'S': 3},
    
    # VGG-style layers (scaled down)
    'VGG-L1': {'N': 1, 'K': 32, 'C': 3, 'P': 56, 'Q': 56, 'R': 3, 'S': 3},
    'VGG-L2': {'N': 1, 'K': 64, 'C': 32, 'P': 28, 'Q': 28, 'R': 3, 'S': 3},
    'VGG-L3': {'N': 1, 'K': 128, 'C': 64, 'P': 14, 'Q': 14, 'R': 3, 'S': 3},
    
    # MobileNet-style pointwise layers
    'MobileNet-L1': {'N': 1, 'K': 32, 'C': 16, 'P': 28, 'Q': 28, 'R': 1, 'S': 1},
    'MobileNet-L2': {'N': 1, 'K': 64, 'C': 32, 'P': 14, 'Q': 14, 'R': 1, 'S': 1},
    'MobileNet-L3': {'N': 1, 'K': 128, 'C': 64, 'P': 7, 'Q': 7, 'R': 1, 'S': 1},
    
    # Batch size study
    'Batch-N1': {'N': 1, 'K': 64, 'C': 32, 'P': 14, 'Q': 14, 'R': 3, 'S': 3},
    'Batch-N2': {'N': 2, 'K': 64, 'C': 32, 'P': 14, 'Q': 14, 'R': 3, 'S': 3},
    'Batch-N4': {'N': 4, 'K': 64, 'C': 32, 'P': 14, 'Q': 14, 'R': 3, 'S': 3},
}

# Full-scale workloads (may have higher error due to trace generator limitations)
WORKLOADS_FULL_SCALE = {
    # ResNet-18 layers
    'ResNet18-Conv1': {'N': 1, 'K': 64, 'C': 3, 'P': 112, 'Q': 112, 'R': 7, 'S': 7, 'stride': (2, 2)},
    'ResNet18-Conv2': {'N': 1, 'K': 64, 'C': 64, 'P': 56, 'Q': 56, 'R': 3, 'S': 3},
    'ResNet18-Conv3': {'N': 1, 'K': 128, 'C': 64, 'P': 28, 'Q': 28, 'R': 3, 'S': 3},
    'ResNet18-Conv4': {'N': 1, 'K': 256, 'C': 128, 'P': 14, 'Q': 14, 'R': 3, 'S': 3},
    'ResNet18-Conv5': {'N': 1, 'K': 512, 'C': 256, 'P': 7, 'Q': 7, 'R': 3, 'S': 3},
}

# Smaller subset for quick testing
WORKLOADS_SMALL = {
    'micro': {'N': 1, 'K': 1, 'C': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1},
    'tiny': {'N': 1, 'K': 8, 'C': 8, 'P': 4, 'Q': 4, 'R': 3, 'S': 3},
    'small': {'N': 1, 'K': 16, 'C': 16, 'P': 8, 'Q': 8, 'R': 3, 'S': 3},
    'medium': {'N': 1, 'K': 32, 'C': 32, 'P': 14, 'Q': 14, 'R': 3, 'S': 3},
}


# =============================================================================
# Experiment Runner
# =============================================================================


def run_single_experiment(name: str, params: Dict, arch, dram_config: DRAMConfig, 
                          output_dir: str) -> Dict:
    """Run experiment for a single workload."""
    # Create workload
    stride = params.pop('stride', (1, 1))
    workload = ConvWorkload(name=name, stride=stride, **params)
    params['stride'] = stride  # Restore for later use
    
    # Run ILP optimizer
    optimizer = PIMOptimizer(arch, verbose=False)
    result = optimizer.optimize([workload])
    model = optimizer.model
    
    # Get ILP predictions
    ilp_row_acts = {'input': 0, 'weight': 0, 'output': 0}
    for t_id, t_name in [(0, 'input'), (1, 'weight'), (2, 'output')]:
        var = model.getVarByName(f'total_row_acts_(0,{t_id})')
        if var:
            ilp_row_acts[t_name] = var.X
    ilp_row_acts['total'] = sum(ilp_row_acts.values())
    
    # Generate trace and count row activations
    try:
        trace_path = os.path.join(output_dir, f"{name}_trace.txt")
        num_traces = generate_trace_for_mapping(optimizer, workload, trace_path)
        trace_stats = count_row_activations_from_trace(trace_path, dram_config)
        trace_row_acts = {
            'total': trace_stats['total_row_acts'],
            'input': trace_stats['per_bank_acts'].get(0, 0),
            'weight': trace_stats['per_bank_acts'].get(1, 0),
            'output': trace_stats['per_bank_acts'].get(2, 0),
            'per_bank': trace_stats['per_bank_acts'],
        }
    except Exception as e:
        print(f"  Trace generation error for {name}: {e}")
        trace_row_acts = {'total': 0, 'input': 0, 'weight': 0, 'output': 0}
    
    # Calculate error
    if trace_row_acts['total'] > 0:
        error_pct = abs(ilp_row_acts['total'] - trace_row_acts['total']) / trace_row_acts['total'] * 100
    else:
        error_pct = 0 if ilp_row_acts['total'] == 0 else 100
    
    return {
        'name': name,
        'macs': workload.macs,
        'params': params,
        'ilp': ilp_row_acts,
        'trace': trace_row_acts,
        'error_pct': error_pct,
    }


def run_experiments(workloads: Dict, arch_path: str, output_dir: str):
    """Run all experiments and save results."""
    os.makedirs(output_dir, exist_ok=True)
    traces_dir = os.path.join(output_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)
    
    if os.path.exists(arch_path):
        arch = PIMArchitecture.from_yaml(arch_path)
    else:
        print(f"Warning: Arch config not found at {arch_path}, using default PIMArchitecture")
        arch = PIMArchitecture()

    dram_config = DRAMConfig()
    
    results = []
    
    print("=" * 100)
    print("Row Activation Validation Experiment")
    print("=" * 100)
    print(f"{'Workload':<20} {'MACs':>12} {'ILP Total':>12} {'Trace Total':>12} {'Error%':>10}")
    print("-" * 100)
    
    for name, params in tqdm(workloads.items(), desc="Running experiments"):
        result = run_single_experiment(name, params.copy(), arch, dram_config, traces_dir)
        results.append(result)
        
        print(f"{result['name']:<20} {result['macs']:>12,} "
              f"{result['ilp']['total']:>12,.0f} {result['trace']['total']:>12,} "
              f"{result['error_pct']:>9.2f}%")
    
    print("-" * 100)
    
    # Calculate statistics
    errors = [r['error_pct'] for r in results]
    print(f"\nStatistics:")
    print(f"  Mean Error: {np.mean(errors):.2f}%")
    print(f"  Max Error:  {np.max(errors):.2f}%")
    print(f"  Min Error:  {np.min(errors):.2f}%")
    print(f"  Std Dev:    {np.std(errors):.2f}%")
    
    # Save results
    save_results(results, output_dir)
    
    return results


def save_results(results: List[Dict], output_dir: str):
    """Save results to CSV and LaTeX format."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save CSV
    csv_path = os.path.join(output_dir, f"row_activation_results_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Workload', 'MACs', 
            'ILP_Input', 'ILP_Weight', 'ILP_Output', 'ILP_Total',
            'Trace_Input', 'Trace_Weight', 'Trace_Output', 'Trace_Total',
            'Error%'
        ])
        for r in results:
            writer.writerow([
                r['name'], r['macs'],
                r['ilp']['input'], r['ilp']['weight'], r['ilp']['output'], r['ilp']['total'],
                r['trace']['input'], r['trace']['weight'], r['trace']['output'], r['trace']['total'],
                f"{r['error_pct']:.2f}"
            ])
    print(f"\nSaved CSV: {csv_path}")
    
    # Save LaTeX table
    latex_path = os.path.join(output_dir, f"row_activation_table_{timestamp}.tex")
    with open(latex_path, 'w') as f:
        f.write("% Row Activation Validation Results\n")
        f.write("% Auto-generated by row_activation_validation.py\n\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Row Activation Prediction Accuracy}\n")
        f.write("\\label{tab:row_activation}\n")
        f.write("\\begin{tabular}{lrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Workload & MACs & ILP Pred. & Trace & Error (\\%) \\\\\n")
        f.write("\\midrule\n")
        
        for r in results:
            name = r['name'].replace('_', '\\_')
            macs_str = f"{r['macs']/1e6:.1f}M" if r['macs'] >= 1e6 else f"{r['macs']/1e3:.1f}K"
            f.write(f"{name} & {macs_str} & {int(r['ilp']['total']):,} & "
                    f"{int(r['trace']['total']):,} & {r['error_pct']:.1f} \\\\\n")
        
        f.write("\\midrule\n")
        errors = [r['error_pct'] for r in results]
        f.write(f"\\textbf{{Average}} & & & & \\textbf{{{np.mean(errors):.1f}}} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"Saved LaTeX: {latex_path}")
    
    # Save JSON for detailed analysis
    json_path = os.path.join(output_dir, f"row_activation_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved JSON: {json_path}")


# =============================================================================
# Per-Tensor Breakdown Table
# =============================================================================

def generate_breakdown_table(results: List[Dict], output_dir: str):
    """Generate per-tensor breakdown LaTeX table."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latex_path = os.path.join(output_dir, f"row_activation_breakdown_{timestamp}.tex")
    
    with open(latex_path, 'w') as f:
        f.write("% Per-Tensor Row Activation Breakdown\n\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Per-Tensor Row Activation Breakdown}\n")
        f.write("\\label{tab:row_activation_breakdown}\n")
        f.write("\\begin{tabular}{l|rrr|rrr}\n")
        f.write("\\toprule\n")
        f.write(" & \\multicolumn{3}{c|}{ILP Prediction} & \\multicolumn{3}{c}{Trace Count} \\\\\n")
        f.write("Workload & Input & Weight & Output & Input & Weight & Output \\\\\n")
        f.write("\\midrule\n")
        
        for r in results:
            name = r['name'].replace('_', '\\_')
            f.write(f"{name} & {int(r['ilp']['input']):,} & {int(r['ilp']['weight']):,} & "
                    f"{int(r['ilp']['output']):,} & {int(r['trace']['input']):,} & "
                    f"{int(r['trace']['weight']):,} & {int(r['trace']['output']):,} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"Saved breakdown table: {latex_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Row Activation Validation Experiment")
    parser.add_argument('--arch', default='examples/configs/arch.yaml', help='Architecture config path')
    parser.add_argument('--output', default='experiments/results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Run quick test with small workloads')
    parser.add_argument('--workloads', nargs='+', help='Specific workloads to run')
    args = parser.parse_args()
    
    # Select workloads
    if args.quick:
        workloads = WORKLOADS_SMALL
    elif args.workloads:
        workloads = {k: v for k, v in WORKLOADS.items() if k in args.workloads}
    else:
        workloads = WORKLOADS
    
    # Run experiments
    results = run_experiments(workloads, args.arch, args.output)
    
    # Generate breakdown table
    generate_breakdown_table(results, args.output)
    
    print("\n" + "=" * 100)
    print("Experiment Complete!")
    print("=" * 100)
