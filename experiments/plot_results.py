#!/usr/bin/env python3
"""
Generate plots for Row Activation Validation experiment.

Creates publication-quality figures:
1. ILP vs Trace comparison bar chart
2. Error distribution histogram
3. Per-tensor breakdown stacked bar chart
4. Scalability analysis (batch size)
"""

import sys
import os
import json
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Set publication-quality style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette (colorblind-friendly)
COLORS = {
    'ilp': '#1f77b4',      # Blue
    'trace': '#ff7f0e',    # Orange
    'input': '#2ca02c',    # Green
    'weight': '#d62728',   # Red
    'output': '#9467bd',   # Purple
}


def load_latest_results(results_dir: str) -> dict:
    """Load the most recent JSON results file."""
    pattern = os.path.join(results_dir, "row_activation_results_*.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No results found in {results_dir}")
    
    latest = max(files, key=os.path.getctime)
    print(f"Loading: {latest}")
    
    with open(latest, 'r') as f:
        return json.load(f)


def plot_comparison_bar(results: list, output_dir: str):
    """
    Create grouped bar chart comparing ILP predictions vs Trace counts.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    names = [r['name'] for r in results]
    ilp_totals = [r['ilp']['total'] for r in results]
    trace_totals = [r['trace']['total'] for r in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ilp_totals, width, label='ILP Model', color=COLORS['ilp'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, trace_totals, width, label='Trace Count', color=COLORS['trace'], edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Row Activations')
    ax.set_xlabel('Workload')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Format y-axis with K suffix
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x >= 1000 else f'{x:.0f}'))
    
    plt.tight_layout()
    
    path = os.path.join(output_dir, "comparison_bar.pdf")
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"Saved: {path}")
    plt.close()


def plot_error_distribution(results: list, output_dir: str):
    """
    Create histogram of prediction errors.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    errors = [r['error_pct'] for r in results]
    
    bins = np.arange(0, max(errors) + 2, 1)
    ax.hist(errors, bins=bins, color=COLORS['ilp'], edgecolor='black', alpha=0.7)
    
    # Add statistics
    mean_err = np.mean(errors)
    ax.axvline(mean_err, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_err:.1f}%')
    
    ax.set_xlabel('Prediction Error (%)')
    ax.set_ylabel('Number of Workloads')
    ax.set_title('Distribution of ILP Prediction Errors')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    path = os.path.join(output_dir, "error_distribution.pdf")
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"Saved: {path}")
    plt.close()


def plot_tensor_breakdown(results: list, output_dir: str):
    """
    Create stacked bar chart showing per-tensor breakdown.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    names = [r['name'] for r in results]
    x = np.arange(len(names))
    width = 0.6
    
    # ILP predictions
    ax = axes[0]
    input_vals = [r['ilp']['input'] for r in results]
    weight_vals = [r['ilp']['weight'] for r in results]
    output_vals = [r['ilp']['output'] for r in results]
    
    ax.bar(x, input_vals, width, label='Input', color=COLORS['input'])
    ax.bar(x, weight_vals, width, bottom=input_vals, label='Weight', color=COLORS['weight'])
    ax.bar(x, output_vals, width, bottom=np.array(input_vals)+np.array(weight_vals), label='Output', color=COLORS['output'])
    
    ax.set_ylabel('Row Activations')
    ax.set_xlabel('Workload')
    ax.set_title('ILP Model Prediction')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Trace counts
    ax = axes[1]
    input_vals = [r['trace']['input'] for r in results]
    weight_vals = [r['trace']['weight'] for r in results]
    output_vals = [r['trace']['output'] for r in results]
    
    ax.bar(x, input_vals, width, label='Input', color=COLORS['input'])
    ax.bar(x, weight_vals, width, bottom=input_vals, label='Weight', color=COLORS['weight'])
    ax.bar(x, output_vals, width, bottom=np.array(input_vals)+np.array(weight_vals), label='Output', color=COLORS['output'])
    
    ax.set_ylabel('Row Activations')
    ax.set_xlabel('Workload')
    ax.set_title('Trace Count')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    path = os.path.join(output_dir, "tensor_breakdown.pdf")
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"Saved: {path}")
    plt.close()


def plot_accuracy_vs_macs(results: list, output_dir: str):
    """
    Scatter plot of error vs workload size (MACs).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    macs = [r['macs'] for r in results]
    errors = [r['error_pct'] for r in results]
    
    ax.scatter(macs, errors, s=80, c=COLORS['ilp'], alpha=0.7, edgecolors='black')
    
    # Add labels for outliers
    for r in results:
        if r['error_pct'] > np.mean(errors) + np.std(errors):
            ax.annotate(r['name'], (r['macs'], r['error_pct']), 
                       textcoords="offset points", xytext=(5, 5), fontsize=9)
    
    ax.set_xlabel('MACs')
    ax.set_ylabel('Prediction Error (%)')
    ax.set_title('Prediction Accuracy vs Workload Size')
    ax.set_xscale('log')
    ax.grid(alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, p: f'{x/1e9:.1f}G' if x >= 1e9 else f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
    
    plt.tight_layout()
    
    path = os.path.join(output_dir, "accuracy_vs_macs.pdf")
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"Saved: {path}")
    plt.close()


def plot_batch_scalability(results: list, output_dir: str):
    """
    Line plot showing scalability with batch size.
    """
    # Filter batch-related workloads
    batch_results = [r for r in results if r['name'].startswith('Batch-')]
    
    if not batch_results:
        print("No batch scalability data found, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Extract batch sizes from names (e.g., "Batch-N4" -> 4)
    batch_sizes = []
    ilp_totals = []
    trace_totals = []
    
    for r in sorted(batch_results, key=lambda x: int(x['name'].split('N')[1])):
        batch_sizes.append(int(r['name'].split('N')[1]))
        ilp_totals.append(r['ilp']['total'])
        trace_totals.append(r['trace']['total'])
    
    ax.plot(batch_sizes, ilp_totals, 'o-', label='ILP Model', color=COLORS['ilp'], linewidth=2, markersize=8)
    ax.plot(batch_sizes, trace_totals, 's--', label='Trace Count', color=COLORS['trace'], linewidth=2, markersize=8)
    
    ax.set_xlabel('Batch Size (N)')
    ax.set_ylabel('Row Activations')
    ax.set_title('Row Activation Scalability with Batch Size')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    path = os.path.join(output_dir, "batch_scalability.pdf")
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Row Activation Validation Plots")
    parser.add_argument('--results', default='experiments/results', help='Results directory')
    parser.add_argument('--output', default='experiments/figures', help='Output directory for figures')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load results
    results = load_latest_results(args.results)
    
    print(f"\nGenerating plots for {len(results)} workloads...")
    
    # Generate plots
    plot_comparison_bar(results, args.output)
    plot_error_distribution(results, args.output)
    plot_tensor_breakdown(results, args.output)
    plot_accuracy_vs_macs(results, args.output)
    plot_batch_scalability(results, args.output)
    
    print(f"\nAll figures saved to: {args.output}")


if __name__ == "__main__":
    main()
