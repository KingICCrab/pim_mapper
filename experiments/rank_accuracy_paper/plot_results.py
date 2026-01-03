
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def plot_results():
    # Setup paths
    base_dir = Path(__file__).parent
    results_dir = base_dir / "results"
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    raw_csv = results_dir / "rank_accuracy_raw.csv"
    summary_csv = results_dir / "rank_accuracy_summary.csv"
    
    if not raw_csv.exists():
        print(f"Error: {raw_csv} not found. Run experiment first.")
        return

    # Load data
    df = pd.read_csv(raw_csv)
    summary_df = pd.read_csv(summary_csv)
    
    workloads = df["Workload"].unique()
    tensors = ["input", "weight", "output"]
    
    # 1. Scatter Plots Grid
    # Rows = Workloads, Cols = Tensors
    n_rows = len(workloads)
    n_cols = len(tensors)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    fig.suptitle("ILP vs Trace Row Activations (Rank Accuracy)", fontsize=16)
    
    for i, workload in enumerate(workloads):
        for j, tensor in enumerate(tensors):
            ax = axes[i, j]
            subset = df[(df["Workload"] == workload) & (df["Tensor"] == tensor)]
            
            if subset.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                continue
                
            # Get correlation from summary
            corr_row = summary_df[(summary_df["Workload"] == workload) & (summary_df["Tensor"] == tensor)]
            if not corr_row.empty:
                r_val = corr_row.iloc[0]["Spearman_R"]
                title_suffix = f"\nR={r_val}"
            else:
                title_suffix = ""
            
            # Scatter plot
            sns.scatterplot(data=subset, x="Trace_Value", y="ILP_Value", ax=ax, alpha=0.7)
            
            # Add diagonal line (y=x) for reference
            min_val = min(subset["Trace_Value"].min(), subset["ILP_Value"].min())
            max_val = max(subset["Trace_Value"].max(), subset["ILP_Value"].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
            
            ax.set_title(f"{workload} - {tensor.title()}{title_suffix}")
            ax.set_xlabel("Trace Simulation")
            ax.set_ylabel("ILP Model")
            ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(figures_dir / "rank_accuracy_scatter_grid.png", dpi=300)
    print(f"Saved scatter plots to {figures_dir / 'rank_accuracy_scatter_grid.png'}")
    
    # 2. Correlation Bar Chart
    plt.figure(figsize=(12, 6))
    
    # Filter out N/A
    plot_df = summary_df[summary_df["Spearman_R"] != "N/A"].copy()
    plot_df["Spearman_R"] = pd.to_numeric(plot_df["Spearman_R"])
    
    sns.barplot(data=plot_df, x="Workload", y="Spearman_R", hue="Tensor")
    plt.title("Spearman Rank Correlation by Workload")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(-1.1, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(figures_dir / "rank_accuracy_correlation_bar.png", dpi=300)
    print(f"Saved correlation bar chart to {figures_dir / 'rank_accuracy_correlation_bar.png'}")

    # 3. Input Only Correlation Bar Chart
    plt.figure(figsize=(10, 6))
    
    input_df = plot_df[plot_df["Tensor"] == "input"].copy()
    if not input_df.empty:
        sns.barplot(data=input_df, x="Workload", y="Spearman_R", color="skyblue")
        plt.title("Input Tensor: Spearman Rank Correlation by Workload")
        plt.xticks(rotation=45, ha='right')
        plt.ylim(-1.1, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(figures_dir / "rank_accuracy_correlation_input.png", dpi=300)
        print(f"Saved input correlation bar chart to {figures_dir / 'rank_accuracy_correlation_input.png'}")

if __name__ == "__main__":
    plot_results()
