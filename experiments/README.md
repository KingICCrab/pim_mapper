# PIM Optimizer Experimental Evaluation

This directory contains experimental scripts for validating and evaluating the ILP-based PIM optimizer.

## Files

| File | Description |
|------|-------------|
| `experimental_evaluation.py` | **Main experiment script** - ILP optimization results |
| `row_activation_validation.py` | Row activation ILP vs Trace validation |
| `plot_results.py` | Generate publication-quality figures |
| `results/` | Experiment outputs (CSV, JSON, LaTeX tables) |
| `figures/` | Generated plots (PDF and PNG) |

## Quick Start

```bash
cd /Users/haochenzhao/Projects/pim_optimizer
source .venv/bin/activate

# Run main experiment (CNN workloads)
python experiments/experimental_evaluation.py --workloads cnn

# Run batch size study
python experiments/experimental_evaluation.py --workloads batch

# Run all workloads
python experiments/experimental_evaluation.py --workloads all

# Generate plots
python experiments/plot_results.py
```

## Experiment Design

### 1. ILP Optimization Results (Main Experiment)

**Script:** `experimental_evaluation.py`

**Metrics:**
- Row Activations (per tensor: Input, Weight, Output)
- ILP Solver Time
- Optimization Objective Value

**Workloads:**

| Category | Layers | Description |
|----------|--------|-------------|
| ResNet-18 | Conv1-Conv5 | 7×7 and 3×3 convolutions |
| VGG-16 | Conv1-Conv3 | 3×3 convolutions, large spatial |
| MobileNet | PW1-PW3 | 1×1 pointwise convolutions |

**Output:** 
- `optimization_results_*.tex` - LaTeX table
- `optimization_results_*.csv` - CSV data
- `optimization_results_*.json` - Full JSON data

### 2. Batch Size Scalability

**Script:** `experimental_evaluation.py --workloads batch`

Studies how row activations scale with batch size (N=1,2,4,8,16).

**Output:**
- `batch_study_*.tex` - LaTeX table

### 3. ILP Accuracy Validation (Optional)

**Script:** `row_activation_validation.py`

Validates ILP predictions against trace-based counting. Note: Large workloads may have higher error due to trace generator limitations.

```bash
# Quick test (small workloads)
python experiments/row_activation_validation.py --quick
```

## Output Files

### LaTeX Tables

```latex
% Include in your paper:
\input{experiments/results/optimization_results_TIMESTAMP.tex}
```

### CSV Data

Columns:
- `name`, `macs`, `N`, `K`, `C`, `P`, `Q`, `R`, `S`
- `row_acts_input`, `row_acts_weight`, `row_acts_output`, `row_acts_total`
- `solve_time`, `objective`

### JSON Data

Complete experiment data with all parameters.

## Example Results

```
Workload            MACs    Input RA  Weight RA  Output RA   Total RA  Time(s)
ResNet-Conv1    118.0M     37,632         10        784      38,426     0.77
ResNet-Conv2    115.6M     25,088         36      3,136      28,260     0.49
VGG-Conv1        86.7M     18,816     12,546      6,272      37,634     2.23
MobileNet-PW1     6.4M      1,568          2        728       2,298     0.73
```

## Key Observations

1. **Input Tensor Dominates**: Input tensor typically accounts for >90% of row activations
2. **Weight Reuse**: Weight row activations are minimal due to data reuse
3. **Fast Solving**: Average ILP solve time <1s for typical CNN layers
4. **1x1 Convolutions**: MobileNet pointwise layers have much lower row activations

## Architecture Configuration

Default `arch.yaml`:
- Row Buffer: 1024 bytes
- 4 DRAM Banks
- 16384 rows per bank
- PE Array: 128×128
