# PIM Optimizer

PIM (Processing-In-Memory) Dataflow Optimization via Integer Linear Programming.

## Overview

This project provides an ILP-based optimizer for finding optimal dataflow mappings on PIM accelerator architectures. It models:

- **Memory hierarchy**: PE local buffer → Global buffer → Row buffer → DRAM
- **Data layout optimization**: Sequential vs Row-Aligned layouts
- **Row activation modeling**: Accurate DRAM row activation cost estimation
- **Crossing tiles analysis**: Input tile crossing ratio based on GCD periodic analysis

## Project Structure

```
pim_optimizer/
├── src/pim_optimizer/
│   ├── __init__.py
│   ├── arch/                    # Architecture definition
│   │   ├── __init__.py
│   │   ├── pim_arch.py          # PIMArchitecture class
│   │   └── memory.py            # Memory level definitions
│   ├── workload/                # Workload definition
│   │   ├── __init__.py
│   │   └── conv.py              # Convolution workload
│   ├── model/                   # ILP model components
│   │   ├── __init__.py
│   │   ├── variables.py         # Decision variables
│   │   ├── constraints.py       # Constraint builders
│   │   ├── expressions.py       # Expression builders
│   │   └── objective.py         # Objective function
│   ├── row_activation/          # Row activation modeling
│   │   ├── __init__.py
│   │   ├── crossing.py          # Crossing ratio calculation
│   │   └── model.py             # Row activation ILP model
│   ├── optimizer.py             # Main optimizer entry
│   ├── mapping.py               # Mapping result class
│   └── utils.py                 # Utility functions
├── examples/                    # Usage examples
│   ├── simple_conv.py
│   └── configs/
│       └── arch.yaml
└── tests/                       # Unit tests
    └── test_optimizer.py
```

## Installation

```bash
# From project root
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

## Requirements

- Python >= 3.10
- Gurobi Optimizer (with valid license)
- NumPy
- PyYAML

## Quick Start

```python
from pim_optimizer import PIMArchitecture, ConvWorkload, optimize

# Create architecture
arch = PIMArchitecture()

# Create workload
workload = ConvWorkload(
    R=3, S=3, P=56, Q=56,
    C=64, K=128, N=1,
    stride=(1, 1),
    dilation=(1, 1),
)

# Run optimization
mapping = optimize(arch, [workload])

# Print results
mapping.print_summary()
```

## Key Features

### 1. Accurate Row Activation Model

Uses the `xj` variable method from the original Lemon project to precisely track data reuse patterns.

### 2. Input Crossing Ratio Analysis

Based on GCD periodic analysis for accurate crossing tile estimation:

```python
# Crossing ratio formula
g = gcd(step, block_h)
period = block_h // g
cross_count = period - ceil((block_h - tile_h + 1) / g)
crossing_ratio = cross_count / period
```

### 3. Data Layout Optimization

Supports two layout modes:
- **Sequential**: Tiles stored contiguously
- **Row-Aligned**: Tiles aligned to DRAM row boundaries

## License

MIT License
