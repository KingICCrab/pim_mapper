# Golden Model for PIM Optimizer Verification

This module provides a verification framework to validate:
1. **Cost Model Correctness**: Verify ILP cost formulas match analytical references
2. **ILP Optimality**: Verify ILP finds globally optimal solutions
3. **Row Activation Model**: Verify DRAM row buffer crossing calculations

## Reference Projects

The golden model is based on analysis of these reference projects:

| Project | Language | Approach | Key Insight |
|---------|----------|----------|-------------|
| OptiPIM | C++ | Gurobi MILP | GCD LUT for Input access patterns |
| Interstellar | Python | Exhaustive | Access count = ∏(irrelevant loop factors) |
| nn_dataflow | Python | Solver | Memory hierarchy cost model |
| UniNDP | Python | Simulation | Instruction-level counting |

## Directory Structure

```
golden_model/
├── __init__.py           # Main exports
├── README.md             # This file
├── analytical/           # Analytical cost formulas
│   ├── cost_formulas.py  # Memory access formulas (from Interstellar)
│   └── row_activation.py # GCD-based crossing ratio (from OptiPIM)
├── exhaustive/           # Exhaustive search
│   ├── brute_force.py    # Full enumeration for small problems
│   └── sampler.py        # Sampling for large spaces
├── comparison/           # Verification tools
│   ├── compare.py        # Core comparison logic
│   └── report.py         # Report generation
└── examples/             # Usage examples
    └── verify_ilp.py     # Complete verification example
```

## Key Formulas

### Access Count (from Interstellar)

For each data type, access count = product of **irrelevant** loop factors at outer levels:

| Data Type | Irrelevant Dimensions |
|-----------|----------------------|
| Input     | K (output channels) |
| Weight    | N, P, Q (batch, spatial) |
| Output    | R, S, C (kernel, input channels) |

```python
# Input access count (K is irrelevant)
input_access = K_outer_factor
# where K_outer = K_bound / K_tile

# Weight access count (N, P, Q are irrelevant)
weight_access = N_outer × P_outer × Q_outer

# Output access count (R, S, C are irrelevant)
output_access = R_outer × S_outer × C_outer
```

### Memory Reads

```python
total_reads = tile_size × access_count
```

### Crossing Ratio (from OptiPIM)

Uses GCD-based periodic analysis:

```python
# Sequential access
period = row_size / gcd(tile_size, row_size)
crossing_ratio = (period - threshold) / period

# Sliding window (for Input)
for k in range(period):
    if (k * step) % block_h + tile_h > block_h:
        crossing_count += 1
crossing_ratio = crossing_count / period
```

## Usage

### Basic Verification

```python
from golden_model import (
    LoopBounds, TileFactors,
    compute_analytical_memory_reads,
    verify_cost_model,
    compare_with_ilp,
)

# Define problem
bounds = LoopBounds(N=4, C=8, K=8, P=4, Q=4, R=3, S=3)
factors = TileFactors(N=2, C=4, K=4, P=2, Q=2, R=1, S=1)
loop_order = ['N', 'K', 'C', 'P', 'Q', 'R', 'S']

# Your ILP results
ilp_results = {
    'input_reads': 1024,
    'weight_reads': 512,
    'output_reads': 256,
    'latency': 10000,
}

# Verify
result = compare_with_ilp(
    test_name="test_conv",
    bounds=bounds,
    ilp_factors=factors,
    ilp_loop_order=loop_order,
    ilp_results=ilp_results,
)

print(f"Passed: {result.overall_passed}")
```

### Optimality Verification

```python
from golden_model import (
    find_optimal_exhaustive,
    verify_ilp_optimality,
    Mapping,
)

# Find true optimal by exhaustive search
optimal, all_mappings = find_optimal_exhaustive(
    bounds,
    objective='latency',
    verbose=True,
)

# Verify your ILP solution
ilp_mapping = Mapping(l1_factors=your_factors, loop_order=your_order)
is_optimal, details = verify_ilp_optimality(ilp_mapping, bounds, 'latency')
```

### Row Activation Verification

```python
from golden_model import (
    RowBufferConfig,
    compute_crossing_ratio_sequential,
    verify_row_activation_model,
)

# Configure row buffer
row_config = RowBufferConfig(
    row_size_bytes=1024,
    element_bytes=1,
    num_banks=8,
    activation_latency=25.0,
)

# Verify
passed, details = verify_row_activation_model(
    ilp_row_acts=your_ilp_row_activations,
    memory_reads=your_memory_reads,
    tile_info={'tile_bytes': 256},
    row_config=row_config,
    datatype='weight',
)
```

## Running Examples

```bash
cd golden_model
python examples/verify_ilp.py
```

## Verification Workflow

```
┌─────────────────┐
│  Your ILP Model │
│  (pim_optimizer)│
└────────┬────────┘
         │ Extract solution
         ▼
┌─────────────────┐     ┌──────────────────┐
│ Cost Model      │────▶│ Analytical       │
│ Verification    │     │ Formulas         │
└────────┬────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│ Optimality      │────▶│ Exhaustive       │
│ Verification    │     │ Search           │
└────────┬────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│ Row Activation  │────▶│ GCD Analysis     │
│ Verification    │     │                  │
└────────┬────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐
│    Report       │
└─────────────────┘
```

## Limitations

1. **Exhaustive search** is only feasible for small problems (≤10K mappings)
2. **Loop order enumeration** multiplies search space by 5040 (7!)
3. **Two-level tiling** exponentially increases search space

For large problems, use sampling strategies:
- `sample_mappings_random()`: Random sampling
- `sample_mappings_latin_hypercube()`: Better coverage
- `sample_boundary_cases()`: Extreme points
