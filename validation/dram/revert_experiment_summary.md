
# Validation Report (Reuse Penalty Fix Reverted)

## Experiment Setup
- **Workload**: Medium (Conv Layer)
- **Configuration**: 
    - `outer_irr_product` fix: **Applied**
    - `row_aligned` fix: **Applied**
    - `reuse_penalty` (lower levels) fix: **REVERTED** (Current State)

## Results (Medium)

| Tensor | ILP (Predicted) | Trace (Actual) | Ratio | Error |
|:-------|:----------------|:---------------|:------|:------|
| Input  | 788.00          | 1776           | 2.25x | +125% |
| Weight | 9.00            | 777            | 86.3x | +8533%|
| Output | 25.00           | 400            | 16.0x | +1500%|

## Analysis
The huge discrepancy in Weight (9 vs 777) and Output (25 vs 400) confirms that **ignoring lower-level loops (Level 0-2) in reuse penalty calculation leads to massive underestimation**.

- **ILP Prediction (9)**: Only counts row switches caused by Level 3 (DRAM) loops.
- **Trace Reality (777)**: The hardware executes Level 2 (Row Buffer) loops *inside* the Level 3 loops. If a tile crosses a row boundary, the Level 2 loops cause repeated switching between the two rows (thrashing).

## Conclusion
The `reuse_penalty` MUST account for lower-level loops to match the hardware behavior modeled by the trace generator.
