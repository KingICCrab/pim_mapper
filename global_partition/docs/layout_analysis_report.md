# Neural Network Data Layout Strategy Analysis Report

## Executive Summary

This analysis compares three data layout handling strategies for neural network accelerators:

1. **Propagation**: Keep native layout throughout, let dataflow adapt to suboptimal access patterns
2. **Adaptation**: Transform layout at boundaries where dataflow preference changes
3. **Transformation**: Insert explicit transform operators at strategic positions

## Key Findings

### 1. Strategy Selection is Pattern-Dependent

| Preference Pattern | Best Strategy | Reason |
|-------------------|---------------|--------|
| Uniform (NCHW or NHWC) | Propagation | No transforms needed, all sequential access |
| Alternating | Propagation | Transform cost > strided access cost |
| Block-based (every 3 layers) | **Adaptation** | Fewer transforms, amortized cost |

### 2. Adaptation Wins When Transform Count is Low

The "block_3" pattern (layout preference changes every 3 layers) consistently shows
**Adaptation** as the best strategy across all networks and DRAM types:

| Network | DRAM | Adaptation Latency | Propagation (Best) | Speedup |
|---------|------|-------------------|-------------------|--------|
| ResNet-50 | DDR4 | 6.91 ms | 12.76 ms | 1.85x |
| ResNet-50 | HBM2 | 10.61 ms | 17.60 ms | 1.66x |
| VGG-16 | DDR4 | 71.07 ms | 90.23 ms | 1.27x |
| VGG-16 | HBM2 | 87.67 ms | 121.79 ms | 1.39x |
| MobileNet | DDR4 | 7.58 ms | 14.49 ms | 1.91x |
| MobileNet | HBM2 | 9.93 ms | 18.68 ms | 1.88x |


### 3. HBM vs DDR4 Trade-offs

- **HBM2** (256 GB/s, 2KB row buffer): 
  - Higher bandwidth reduces sequential access time
  - Smaller row buffer increases transform cost (more row activations)
  
- **DDR4** (19.2 GB/s, 8KB row buffer):
  - Lower bandwidth but larger row buffer
  - Better for strided access patterns

### 4. Operator Fusion Impact

When consecutive operators are fused (e.g., bottleneck blocks in ResNet):
- **76.7%** reduction in transform overhead (ResNet-50 example)
- Transform cost is only paid at fusion boundaries

## Detailed Analysis

### ResNet-50 (DDR4)

| Pattern | Prop(NCHW) | Prop(NHWC) | Adaptation | Transforms | Best |
|---------|------------|------------|------------|------------|------|
| uniform_nchw | 0.28ms | 21.83ms | 0.28ms | 0 | prop_nchw |
| uniform_nhwc | 40.91ms | 0.28ms | 3.30ms | 1 | prop_nhwc |
| alternating | 18.75ms | 11.14ms | 25.33ms | 11 | prop_nhwc |
| random_like | 20.78ms | 13.04ms | 15.93ms | 8 | prop_nhwc |
| block_3 | 12.76ms | 12.87ms | 6.91ms | 3 | adaptation |

### ResNet-50 (HBM2)

| Pattern | Prop(NCHW) | Prop(NHWC) | Adaptation | Transforms | Best |
|---------|------------|------------|------------|------------|------|
| uniform_nchw | 0.06ms | 31.63ms | 0.06ms | 0 | prop_nchw |
| uniform_nhwc | 49.47ms | 0.06ms | 3.58ms | 1 | prop_nhwc |
| alternating | 23.78ms | 15.84ms | 33.21ms | 11 | prop_nhwc |
| random_like | 22.94ms | 20.06ms | 22.65ms | 8 | prop_nhwc |
| block_3 | 18.59ms | 17.60ms | 10.61ms | 3 | adaptation |

### VGG-16 (DDR4)

| Pattern | Prop(NCHW) | Prop(NHWC) | Adaptation | Transforms | Best |
|---------|------------|------------|------------|------------|------|
| uniform_nchw | 1.95ms | 147.47ms | 1.95ms | 0 | prop_nchw |
| uniform_nhwc | 355.27ms | 1.95ms | 4.09ms | 1 | prop_nhwc |
| alternating | 251.94ms | 48.17ms | 202.22ms | 12 | prop_nhwc |
| random_like | 192.89ms | 75.06ms | 177.64ms | 9 | prop_nhwc |
| block_3 | 132.19ms | 90.23ms | 71.07ms | 4 | adaptation |

### VGG-16 (HBM2)

| Pattern | Prop(NCHW) | Prop(NHWC) | Adaptation | Transforms | Best |
|---------|------------|------------|------------|------------|------|
| uniform_nchw | 0.39ms | 211.61ms | 0.39ms | 0 | prop_nchw |
| uniform_nhwc | 377.37ms | 0.39ms | 2.53ms | 1 | prop_nhwc |
| alternating | 264.34ms | 72.69ms | 246.78ms | 12 | prop_nhwc |
| random_like | 199.76ms | 119.02ms | 200.30ms | 9 | prop_nhwc |
| block_3 | 140.79ms | 121.79ms | 87.67ms | 4 | adaptation |

### MobileNet (DDR4)

| Pattern | Prop(NCHW) | Prop(NHWC) | Adaptation | Transforms | Best |
|---------|------------|------------|------------|------------|------|
| uniform_nchw | 0.35ms | 25.70ms | 0.35ms | 0 | prop_nchw |
| uniform_nhwc | 59.07ms | 0.35ms | 2.49ms | 1 | prop_nhwc |
| alternating | 40.63ms | 8.29ms | 30.63ms | 8 | prop_nhwc |
| random_like | 29.60ms | 13.46ms | 32.62ms | 6 | prop_nhwc |
| block_3 | 25.18ms | 14.49ms | 7.58ms | 2 | adaptation |

### MobileNet (HBM2)

| Pattern | Prop(NCHW) | Prop(NHWC) | Adaptation | Transforms | Best |
|---------|------------|------------|------------|------------|------|
| uniform_nchw | 0.07ms | 34.82ms | 0.07ms | 0 | prop_nchw |
| uniform_nhwc | 65.78ms | 0.07ms | 2.21ms | 1 | prop_nhwc |
| alternating | 45.00ms | 10.27ms | 39.21ms | 8 | prop_nhwc |
| random_like | 32.36ms | 18.69ms | 38.50ms | 6 | prop_nhwc |
| block_3 | 29.55ms | 18.68ms | 9.93ms | 2 | adaptation |


## Recommendations

### For Accelerator Designers

1. **Support efficient layout transforms in hardware**
   - A dedicated layout transformation unit can reduce overhead by 50%+
   - Consider on-chip transform during DMA operations

2. **Minimize dataflow preference changes**
   - Design dataflows with consistent layout preferences
   - The "block_3" pattern shows adaptation can win when changes are infrequent

3. **Enable operator fusion**
   - Fusing consecutive operations eliminates intermediate transforms
   - 76.7% overhead reduction demonstrated in ResNet-50

### For Compiler/Runtime Developers

1. **Implement ILP-based layout optimization**
   - Formulate as graph optimization problem
   - Find optimal transform insertion points

2. **Consider memory hierarchy**
   - Transform cost varies with DRAM type
   - HBM benefits from higher bandwidth but smaller row buffer

3. **Profile-guided optimization**
   - Actual costs depend on specific hardware
   - Use profiling to calibrate analytical models

## Methodology

### DRAM Access Model

- **Sequential Access**: Row buffer hit rate approaches 100%
- **Strided Access**: Row buffer hit rate depends on stride vs row buffer size
- **Transform Cost**: Read (sequential) + Write (strided) 

### Cost Calculation

```
Transform_Cost = Read_Time + Write_Time
Read_Time = Data_Size / Bandwidth + Row_Activation_Cost
Write_Time = Data_Size / Bandwidth + (Num_Row_Misses × Row_Miss_Latency)
```

Row miss rate for write depends on output stride relative to row buffer size.

### DRAM Configurations

| Type | Bandwidth | Row Buffer | tRCD | tRP | tCAS |
|------|-----------|------------|------|-----|------|
| DDR4-2400 | 19.2 GB/s | 8 KB | 13.75ns | 13.75ns | 13.75ns |
| HBM2 | 256 GB/s | 2 KB | 14ns | 14ns | 14ns |

## Appendix: Source Code

Analysis performed using Python scripts:
- `dram_layout_analysis.py` - Core DRAM access model
- `extended_layout_analysis.py` - Network analysis and comparison
- `layout_report.py` - Report generation (this file)
