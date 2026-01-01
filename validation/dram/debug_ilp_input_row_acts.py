"""
Debug script to understand why ILP predicts 2392 row activations for Input.

Based on row_activation.py, ILP calculates Input row activations as:
    total = row_acts_aligned + block_crossing_acts

Where:
    row_acts_aligned = Π_{j ∈ dims} bound_j^{xj}  (all dims, product of DRAM loop factors)
    block_crossing_acts = (H_crossing + W_crossing) × reuse_penalty
    reuse_penalty = Π_{j ∈ irrelevant} bound_j^{xj}

For Input, irrelevant dims = [K, N] (K and N don't affect Input)
Relevant dims for aligned = all 7 dims: R, S, P, Q, C, K, N

Let's verify this formula with the actual mapping.
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer.model.row_activation import compute_input_block_crossing_count
import math

# Dimension indices
DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6
DIM_NAMES = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']

def debug_ilp_input():
    """Debug ILP Input row activation calculation."""
    
    # Create workload and get mapping
    workload = ConvWorkload(name="ResNet-L1", R=7, S=7, P=56, Q=56, C=3, K=64, N=1)
    optimizer = PIMOptimizer()
    result = optimizer.optimize([workload], objective="latency")
    mapping = result.mappings[0]
    
    print("=" * 80)
    print("ILP Input Row Activation Debug for ResNet-L1")
    print("=" * 80)
    
    # Print workload info
    print(f"\nWorkload bounds: R={workload.bounds[0]}, S={workload.bounds[1]}, "
          f"P={workload.bounds[2]}, Q={workload.bounds[3]}, "
          f"C={workload.bounds[4]}, K={workload.bounds[5]}, N={workload.bounds[6]}")
    
    # Print ILP result
    ilp_input = mapping.metrics.get('row_activations_input', 0)
    print(f"\nILP row_activations_input = {ilp_input}")
    
    # Extract DRAM factors (Level 3)
    dram_factors = {d: 1 for d in range(7)}
    if 3 in mapping.loop_bounds:
        for key in ['spatial', 'temporal']:
            if key in mapping.loop_bounds[3]:
                for d, bound in mapping.loop_bounds[3][key].items():
                    dram_factors[d] *= bound
    
    print(f"\n--- DRAM Loop Factors (Level 3) ---")
    for d in range(7):
        print(f"  {DIM_NAMES[d]}: {dram_factors[d]}")
    
    # Extract Level 2 factors
    level2_factors = {d: 1 for d in range(7)}
    if 2 in mapping.loop_bounds:
        for key in ['spatial', 'temporal']:
            if key in mapping.loop_bounds[2]:
                for d, bound in mapping.loop_bounds[2][key].items():
                    level2_factors[d] *= bound
    
    print(f"\n--- Level 2 Factors ---")
    for d in range(7):
        if level2_factors[d] > 1:
            print(f"  {DIM_NAMES[d]}: {level2_factors[d]}")
    
    # Calculate row_acts_aligned = product of DRAM factors over all dims
    print(f"\n{'='*80}")
    print("Step 1: Calculate row_acts_aligned")
    print("      = Π_{j ∈ all_dims} dram_factor_j")
    print("{'='*80}")
    
    row_acts_aligned = 1
    for d in range(7):
        row_acts_aligned *= dram_factors[d]
    
    print(f"\n  row_acts_aligned = K × C × P × Q (DRAM factors only)")
    print(f"                   = {dram_factors[DIM_K]} × {dram_factors[DIM_C]} × "
          f"{dram_factors[DIM_P]} × {dram_factors[DIM_Q]}")
    print(f"                   = {dram_factors[DIM_K] * dram_factors[DIM_C] * dram_factors[DIM_P] * dram_factors[DIM_Q]}")
    print(f"\n  Full product (R, S, P, Q, C, K, N):")
    product_str = " × ".join([f"{dram_factors[d]}" for d in range(7)])
    print(f"                   = {product_str}")
    print(f"                   = {row_acts_aligned}")
    
    # Calculate reuse_penalty = product of irrelevant dims (K, N for Input)
    print(f"\n{'='*80}")
    print("Step 2: Calculate reuse_penalty (for Input: K, N irrelevant)")
    print("      = Π_{j ∈ [K, N]} dram_factor_j")
    print("{'='*80}")
    
    irrelevant_dims = [DIM_K, DIM_N]  # K and N don't affect Input
    reuse_penalty = 1
    for d in irrelevant_dims:
        reuse_penalty *= dram_factors[d]
    
    print(f"\n  reuse_penalty = K_factor × N_factor")
    print(f"                = {dram_factors[DIM_K]} × {dram_factors[DIM_N]}")
    print(f"                = {reuse_penalty}")
    
    # Calculate block crossing
    print(f"\n{'='*80}")
    print("Step 3: Calculate Input Block Crossing")
    print("{'='*80}")
    
    # Get buffer tile dimensions
    buffer_tile = {d: 1 for d in range(7)}
    for level in [0, 1]:
        if level not in mapping.loop_bounds:
            continue
        level_bounds = mapping.loop_bounds[level]
        if level == 0:
            for key in ['H', 'W', 'Internal', 'temporal']:
                if key in level_bounds:
                    for d, bound in level_bounds[key].items():
                        buffer_tile[d] *= bound
        else:
            for key in ['spatial', 'temporal']:
                if key in level_bounds:
                    for d, bound in level_bounds[key].items():
                        buffer_tile[d] *= bound
    
    stride_h, stride_w = 1, 1
    dilation_h, dilation_w = 1, 1
    
    P_per_tile = buffer_tile[DIM_P]
    Q_per_tile = buffer_tile[DIM_Q]
    R_per_tile = buffer_tile[DIM_R]
    S_per_tile = buffer_tile[DIM_S]
    
    H_per_tile = (P_per_tile - 1) * stride_h + R_per_tile
    W_per_tile = (Q_per_tile - 1) * stride_w + S_per_tile
    
    print(f"\n  Buffer tile (Level 0+1):")
    print(f"    P_per_tile={P_per_tile}, Q_per_tile={Q_per_tile}")
    print(f"    R_per_tile={R_per_tile}, S_per_tile={S_per_tile}")
    print(f"    H_per_tile={H_per_tile}, W_per_tile={W_per_tile}")
    
    # Block sizes
    block_h = mapping.tile_info.get('block_h', 31)
    block_w = mapping.tile_info.get('block_w', 31)
    print(f"\n  Block sizes: block_h={block_h}, block_w={block_w}")
    
    # H direction crossing
    print(f"\n  --- H Direction (P, R) ---")
    P_factor = dram_factors[DIM_P]  # Number of tiles in P direction
    R_factor = level2_factors[DIM_R]  # R loop at level 2
    
    # tile_h in input space
    tile_h = (P_per_tile - 1) * stride_h + R_per_tile * dilation_h
    step_h = P_per_tile * stride_h  # step between tiles
    
    print(f"    P_factor (DRAM)={P_factor}, R_factor (L2)={R_factor}")
    print(f"    tile_h={tile_h}, step_h={step_h}")
    
    # ILP calls compute_input_block_crossing_count with num_tiles=P_factor
    h_crossing, h_total = compute_input_block_crossing_count(
        block_h=block_h,
        tile_h=tile_h,
        step=step_h,
        tile_s=R_per_tile,
        total_S=workload.bounds[DIM_R],  # Total R
        dilation=dilation_h,
        num_tiles=P_factor,  # ILP uses DRAM factor as num_tiles
    )
    print(f"    H_crossing = {h_crossing} (with num_tiles={P_factor})")
    
    # W direction crossing
    print(f"\n  --- W Direction (Q, S) ---")
    Q_factor = dram_factors[DIM_Q]  # Number of tiles in Q direction
    S_factor = level2_factors.get(DIM_S, 1)
    
    tile_w = (Q_per_tile - 1) * stride_w + S_per_tile * dilation_w
    step_w = Q_per_tile * stride_w
    
    print(f"    Q_factor (DRAM)={Q_factor}, S_factor (L2)={S_factor}")
    print(f"    tile_w={tile_w}, step_w={step_w}")
    
    w_crossing, w_total = compute_input_block_crossing_count(
        block_h=block_w,
        tile_h=tile_w,
        step=step_w,
        tile_s=S_per_tile,
        total_S=workload.bounds[DIM_S],  # Total S
        dilation=dilation_w,
        num_tiles=Q_factor,  # ILP uses DRAM factor as num_tiles
    )
    print(f"    W_crossing = {w_crossing} (with num_tiles={Q_factor})")
    
    # Total block crossing (ILP formula)
    print(f"\n{'='*80}")
    print("Step 4: Final ILP Calculation")
    print("{'='*80}")
    
    block_crossing_count = h_crossing + w_crossing
    block_crossing_acts = block_crossing_count * 2 * reuse_penalty
    
    print(f"\n  block_crossing_count = H_crossing + W_crossing")
    print(f"                       = {h_crossing} + {w_crossing}")
    print(f"                       = {block_crossing_count}")
    
    print(f"\n  block_crossing_acts = crossing_count × 2 × reuse_penalty")
    print(f"                      = {block_crossing_count} × 2 × {reuse_penalty}")
    print(f"                      = {block_crossing_acts}")
    
    # Total (for row_aligned mode)
    total_ilp = row_acts_aligned + block_crossing_acts
    
    print(f"\n  total_row_acts = row_acts_aligned + block_crossing_acts")
    print(f"                 = {row_acts_aligned} + {block_crossing_acts}")
    print(f"                 = {total_ilp}")
    
    print(f"\n{'='*80}")
    print("Comparison")
    print("{'='*80}")
    print(f"\n  ILP Predicted: {ilp_input}")
    print(f"  Manual Calc:   {total_ilp}")
    print(f"  Trace Result:  5376 (after fix)")
    
    # Analyze discrepancy
    print(f"\n{'='*80}")
    print("Discrepancy Analysis")
    print("{'='*80}")
    
    print(f"""
  ILP 公式:
    total = row_acts_aligned + block_crossing_acts
          = {row_acts_aligned} + {block_crossing_acts}
          = {total_ilp}
    
  问题: ILP 的 row_acts_aligned = {row_acts_aligned}
        但这只考虑了 DRAM 层(Level 3)的因子!
        
  实际应该考虑:
    - Level 3 (DRAM): K={dram_factors[DIM_K]}, C={dram_factors[DIM_C]}, P={dram_factors[DIM_P]}, Q={dram_factors[DIM_Q]}
    - Level 2: R={level2_factors[DIM_R]}
    
  正确的 row_acts_aligned 应该是 DRAM + Level 2 的乘积:
    = K × C × P × Q × R
    = {dram_factors[DIM_K]} × {dram_factors[DIM_C]} × {dram_factors[DIM_P]} × {dram_factors[DIM_Q]} × {level2_factors[DIM_R]}
    = {dram_factors[DIM_K] * dram_factors[DIM_C] * dram_factors[DIM_P] * dram_factors[DIM_Q] * level2_factors[DIM_R]}
""")
    
    # Let's compute what Trace does
    print(f"\n{'='*80}")
    print("Trace 的计算 (修复后)")
    print("{'='*80}")
    
    # Trace iterates: K_l3 × C_l3 × P_l3 × Q_l3 × R_l2 = 4 × 3 × 28 × 7 × 7 = 16464 tiles
    total_tile_iters = (dram_factors[DIM_K] * dram_factors[DIM_C] * 
                        dram_factors[DIM_P] * dram_factors[DIM_Q] * level2_factors[DIM_R])
    print(f"  Total tile iterations = K×C×P×Q×R = {total_tile_iters}")
    
    # Each (p_tile, q_tile, r_tile) combination accesses a unique input region
    # For row_aligned mode, row switches = number of unique block positions
    # With R sliding, the block positions change as R slides
    
    print(f"""
  Trace 分析:
    - 总迭代次数: {total_tile_iters}
    - 但由于 K 和 N 不影响 Input,实际 unique 访问 = C×P×Q×R
    - = {dram_factors[DIM_C]} × {dram_factors[DIM_P]} × {dram_factors[DIM_Q]} × {level2_factors[DIM_R]}
    - = {dram_factors[DIM_C] * dram_factors[DIM_P] * dram_factors[DIM_Q] * level2_factors[DIM_R]}
    
  Block Crossing (R滑动):
    - 196 个 (p,q) 组合 × 7 个 R 位置
    - 其中一部分会 crossing block boundary
    - 我们之前计算: 5376 = 196 × 7 × 4 (每个都 crossing 约 4 次?)
    
  实际 Trace 结果: 5376
""")


if __name__ == "__main__":
    debug_ilp_input()
