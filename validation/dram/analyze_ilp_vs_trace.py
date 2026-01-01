"""
深入分析 ILP 和 Trace 的 Input Row Activation 差异

已知信息:
- ILP: 2392 = 2352 (row_acts_aligned) + 40 (block_crossing)
- Trace: 5376

关键差异:
1. ILP 的 row_acts_aligned 只用 Level 3 DRAM factors
2. Trace 实际遍历 Level 3 + Level 2

本脚本将验证 ILP 公式是否正确,以及如何修复
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

def analyze_ilp_vs_trace():
    """深入分析 ILP 和 Trace 的计算逻辑差异."""
    
    # Create workload and get mapping
    workload = ConvWorkload(name="ResNet-L1", R=7, S=7, P=56, Q=56, C=3, K=64, N=1)
    optimizer = PIMOptimizer()
    result = optimizer.optimize([workload], objective="latency")
    mapping = result.mappings[0]
    
    print("=" * 80)
    print("ILP vs Trace: Input Row Activation 深入分析")
    print("=" * 80)
    
    # Extract factors
    dram_factors = {d: 1 for d in range(7)}
    if 3 in mapping.loop_bounds:
        for key in ['spatial', 'temporal']:
            if key in mapping.loop_bounds[3]:
                for d, bound in mapping.loop_bounds[3][key].items():
                    dram_factors[d] *= bound
    
    level2_factors = {d: 1 for d in range(7)}
    if 2 in mapping.loop_bounds:
        for key in ['spatial', 'temporal']:
            if key in mapping.loop_bounds[2]:
                for d, bound in mapping.loop_bounds[2][key].items():
                    level2_factors[d] *= bound
    
    # Buffer tile
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
    
    P_per_tile = buffer_tile[DIM_P]
    Q_per_tile = buffer_tile[DIM_Q]
    R_per_tile = buffer_tile[DIM_R]
    S_per_tile = buffer_tile[DIM_S]
    
    stride_h, stride_w = 1, 1
    dilation_h, dilation_w = 1, 1
    
    H_per_tile = (P_per_tile - 1) * stride_h + R_per_tile
    W_per_tile = (Q_per_tile - 1) * stride_w + S_per_tile
    
    block_h = mapping.tile_info.get('block_h', 31)
    block_w = mapping.tile_info.get('block_w', 31)
    
    print(f"\n配置参数:")
    print(f"  DRAM (L3): K={dram_factors[DIM_K]}, C={dram_factors[DIM_C]}, "
          f"P={dram_factors[DIM_P]}, Q={dram_factors[DIM_Q]}, R={dram_factors[DIM_R]}")
    print(f"  Level 2:   R={level2_factors[DIM_R]}")
    print(f"  Buffer:    P={P_per_tile}, Q={Q_per_tile}, R={R_per_tile}, S={S_per_tile}")
    print(f"  Input tile: H={H_per_tile}, W={W_per_tile}")
    print(f"  Block size: {block_h} × {block_w}")
    
    print("\n" + "=" * 80)
    print("理论分析: row_aligned 模式下的 Row Activation")
    print("=" * 80)
    
    print("""
    在 row_aligned 模式下:
    - 每个数据 block 正好占用一个 DRAM row
    - 因此 "row activation" ≈ "block activation"
    - 每次访问一个新的 block 就需要一次 row activation
    
    Input 的 block 布局 (row_aligned):
    - block_h = 31 (H 方向)
    - block_w = 31 (W 方向)  
    - 每个 block = 31 × 31 = 961 elements
    - 每个 DRAM row = 1024 bytes = 1024 elements (假设 1 byte/element)
    - 因此每个 block 几乎占满一个 row
    
    关键问题: 一次 DRAM tile 访问会触发多少 block?
    """)
    
    # Analysis 1: ILP 的逻辑
    print("\n" + "-" * 60)
    print("方法 A: ILP 的计算 (只看 DRAM Level)")
    print("-" * 60)
    
    print(f"""
    ILP 认为:
    - row_acts_aligned = 所有 DRAM 维度因子的乘积 (L3 层)
    - = K × C × P × Q × R × S × N (DRAM factors)
    - = {dram_factors[DIM_K]} × {dram_factors[DIM_C]} × {dram_factors[DIM_P]} × {dram_factors[DIM_Q]} × 1 × 1 × 1
    - = 2352
    
    问题: ILP 假设每次 DRAM tile iteration 只触发 1 次 row activation
    但实际上, R 在 Level 2 循环, 每次 R 滑动都可能改变访问的 block!
    """)
    
    # Analysis 2: 正确的理解
    print("\n" + "-" * 60)
    print("方法 B: 考虑 Level 2 的 R 循环")
    print("-" * 60)
    
    # P direction blocks
    total_h = 62  # H_in
    num_h_blocks = (total_h + block_h - 1) // block_h  # ceil(62/31) = 2
    
    # Q direction blocks
    total_w = 62  # W_in
    num_w_blocks = (total_w + block_w - 1) // block_w  # ceil(62/31) = 2
    
    print(f"""
    Input 维度: H={total_h}, W={total_w}, C=3
    Block 数量: {num_h_blocks} × {num_w_blocks} × 3 = {num_h_blocks * num_w_blocks * 3} blocks
    
    DRAM 循环结构:
    for k in range(K_l3=4):           # 不影响 Input
      for c in range(C_l3=3):         # C 循环
        for p in range(P_l3=28):      # P tile 循环
          for q in range(Q_l3=7):     # Q tile 循环
            for r in range(R_l2=7):   # R 滑动循环 (Level 2)
              # 访问 Input tile at (c, p_tile, q_tile, r_offset)
    
    每个 (p_tile, q_tile, r_offset) 组合访问的 Input 区域:
    - h_start = p_tile × P_per_tile × stride + r_offset × dilation
             = p_tile × 2 × 1 + r_offset × 1
             = p_tile × 2 + r_offset
    - h_end = h_start + H_per_tile - 1
           = h_start + 2 - 1 = h_start + 1
    
    因此 H 访问范围 [h_start, h_start+1]:
    - r_offset=0: h = [2p, 2p+1]
    - r_offset=1: h = [2p+1, 2p+2]
    - r_offset=2: h = [2p+2, 2p+3]
    - ...
    - r_offset=6: h = [2p+6, 2p+7]
    """)
    
    # 计算每个 (p_tile, q_tile) 在不同 R 下的 block crossing
    print("\n" + "-" * 60)
    print("方法 C: 逐 R 计算 block 访问")
    print("-" * 60)
    
    P_l3 = dram_factors[DIM_P]  # 28
    Q_l3 = dram_factors[DIM_Q]  # 7
    R_l2 = level2_factors[DIM_R]  # 7
    C_l3 = dram_factors[DIM_C]  # 3
    
    total_row_switches = 0
    h_block_visits = {}  # Track (h_block, w_block, c) visits
    
    for c in range(C_l3):
        for p in range(P_l3):
            for q in range(Q_l3):
                prev_blocks = None
                for r in range(R_l2):
                    # Calculate h,w range
                    h_start = p * P_per_tile * stride_h + r * dilation_h
                    h_end = h_start + H_per_tile - 1
                    w_start = q * Q_per_tile * stride_w
                    w_end = w_start + W_per_tile - 1
                    
                    # Which blocks are accessed?
                    h_blocks = set(range(h_start // block_h, (h_end // block_h) + 1))
                    w_blocks = set(range(w_start // block_w, (w_end // block_w) + 1))
                    
                    # Current blocks
                    curr_blocks = set()
                    for hb in h_blocks:
                        for wb in w_blocks:
                            curr_blocks.add((hb, wb, c))
                    
                    if prev_blocks is None:
                        # First R: count all blocks
                        total_row_switches += len(curr_blocks)
                    else:
                        # Subsequent R: count new blocks
                        new_blocks = curr_blocks - prev_blocks
                        total_row_switches += len(new_blocks)
                    
                    prev_blocks = curr_blocks
    
    print(f"  逐 R 计算的 row switches (考虑 block 复用): {total_row_switches}")
    
    # Method D: 不考虑 R 之间的复用
    print("\n" + "-" * 60)
    print("方法 D: 每个 R 都重新计算 (不考虑复用)")
    print("-" * 60)
    
    total_without_reuse = 0
    for c in range(C_l3):
        for p in range(P_l3):
            for q in range(Q_l3):
                for r in range(R_l2):
                    h_start = p * P_per_tile * stride_h + r * dilation_h
                    h_end = h_start + H_per_tile - 1
                    w_start = q * Q_per_tile * stride_w
                    w_end = w_start + W_per_tile - 1
                    
                    h_blocks = set(range(h_start // block_h, (h_end // block_h) + 1))
                    w_blocks = set(range(w_start // block_w, (w_end // block_w) + 1))
                    
                    num_blocks = len(h_blocks) * len(w_blocks)
                    total_without_reuse += num_blocks
    
    print(f"  每个 (c,p,q,r) 都计算 blocks: {total_without_reuse}")
    
    # Consider K multiplication
    K_l3 = dram_factors[DIM_K]
    total_with_k = total_without_reuse * K_l3
    total_c_with_k = total_row_switches * K_l3
    
    print(f"\n  乘以 K (每个 K 都重新读 Input,但 Input 不变):")
    print(f"    方法 C × K = {total_row_switches} × {K_l3} = {total_c_with_k}")
    print(f"    方法 D × K = {total_without_reuse} × {K_l3} = {total_with_k}")
    
    print("\n" + "=" * 80)
    print("结果对比")
    print("=" * 80)
    
    print(f"""
    ILP 结果:           2392  (= 2352 aligned + 40 crossing)
    方法 C × K:         {total_c_with_k}
    方法 D × K:         {total_with_k}
    Trace 结果 (修复后): 5376
    
    分析:
    - ILP: 只考虑 DRAM Level (L3) 的迭代,忽略了 Level 2 的 R 循环
    - 方法 C: 考虑 R 之间的 block 复用
    - 方法 D: 不考虑复用,与 Trace 更接近
    """)
    
    # 计算 Trace 的逻辑
    print("\n" + "=" * 80)
    print("Trace 的计算逻辑分析")
    print("=" * 80)
    
    print(f"""
    Trace 计算 row activations = row switches
    - 每次访问不同的 row,就计一次 switch
    - Trace 迭代所有 Level 2+3 循环
    
    但是,Trace 可能计算的是:
    - 每个 tile 访问的所有 rows (包括 crossing 时的多 rows)
    - 而不是 "unique block" 访问次数
    
    让我们验证 5376 的来源:
    - C × P × Q × R × avg_blocks_per_tile
    - = 3 × 28 × 7 × 7 × avg
    - = 4116 × avg
    
    如果 avg ≈ 1.3,则 4116 × 1.3 ≈ 5376
    
    这说明平均每个 tile 访问 ~1.3 个 rows (因为有些 tile crossing block)
    """)
    
    # 详细验证
    print("\n验证: 计算 5376 的组成")
    
    # Count detailed row switches
    total_detail = 0
    crossing_h = 0
    crossing_w = 0
    crossing_both = 0
    non_crossing = 0
    
    for c in range(C_l3):
        for p in range(P_l3):
            for q in range(Q_l3):
                for r in range(R_l2):
                    h_start = p * P_per_tile * stride_h + r * dilation_h
                    h_end = h_start + H_per_tile - 1
                    w_start = q * Q_per_tile * stride_w
                    w_end = w_start + W_per_tile - 1
                    
                    h_block_start = h_start // block_h
                    h_block_end = h_end // block_h
                    w_block_start = w_start // block_w
                    w_block_end = w_end // block_w
                    
                    crosses_h = h_block_end > h_block_start
                    crosses_w = w_block_end > w_block_start
                    
                    if crosses_h and crosses_w:
                        crossing_both += 1
                        num_rows = 4  # 2x2 blocks
                    elif crosses_h:
                        crossing_h += 1
                        num_rows = 2  # 2 H blocks
                    elif crosses_w:
                        crossing_w += 1
                        num_rows = 2  # 2 W blocks
                    else:
                        non_crossing += 1
                        num_rows = 1
                    
                    total_detail += num_rows
    
    print(f"  C × P × Q × R = 3 × 28 × 7 × 7 = {C_l3 * P_l3 * Q_l3 * R_l2} tiles")
    print(f"  - 不跨 block: {non_crossing} tiles × 1 row = {non_crossing}")
    print(f"  - 跨 H block: {crossing_h} tiles × 2 rows = {crossing_h * 2}")
    print(f"  - 跨 W block: {crossing_w} tiles × 2 rows = {crossing_w * 2}")
    print(f"  - 跨 H+W:     {crossing_both} tiles × 4 rows = {crossing_both * 4}")
    print(f"  Total = {total_detail}")
    
    # K factor
    print(f"\n  乘以 K = {K_l3}:")
    print(f"  Total × K = {total_detail} × {K_l3} = {total_detail * K_l3}")
    
    # 这才是 Trace 计算的逻辑!
    # 但 Trace 修复后是 5376, 等于 total_detail (不乘 K)
    # 因为 Input 数据对 K 是 reuse 的!
    
    print(f"\n  实际 Trace = {total_detail} (不乘 K, 因为 Input 对 K reuse)")


if __name__ == "__main__":
    analyze_ilp_vs_trace()
