"""
最终验证: 理解 5376 的来源

方法 1 (prev_row tracking with K) = 5376 ✓
方法 4 (per tile without K) = 4800

差异: 5376 - 4800 = 576

这 576 来自哪里?
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload

DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6

def analyze_5376():
    workload = ConvWorkload(name="ResNet-L1", R=7, S=7, P=56, Q=56, C=3, K=64, N=1)
    optimizer = PIMOptimizer()
    result = optimizer.optimize([workload], objective="latency")
    mapping = result.mappings[0]
    
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
    
    K_l3 = dram_factors[DIM_K]   # 4
    C_l3 = dram_factors[DIM_C]   # 3
    P_l3 = dram_factors[DIM_P]   # 28
    Q_l3 = dram_factors[DIM_Q]   # 7
    R_l2 = level2_factors[DIM_R] # 7
    
    P_per_tile = buffer_tile[DIM_P]  # 2
    Q_per_tile = buffer_tile[DIM_Q]  # 8
    R_per_tile = buffer_tile[DIM_R]  # 1
    
    stride_h, stride_w = 1, 1
    dilation_h, dilation_w = 1, 1
    
    H_per_tile = (P_per_tile - 1) * stride_h + R_per_tile * dilation_h  # 2
    W_per_tile = (Q_per_tile - 1) * stride_w + buffer_tile[DIM_S] * dilation_w  # 14
    
    block_h = mapping.tile_info.get('block_h', 31)
    block_w = mapping.tile_info.get('block_w', 31)
    
    num_h_blocks = 2
    num_w_blocks = 2
    
    print("=" * 80)
    print("分析 5376 的组成")
    print("=" * 80)
    
    print(f"""
循环结构:
  for k in range({K_l3}):        # K loop (外层)
    for c in range({C_l3}):      # C loop
      for p in range({P_l3}):    # P loop
        for q in range({Q_l3}):  # Q loop
          for r in range({R_l2}): # R loop (内层)

关键问题: K 循环在外层,每次 K 改变后:
  - 重新从 c=0, p=0, q=0, r=0 开始
  - 此时访问的 row 与 K 结束前的 row 不同
  - 会触发一次 row switch
""")
    
    # 分析每次 K 开始时的 switch
    print("-" * 60)
    print("分析 K 循环边界的 switches")
    print("-" * 60)
    
    # 在 k 结束时, c=2, p=27, q=6, r=6
    # 此时访问的 row
    c, p, q, r = C_l3 - 1, P_l3 - 1, Q_l3 - 1, R_l2 - 1
    h_start = p * P_per_tile * stride_h + r * R_per_tile * dilation_h
    w_start = q * Q_per_tile * stride_w
    h_end = h_start + H_per_tile - 1
    w_end = w_start + W_per_tile - 1
    
    h_block_end = h_end // block_h
    w_block_end = w_end // block_w
    
    last_row_of_k = (c * num_h_blocks + h_block_end) * num_w_blocks + w_block_end
    
    # 在 k+1 开始时, c=0, p=0, q=0, r=0
    c, p, q, r = 0, 0, 0, 0
    h_start = p * P_per_tile * stride_h + r * R_per_tile * dilation_h
    w_start = q * Q_per_tile * stride_w
    h_end = h_start + H_per_tile - 1
    w_end = w_start + W_per_tile - 1
    
    h_block_start = h_start // block_h
    w_block_start = w_start // block_w
    
    first_row_of_k = (c * num_h_blocks + h_block_start) * num_w_blocks + w_block_start
    
    print(f"  K 结束时 (c=2, p=27, q=6, r=6): row = {last_row_of_k}")
    print(f"  K+1 开始时 (c=0, p=0, q=0, r=0): row = {first_row_of_k}")
    print(f"  这两个 row 不同, 会触发 switch")
    print(f"  K 循环有 {K_l3} 次, 所以 K 边界产生 {K_l3 - 1} 次额外 switch? (不一定)")
    
    # 详细计算每个 K 内部的 switches
    print("\n" + "-" * 60)
    print("详细统计每个 K iteration 内的 switches")
    print("-" * 60)
    
    switches_per_k = []
    
    for k in range(K_l3):
        switches_in_k = 0
        prev_row = -1
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r in range(R_l2):
                        h_start = p * P_per_tile * stride_h + r * R_per_tile * dilation_h
                        w_start = q * Q_per_tile * stride_w
                        h_end = h_start + H_per_tile - 1
                        w_end = w_start + W_per_tile - 1
                        
                        h_blocks = list(range(h_start // block_h, (h_end // block_h) + 1))
                        w_blocks = list(range(w_start // block_w, (w_end // block_w) + 1))
                        
                        rows = set()
                        for hb in h_blocks:
                            for wb in w_blocks:
                                row = (c * num_h_blocks + hb) * num_w_blocks + wb
                                rows.add(row)
                        
                        for row in sorted(rows):
                            if row != prev_row:
                                switches_in_k += 1
                                prev_row = row
        
        switches_per_k.append(switches_in_k)
        print(f"  K={k}: {switches_in_k} switches")
    
    total_within_k = sum(switches_per_k)
    print(f"\n  Sum of switches within each K: {total_within_k}")
    
    # 计算 K 边界的 switches
    print("\n" + "-" * 60)
    print("计算 K 边界的额外 switches")
    print("-" * 60)
    
    # 模拟完整循环
    prev_row = -1
    total_with_k_boundary = 0
    k_boundary_switches = 0
    
    for k in range(K_l3):
        is_first_in_k = True
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r in range(R_l2):
                        h_start = p * P_per_tile * stride_h + r * R_per_tile * dilation_h
                        w_start = q * Q_per_tile * stride_w
                        h_end = h_start + H_per_tile - 1
                        w_end = w_start + W_per_tile - 1
                        
                        h_blocks = list(range(h_start // block_h, (h_end // block_h) + 1))
                        w_blocks = list(range(w_start // block_w, (w_end // block_w) + 1))
                        
                        rows = set()
                        for hb in h_blocks:
                            for wb in w_blocks:
                                row = (c * num_h_blocks + hb) * num_w_blocks + wb
                                rows.add(row)
                        
                        for row in sorted(rows):
                            if row != prev_row:
                                total_with_k_boundary += 1
                                if is_first_in_k and k > 0:
                                    k_boundary_switches += 1
                                prev_row = row
                        
                        is_first_in_k = False
    
    print(f"  Total switches (with K boundary): {total_with_k_boundary}")
    print(f"  K boundary switches: {k_boundary_switches}")
    print(f"  Switches within K only: {total_with_k_boundary - k_boundary_switches}")
    
    print("\n" + "=" * 80)
    print("最终分析")
    print("=" * 80)
    
    print(f"""
    5376 的组成:
    - 每个 K 内部的 switches: {switches_per_k[0]} (每个 K 相同)
    - K 次数: {K_l3}
    - 总计: {switches_per_k[0]} × {K_l3} = {switches_per_k[0] * K_l3}
    
    但实际是 {total_with_k_boundary}, 因为:
    - K=0 开始时有 1 次初始 switch (prev_row = -1 → first row)
    - 后续 K 开始时,如果 row 变化,也计 switch
    
    公式:
    5376 = 4800 (per tile, no K) × K × (一些系数) ?
    
    让我们验证:
    - 4800 / 3 = 1600 (每个 C 的 tiles)
    - 1600 = P×Q×R = 28×7×7 = 1372? 不对
    - 1600 = 28×7×8 = 1568? 不对
    
    实际上:
    - C×P×Q×R = 3×28×7×7 = 4116 tiles
    - 每个 tile 平均 4800/4116 ≈ 1.166 blocks
    
    5376 = 1344 (C-aware switches) × K ?
    - 1344 × 4 = 5376 ✓
    
    这就对了! 因为:
    - 对于 Input, K 是 irrelevant dim
    - 每次 K 循环, Input 的访问模式完全重复
    - 但由于 prev_row tracking, 每次 K 开始都是新的 switch
    - 所以 switches = switches_within_C × K
    """)
    
    # 验证 1344
    print("\n验证 1344:")
    
    prev_row = -1
    switches_one_k = 0
    for c in range(C_l3):
        for p in range(P_l3):
            for q in range(Q_l3):
                for r in range(R_l2):
                    h_start = p * P_per_tile * stride_h + r * R_per_tile * dilation_h
                    w_start = q * Q_per_tile * stride_w
                    h_end = h_start + H_per_tile - 1
                    w_end = w_start + W_per_tile - 1
                    
                    h_blocks = list(range(h_start // block_h, (h_end // block_h) + 1))
                    w_blocks = list(range(w_start // block_w, (w_end // block_w) + 1))
                    
                    rows = set()
                    for hb in h_blocks:
                        for wb in w_blocks:
                            row = (c * num_h_blocks + hb) * num_w_blocks + wb
                            rows.add(row)
                    
                    for row in sorted(rows):
                        if row != prev_row:
                            switches_one_k += 1
                            prev_row = row
    
    print(f"  Switches in one K (no prev_row reset): {switches_one_k}")
    print(f"  × K = {switches_one_k} × {K_l3} = {switches_one_k * K_l3}")
    
    # 如果每次 K 都重置 prev_row
    print("\n  如果每次 K 都重置 prev_row:")
    
    total_with_reset = 0
    for k in range(K_l3):
        prev_row = -1  # Reset at each K
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r in range(R_l2):
                        h_start = p * P_per_tile * stride_h + r * R_per_tile * dilation_h
                        w_start = q * Q_per_tile * stride_w
                        h_end = h_start + H_per_tile - 1
                        w_end = w_start + W_per_tile - 1
                        
                        h_blocks = list(range(h_start // block_h, (h_end // block_h) + 1))
                        w_blocks = list(range(w_start // block_w, (w_end // block_w) + 1))
                        
                        rows = set()
                        for hb in h_blocks:
                            for wb in w_blocks:
                                row = (c * num_h_blocks + hb) * num_w_blocks + wb
                                rows.add(row)
                        
                        for row in sorted(rows):
                            if row != prev_row:
                                total_with_reset += 1
                                prev_row = row
    
    print(f"  Total with reset at each K: {total_with_reset}")


if __name__ == "__main__":
    analyze_5376()
