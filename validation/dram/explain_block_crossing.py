"""
详细解释 Block Crossing 的计算逻辑

目标：理解为什么有 162 个不跨 block，6 个跨 H，27 个跨 W，1 个跨两者
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

def explain_block_crossing():
    """详细解释 block crossing 的计算"""
    
    print("=" * 80)
    print("BLOCK CROSSING 详细解释")
    print("=" * 80)
    
    # ==========================================================================
    # 1. 基本参数
    # ==========================================================================
    print("\n【1. 基本参数】")
    
    # Workload 参数
    H_in = 62  # Input Height
    W_in = 62  # Input Width
    
    # Block 大小 (数据布局单位)
    block_h = 31
    block_w = 31
    
    # Access tile 大小 (每次 DRAM 访问的范围)
    # 来自 buffer_tile: P_per_tile=2, Q_per_tile=8, S=7
    P_per_tile = 2
    Q_per_tile = 8
    S_per_tile = 7  # S 在 Level 1 循环
    
    # H_per_tile = P_per_tile (因为 stride=1, R 在内层)
    # W_per_tile = Q_per_tile + (S_per_tile - 1) = 8 + 6 = 14 (滑动窗口)
    H_per_tile = P_per_tile  # = 2
    W_per_tile = Q_per_tile + S_per_tile - 1  # = 8 + 7 - 1 = 14
    
    # DRAM level 的 tile 数量
    P_l3 = 28  # 56 / 2 = 28
    Q_l3 = 7   # 56 / 8 = 7
    
    print(f"  Input size: H={H_in}, W={W_in}")
    print(f"  Block size: block_h={block_h}, block_w={block_w}")
    print(f"  Access tile: H_per_tile={H_per_tile}, W_per_tile={W_per_tile}")
    print(f"  DRAM tiles: P_l3={P_l3}, Q_l3={Q_l3}")
    print(f"  Total (p,q) combinations: {P_l3} × {Q_l3} = {P_l3 * Q_l3}")
    
    # ==========================================================================
    # 2. Block 边界位置
    # ==========================================================================
    print("\n【2. Block 边界位置】")
    
    # H 方向的 block 边界
    h_boundaries = [i * block_h for i in range(1, (H_in + block_h - 1) // block_h)]
    print(f"  H 方向 block 边界: {h_boundaries}")
    print(f"    Block 0: h ∈ [0, 31)")
    print(f"    Block 1: h ∈ [31, 62)")
    
    # W 方向的 block 边界
    w_boundaries = [i * block_w for i in range(1, (W_in + block_w - 1) // block_w)]
    print(f"\n  W 方向 block 边界: {w_boundaries}")
    print(f"    Block 0: w ∈ [0, 31)")
    print(f"    Block 1: w ∈ [31, 62)")
    
    # ==========================================================================
    # 3. P tile 到 H 坐标的映射
    # ==========================================================================
    print("\n【3. P tile → H 坐标映射】")
    print(f"  公式: h_start = p_tile × P_per_tile = p_tile × {P_per_tile}")
    print(f"        h_end = h_start + H_per_tile = h_start + {H_per_tile}")
    print(f"  边界 h=31 落在 block 0 和 block 1 之间")
    print()
    
    # 找出哪些 p_tile 的 H 范围跨越 h=31
    p_tiles_crossing_h = []
    print(f"  详细 P tile 映射 (标注跨越边界的):")
    for p in range(P_l3):
        h_start = p * P_per_tile
        h_end = h_start + H_per_tile
        h_block_start = h_start // block_h
        h_block_end = (h_end - 1) // block_h
        crosses = h_block_start != h_block_end
        
        if crosses:
            p_tiles_crossing_h.append(p)
            marker = " ← CROSSES h=31!"
        else:
            marker = ""
        
        if p < 5 or p >= P_l3 - 5 or crosses:
            print(f"    p={p:2d}: h ∈ [{h_start:2d}, {h_end:2d}) → h_blocks=[{h_block_start}, {h_block_end}]{marker}")
        elif p == 5:
            print(f"    ... (p=5 到 p={P_l3-5} 省略)")
    
    print(f"\n  跨越 H block 边界的 p_tile: {p_tiles_crossing_h}")
    print(f"  数量: {len(p_tiles_crossing_h)}")
    
    # 验证
    # h=31 是边界，如果 h_start < 31 且 h_end > 31，则跨越
    # h_start = p × 2 < 31 → p < 15.5 → p ≤ 15
    # h_end = p × 2 + 2 > 31 → p > 14.5 → p ≥ 15
    # 所以只有 p=15 满足 (h_start=30, h_end=32)
    print(f"\n  数学验证:")
    print(f"    h_start = p × 2 < 31 → p ≤ 15")
    print(f"    h_end = p × 2 + 2 > 31 → p ≥ 15")
    print(f"    只有 p=15 满足两个条件")
    print(f"    p=15: h ∈ [30, 32)，确实跨越 h=31")
    
    # ==========================================================================
    # 4. Q tile 到 W 坐标的映射
    # ==========================================================================
    print("\n【4. Q tile → W 坐标映射】")
    print(f"  公式: w_start = q_tile × Q_per_tile = q_tile × {Q_per_tile}")
    print(f"        w_end = w_start + W_per_tile = w_start + {W_per_tile}")
    print(f"  边界 w=31 落在 block 0 和 block 1 之间")
    print()
    
    # 找出哪些 q_tile 的 W 范围跨越 w=31
    q_tiles_crossing_w = []
    print(f"  详细 Q tile 映射:")
    for q in range(Q_l3):
        w_start = q * Q_per_tile
        w_end = w_start + W_per_tile
        w_block_start = w_start // block_w
        w_block_end = (w_end - 1) // block_w
        crosses = w_block_start != w_block_end
        
        if crosses:
            q_tiles_crossing_w.append(q)
            marker = " ← CROSSES w=31!"
        else:
            marker = ""
        
        print(f"    q={q}: w ∈ [{w_start:2d}, {w_end:2d}) → w_blocks=[{w_block_start}, {w_block_end}]{marker}")
    
    print(f"\n  跨越 W block 边界的 q_tile: {q_tiles_crossing_w}")
    print(f"  数量: {len(q_tiles_crossing_w)}")
    
    # 验证
    # w=31 是边界，如果 w_start < 31 且 w_end > 31，则跨越
    # w_start = q × 8 < 31 → q < 3.875 → q ≤ 3
    # w_end = q × 8 + 14 > 31 → q > 2.125 → q ≥ 3
    # 所以只有 q=3 满足 (w_start=24, w_end=38)
    print(f"\n  数学验证:")
    print(f"    w_start = q × 8 < 31 → q ≤ 3")
    print(f"    w_end = q × 8 + 14 > 31 → q ≥ 3 (因为 2×8+14=30 ≤ 31)")
    print(f"    只有 q=3 满足两个条件")
    print(f"    q=3: w ∈ [24, 38)，确实跨越 w=31")
    
    # ==========================================================================
    # 5. 组合分析
    # ==========================================================================
    print("\n【5. 组合分析】")
    
    # 统计四种情况
    count_no_cross = 0
    count_cross_h_only = 0
    count_cross_w_only = 0
    count_cross_both = 0
    
    details = {
        'no_cross': [],
        'cross_h': [],
        'cross_w': [],
        'cross_both': []
    }
    
    for p in range(P_l3):
        crosses_h = p in p_tiles_crossing_h
        for q in range(Q_l3):
            crosses_w = q in q_tiles_crossing_w
            
            if crosses_h and crosses_w:
                count_cross_both += 1
                details['cross_both'].append((p, q))
            elif crosses_h:
                count_cross_h_only += 1
                details['cross_h'].append((p, q))
            elif crosses_w:
                count_cross_w_only += 1
                details['cross_w'].append((p, q))
            else:
                count_no_cross += 1
                details['no_cross'].append((p, q))
    
    print(f"  不跨 block:        {count_no_cross} 个")
    print(f"  仅跨 H block:      {count_cross_h_only} 个")
    print(f"  仅跨 W block:      {count_cross_w_only} 个")
    print(f"  跨 H 和 W block:   {count_cross_both} 个")
    print(f"  总计:              {count_no_cross + count_cross_h_only + count_cross_w_only + count_cross_both}")
    
    # 验证公式
    print(f"\n  公式验证:")
    num_p_cross_h = len(p_tiles_crossing_h)  # 1
    num_p_no_cross = P_l3 - num_p_cross_h    # 27
    num_q_cross_w = len(q_tiles_crossing_w)  # 1
    num_q_no_cross = Q_l3 - num_q_cross_w    # 6
    
    print(f"    P tiles: {num_p_no_cross} 不跨 + {num_p_cross_h} 跨 = {P_l3}")
    print(f"    Q tiles: {num_q_no_cross} 不跨 + {num_q_cross_w} 跨 = {Q_l3}")
    print()
    print(f"    不跨 block     = (P不跨) × (Q不跨) = {num_p_no_cross} × {num_q_no_cross} = {num_p_no_cross * num_q_no_cross}")
    print(f"    仅跨 H block   = (P跨) × (Q不跨)   = {num_p_cross_h} × {num_q_no_cross} = {num_p_cross_h * num_q_no_cross}")
    print(f"    仅跨 W block   = (P不跨) × (Q跨)   = {num_p_no_cross} × {num_q_cross_w} = {num_p_no_cross * num_q_cross_w}")
    print(f"    跨 H 和 W      = (P跨) × (Q跨)     = {num_p_cross_h} × {num_q_cross_w} = {num_p_cross_h * num_q_cross_w}")
    
    # ==========================================================================
    # 6. 详细列出跨边界的 tiles
    # ==========================================================================
    print("\n【6. 跨边界的 tiles 详细列表】")
    
    print(f"\n  仅跨 H block 的 tiles ({len(details['cross_h'])} 个):")
    for p, q in details['cross_h']:
        h_start = p * P_per_tile
        h_end = h_start + H_per_tile
        w_start = q * Q_per_tile
        w_end = w_start + W_per_tile
        print(f"    (p={p}, q={q}): h ∈ [{h_start}, {h_end}), w ∈ [{w_start}, {w_end})")
    
    print(f"\n  仅跨 W block 的 tiles ({len(details['cross_w'])} 个):")
    for p, q in details['cross_w']:
        h_start = p * P_per_tile
        h_end = h_start + H_per_tile
        w_start = q * Q_per_tile
        w_end = w_start + W_per_tile
        print(f"    (p={p}, q={q}): h ∈ [{h_start}, {h_end}), w ∈ [{w_start}, {w_end})")
    
    print(f"\n  跨 H 和 W block 的 tiles ({len(details['cross_both'])} 个):")
    for p, q in details['cross_both']:
        h_start = p * P_per_tile
        h_end = h_start + H_per_tile
        w_start = q * Q_per_tile
        w_end = w_start + W_per_tile
        print(f"    (p={p}, q={q}): h ∈ [{h_start}, {h_end}), w ∈ [{w_start}, {w_end})")
    
    # ==========================================================================
    # 7. 可视化
    # ==========================================================================
    print("\n【7. 可视化 (H-W 空间)】")
    print()
    print("  H (纵向)")
    print("  ^")
    print("  |")
    print("  62 +-----------------------------+")
    print("     |         Block 1,0           |         Block 1,1           |")
    print("     |  (h ∈ [31,62), w ∈ [0,31))  |  (h ∈ [31,62), w ∈ [31,62)) |")
    print("  31 +-----------------------------+-----------------------------+")
    print("     |         Block 0,0           |         Block 0,1           |")
    print("     |  (h ∈ [0,31), w ∈ [0,31))   |  (h ∈ [0,31), w ∈ [31,62))  |")
    print("   0 +-----------------------------+-----------------------------+---> W")
    print("     0                            31                            62")
    print()
    print("  Access tile 大小: 2 × 14")
    print("  当 p=15: h ∈ [30, 32) 跨越 h=31 边界")
    print("  当 q=3:  w ∈ [24, 38) 跨越 w=31 边界")
    
    # ==========================================================================
    # 8. Row Switch 计算
    # ==========================================================================
    print("\n【8. Row Switch 计算】")
    
    # row_aligned 模式下：
    # 每个 (h_block, w_block, c) 组合对应一个独立的 row
    # stride_q_l3 = 1024 = 1 row
    # stride_p_l3 = 7168 = 7 rows (因为 Q_l3=7)
    
    print(f"\n  row_aligned 模式下的 row 分配:")
    print(f"    每个 (p_tile, q_tile, c_tile) 基址对齐到 row 边界")
    print(f"    但当 tile 跨 block 时，需要访问多个 block")
    print(f"    每个 block 在不同的 row！")
    
    print(f"\n  Row switch 来源:")
    print(f"    1. Tile 之间: 相邻 tile 的 base row 可能不同")
    print(f"    2. Tile 内部: 跨 block 时需要访问不同的 block (不同 row)")
    
    # 每种 tile 访问的 block 数量
    print(f"\n  每种 tile 访问的 block 数:")
    print(f"    不跨 block:     1 个 block")
    print(f"    仅跨 H block:   2 个 block (2 h_blocks × 1 w_block)")
    print(f"    仅跨 W block:   2 个 block (1 h_block × 2 w_blocks)")
    print(f"    跨 H 和 W:      4 个 block (2 h_blocks × 2 w_blocks)")
    
    # ==========================================================================
    # 9. 推导公式
    # ==========================================================================
    print("\n【9. 推导 Row Activation 公式】")
    
    C_l3 = 3
    K_l3 = 4
    R_l2 = 7
    
    # 方法 B 的计算：遍历每个 block
    # 每次遇到新的 (h_block, w_block, c) 就是一次 row switch
    
    print(f"\n  参数: C_l3={C_l3}, K_l3={K_l3}, R_l2={R_l2}")
    print(f"        P_l3={P_l3}, Q_l3={Q_l3}")
    print(f"        不跨={count_no_cross}, 跨H={count_cross_h_only}, 跨W={count_cross_w_only}, 跨两者={count_cross_both}")
    
    # 每个 (p,q) 组合在一次 C 循环中访问的 block 数
    blocks_per_tile = {
        'no_cross': 1,
        'cross_h': 2,
        'cross_w': 2,
        'cross_both': 4
    }
    
    # 总 block 访问数 (不考虑 row switch，只是计数)
    total_block_visits = (
        count_no_cross * blocks_per_tile['no_cross'] +
        count_cross_h_only * blocks_per_tile['cross_h'] +
        count_cross_w_only * blocks_per_tile['cross_w'] +
        count_cross_both * blocks_per_tile['cross_both']
    )
    
    print(f"\n  一次 P × Q 迭代访问的 block 数:")
    print(f"    = {count_no_cross} × 1 + {count_cross_h_only} × 2 + {count_cross_w_only} × 2 + {count_cross_both} × 4")
    print(f"    = {count_no_cross} + {count_cross_h_only * 2} + {count_cross_w_only * 2} + {count_cross_both * 4}")
    print(f"    = {total_block_visits}")
    
    # 一次完整的 K × C × P × Q × R 迭代
    total_iterations = K_l3 * C_l3 * R_l2
    
    print(f"\n  总迭代次数: K_l3 × C_l3 × R_l2 = {K_l3} × {C_l3} × {R_l2} = {total_iterations}")
    print(f"  每次迭代遍历所有 P × Q = {P_l3} × {Q_l3} = {P_l3 * Q_l3} tiles")
    
    # 但是 row switch 不是简单的 block 访问数！
    # 需要考虑访问顺序和 row buffer 状态
    
    print(f"\n  注意: Row switch ≠ Block 访问数")
    print(f"       Row switch 取决于访问顺序和是否与前一次访问在同一 row")


if __name__ == "__main__":
    explain_block_crossing()
