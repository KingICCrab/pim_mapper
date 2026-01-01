"""
推导 5880 Row Activation 的完整公式

已知 block crossing 分类:
- 不跨 block: 162 个 (p,q)
- 仅跨 H: 6 个 (p,q)  
- 仅跨 W: 27 个 (p,q)
- 跨两者: 1 个 (p,q)

现在推导这如何变成 5880 次 row activation
"""

def derive_5880_formula():
    print("=" * 80)
    print("推导 5880 Row Activation 公式")
    print("=" * 80)
    
    # ==========================================================================
    # 基本参数
    # ==========================================================================
    print("\n【基本参数】")
    
    P_l3 = 28
    Q_l3 = 7
    C_l3 = 3
    K_l3 = 4
    R_l2 = 7
    
    row_size = 1024
    stride_q = 1024      # 每个 q_tile 跳 1 row
    stride_p = 7168      # 每个 p_tile 跳 7 rows (= Q_l3 rows)
    stride_c = 200704    # 每个 c_tile 跳 196 rows (= P_l3 × Q_l3 rows)
    
    # Block crossing 分类
    count_no_cross = 162
    count_cross_h = 6
    count_cross_w = 27
    count_cross_both = 1
    
    print(f"  DRAM 循环: K={K_l3} × C={C_l3} × P={P_l3} × Q={Q_l3} × R={R_l2}")
    print(f"  总迭代数: {K_l3 * C_l3 * P_l3 * Q_l3 * R_l2}")
    print(f"  Block crossing: 不跨={count_no_cross}, 跨H={count_cross_h}, 跨W={count_cross_w}, 跨两者={count_cross_both}")
    
    # ==========================================================================
    # 理解循环结构对 Input 的影响
    # ==========================================================================
    print("\n【循环结构对 Input 的影响】")
    
    print(f"""
  循环顺序 (outer → inner): K → C → P → Q → R
  
  Input 相关维度: C, P, Q (N=1 忽略)
  Input 不相关维度: K, R
  
  关键洞察:
  - K 对 Input 不相关，但 K 循环会导致 (C,P,Q) 重新从 (0,0,0) 开始
  - R 对 Input 不相关，但 R 是最内层循环，每次 R 迭代访问相同的 Input tile
  - 所以实际上：每 R_l2=7 次迭代访问相同的 Input 数据
    """)
    
    # ==========================================================================
    # 分析一次 P × Q 扫描的 row switch
    # ==========================================================================
    print("\n【分析一次 P × Q 扫描的 row switch】")
    
    # 在一次 C 固定的 P × Q 扫描中：
    # - 每个 (p,q) 有一个 base row
    # - 如果 tile 跨 block，访问多个 row
    
    # 模拟一次 P × Q 扫描
    def get_base_row(p, q, c):
        """计算 (p, q, c) 的 base row"""
        addr = p * stride_p + q * stride_q + c * stride_c
        return addr // row_size
    
    def get_blocks_for_tile(p, q):
        """获取 tile (p,q) 访问的所有 (h_block, w_block) 对"""
        block_h, block_w = 31, 31
        P_per_tile, Q_per_tile = 2, 8
        H_per_tile, W_per_tile = 2, 14
        
        h_start = p * P_per_tile
        h_end = h_start + H_per_tile
        w_start = q * Q_per_tile
        w_end = w_start + W_per_tile
        
        h_block_start = h_start // block_h
        h_block_end = (h_end - 1) // block_h
        w_block_start = w_start // block_w
        w_block_end = (w_end - 1) // block_w
        
        blocks = []
        for hb in range(h_block_start, h_block_end + 1):
            for wb in range(w_block_start, w_block_end + 1):
                blocks.append((hb, wb))
        return blocks
    
    def get_rows_for_tile(p, q, c):
        """获取 tile (p,q,c) 访问的所有 row"""
        blocks = get_blocks_for_tile(p, q)
        rows = []
        for hb, wb in blocks:
            addr = hb * stride_p + wb * stride_q + c * stride_c
            row = addr // row_size
            rows.append(row)
        return rows
    
    # 统计一次 C=0 的 P × Q 扫描
    print("  模拟 C=0 时的 P × Q 扫描:")
    
    c = 0
    current_row = None
    switches_in_pq = 0
    
    for p in range(P_l3):
        for q in range(Q_l3):
            rows = get_rows_for_tile(p, q, c)
            for row in rows:
                if current_row != row:
                    switches_in_pq += 1
                    current_row = row
    
    print(f"    C=0 时的 row switches: {switches_in_pq}")
    
    # 验证: 逐个 tile 分析
    print("\n  逐个 tile 分析 row switch 贡献:")
    
    # 重新统计，区分 tile 间 switch 和 tile 内 switch
    current_row = None
    inter_tile_switches = 0  # tile 间 switch
    intra_tile_switches = 0  # tile 内 switch (跨 block)
    
    for p in range(P_l3):
        for q in range(Q_l3):
            rows = get_rows_for_tile(p, q, c)
            
            for i, row in enumerate(rows):
                if current_row != row:
                    if i == 0:
                        inter_tile_switches += 1  # 第一个 row，属于 tile 间
                    else:
                        intra_tile_switches += 1  # 不是第一个，属于 tile 内
                    current_row = row
    
    print(f"    Tile 间 switches: {inter_tile_switches}")
    print(f"    Tile 内 switches (跨 block): {intra_tile_switches}")
    print(f"    总计: {inter_tile_switches + intra_tile_switches}")
    
    # ==========================================================================
    # 分析 tile 内 switch
    # ==========================================================================
    print("\n【分析 Tile 内 switch】")
    
    # 每种 tile 的内部 switch 数:
    # - 不跨 block: 访问 1 个 block，0 次内部 switch
    # - 仅跨 H: 访问 2 个 block，但按什么顺序？
    # - 仅跨 W: 访问 2 个 block
    # - 跨两者: 访问 4 个 block
    
    print("  Block 访问顺序 (h_block 外层, w_block 内层):")
    print("    不跨: [(hb, wb)] → 0 次内部 switch")
    print("    仅跨 H: [(hb0, wb), (hb1, wb)] → 1 次内部 switch")
    print("    仅跨 W: [(hb, wb0), (hb, wb1)] → 1 次内部 switch")
    print("    跨两者: [(hb0, wb0), (hb0, wb1), (hb1, wb0), (hb1, wb1)] → 3 次内部 switch")
    
    # 但是！这里有个关键问题：tile 之间的切换
    # 如果当前 tile 的最后一个 row 和下一个 tile 的第一个 row 相同，就没有 switch
    
    # ==========================================================================
    # 精确计算一次 C × P × Q 扫描
    # ==========================================================================
    print("\n【精确计算一次 C × P × Q 扫描】")
    
    current_row = None
    total_switches_cpq = 0
    
    for c in range(C_l3):
        for p in range(P_l3):
            for q in range(Q_l3):
                rows = get_rows_for_tile(p, q, c)
                for row in rows:
                    if current_row != row:
                        total_switches_cpq += 1
                        current_row = row
    
    print(f"  一次 C × P × Q 扫描的 row switches: {total_switches_cpq}")
    
    # ==========================================================================
    # 考虑 K 和 R 循环
    # ==========================================================================
    print("\n【考虑 K 和 R 循环】")
    
    print(f"""
  循环结构:
    for k in range(K_l3={K_l3}):
        for c in range(C_l3={C_l3}):
            for p in range(P_l3={P_l3}):
                for q in range(Q_l3={Q_l3}):
                    for r in range(R_l2={R_l2}):
                        # 访问 Input tile (p, q, c)
    
  关键问题：
  1. R 循环：每次 R 迭代访问相同的 Input，所以 R 循环内有 row switch 吗？
  2. K 循环：K 不相关，但从 K=k 到 K=k+1 时，(C,P,Q) 回到 (0,0,0)
    """)
    
    # 完整模拟
    print("  完整模拟 K × C × P × Q × R 循环:")
    
    current_row = None
    total_switches = 0
    
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r in range(R_l2):
                        rows = get_rows_for_tile(p, q, c)
                        for row in rows:
                            if current_row != row:
                                total_switches += 1
                                current_row = row
    
    print(f"    总 row switches: {total_switches}")
    
    # ==========================================================================
    # 分析 R 循环的影响
    # ==========================================================================
    print("\n【分析 R 循环的影响】")
    
    # 在一个 (k, c, p, q) 固定时，R 循环 7 次
    # 每次 R 迭代访问相同的 Input tile
    # 所以只有第一次 R=0 时产生 switch，R=1,2,...,6 不产生新 switch
    
    # 验证
    print("  验证: 固定 (k=0, c=0, p=0, q=0)，遍历 R:")
    
    current_row = None
    switches_in_r = 0
    
    p, q, c = 0, 0, 0
    rows = get_rows_for_tile(p, q, c)
    
    for r in range(R_l2):
        for row in rows:
            if current_row != row:
                switches_in_r += 1
                current_row = row
        print(f"    r={r}: current_row={current_row}, total_switches={switches_in_r}")
    
    print(f"\n  结论: R 循环不产生额外 switch（访问相同的 row）")
    print(f"         所以 R_l2 因子不影响 row switch 数量")
    
    # ==========================================================================
    # 分析 K 循环的影响
    # ==========================================================================
    print("\n【分析 K 循环的影响】")
    
    # 当 K 从 k 变到 k+1 时：
    # - 最后一个 tile 是 (c=2, p=27, q=6)
    # - 下一个 tile 是 (c=0, p=0, q=0)
    # - 如果它们的最后/第一个 row 不同，产生 1 次 switch
    
    last_tile_rows = get_rows_for_tile(P_l3-1, Q_l3-1, C_l3-1)
    first_tile_rows = get_rows_for_tile(0, 0, 0)
    
    print(f"  最后一个 tile (p={P_l3-1}, q={Q_l3-1}, c={C_l3-1}) 的 rows: {last_tile_rows}")
    print(f"  第一个 tile (p=0, q=0, c=0) 的 rows: {first_tile_rows}")
    
    last_row_of_cpq = last_tile_rows[-1]
    first_row_of_cpq = first_tile_rows[0]
    
    print(f"  K 回绕时: row {last_row_of_cpq} → {first_row_of_cpq}")
    if last_row_of_cpq != first_row_of_cpq:
        print(f"  K 回绕产生 1 次 switch")
        k_wraparound_switch = 1
    else:
        print(f"  K 回绕不产生 switch")
        k_wraparound_switch = 0
    
    # ==========================================================================
    # 推导公式
    # ==========================================================================
    print("\n【推导公式】")
    
    # 一次 C × P × Q 扫描的 switches
    switches_per_cpq = total_switches_cpq
    
    # K 回绕的 switches (K-1 次回绕)
    k_wraparound_total = k_wraparound_switch * (K_l3 - 1)
    
    # 总公式
    # 注意：R 循环不产生新 switch，但会重复访问
    # 关键：trace 是否对 R 循环也生成访问？
    
    print(f"  一次 C × P × Q 扫描: {switches_per_cpq} switches")
    print(f"  K 循环: {K_l3} 次完整扫描")
    print(f"  K 回绕 switches: {k_wraparound_total} (K_l3-1 次回绕 × {k_wraparound_switch})")
    
    # 公式 1: 假设 R 循环不产生 switch
    formula1 = switches_per_cpq * K_l3 + k_wraparound_total
    print(f"\n  公式 1 (R 不产生 switch):")
    print(f"    = {switches_per_cpq} × {K_l3} + {k_wraparound_total}")
    print(f"    = {formula1}")
    
    # 但实际结果是 5880，让我们验证
    print(f"\n  实际结果: {total_switches}")
    print(f"  差异: {total_switches - formula1}")
    
    # ==========================================================================
    # 重新分析：R 循环的真实影响
    # ==========================================================================
    print("\n【重新分析 R 循环】")
    
    # R 循环在最内层，每次 R 迭代确实访问 Input
    # 但访问的是相同的 (p, q, c) tile
    # 问题是：访问顺序是什么？
    
    # 如果每次 R 迭代都重新访问 tile 的所有 block...
    print("  Trace 的访问模式分析:")
    print("  假设每次 R 迭代都重新遍历 tile 内所有 block")
    print("  那么跨 block 的 tile 每次 R 迭代都会产生 switch")
    
    # 重新模拟，考虑每次 R 迭代都重新遍历
    current_row = None
    total_with_r = 0
    
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r in range(R_l2):
                        rows = get_rows_for_tile(p, q, c)
                        for row in rows:
                            if current_row != row:
                                total_with_r += 1
                                current_row = row
    
    print(f"\n  考虑 R 循环后的总 switches: {total_with_r}")
    
    # ==========================================================================
    # 深入分析：R 循环内 tile 的重复访问
    # ==========================================================================
    print("\n【深入分析 R 循环内的 tile 访问】")
    
    # 跟踪一个跨 block 的 tile
    print("  跟踪跨 W 的 tile (p=0, q=3, c=0) 在 R 循环中:")
    
    p, q, c = 0, 3, 0
    rows = get_rows_for_tile(p, q, c)
    print(f"    访问的 rows: {rows}")
    
    current_row = None
    for r in range(R_l2):
        switches_this_r = 0
        for row in rows:
            if current_row != row:
                switches_this_r += 1
                current_row = row
        print(f"    r={r}: 访问 rows={rows}, switches={switches_this_r}, current_row={current_row}")
    
    print(f"\n  观察: 只有 r=0 时产生 switch，r=1..6 不产生 (因为 current_row 没变)")
    
    # ==========================================================================
    # 最终公式推导
    # ==========================================================================
    print("\n【最终公式推导】")
    
    # 经过分析，R 循环不产生额外 switch
    # 所以总 switch = K_l3 × (一次 C × P × Q 扫描的 switches) + K 回绕
    
    print(f"  一次 C × P × Q 扫描: {switches_per_cpq} switches")
    print(f"  K 循环: {K_l3} 次")
    print(f"  K 回绕: (K_l3-1) × {k_wraparound_switch} = {k_wraparound_total}")
    
    predicted = switches_per_cpq * K_l3 + k_wraparound_total
    print(f"\n  预测总 switches = {switches_per_cpq} × {K_l3} + {k_wraparound_total} = {predicted}")
    print(f"  实际总 switches = {total_switches}")
    
    if predicted == total_switches:
        print(f"\n  ✓ 公式正确!")
    else:
        print(f"\n  ✗ 有差异，需要进一步分析")
        
        # 检查是否漏算了什么
        print(f"\n  进一步检查...")
        
        # 打印每个 K 迭代的 switch 数
        current_row = None
        for k in range(K_l3):
            switches_this_k = 0
            for c in range(C_l3):
                for p in range(P_l3):
                    for q in range(Q_l3):
                        for r in range(R_l2):
                            rows = get_rows_for_tile(p, q, c)
                            for row in rows:
                                if current_row != row:
                                    switches_this_k += 1
                                    current_row = row
            print(f"    K={k}: {switches_this_k} switches")
    
    # ==========================================================================
    # 分解 switches_per_cpq
    # ==========================================================================
    print("\n【分解一次 C × P × Q 扫描的 switches】")
    
    # 分解为: tile 间 switch + tile 内 switch
    
    # Tile 内 switch (跨 block 产生):
    # - 仅跨 H: 1 次内部 switch × 6 tiles × C_l3 = 1 × 6 × 3 = 18
    # - 仅跨 W: 1 次内部 switch × 27 tiles × C_l3 = 1 × 27 × 3 = 81
    # - 跨两者: 3 次内部 switch × 1 tile × C_l3 = 3 × 1 × 3 = 9
    
    intra_switch_h = 1 * count_cross_h * C_l3
    intra_switch_w = 1 * count_cross_w * C_l3
    intra_switch_both = 3 * count_cross_both * C_l3
    total_intra = intra_switch_h + intra_switch_w + intra_switch_both
    
    print(f"  Tile 内 switch (跨 block):")
    print(f"    仅跨 H: 1 × {count_cross_h} × {C_l3} = {intra_switch_h}")
    print(f"    仅跨 W: 1 × {count_cross_w} × {C_l3} = {intra_switch_w}")
    print(f"    跨两者: 3 × {count_cross_both} × {C_l3} = {intra_switch_both}")
    print(f"    小计: {total_intra}")
    
    # Tile 间 switch:
    inter_switch = switches_per_cpq - total_intra
    print(f"\n  Tile 间 switch: {switches_per_cpq} - {total_intra} = {inter_switch}")
    
    # 验证 tile 间 switch
    # 每个 unique (c, p, q) 组合在访问时产生 1 次 tile 间 switch (如果 row 变化)
    unique_tiles = C_l3 * P_l3 * Q_l3
    print(f"  Unique tiles: {C_l3} × {P_l3} × {Q_l3} = {unique_tiles}")
    print(f"  如果每个 tile 的 base row 都不同，应该有 {unique_tiles} 次 tile 间 switch")
    print(f"  实际 tile 间 switch: {inter_switch}")
    
    # ==========================================================================
    # 最终公式
    # ==========================================================================
    print("\n" + "=" * 80)
    print("【最终公式】")
    print("=" * 80)
    
    print(f"""
  Row Activation = K_l3 × (Tile间 + Tile内) + K回绕
  
  其中:
    Tile间 = {inter_switch} (每个 unique tile 产生 1 次 switch)
    Tile内 = {total_intra} (跨 block 的额外 switch)
    K回绕 = {k_wraparound_total}
  
  计算:
    = {K_l3} × ({inter_switch} + {total_intra}) + {k_wraparound_total}
    = {K_l3} × {switches_per_cpq} + {k_wraparound_total}
    = {K_l3 * switches_per_cpq} + {k_wraparound_total}
    = {K_l3 * switches_per_cpq + k_wraparound_total}
  
  实际结果: {total_switches}
    """)


if __name__ == "__main__":
    derive_5880_formula()
