"""
深入分析 R 循环对 Row Activation 的影响

发现：每个 K 迭代有 1470 switches，而不是预期的 210
1470 = 210 × 7 (R_l2 = 7)

这意味着 R 循环确实产生 switch！让我们深入分析
"""

def analyze_r_loop():
    print("=" * 80)
    print("深入分析 R 循环对 Row Activation 的影响")
    print("=" * 80)
    
    # 参数
    P_l3 = 28
    Q_l3 = 7
    C_l3 = 3
    K_l3 = 4
    R_l2 = 7
    
    row_size = 1024
    stride_q = 1024
    stride_p = 7168
    stride_c = 200704
    
    block_h, block_w = 31, 31
    P_per_tile, Q_per_tile = 2, 8
    H_per_tile, W_per_tile = 2, 14
    
    def get_blocks_for_tile(p, q):
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
        blocks = get_blocks_for_tile(p, q)
        rows = []
        for hb, wb in blocks:
            addr = hb * stride_p + wb * stride_q + c * stride_c
            row = addr // row_size
            rows.append(row)
        return rows
    
    # ==========================================================================
    # 关键发现
    # ==========================================================================
    print("\n【关键发现】")
    print(f"  每个 K 迭代的 switches: 1470")
    print(f"  一次 C × P × Q 扫描: 210 switches")
    print(f"  1470 / 210 = {1470 / 210} = R_l2!")
    print(f"\n  这意味着 R 循环确实产生 switches！")
    
    # ==========================================================================
    # 分析跨 W 的 tile 在 R 循环中的行为
    # ==========================================================================
    print("\n【分析跨 W 的 tile 在 R 循环中的行为】")
    
    # Tile (p=0, q=3, c=0) 跨越 W block 边界
    # 访问的 rows: [0, 1]
    
    p, q, c = 0, 3, 0
    rows = get_rows_for_tile(p, q, c)
    print(f"  Tile (p={p}, q={q}, c={c}) 访问的 rows: {rows}")
    
    # 在 R 循环中：
    # r=0: 访问 row 0, 1 → 2 switches (从上一个 tile 切换来，再内部切换)
    # r=1: 访问 row 0, 1 → ?
    
    print("\n  问题：r=1..6 时，为什么还会有 switch？")
    print("  答案：因为 Q 循环在 R 循环外面!")
    
    # ==========================================================================
    # 重新理解循环顺序
    # ==========================================================================
    print("\n【重新理解循环顺序】")
    
    print("""
  正确的循环顺序:
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r in range(R_l2):
                        access(p, q, c)
                        
  关键点：
  - Q 在外，R 在内
  - 所以访问序列是:
    (p=0, q=0, r=0), (p=0, q=0, r=1), ..., (p=0, q=0, r=6)  # 7 次访问 row 0
    (p=0, q=1, r=0), (p=0, q=1, r=1), ..., (p=0, q=1, r=6)  # 7 次访问 row 1
    ...
    """)
    
    # ==========================================================================
    # 跟踪详细的访问序列
    # ==========================================================================
    print("\n【跟踪详细的访问序列】")
    
    print("  跟踪 c=0, p=0 的所有 q 和 r:")
    
    current_row = None
    total_switches = 0
    
    p, c = 0, 0
    for q in range(Q_l3):
        for r in range(R_l2):
            rows = get_rows_for_tile(p, q, c)
            # 对于每个 R 迭代，访问所有 block
            for row in rows:
                if current_row != row:
                    total_switches += 1
                    current_row = row
            
            if r == 0 or r == R_l2 - 1:
                print(f"    (p={p}, q={q}, r={r}): rows={rows}, current_row={current_row}, total_switches={total_switches}")
    
    print(f"\n  在 (c=0, p=0) 完成后，total_switches = {total_switches}")
    
    # ==========================================================================
    # 发现问题：每次 R 迭代都重新访问所有 block！
    # ==========================================================================
    print("\n【发现关键问题】")
    
    print("""
  Trace 的行为：
  - 对于跨 block 的 tile，每次 R 迭代都按顺序访问所有 block
  - 这意味着:
    r=0: row 0 → row 1 (2 switches: 进入 + 内部)
    r=1: row 0 → row 1 (2 switches: 回到起点 + 内部)
    r=2: row 0 → row 1 (2 switches)
    ...
    
  等等！让我重新验证...
    """)
    
    # ==========================================================================
    # 更详细的跟踪
    # ==========================================================================
    print("\n【更详细的跟踪单个跨 block tile】")
    
    # 跟踪 (p=0, q=3, c=0)
    # 但先看它前后的 tile
    
    c = 0
    p = 0
    
    print(f"  跟踪 p={p}, c={c} 的 q=2,3,4 在 R 循环中:")
    
    current_row = None
    
    # 从 q=2 开始（不跨 block）
    for q in [2, 3, 4]:
        print(f"\n  Tile (q={q}):")
        rows = get_rows_for_tile(p, q, c)
        print(f"    访问 rows: {rows}")
        
        for r in range(R_l2):
            switches_this_r = 0
            rows_accessed = []
            for row in rows:
                rows_accessed.append(row)
                if current_row != row:
                    switches_this_r += 1
                    current_row = row
            
            print(f"      r={r}: 访问 {rows_accessed}, switches={switches_this_r}, 结束 row={current_row}")
    
    # ==========================================================================
    # 关键洞察
    # ==========================================================================
    print("\n" + "=" * 80)
    print("【关键洞察】")
    print("=" * 80)
    
    print("""
  问题在于：跨 block 的 tile 每次 R 迭代都要切换 row!
  
  对于 tile (p=0, q=3, c=0)，访问 rows [0, 1]:
  
  假设 q=2 的最后状态是 current_row = 0
  
  R 循环进入 q=3:
    r=0: row 0 (没变) → row 1 (switch!) → 结束 row=1
    r=1: row 0 (switch!) → row 1 (switch!) → 结束 row=1  
    r=2: row 0 (switch!) → row 1 (switch!) → 结束 row=1
    ...
    r=6: row 0 (switch!) → row 1 (switch!) → 结束 row=1
    
  所以对于跨 W 的 tile:
  - r=0: 1 次 switch (0→1)
  - r=1: 2 次 switch (1→0, 0→1)
  - r=2: 2 次 switch
  - ...
  - r=6: 2 次 switch
  
  总计: 1 + 2×6 = 13 次 switch！
  
  但是！q=4 也需要考虑：
    r=0: row 1 (没变) → 结束 row=1
    r=1: row 1 (没变) → 结束 row=1
    ...
    
  所以从 q=3 到 q=4 的过渡中，没有额外的 switch
    """)
    
    # ==========================================================================
    # 重新模拟，精确跟踪
    # ==========================================================================
    print("\n【精确模拟 c=0, p=0 的完整序列】")
    
    c = 0
    p = 0
    current_row = None
    
    total_switches_p0 = 0
    switches_per_q = []
    
    for q in range(Q_l3):
        switches_this_q = 0
        rows = get_rows_for_tile(p, q, c)
        
        for r in range(R_l2):
            for row in rows:
                if current_row != row:
                    switches_this_q += 1
                    current_row = row
        
        switches_per_q.append(switches_this_q)
        total_switches_p0 += switches_this_q
        print(f"  q={q}: rows={rows}, switches={switches_this_q}")
    
    print(f"\n  c=0, p=0 总 switches: {total_switches_p0}")
    
    # ==========================================================================
    # 公式推导
    # ==========================================================================
    print("\n" + "=" * 80)
    print("【公式推导】")
    print("=" * 80)
    
    # 对于不跨 block 的 tile：
    # - 只访问 1 个 row
    # - 每次 R 迭代访问相同 row
    # - 只有第一次 R=0 时可能切换（如果和前一个 tile 的 row 不同）
    
    # 对于跨 block 的 tile：
    # - 访问多个 row
    # - r=0: 进入 + 内部切换
    # - r=1..R-1: 每次都要回到起点 + 内部切换
    
    # 让我计算跨 block tile 的 switch 数
    
    # 跨 W 的 tile (访问 2 个 row):
    # - r=0: 最多 2 次 (进入 + 内部)，但进入可能是 0（如果和前 tile 的 row 相同）
    # - r=1..6: 每次 2 次 (回起点 + 内部)
    
    print("  跨 W 的 tile (rows = 2):")
    print(f"    r=0: 最多 2 次")
    print(f"    r=1..{R_l2-1}: 每次 2 次")
    print(f"    最大 switches per tile: 2 + 2×{R_l2-1} = {2 + 2*(R_l2-1)}")
    
    # 但实际上，q=3 跨 W，访问 rows [0, 1]
    # r=0 时，前一个 tile q=2 的 row 是 0
    # 所以 r=0: 0→1 = 1 次
    # r=1: 1→0→1 = 2 次
    # ...
    # 总计: 1 + 2×6 = 13 次
    
    print(f"\n  实际 q=3 的 switches: 13 (从 q=2 过来，起始 row=0)")
    
    # ==========================================================================
    # 完整公式
    # ==========================================================================
    print("\n【完整公式推导】")
    
    # 统计不同情况
    count_no_cross = 162  # 不跨 block
    count_cross_h = 6     # 仅跨 H (访问 2 row)
    count_cross_w = 27    # 仅跨 W (访问 2 row)
    count_cross_both = 1  # 跨两者 (访问 4 row)
    
    # 每次 C × P × Q 扫描:
    # 假设所有 tile 间都有 row switch
    
    # 不跨 block: 1 次 tile 间 switch × 162 tiles = 162
    # 跨 H: (1 tile间 + 1 内部) × R_l2 × 6 = ?
    # 跨 W: (1 tile间 + 1 内部) × R_l2 × 27 = ?
    # 跨两者: (1 tile间 + 3 内部) × R_l2 × 1 = ?
    
    # 但这不对... 让我精确计算
    
    print("  精确计算每种 tile 的贡献:")
    
    # 模拟完整的 C × P × Q × R 循环
    current_row = None
    total = 0
    
    no_cross_switches = 0
    cross_h_switches = 0
    cross_w_switches = 0
    cross_both_switches = 0
    
    for c in range(C_l3):
        for p in range(P_l3):
            for q in range(Q_l3):
                rows = get_rows_for_tile(p, q, c)
                tile_switches = 0
                
                for r in range(R_l2):
                    for row in rows:
                        if current_row != row:
                            tile_switches += 1
                            current_row = row
                
                total += tile_switches
                
                # 分类
                blocks = get_blocks_for_tile(p, q)
                cross_h = len(set(b[0] for b in blocks)) > 1
                cross_w = len(set(b[1] for b in blocks)) > 1
                
                if not cross_h and not cross_w:
                    no_cross_switches += tile_switches
                elif cross_h and not cross_w:
                    cross_h_switches += tile_switches
                elif not cross_h and cross_w:
                    cross_w_switches += tile_switches
                else:
                    cross_both_switches += tile_switches
    
    print(f"    不跨 block 的 switches: {no_cross_switches} (tiles: {count_no_cross*C_l3})")
    print(f"    仅跨 H 的 switches: {cross_h_switches} (tiles: {count_cross_h*C_l3})")
    print(f"    仅跨 W 的 switches: {cross_w_switches} (tiles: {count_cross_w*C_l3})")
    print(f"    跨两者的 switches: {cross_both_switches} (tiles: {count_cross_both*C_l3})")
    print(f"    总计: {total}")
    
    print(f"\n  平均 switches per tile:")
    print(f"    不跨: {no_cross_switches / (count_no_cross*C_l3):.2f}")
    print(f"    跨H: {cross_h_switches / (count_cross_h*C_l3):.2f}")
    print(f"    跨W: {cross_w_switches / (count_cross_w*C_l3):.2f}")
    print(f"    跨两者: {cross_both_switches / (count_cross_both*C_l3):.2f}")
    
    # ==========================================================================
    # 推导通用公式
    # ==========================================================================
    print("\n【推导通用公式】")
    
    # 不跨 block 的 tile:
    # - 只有 1 个 row
    # - 只有第一次访问时可能 switch
    # - 平均约 0.21 次 switch/tile... 这是因为 row 有重复
    
    # 跨 H/W 的 tile:
    # - 有 2 个 row
    # - r=0: 最多 2 次 switch
    # - r=1..6: 每次 2 次 switch
    # - 但实际可能更少（如果和前 tile 的 row 相同）
    
    # 跨两者的 tile:
    # - 有 4 个 row
    # - r=0: 最多 4 次 switch
    # - r=1..6: 每次 4 次 switch
    
    # 尝试推导
    print("""
  对于跨 block 的 tile (访问 n 个 row):
  - r=0: 第一次访问，最多 n 次 switch (进入 + n-1 内部)
  - r=1..R-1: 每次需要回到起点，n 次 switch
  
  理论上限:
    跨 2 row: n×R = 2×7 = 14 次/tile
    跨 4 row: n×R = 4×7 = 28 次/tile
    
  但实际:
    跨H: {:.2f} 次/tile
    跨W: {:.2f} 次/tile
    跨两者: {:.2f} 次/tile
    """.format(
        cross_h_switches / (count_cross_h*C_l3),
        cross_w_switches / (count_cross_w*C_l3),
        cross_both_switches / (count_cross_both*C_l3)
    ))
    
    # ==========================================================================
    # 验证公式
    # ==========================================================================
    print("\n【验证公式】")
    
    # 假设:
    # - 不跨 tile: 平均 x 次 switch
    # - 跨 2 row tile: (n-1)×R + 进入 = 1×7 + 进入 = 7 + 进入
    # - 跨 4 row tile: (n-1)×R + 进入 = 3×7 + 进入 = 21 + 进入
    
    # 但 "进入" 取决于前一个 tile 的最后 row
    
    # 更简单的公式:
    # 每次 R 迭代访问 tile 时，row switches = tile 内的 row 数
    # 但第一次 R 迭代可能少 1 次（如果和前 tile 相同）
    
    # 让我用另一种方法:
    # 总 row switches = tile 间 + tile 内
    # tile 间: 每个 unique (c, p, q) 最多 1 次
    # tile 内: 跨 block 的 tile，每次 R 都要内部切换
    
    print("  另一种分解:")
    print(f"    总 switches: {total}")
    
    # tile 内 switches（仅考虑内部切换）
    # 跨 2 row: (R_l2 - 1) × 2 + 1 = 13 per tile (r=1..6 每次切 2 次回起点再到终点, r=0 切 1 次)
    # 不对... 
    
    # 实际：
    # 跨 2 row tile, rows = [a, b], 假设 a < b
    # r=0: 如果前 tile 在 row a → switch 0 次到 a，再 1 次到 b
    #      如果前 tile 在 row b → switch 1 次到 a，再 1 次到 b
    # r=1: 当前在 row b，要访问 a → switch 1 次，再访问 b → switch 1 次
    # ...
    
    # 简化：假设每次 R 迭代都重新从起点访问
    # 那么跨 2 row tile: 
    #   r=0: 2 次 (假设切换到起点)
    #   r=1: 2 次
    #   ...
    #   总计: 2 × 7 = 14 次
    # 但实际平均 13.25 次，说明有些 r=0 时没有切换
    
    # ==========================================================================
    # 最终公式
    # ==========================================================================
    print("\n" + "=" * 80)
    print("【最终公式】")
    print("=" * 80)
    
    # 一次 C × P × Q × R 扫描的 switches
    switches_per_cpqr = total
    
    # K 循环
    k_wraparound = 1  # 假设每次 K 回绕都有 switch
    
    total_formula = switches_per_cpqr * K_l3 + k_wraparound * (K_l3 - 1)
    
    print(f"""
  Row Activation = K_l3 × (C × P × Q × R 扫描的 switches) + K回绕
  
  其中:
    C × P × Q × R 扫描 = {switches_per_cpqr}
      = 不跨({no_cross_switches}) + 跨H({cross_h_switches}) + 跨W({cross_w_switches}) + 跨两者({cross_both_switches})
    
    K回绕 = {k_wraparound * (K_l3 - 1)}
  
  计算:
    = {K_l3} × {switches_per_cpqr} + {k_wraparound * (K_l3 - 1)}
    = {K_l3 * switches_per_cpqr + k_wraparound * (K_l3 - 1)}
  
  实际结果: {K_l3 * switches_per_cpqr + k_wraparound * (K_l3 - 1)}
    """)
    
    # 每个 K 迭代
    print(f"  每个 K 迭代的 switches: {switches_per_cpqr}")
    print(f"  4 个 K 迭代: 4 × {switches_per_cpqr} = {4 * switches_per_cpqr}")
    print(f"  加上 K 回绕: {4 * switches_per_cpqr} + 3 = {4 * switches_per_cpqr + 3}")
    
    # 但实际是 5880，让我验证
    print(f"\n  验证：模拟完整循环")
    current_row = None
    total_full = 0
    
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r in range(R_l2):
                        rows = get_rows_for_tile(p, q, c)
                        for row in rows:
                            if current_row != row:
                                total_full += 1
                                current_row = row
    
    print(f"  完整循环总 switches: {total_full}")


if __name__ == "__main__":
    analyze_r_loop()
