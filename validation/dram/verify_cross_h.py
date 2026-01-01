"""
验证跨 H 的 tile 的实际 switches 数

之前的表格说 跨H = 14 switches/tile，但模拟显示只有 2
让我找出差异在哪
"""

def verify_cross_h_switches():
    print("=" * 80)
    print("验证跨 H tile 的 switches")
    print("=" * 80)
    
    # 参数
    P_l3 = 28
    Q_l3 = 7
    C_l3 = 3
    R_l2 = 7
    
    H_per_tile = 2
    P_per_tile = 2
    Q_per_tile = 8
    W_per_tile = 14
    
    block_h = 31
    block_w = 31
    
    row_size = 1024
    stride_q = 1024
    stride_p = 7168
    stride_c = 200704
    
    def get_blocks_and_rows(p_tile, q_tile, c, r):
        h_start = p_tile * P_per_tile + r
        h_end = h_start + H_per_tile
        w_start = q_tile * Q_per_tile
        w_end = w_start + W_per_tile
        
        h_block_start = h_start // block_h
        h_block_end = (h_end - 1) // block_h
        w_block_start = w_start // block_w
        w_block_end = (w_end - 1) // block_w
        
        rows = []
        for hb in range(h_block_start, h_block_end + 1):
            for wb in range(w_block_start, w_block_end + 1):
                addr = hb * stride_p + wb * stride_q + c * stride_c
                row = addr // row_size
                rows.append(row)
        
        return rows
    
    # ==========================================================================
    # 统计所有 "跨 H" 的 tile
    # ==========================================================================
    print("\n【1. 找出所有跨 H 的 (p, q) 组合】")
    
    # 跨 H 的条件: p=15 (H 起点=30，H 范围 30:32 跨 h_block 边界 31)
    # 不跨 W 的条件: q ∈ {0, 1, 2, 4, 5, 6}
    
    cross_h_only_tiles = []
    
    for p in range(P_l3):
        for q in range(Q_l3):
            # 检查 r=0 时是否跨 H
            h_start = p * P_per_tile
            h_end = h_start + H_per_tile
            w_start = q * Q_per_tile
            w_end = w_start + W_per_tile
            
            h_blocks = set()
            for h in range(h_start, h_end):
                h_blocks.add(h // block_h)
            
            w_blocks = set()
            for w in range(w_start, w_end):
                w_blocks.add(w // block_w)
            
            cross_h = len(h_blocks) > 1
            cross_w = len(w_blocks) > 1
            
            if cross_h and not cross_w:
                cross_h_only_tiles.append((p, q))
    
    print(f"  跨 H 但不跨 W 的 (p, q): {cross_h_only_tiles}")
    print(f"  数量: {len(cross_h_only_tiles)}")
    
    # ==========================================================================
    # 模拟所有跨 H tile 的 switches
    # ==========================================================================
    print("\n【2. 模拟跨 H tiles 的 switches】")
    
    # 完整模拟 C × P × Q × R 循环，只统计跨 H tiles 的贡献
    current_row = None
    cross_h_switches = 0
    
    for c in range(C_l3):
        for p in range(P_l3):
            for q in range(Q_l3):
                # 判断是否跨 H only
                is_cross_h_only = (p, q) in cross_h_only_tiles
                
                for r in range(R_l2):
                    rows = get_blocks_and_rows(p, q, c, r)
                    
                    for row in rows:
                        if current_row != row:
                            if is_cross_h_only:
                                cross_h_switches += 1
                            current_row = row
    
    tiles_count = len(cross_h_only_tiles) * C_l3
    print(f"  跨 H tiles 总数: {len(cross_h_only_tiles)} × {C_l3} = {tiles_count}")
    print(f"  跨 H tiles 的总 switches: {cross_h_switches}")
    print(f"  平均 switches/tile: {cross_h_switches / tiles_count:.2f}")
    
    # ==========================================================================
    # 详细跟踪单个跨 H tile
    # ==========================================================================
    print("\n【3. 详细跟踪跨 H tile (p=15, q=0) 在完整循环中的行为】")
    
    # 找到 (p=15, q=0) 在循环中的位置
    # 前一个 tile 是 (p=14, q=6) 或者如果是新 p，前一个是 (p=15, q=-1) 不存在
    
    # 让我模拟 c=0, p=14..15 的访问
    print("  模拟 c=0, p=14..15 的访问序列:")
    
    current_row = None
    
    for p in range(14, 16):
        for q in range(Q_l3):
            is_target = (p == 15 and q == 0)
            
            for r in range(R_l2):
                rows = get_blocks_and_rows(p, q, 0, r)
                
                switches_this_iter = 0
                for row in rows:
                    if current_row != row:
                        switches_this_iter += 1
                        current_row = row
                
                if is_target:
                    print(f"    p=15, q=0, r={r}: rows={rows}, switches={switches_this_iter}")
    
    # ==========================================================================
    # 分析为什么 switches/tile = 14
    # ==========================================================================
    print("\n【4. 分析为什么平均是 14 switches/tile】")
    
    # 逐个分析每个跨 H tile
    print("  每个跨 H tile 的 switches:")
    
    for c in range(C_l3):
        for p, q in cross_h_only_tiles:
            # 获取前一个 tile 的最后 row
            if q > 0:
                prev_rows = get_blocks_and_rows(p, q-1, c, R_l2-1)
                current_row = prev_rows[-1]
            else:
                if p > 0:
                    prev_rows = get_blocks_and_rows(p-1, Q_l3-1, c, R_l2-1)
                    current_row = prev_rows[-1]
                else:
                    if c > 0:
                        prev_rows = get_blocks_and_rows(P_l3-1, Q_l3-1, c-1, R_l2-1)
                        current_row = prev_rows[-1]
                    else:
                        current_row = None
            
            tile_switches = 0
            rows_sequence = []
            
            for r in range(R_l2):
                rows = get_blocks_and_rows(p, q, c, r)
                rows_sequence.append(rows)
                
                for row in rows:
                    if current_row != row:
                        tile_switches += 1
                        current_row = row
            
            print(f"    (c={c}, p={p}, q={q}): prev_row={prev_rows[-1] if 'prev_rows' in dir() else 'None'}, "
                  f"rows_seq={rows_sequence}, switches={tile_switches}")


if __name__ == "__main__":
    verify_cross_h_switches()
