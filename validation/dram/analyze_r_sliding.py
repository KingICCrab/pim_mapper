"""
分析 R 循环中 Input tile 的实际访问模式

R 循环是滤波器在 H 方向的滑动，每次 r 迭代访问 Input 的不同行
"""

def analyze_r_loop_sliding():
    print("=" * 80)
    print("R 循环的滑动窗口分析")
    print("=" * 80)
    
    # 参数
    R = 7           # filter height
    S = 7           # filter width  
    P = 56          # output height
    Q = 56          # output width
    H = 62          # input height (= P + R - 1)
    W = 62          # input width (= Q + S - 1)
    
    P_l3 = 28
    Q_l3 = 7
    R_l2 = 7
    
    # Tile sizes
    P_per_tile = 2   # 每个 p_tile 覆盖的 P 范围
    Q_per_tile = 8   # 每个 q_tile 覆盖的 Q 范围
    
    # 对应的 Input 范围
    H_per_tile = P_per_tile  # = 2 (不含 R 的滑动)
    W_per_tile = Q_per_tile + S - 1  # = 8 + 6 = 14
    
    print("\n【1. 基本参数】")
    print(f"  Output: P={P}, Q={Q}")
    print(f"  Input: H={H}, W={W}")
    print(f"  Filter: R={R}, S={S}")
    print(f"  DRAM loops: P_l3={P_l3}, Q_l3={Q_l3}, R_l2={R_l2}")
    print(f"  Tile size: P_per_tile={P_per_tile}, Q_per_tile={Q_per_tile}")
    
    # ==========================================================================
    # R 循环的滑动窗口
    # ==========================================================================
    print("\n【2. R 循环的滑动窗口】")
    
    print("""
  卷积关系: output[p][q] = sum over r,s of input[p+r][q+s] * filter[r][s]
  
  在 R 循环中:
    for r in range(R_l2):
        # 访问 Input[p_tile*2 + r : p_tile*2 + r + H_per_tile, ...]
        
  所以每次 r 迭代，访问的 H 范围会滑动 1 行
    """)
    
    # 示例：p_tile = 0, q_tile = 0
    print("  示例：p_tile=0, q_tile=0 的 R 循环")
    print()
    
    p_tile = 0
    q_tile = 0
    
    p_start = p_tile * P_per_tile  # = 0
    q_start = q_tile * Q_per_tile  # = 0
    
    for r in range(R_l2):
        h_start = p_start + r
        h_end = h_start + H_per_tile
        w_start = q_start
        w_end = w_start + W_per_tile
        
        print(f"    r={r}: Input[{h_start}:{h_end}, {w_start}:{w_end}]")
    
    # ==========================================================================
    # 分析单个 tile 在 R 循环中访问的所有行
    # ==========================================================================
    print("\n【3. 单个 tile 在 R 循环中访问的所有行】")
    
    # 对于 (p_tile=0, q_tile=0):
    # r=0: H[0:2], W[0:14]
    # r=1: H[1:3], W[0:14]
    # ...
    # r=6: H[6:8], W[0:14]
    
    # 总共覆盖 H[0:8]，即 8 行
    
    total_h_range = H_per_tile + R_l2 - 1
    print(f"  每个 (p, q) tile 在整个 R 循环中覆盖的 H 范围:")
    print(f"    H_per_tile + R_l2 - 1 = {H_per_tile} + {R_l2} - 1 = {total_h_range} 行")
    
    # ==========================================================================
    # 重新理解 block crossing 和 row activation
    # ==========================================================================
    print("\n【4. 重新理解 block crossing】")
    
    block_h = 31
    block_w = 31
    
    print(f"  Block size: {block_h} × {block_w}")
    print(f"  每个 tile 在 R 循环中覆盖 {total_h_range} 行 (H方向)")
    print(f"  每个 tile 覆盖 {W_per_tile} 列 (W方向)")
    
    # 检查 (p_tile=0, q_tile=0) 跨越哪些 block
    print(f"\n  检查 (p_tile=0, q_tile=0) 的 block crossing:")
    
    p_tile = 0
    q_tile = 0
    
    h_start_total = p_tile * P_per_tile
    h_end_total = h_start_total + total_h_range
    w_start = q_tile * Q_per_tile
    w_end = w_start + W_per_tile
    
    print(f"    H 范围 (整个 R 循环): [{h_start_total}, {h_end_total}) = [0, 8)")
    print(f"    W 范围: [{w_start}, {w_end}) = [0, 14)")
    
    h_blocks = set()
    for h in range(h_start_total, h_end_total):
        h_blocks.add(h // block_h)
    
    w_blocks = set()
    for w in range(w_start, w_end):
        w_blocks.add(w // block_w)
    
    print(f"    H blocks: {h_blocks}")
    print(f"    W blocks: {w_blocks}")
    
    # ==========================================================================
    # R 循环中的逐次访问
    # ==========================================================================
    print("\n【5. R 循环中的逐次访问】")
    
    print("  分析 (p=0, q=3, c=0) 在 R 循环中的访问 (这是跨 W 的 tile):")
    
    p_tile = 0
    q_tile = 3
    c = 0
    
    w_start = q_tile * Q_per_tile  # = 24
    w_end = w_start + W_per_tile   # = 38
    
    print(f"    W 范围: [{w_start}, {w_end}) = [24, 38)")
    print(f"    W block: {w_start // block_w} 到 {(w_end-1) // block_w}")
    print(f"    跨越 W block 边界 (w=31)")
    
    # Row 计算参数
    row_size = 1024
    stride_q = 1024      # 每个 w_block 跳 1 row
    stride_p = 7168      # 每个 h_block 跳 7 rows
    stride_c = 200704
    
    print(f"\n    R 循环中每次访问的 rows:")
    
    for r in range(R_l2):
        h_start = p_tile * P_per_tile + r
        h_end = h_start + H_per_tile
        
        # 计算这次访问的 block
        h_block_start = h_start // block_h
        h_block_end = (h_end - 1) // block_h
        w_block_start = w_start // block_w
        w_block_end = (w_end - 1) // block_w
        
        # 计算 rows
        rows = []
        for hb in range(h_block_start, h_block_end + 1):
            for wb in range(w_block_start, w_block_end + 1):
                addr = hb * stride_p + wb * stride_q + c * stride_c
                row = addr // row_size
                rows.append(row)
        
        print(f"      r={r}: H[{h_start}:{h_end}], h_blocks={list(range(h_block_start, h_block_end+1))}, "
              f"w_blocks={list(range(w_block_start, w_block_end+1))}, rows={rows}")
    
    # ==========================================================================
    # 关键发现
    # ==========================================================================
    print("\n【6. 关键发现】")
    
    print("""
  观察上面的输出:
  
  对于 (p=0, q=3) 这个跨 W 的 tile:
    - H 范围固定为 2 行 (H_per_tile=2)
    - 但 R 循环会滑动 H 的起点
    - r=0: H[0:2], r=1: H[1:3], ..., r=6: H[6:8]
    
  每次 r 迭代:
    - H 方向: 只访问 2 行，都在 h_block=0 内 (因为 0-7 都 < 31)
    - W 方向: 访问 14 列，跨越 w_block=0 和 w_block=1
    - 所以每次 r 都访问 2 个 row (row 0 和 row 1)
    
  这就是为什么跨 W 的 tile 每次 R 迭代都有 row switch!
    """)
    
    # ==========================================================================
    # 验证模拟
    # ==========================================================================
    print("\n【7. 验证模拟】")
    
    def get_rows_for_r_iteration(p_tile, q_tile, c, r):
        """获取单次 r 迭代访问的 rows"""
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
    
    # 模拟 (p=0, q=3, c=0) 的 R 循环
    print("  模拟 (p=0, q=3, c=0) 的 R 循环，跟踪 row switch:")
    
    current_row = 0  # 假设从 q=2 结束时在 row 0
    total_switches = 0
    
    p_tile, q_tile, c = 0, 3, 0
    
    for r in range(R_l2):
        rows = get_rows_for_r_iteration(p_tile, q_tile, c, r)
        switches_this_r = 0
        
        for row in rows:
            if current_row != row:
                switches_this_r += 1
                current_row = row
        
        total_switches += switches_this_r
        print(f"    r={r}: rows={rows}, switches={switches_this_r}, total={total_switches}")
    
    print(f"\n  (p=0, q=3, c=0) 在 R 循环中总 switches: {total_switches}")
    
    # ==========================================================================
    # 对比不同 tile 类型
    # ==========================================================================
    print("\n【8. 对比不同 tile 类型在 R 循环中的行为】")
    
    test_tiles = [
        (0, 0, 0, "不跨 block"),
        (0, 3, 0, "跨 W"),
        (15, 0, 0, "跨 H"),
        (15, 3, 0, "跨两者"),
    ]
    
    for p_tile, q_tile, c, desc in test_tiles:
        print(f"\n  {desc} (p={p_tile}, q={q_tile}, c={c}):")
        
        # 假设从某个合理的初始 row 开始
        # 获取前一个 tile 的最后 row
        if q_tile > 0:
            prev_rows = get_rows_for_r_iteration(p_tile, q_tile-1, c, R_l2-1)
            current_row = prev_rows[-1]
        else:
            current_row = None
        
        total_switches = 0
        
        for r in range(R_l2):
            rows = get_rows_for_r_iteration(p_tile, q_tile, c, r)
            
            for row in rows:
                if current_row != row:
                    total_switches += 1
                    current_row = row
            
            if r == 0 or r == R_l2 - 1:
                print(f"    r={r}: rows={rows}")
        
        print(f"    总 switches: {total_switches}")


if __name__ == "__main__":
    analyze_r_loop_sliding()
