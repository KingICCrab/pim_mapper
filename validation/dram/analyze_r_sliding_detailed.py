"""
深入分析 R 循环滑动时的 block crossing 变化

关键发现：R 循环滑动时，h_block 可能会变化！
"""

def analyze_r_sliding_blocks():
    print("=" * 80)
    print("R 循环滑动时的 Block Crossing 变化")
    print("=" * 80)
    
    # 参数
    R_l2 = 7
    H_per_tile = 2
    block_h = 31
    block_w = 31
    P_per_tile = 2
    Q_per_tile = 8
    W_per_tile = 14
    
    row_size = 1024
    stride_q = 1024
    stride_p = 7168
    stride_c = 200704
    
    def get_blocks_and_rows(p_tile, q_tile, c, r):
        """获取单次 (p, q, c, r) 访问的 blocks 和 rows"""
        h_start = p_tile * P_per_tile + r
        h_end = h_start + H_per_tile
        w_start = q_tile * Q_per_tile
        w_end = w_start + W_per_tile
        
        h_block_start = h_start // block_h
        h_block_end = (h_end - 1) // block_h
        w_block_start = w_start // block_w
        w_block_end = (w_end - 1) // block_w
        
        blocks = []
        rows = []
        for hb in range(h_block_start, h_block_end + 1):
            for wb in range(w_block_start, w_block_end + 1):
                blocks.append((hb, wb))
                addr = hb * stride_p + wb * stride_q + c * stride_c
                row = addr // row_size
                rows.append(row)
        
        return h_start, h_end, blocks, rows
    
    # ==========================================================================
    # 分析跨 H 的 tile (p=15)
    # ==========================================================================
    print("\n【1. 分析跨 H 的 tile (p=15, q=0, c=0)】")
    
    p_tile, q_tile, c = 15, 0, 0
    
    print(f"  p_tile={p_tile} 的 H 起点: {p_tile * P_per_tile} = 30")
    print(f"  block_h = 31")
    print()
    
    print("  R 循环中每次访问:")
    for r in range(R_l2):
        h_start, h_end, blocks, rows = get_blocks_and_rows(p_tile, q_tile, c, r)
        cross_h = len(set(b[0] for b in blocks)) > 1
        print(f"    r={r}: H[{h_start}:{h_end}], blocks={blocks}, rows={rows}, 跨H={cross_h}")
    
    print("""
  关键观察:
  - r=0: H[30:32] 跨越 h_block=0 和 h_block=1，访问 2 个 row
  - r=1: H[31:33] 全在 h_block=1 内，访问 1 个 row
  - ...
  - r=6: H[36:38] 全在 h_block=1 内，访问 1 个 row
  
  所以 "跨 H" 只在 r=0 时发生！
    """)
    
    # ==========================================================================
    # 分析跨 W 的 tile (q=3)
    # ==========================================================================
    print("\n【2. 分析跨 W 的 tile (p=0, q=3, c=0)】")
    
    p_tile, q_tile, c = 0, 3, 0
    
    w_start = q_tile * Q_per_tile
    w_end = w_start + W_per_tile
    
    print(f"  W 范围: [{w_start}, {w_end}) = [24, 38)")
    print(f"  block_w = 31")
    print(f"  W 跨越 w=31 边界")
    print()
    
    print("  R 循环中每次访问:")
    for r in range(R_l2):
        h_start, h_end, blocks, rows = get_blocks_and_rows(p_tile, q_tile, c, r)
        cross_w = len(set(b[1] for b in blocks)) > 1
        print(f"    r={r}: H[{h_start}:{h_end}], blocks={blocks}, rows={rows}, 跨W={cross_w}")
    
    print("""
  关键观察:
  - 每次 r 迭代都跨 W (因为 W 范围不变)
  - H 范围虽然滑动，但都在 h_block=0 内
  - 所以每次 r 都访问 2 个 row
    """)
    
    # ==========================================================================
    # 分析跨两者的 tile (p=15, q=3)
    # ==========================================================================
    print("\n【3. 分析跨两者的 tile (p=15, q=3, c=0)】")
    
    p_tile, q_tile, c = 15, 3, 0
    
    print("  R 循环中每次访问:")
    for r in range(R_l2):
        h_start, h_end, blocks, rows = get_blocks_and_rows(p_tile, q_tile, c, r)
        cross_h = len(set(b[0] for b in blocks)) > 1
        cross_w = len(set(b[1] for b in blocks)) > 1
        print(f"    r={r}: H[{h_start}:{h_end}], blocks={blocks}, rows={rows}, 跨H={cross_h}, 跨W={cross_w}")
    
    print("""
  关键观察:
  - r=0: H[30:32] 跨 H，W[24:38] 跨 W → 4 个 block，4 个 row
  - r=1..6: H 全在 h_block=1 内，只跨 W → 2 个 block，2 个 row
    """)
    
    # ==========================================================================
    # 重新计算每 tile switches
    # ==========================================================================
    print("\n" + "=" * 80)
    print("【4. 重新计算每 tile switches (考虑 R 滑动)】")
    print("=" * 80)
    
    test_cases = [
        (0, 0, 0, "不跨 block"),
        (15, 0, 0, "仅跨 H"),
        (0, 3, 0, "仅跨 W"),
        (15, 3, 0, "跨两者"),
    ]
    
    for p_tile, q_tile, c, desc in test_cases:
        print(f"\n  {desc} (p={p_tile}, q={q_tile}, c={c}):")
        
        # 获取前一个 tile 的最后 row
        if q_tile > 0:
            _, _, _, prev_rows = get_blocks_and_rows(p_tile, q_tile-1, c, R_l2-1)
            current_row = prev_rows[-1]
            print(f"    前一个 tile 结束在 row {current_row}")
        else:
            current_row = None
            print(f"    没有前一个 tile")
        
        total_switches = 0
        rows_per_r = []
        
        for r in range(R_l2):
            _, _, blocks, rows = get_blocks_and_rows(p_tile, q_tile, c, r)
            rows_per_r.append(rows)
            
            switches_this_r = 0
            for row in rows:
                if current_row != row:
                    switches_this_r += 1
                    current_row = row
            
            total_switches += switches_this_r
        
        print(f"    每次 r 的 rows: {rows_per_r}")
        print(f"    总 switches: {total_switches}")
        
        # 分析公式
        unique_rows_per_r = [len(set(rows)) for rows in rows_per_r]
        print(f"    每次 r 的 unique rows 数: {unique_rows_per_r}")
    
    # ==========================================================================
    # 总结公式
    # ==========================================================================
    print("\n" + "=" * 80)
    print("【5. 修正后的公式】")
    print("=" * 80)
    
    print("""
  关键发现：R 循环滑动时，跨 H 的行为会变化！
  
  1. 不跨 block: 
     - 每次 r 访问 1 个 row
     - 只有 tile 间可能切换
     
  2. 仅跨 H (p=15):
     - r=0: 跨 H 边界，访问 2 个 row
     - r=1..6: 不跨 H，访问 1 个 row
     - 所以不是 "14 = 2×R"，而是更少
     
  3. 仅跨 W (q=3):
     - 每次 r 都跨 W (W 范围不变)
     - 每次访问 2 个 row
     - 公式: 1 + 2×(R-1) = 1 + 12 = 13 ✓
     
  4. 跨两者 (p=15, q=3):
     - r=0: 跨 H 和 W，访问 4 个 row
     - r=1..6: 只跨 W，访问 2 个 row
     
  这解释了为什么之前的模拟结果是:
    - 跨H: 14 (不是简单的 2×7)
    - 跨W: 13 = 2×7 - 1
    - 跨两者: 28 (不是简单的 4×7)
    """)


if __name__ == "__main__":
    analyze_r_sliding_blocks()
