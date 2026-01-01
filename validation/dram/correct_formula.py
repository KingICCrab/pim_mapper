"""
修正后的完整公式推导

之前的表格有错误：
- 跨 H: 14 switches/tile ← 错误！实际是 2
- 跨两者: 28 switches/tile ← 需要验证

关键原因：R 循环是滑动的，跨 H 只在 r=0 时发生
"""

def correct_formula():
    print("=" * 80)
    print("修正后的 5880 Row Activation 公式推导")
    print("=" * 80)
    
    # 参数
    P_l3 = 28
    Q_l3 = 7
    C_l3 = 3
    K_l3 = 4
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
    
    def get_rows(p, q, c, r):
        h_start = p * P_per_tile + r
        h_end = h_start + H_per_tile
        w_start = q * Q_per_tile
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
    
    def classify_tile(p, q):
        """分类 tile (基于 r=0 时的 block crossing)"""
        h_start = p * P_per_tile
        h_end = h_start + H_per_tile
        w_start = q * Q_per_tile
        w_end = w_start + W_per_tile
        
        cross_h = (h_start // block_h) != ((h_end - 1) // block_h)
        cross_w = (w_start // block_w) != ((w_end - 1) // block_w)
        
        if not cross_h and not cross_w:
            return "no_cross"
        elif cross_h and not cross_w:
            return "cross_h"
        elif not cross_h and cross_w:
            return "cross_w"
        else:
            return "cross_both"
    
    # ==========================================================================
    # 完整模拟
    # ==========================================================================
    print("\n【1. 完整模拟统计】")
    
    stats = {
        "no_cross": {"tiles": 0, "switches": 0},
        "cross_h": {"tiles": 0, "switches": 0},
        "cross_w": {"tiles": 0, "switches": 0},
        "cross_both": {"tiles": 0, "switches": 0},
    }
    
    current_row = None
    total_switches = 0
    
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    tile_type = classify_tile(p, q)
                    stats[tile_type]["tiles"] += 1
                    
                    tile_switches = 0
                    for r in range(R_l2):
                        rows = get_rows(p, q, c, r)
                        for row in rows:
                            if current_row != row:
                                tile_switches += 1
                                current_row = row
                    
                    stats[tile_type]["switches"] += tile_switches
                    total_switches += tile_switches
    
    print(f"  总 switches: {total_switches}")
    print()
    
    # ==========================================================================
    # 输出修正后的表格
    # ==========================================================================
    print("【2. 修正后的表格】")
    print()
    print("  ┌──────────┬─────────┬────────┬─────────────┐")
    print("  │ 类型     │ tiles   │ 总 sw  │ sw/tile     │")
    print("  ├──────────┼─────────┼────────┼─────────────┤")
    
    for tile_type in ["no_cross", "cross_h", "cross_w", "cross_both"]:
        tiles = stats[tile_type]["tiles"]
        sw = stats[tile_type]["switches"]
        avg = sw / tiles if tiles > 0 else 0
        
        name = {
            "no_cross": "不跨",
            "cross_h": "跨 H",
            "cross_w": "跨 W",
            "cross_both": "跨两者"
        }[tile_type]
        
        print(f"  │ {name:<8} │ {tiles:>7} │ {sw:>6} │ {avg:>11.2f} │")
    
    print("  ├──────────┼─────────┼────────┼─────────────┤")
    total_tiles = sum(s["tiles"] for s in stats.values())
    print(f"  │ 总计     │ {total_tiles:>7} │ {total_switches:>6} │             │")
    print("  └──────────┴─────────┴────────┴─────────────┘")
    
    # ==========================================================================
    # 分析 R 循环的影响
    # ==========================================================================
    print("\n【3. R 循环滑动的关键影响】")
    
    print("""
  跨 H 的 tile (p=15):
    - r=0: H[30:32] 跨 h_block 边界，访问 2 个 row
    - r=1..6: H[31+] 全在 h_block=1 内，访问 1 个 row
    → 只有 r=0 时跨 H，所以 switches 少
    
  跨 W 的 tile (q=3):
    - W 范围不变 [24:38]，始终跨 w_block 边界
    - 每次 r 都访问 2 个 row
    → 每次 r 都要切换 row
    
  跨两者的 tile (p=15, q=3):
    - r=0: 跨 H 和 W，访问 4 个 row
    - r=1..6: 只跨 W，访问 2 个 row
    """)
    
    # ==========================================================================
    # 推导公式
    # ==========================================================================
    print("\n【4. 推导各类 tile 的 switches 公式】")
    
    # 不跨: 每次 r 访问 1 个 row，只有 tile 间可能切换
    no_cross_sw = stats["no_cross"]["switches"]
    no_cross_tiles = stats["no_cross"]["tiles"]
    
    # 跨 H: r=0 访问 2 row，r=1..6 访问 1 row
    # r=0: 可能 2 次 switch (进入 + 内部)
    # r=1..6: 可能 1 次 switch (tile 间)
    cross_h_sw = stats["cross_h"]["switches"]
    cross_h_tiles = stats["cross_h"]["tiles"]
    
    # 跨 W: 每次 r 都访问 2 row
    # r=0: 可能 1-2 次 switch
    # r=1..6: 每次 2 次 switch (回起点 + 到终点)
    cross_w_sw = stats["cross_w"]["switches"]
    cross_w_tiles = stats["cross_w"]["tiles"]
    
    # 跨两者: r=0 访问 4 row，r=1..6 访问 2 row
    cross_both_sw = stats["cross_both"]["switches"]
    cross_both_tiles = stats["cross_both"]["tiles"]
    
    print(f"  不跨 block:")
    print(f"    tiles = {no_cross_tiles}")
    print(f"    switches = {no_cross_sw}")
    print(f"    avg = {no_cross_sw / no_cross_tiles:.4f}")
    print()
    
    print(f"  跨 H:")
    print(f"    tiles = {cross_h_tiles}")
    print(f"    switches = {cross_h_sw}")
    print(f"    avg = {cross_h_sw / cross_h_tiles:.2f}")
    print(f"    公式: r=0 访问 2 row，r=1..6 访问 1 row")
    print(f"          每 tile: ~2 switches (r=0 的内部切换)")
    print()
    
    print(f"  跨 W:")
    print(f"    tiles = {cross_w_tiles}")
    print(f"    switches = {cross_w_sw}")
    print(f"    avg = {cross_w_sw / cross_w_tiles:.2f}")
    print(f"    公式: 每次 r 访问 2 row")
    print(f"          每 tile: 1 + 2×(R-1) = 1 + 2×6 = 13 switches")
    print()
    
    print(f"  跨两者:")
    print(f"    tiles = {cross_both_tiles}")
    print(f"    switches = {cross_both_sw}")
    print(f"    avg = {cross_both_sw / cross_both_tiles:.2f}")
    print(f"    公式: r=0 访问 4 row，r=1..6 访问 2 row")
    print(f"          每 tile: ~16 switches")
    
    # ==========================================================================
    # 验证公式
    # ==========================================================================
    print("\n【5. 验证公式】")
    
    # 跨 W: (1 + 2×6) × tiles = 13 × 324 = 4212
    predicted_cross_w = 13 * cross_w_tiles
    print(f"  跨 W 预测: 13 × {cross_w_tiles} = {predicted_cross_w}")
    print(f"  跨 W 实际: {cross_w_sw}")
    
    # ==========================================================================
    # 最终公式
    # ==========================================================================
    print("\n" + "=" * 80)
    print("【最终公式】")
    print("=" * 80)
    
    # 每个 K 迭代
    sw_per_k = total_switches // K_l3
    
    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │               修正后的 Row Activation 公式                      │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                 │
  │   Total = K_l3 × (SW_no + SW_H + SW_W + SW_both)               │
  │         = {K_l3} × ({no_cross_sw//K_l3} + {cross_h_sw//K_l3} + {cross_w_sw//K_l3} + {cross_both_sw//K_l3})               │
  │         = {K_l3} × {sw_per_k}                                              │
  │         = {total_switches}                                                        │
  │                                                                 │
  │   各项 switches/tile:                                           │
  │     不跨: ~{no_cross_sw / no_cross_tiles:.2f} (只有 tile 间切换)                │
  │     跨H:  {cross_h_sw / cross_h_tiles:.2f} (r=0 跨 H，r=1..6 不跨)              │
  │     跨W:  {cross_w_sw / cross_w_tiles:.2f} (每次 r 都跨 W)                      │
  │     跨两者: {cross_both_sw / cross_both_tiles:.2f} (r=0 跨两者，r=1..6 只跨 W)    │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    correct_formula()
