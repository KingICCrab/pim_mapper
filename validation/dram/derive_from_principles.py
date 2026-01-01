"""
从原理推导 Row Activation 的正确计算

不把 Trace 当作 golden model，而是从访问模式原理出发推导
"""

def derive_from_principles():
    print("=" * 80)
    print("从原理推导 Row Activation")
    print("=" * 80)
    
    # ==========================================================================
    # 1. 明确参数
    # ==========================================================================
    print("\n【1. 参数定义】")
    
    # Workload
    R, S = 7, 7
    P, Q = 56, 56
    C, K, N = 3, 64, 1
    H, W = P + R - 1, Q + S - 1  # 62, 62
    
    # DRAM loop factors
    K_l3, C_l3, P_l3, Q_l3 = 4, 3, 28, 7
    R_l2 = 7
    
    # Buffer tile (访问的 Output tile 大小)
    P_tile, Q_tile = 2, 8  # Output tile
    
    # 对应的 Input tile (不含 R 滑动)
    H_tile = P_tile  # = 2 (每次 r 迭代访问的 H 行数)
    W_tile = Q_tile + S - 1  # = 14
    
    # Block layout
    block_h, block_w = 31, 31
    row_size = 1024
    
    print(f"  Input: H={H}, W={W}, C={C}")
    print(f"  Output tile: P_tile={P_tile}, Q_tile={Q_tile}")
    print(f"  Input per R iter: H_tile={H_tile}, W_tile={W_tile}")
    print(f"  DRAM loops: K={K_l3}, C={C_l3}, P={P_l3}, Q={Q_l3}, R={R_l2}")
    print(f"  Block: {block_h}×{block_w}, Row size: {row_size}")
    
    # ==========================================================================
    # 2. 分析循环结构
    # ==========================================================================
    print("\n【2. 循环结构分析】")
    
    print("""
  DRAM 循环 (outer → inner):
    for k in K_l3:     # K 对 Input 不相关
        for c in C_l3:     # C 相关
            for p in P_l3:     # P 相关 (决定 H 位置)
                for q in Q_l3:     # Q 相关 (决定 W 位置)
                    for r in R_l2:     # R 相关 (在 H 方向滑动)
                        access_input(p, q, c, r)
                        
  每次迭代访问的 Input 区域:
    H 范围: [p * P_tile + r, p * P_tile + r + H_tile)
    W 范围: [q * Q_tile, q * Q_tile + W_tile)
    """)
    
    # ==========================================================================
    # 3. 核心问题：访问顺序和 Row 计算
    # ==========================================================================
    print("\n【3. Row 计算方式】")
    
    # row_aligned layout:
    # 每个 block (h_block, w_block, c) 有自己的 row
    # block 内部的元素在同一个 row 中
    
    # Row 地址计算:
    # block_base = h_block * stride_p + w_block * stride_q + c * stride_c
    # row = block_base // row_size
    
    # Strides (每个 block 跳过的字节数)
    stride_q = 1024   # 每个 w_block
    stride_p = 7168   # 每个 h_block (= Q_l3 * stride_q = 7 * 1024)
    stride_c = 200704 # 每个 C (= P_l3 * stride_p = 28 * 7168)
    
    print(f"  Strides: q={stride_q}, p={stride_p}, c={stride_c}")
    print(f"  Row 计算: row = (h_block * {stride_p} + w_block * {stride_q} + c * {stride_c}) / {row_size}")
    
    # ==========================================================================
    # 4. 分析不同的 Row Activation 统计方式
    # ==========================================================================
    print("\n【4. 三种可能的统计方式】")
    
    print("""
  方式 A: 只统计 Unique Rows
    - 不考虑访问顺序
    - 结果 = 12 (分析显示只有 12 个不同的 row)
    
  方式 B: 统计 Row Switches (tile 级别)
    - 每个 tile 作为整体，统计 tile 间的 row 切换
    - 每个 tile 的 base row 切换时计 1 次
    - 结果 ≈ 2352 (P_l3 × Q_l3 × C_l3 × K_l3 = 28×7×3×4 = 2352)
    
  方式 C: 统计 Row Switches (element 级别)
    - 每次访问的 row 和上一次不同时计 1 次
    - 需要考虑 block crossing 和 R 滑动
    - 结果取决于具体的访问顺序
    """)
    
    # ==========================================================================
    # 5. 方式 B: Tile 级别统计
    # ==========================================================================
    print("\n【5. 方式 B: Tile 级别统计】")
    
    def get_tile_base_row(p, q, c):
        """获取 tile 的 base row (r=0 时第一个 block 的 row)"""
        h_block = (p * P_tile) // block_h
        w_block = (q * Q_tile) // block_w
        addr = h_block * stride_p + w_block * stride_q + c * stride_c
        return addr // row_size
    
    # 统计 tile 间的 row switch
    current_row = None
    tile_level_switches = 0
    
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    row = get_tile_base_row(p, q, c)
                    if current_row != row:
                        tile_level_switches += 1
                        current_row = row
    
    print(f"  Tile 级别 row switches: {tile_level_switches}")
    print(f"  (这接近 ILP 的 2392)")
    
    # ==========================================================================
    # 6. 方式 C: Element 级别统计 (考虑 R 滑动)
    # ==========================================================================
    print("\n【6. 方式 C: Element 级别统计 (R 滑动)】")
    
    def get_blocks_for_iteration(p, q, c, r):
        """获取单次 (p, q, c, r) 迭代访问的所有 block 和对应的 row"""
        h_start = p * P_tile + r
        h_end = h_start + H_tile
        w_start = q * Q_tile
        w_end = w_start + W_tile
        
        h_block_start = h_start // block_h
        h_block_end = (h_end - 1) // block_h
        w_block_start = w_start // block_w
        w_block_end = (w_end - 1) // block_w
        
        blocks = []
        for hb in range(h_block_start, h_block_end + 1):
            for wb in range(w_block_start, w_block_end + 1):
                addr = hb * stride_p + wb * stride_q + c * stride_c
                row = addr // row_size
                blocks.append((hb, wb, row))
        return blocks
    
    # 完整模拟
    current_row = None
    element_level_switches = 0
    
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r in range(R_l2):
                        blocks = get_blocks_for_iteration(p, q, c, r)
                        for hb, wb, row in blocks:
                            if current_row != row:
                                element_level_switches += 1
                                current_row = row
    
    print(f"  Element 级别 row switches (R 滑动): {element_level_switches}")
    
    # ==========================================================================
    # 7. 方式 D: Element 级别统计 (R 不滑动，重复访问)
    # ==========================================================================
    print("\n【7. 方式 D: Element 级别统计 (R 重复访问同一位置)】")
    
    def get_blocks_no_sliding(p, q, c):
        """获取 tile 访问的所有 block (不考虑 R 滑动)"""
        h_start = p * P_tile
        h_end = h_start + H_tile
        w_start = q * Q_tile
        w_end = w_start + W_tile
        
        h_block_start = h_start // block_h
        h_block_end = (h_end - 1) // block_h
        w_block_start = w_start // block_w
        w_block_end = (w_end - 1) // block_w
        
        blocks = []
        for hb in range(h_block_start, h_block_end + 1):
            for wb in range(w_block_start, w_block_end + 1):
                addr = hb * stride_p + wb * stride_q + c * stride_c
                row = addr // row_size
                blocks.append((hb, wb, row))
        return blocks
    
    # 模拟：每次 R 迭代重复访问同一组 block
    current_row = None
    repeated_switches = 0
    
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    blocks = get_blocks_no_sliding(p, q, c)
                    for r in range(R_l2):
                        for hb, wb, row in blocks:
                            if current_row != row:
                                repeated_switches += 1
                                current_row = row
    
    print(f"  Element 级别 row switches (R 重复): {repeated_switches}")
    print(f"  (这接近 Trace 的 5880)")
    
    # ==========================================================================
    # 8. 总结
    # ==========================================================================
    print("\n" + "=" * 80)
    print("【8. 总结】")
    print("=" * 80)
    
    print(f"""
  三种统计方式的结果:
  
  ┌───────────────────────────────────────────────────┐
  │ 方式                           │ 结果   │ 对应   │
  ├───────────────────────────────────────────────────┤
  │ B. Tile 级别 (只看 base row)   │ {tile_level_switches:>5}  │ ≈ILP   │
  │ C. Element 级别 (R 滑动)       │ {element_level_switches:>5}  │        │
  │ D. Element 级别 (R 重复)       │ {repeated_switches:>5}  │ ≈Trace │
  └───────────────────────────────────────────────────┘
  
  ILP 预测: 2392 (接近方式 B)
  Trace 结果: 5880 (接近方式 D)
  
  关键差异:
  1. ILP 可能只考虑 tile 级别的 row switch
  2. Trace 可能把每次 R 迭代都当作重复访问同一组 block
  
  哪个是"正确的"？
  - 取决于 DRAM 访问的实际行为
  - 如果 R 循环内的多次访问可以被 row buffer 缓存，则方式 B 更合理
  - 如果每次 R 迭代都是独立的 DRAM 访问，则需要看具体实现
    """)
    
    # ==========================================================================
    # 9. 分析 Trace 的行为
    # ==========================================================================
    print("\n【9. 分析 Trace 的具体行为】")
    
    # 看看 Trace 的 5880 是怎么来的
    # 5880 / 4 = 1470 per K
    # 1470 / 3 = 490 per C? 不是整数
    
    print(f"  5880 / K_l3 = 5880 / 4 = {5880 / 4}")
    print(f"  5880 / (K_l3 × C_l3) = 5880 / 12 = {5880 / 12}")
    print(f"  5880 / (K_l3 × C_l3 × P_l3) = 5880 / 336 = {5880 / 336:.2f}")
    
    # 我的方式 D 结果
    print(f"\n  方式 D: {repeated_switches}")
    print(f"  {repeated_switches} / K_l3 = {repeated_switches / 4}")


if __name__ == "__main__":
    derive_from_principles()
