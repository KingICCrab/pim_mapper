#!/usr/bin/env python3
"""
分析哪些 tile 会产生 W/H 边界跨越 (cross)
"""

def analyze_crossing_tiles():
    # 参数 (来自 ResNet-L1 mapping)
    block_h, block_w = 31, 31
    P_l3, Q_l3, C_l3, K_l3 = 7, 28, 3, 4
    S_l2 = 7  # Level 2 的 S 循环

    P_buf, Q_buf = 8, 2  # buffer tile
    R_buf, S_buf = 7, 1

    # 计算 H_per_tile, W_per_tile (access tile size)
    stride_h, stride_w = 1, 1
    dilation_h, dilation_w = 1, 1
    H_per_tile = (P_buf - 1) * stride_h + (R_buf - 1) * dilation_h + 1  # = 14
    W_per_tile = (Q_buf - 1) * stride_w + (S_buf - 1) * dilation_w + 1  # = 2

    print("=" * 80)
    print("分析产生边界跨越 (cross) 的 Tile")
    print("=" * 80)
    print(f"\n参数:")
    print(f"  block_h = {block_h}, block_w = {block_w}")
    print(f"  H_per_tile = {H_per_tile}, W_per_tile = {W_per_tile}")
    print(f"  P_l3 = {P_l3}, Q_l3 = {Q_l3}, C_l3 = {C_l3}, K_l3 = {K_l3}")
    print(f"  S_l2 = {S_l2}")
    print(f"  P_buf = {P_buf}, Q_buf = {Q_buf}")

    print(f"\nDRAM 循环结构:")
    print(f"  for k in range({K_l3}):        # K_l3")
    print(f"    for c in range({C_l3}):      # C_l3")
    print(f"      for q in range({Q_l3}):    # Q_l3")
    print(f"        for p in range({P_l3}):  # P_l3")
    print(f"          for s in range({S_l2}):  # S_l2 (Level 2)")

    # 收集跨越边界的 tile
    w_crossing_tiles = []
    h_crossing_tiles = []

    total_tiles = K_l3 * C_l3 * Q_l3 * P_l3 * S_l2
    tile_idx = 0

    for k in range(K_l3):
        for c in range(C_l3):
            for q in range(Q_l3):
                for p in range(P_l3):
                    for s in range(S_l2):
                        # 计算 H, W 起始位置
                        p_start = p * P_buf
                        q_start = q * Q_buf + s  # S 会影响 W 的起始位置
                        
                        h_start = p_start * stride_h
                        h_end = h_start + H_per_tile - 1
                        w_start = q_start * stride_w
                        w_end = w_start + W_per_tile - 1
                        
                        # 检查 W 是否跨越 block_w 边界
                        w_start_block = w_start // block_w
                        w_end_block = w_end // block_w
                        
                        # 检查 H 是否跨越 block_h 边界
                        h_start_block = h_start // block_h
                        h_end_block = h_end // block_h
                        
                        if w_start_block != w_end_block:
                            w_crossing_tiles.append({
                                'tile_idx': tile_idx,
                                'k': k, 'c': c, 'q': q, 'p': p, 's': s,
                                'w_start': w_start, 'w_end': w_end,
                                'w_blocks': (w_start_block, w_end_block),
                                'h_start': h_start, 'h_end': h_end,
                            })
                        
                        if h_start_block != h_end_block:
                            h_crossing_tiles.append({
                                'tile_idx': tile_idx,
                                'k': k, 'c': c, 'q': q, 'p': p, 's': s,
                                'h_start': h_start, 'h_end': h_end,
                                'h_blocks': (h_start_block, h_end_block),
                            })
                        
                        tile_idx += 1

    print(f"\n" + "=" * 80)
    print(f"W 边界跨越 (W crossing block_w={block_w}) 的 Tile")
    print("=" * 80)
    print(f"\n总 tile 数: {total_tiles}")
    print(f"W 跨越边界的 tile 数: {len(w_crossing_tiles)}")

    # 按 Q 分组显示
    print(f"\n按 Q_tile 分组的 W 跨越情况:")
    q_crossing = {}
    for t in w_crossing_tiles:
        q_val = t['q']
        if q_val not in q_crossing:
            q_crossing[q_val] = []
        q_crossing[q_val].append(t)

    for q_val in sorted(q_crossing.keys()):
        tiles = q_crossing[q_val]
        sample = tiles[0]
        count = len(tiles)
        # 获取这个 Q 下所有的 S 值
        s_values = sorted(set(t['s'] for t in tiles))
        print(f"  Q={q_val}: W范围跨越 block {sample['w_blocks'][0]} -> {sample['w_blocks'][1]}, "
              f"共 {count} 个 tile, S值={s_values}")

    # 显示具体的 (Q, S) 组合
    print(f"\n具体的 (Q, S) 组合导致 W 跨越:")
    qs_crossing = set()
    for t in w_crossing_tiles:
        qs_crossing.add((t['q'], t['s']))

    for q_val, s_val in sorted(qs_crossing):
        w_start = q_val * Q_buf + s_val
        w_end = w_start + W_per_tile - 1
        boundary = (w_end // block_w) * block_w
        print(f"  Q={q_val:2d}, S={s_val}: W=[{w_start:3d}, {w_end:3d}], "
              f"边界={boundary}, blocks=[{w_start // block_w}, {w_end // block_w}]")

    print(f"\n" + "=" * 80)
    print(f"H 边界跨越 (H crossing block_h={block_h}) 的 Tile")
    print("=" * 80)
    print(f"H 跨越边界的 tile 数: {len(h_crossing_tiles)}")

    # 按 P 分组显示
    p_crossing = {}
    for t in h_crossing_tiles:
        p_val = t['p']
        if p_val not in p_crossing:
            p_crossing[p_val] = []
        p_crossing[p_val].append(t)

    for p_val in sorted(p_crossing.keys()):
        tiles = p_crossing[p_val]
        sample = tiles[0]
        count = len(tiles)
        print(f"  P={p_val}: H=[{sample['h_start']}, {sample['h_end']}], "
              f"跨越 block {sample['h_blocks'][0]} -> {sample['h_blocks'][1]}, "
              f"共 {count} 个 tile")

    print(f"\n" + "=" * 80)
    print(f"Row Switch 估算 (简化模型)")
    print("=" * 80)

    # 估算 W crossing 带来的 row switch
    # 每个 W crossing tile，访问模式 for h: for w 导致每个 h 都会切换
    # 假设每行内访问 w_start 和 w_end，跨边界时会切换 row
    w_crossing_row_switches = len(w_crossing_tiles) * H_per_tile  # 每个 h 切换一次
    print(f"W crossing tiles: {len(w_crossing_tiles)}")
    print(f"  每个 tile 产生的 row switches: ~{H_per_tile} (H_per_tile)")
    print(f"  W crossing 总 row switches: ~{w_crossing_row_switches}")

    # H crossing 带来的额外 switch
    h_crossing_row_switches = len(h_crossing_tiles) * W_per_tile
    print(f"\nH crossing tiles: {len(h_crossing_tiles)}")
    print(f"  H crossing 贡献的额外 row switches: ~{h_crossing_row_switches}")

    print(f"\n总估计 row switches: ~{w_crossing_row_switches + h_crossing_row_switches}")
    print(f"实际 trace Input row activations: 21000")
    
    # 更详细的分析
    print(f"\n" + "=" * 80)
    print(f"详细分析: W Crossing 的边界位置")
    print("=" * 80)
    
    # 找出所有 W crossing 的边界
    w_boundaries = set()
    for t in w_crossing_tiles:
        boundary = t['w_blocks'][1] * block_w  # 边界位置
        w_boundaries.add(boundary)
    
    print(f"W crossing 边界位置: {sorted(w_boundaries)}")
    print(f"即 w = 31, 62 处会发生跨越")
    
    # 分析每个 C channel 的情况
    print(f"\n按 Channel 分组:")
    for c_idx in range(C_l3):
        c_w_crossing = [t for t in w_crossing_tiles if t['c'] == c_idx]
        print(f"  C={c_idx}: {len(c_w_crossing)} 个 W crossing tiles")
    
    # 计算每个 unique (Q, S) 的出现次数
    print(f"\n每个 (Q, S) 跨越组合出现次数:")
    qs_count = {}
    for t in w_crossing_tiles:
        key = (t['q'], t['s'])
        qs_count[key] = qs_count.get(key, 0) + 1
    
    sample_qs = list(qs_count.items())[0]
    print(f"  每个 (Q, S) 组合出现 {sample_qs[1]} 次 (= P_l3 × C_l3 × K_l3 = {P_l3} × {C_l3} × {K_l3})")
    
    print(f"\n总 W crossing tiles = {len(qs_crossing)} 个 (Q,S) × {sample_qs[1]} = {len(w_crossing_tiles)}")


if __name__ == "__main__":
    analyze_crossing_tiles()
