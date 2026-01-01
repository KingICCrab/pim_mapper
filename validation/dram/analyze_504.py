#!/usr/bin/env python3
"""
分析 504 non_crossing_switches 的精确来源
"""

P_dram, Q_dram, R_dram = 28, 7, 7
C_dram, K_dram = 3, 4
block_h, block_w = 31, 31
P_tile, Q_tile = 2, 8

def get_tile_info(p, q, r):
    h_start = p * P_tile + r
    w_start = q * Q_tile
    H_per_tile = 2
    W_per_tile = 14
    h_end = h_start + H_per_tile
    w_end = w_start + W_per_tile
    
    h_block_start = h_start // block_h
    h_block_end = (h_end - 1) // block_h
    w_block_start = w_start // block_w
    w_block_end = (w_end - 1) // block_w
    
    h_crosses = (h_block_start != h_block_end)
    w_crosses = (w_block_start != w_block_end)
    
    if not h_crosses and not w_crosses:
        return 'non', h_block_start, w_block_start, h_block_end, w_block_end
    elif h_crosses and not w_crosses:
        return 'h', h_block_start, w_block_start, h_block_end, w_block_end
    elif not h_crosses and w_crosses:
        return 'w', h_block_start, w_block_start, h_block_end, w_block_end
    else:
        return 'both', h_block_start, w_block_start, h_block_end, w_block_end


def main():
    # 分析一个 (K, C) 组合内的 non_crossing switches (不考虑 c)
    prev_block = None
    switches_in_one_pqr = 0
    
    for p in range(P_dram):
        for q in range(Q_dram):
            for r in range(R_dram):
                tile_type, h_s, w_s, h_e, w_e = get_tile_info(p, q, r)
                
                if tile_type == 'non':
                    block = (h_s, w_s)
                    if prev_block is None or block != prev_block:
                        switches_in_one_pqr += 1
                    prev_block = block
                else:
                    # crossing tile 的最后一个 block
                    prev_block = (h_e, w_e)
    
    print(f"=== 一个 P->Q->R 遍历内的 block switches (忽略 c) ===")
    print(f"switches_in_one_pqr = {switches_in_one_pqr}")
    print()
    
    # 完整模拟
    prev_row_id = None
    non_crossing_switches = 0
    
    # 分类统计 switch 原因
    from_none = 0           # prev_row_id 是 None
    c_change = 0            # c 变化导致的 switch
    block_change = 0        # 同一个 c 内 block 变化导致的 switch
    
    for k in range(K_dram):
        for c in range(C_dram):
            for p in range(P_dram):
                for q in range(Q_dram):
                    for r in range(R_dram):
                        tile_type, h_s, w_s, h_e, w_e = get_tile_info(p, q, r)
                        
                        if tile_type == 'non':
                            row_id = (c, h_s, w_s)
                            if prev_row_id is None:
                                from_none += 1
                                non_crossing_switches += 1
                            elif row_id != prev_row_id:
                                non_crossing_switches += 1
                                if prev_row_id[0] != c:
                                    c_change += 1
                                else:
                                    block_change += 1
                            prev_row_id = row_id
                        else:
                            # crossing tile 的最后一个 block
                            prev_row_id = (c, h_e, w_e)
    
    print(f"=== 完整遍历的 non_crossing_switches 分类 ===")
    print(f"Total non_crossing_switches = {non_crossing_switches}")
    print(f"  - from_none (第一次):     {from_none}")
    print(f"  - c_change (c 变化):      {c_change}")
    print(f"  - block_change (block 变): {block_change}")
    print()
    
    # 验证分解
    print(f"=== 验证 ===")
    print(f"from_none + c_change + block_change = {from_none + c_change + block_change}")
    print()
    
    # 分析 c_change 的来源
    # c_change 发生在：从一个 (k,c) 切换到下一个 (k,c') 时，
    # 如果前一个 (k,c) 结束于 crossing tile，下一个 (k,c') 开始于 non_crossing tile
    # 或者在 K 切换时
    
    print(f"=== 进一步分析 ===")
    print(f"K_dram * C_dram = {K_dram * C_dram}")
    print(f"如果每个 (K,C) 的第一个 non_crossing tile 都产生 switch:")
    
    # 找出每个 (K,C) 内的第一个 non_crossing tile
    first_non_crossing_per_KC = []
    for k in range(K_dram):
        for c in range(C_dram):
            for p in range(P_dram):
                for q in range(Q_dram):
                    for r in range(R_dram):
                        tile_type, h_s, w_s, _, _ = get_tile_info(p, q, r)
                        if tile_type == 'non':
                            first_non_crossing_per_KC.append((k, c, p, q, r, h_s, w_s))
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
    
    print(f"每个 (K,C) 的第一个 non_crossing tile:")
    for info in first_non_crossing_per_KC[:6]:
        print(f"  k={info[0]}, c={info[1]}: p={info[2]}, q={info[3]}, r={info[4]} -> block=({info[5]}, {info[6]})")
    print("  ...")
    print()
    
    # 更精确的分析：统计每个 (K,C) 内 non_crossing 产生的 switches
    print(f"=== 每个 (K,C) 内 non_crossing switches ===")
    prev_row_id = None
    switches_per_KC = []
    
    for k in range(K_dram):
        for c in range(C_dram):
            kc_switches = 0
            for p in range(P_dram):
                for q in range(Q_dram):
                    for r in range(R_dram):
                        tile_type, h_s, w_s, h_e, w_e = get_tile_info(p, q, r)
                        
                        if tile_type == 'non':
                            row_id = (c, h_s, w_s)
                            if prev_row_id is None or row_id != prev_row_id:
                                kc_switches += 1
                            prev_row_id = row_id
                        else:
                            prev_row_id = (c, h_e, w_e)
            switches_per_KC.append(kc_switches)
    
    unique_counts = {}
    for s in switches_per_KC:
        unique_counts[s] = unique_counts.get(s, 0) + 1
    
    print(f"switches 分布: {unique_counts}")
    print(f"Total = {sum(switches_per_KC)}")
    
    # 第一个 (K,C) 与后续的区别
    print(f"\n第一个 (K,C) switches: {switches_per_KC[0]}")
    print(f"后续 (K,C) 平均 switches: {sum(switches_per_KC[1:]) / (len(switches_per_KC) - 1):.2f}")


if __name__ == "__main__":
    main()
