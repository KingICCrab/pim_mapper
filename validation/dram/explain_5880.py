#!/usr/bin/env python3
"""
精确分析 Trace 的 5880 row activations 来源
"""

# 参数 (与 verify_crossing.py 一致)
block_h, block_w = 31, 31
P_buf, Q_buf = 8, 2
R_buf, S_buf = 7, 1
stride_h, stride_w = 1, 1

P_l3, Q_l3, C_l3, K_l3 = 7, 28, 3, 4
S_l2 = 7

input_H, input_W = 62, 62

H_per_tile = (P_buf - 1) * stride_h + (R_buf - 1) + 1  # = 14
W_per_tile = (Q_buf - 1) * stride_w + (S_buf - 1) + 1  # = 2

print("=" * 70)
print("Trace 5880 Row Activations 精确分析")
print("=" * 70)

def simulate_trace():
    """模拟 Trace 的 block-wise 访问模式"""
    last_block = None
    total_row_acts = 0
    block_visits = {}
    
    for k in range(K_l3):
        for c in range(C_l3):
            for q in range(Q_l3):
                for p in range(P_l3):
                    for s in range(S_l2):
                        p_start = p * P_buf
                        q_start = q * Q_buf + s
                        
                        h_start = p_start * stride_h
                        h_end = min(h_start + H_per_tile, input_H)
                        w_start = q_start * stride_w
                        w_end = min(w_start + W_per_tile, input_W)
                        
                        h_blk_s = h_start // block_h
                        h_blk_e = (h_end - 1) // block_h
                        w_blk_s = w_start // block_w
                        w_blk_e = (w_end - 1) // block_w
                        
                        # Block-wise 访问
                        for h_blk in range(h_blk_s, h_blk_e + 1):
                            for w_blk in range(w_blk_s, w_blk_e + 1):
                                block = (c, h_blk, w_blk)
                                if last_block is None or block != last_block:
                                    total_row_acts += 1
                                    block_visits[block] = block_visits.get(block, 0) + 1
                                    last_block = block
    
    return total_row_acts, block_visits

total, visits = simulate_trace()

print(f"\n模拟结果: {total} row activations")
print(f"Trace 实际值: 5880")
print()

print("每个 block 的访问次数:")
for block in sorted(visits.keys()):
    c, h_blk, w_blk = block
    print(f"  Channel {c}, Block ({h_blk},{w_blk}): {visits[block]} visits")

# 分析 tile 切换
print("\n" + "=" * 70)
print("Row Activation 来源分解")
print("=" * 70)

def analyze_source():
    """分解 row activation 的来源"""
    last_block = None
    
    # 来源分类
    sources = {
        'first_access': 0,           # 首次访问
        'channel_switch': 0,         # channel 切换
        'q_tile_switch': 0,          # Q tile 切换 (同一 channel)
        'p_tile_switch': 0,          # P tile 切换
        's_iteration_switch': 0,     # S 迭代切换
        'within_tile_crossing': 0,   # tile 内部 crossing
    }
    
    for k in range(K_l3):
        for c in range(C_l3):
            for q in range(Q_l3):
                for p in range(P_l3):
                    for s in range(S_l2):
                        p_start = p * P_buf
                        q_start = q * Q_buf + s
                        
                        h_start = p_start * stride_h
                        h_end = min(h_start + H_per_tile, input_H)
                        w_start = q_start * stride_w
                        w_end = min(w_start + W_per_tile, input_W)
                        
                        h_blk_s = h_start // block_h
                        h_blk_e = (h_end - 1) // block_h
                        w_blk_s = w_start // block_w
                        w_blk_e = (w_end - 1) // block_w
                        
                        num_h_blocks = h_blk_e - h_blk_s + 1
                        num_w_blocks = w_blk_e - w_blk_s + 1
                        num_blocks = num_h_blocks * num_w_blocks
                        
                        for h_blk in range(h_blk_s, h_blk_e + 1):
                            for w_blk in range(w_blk_s, w_blk_e + 1):
                                block = (c, h_blk, w_blk)
                                if last_block is None:
                                    sources['first_access'] += 1
                                elif block != last_block:
                                    # 判断切换类型
                                    prev_c, prev_h, prev_w = last_block
                                    if c != prev_c:
                                        sources['channel_switch'] += 1
                                    elif h_blk != prev_h and w_blk == prev_w:
                                        sources['within_tile_crossing'] += 1  # H crossing
                                    elif w_blk != prev_w and h_blk == prev_h:
                                        sources['within_tile_crossing'] += 1  # W crossing
                                    else:
                                        # 复杂切换 (可能是 tile 边界)
                                        sources['within_tile_crossing'] += 1
                                last_block = block
    
    return sources

sources = analyze_source()
print("\nRow activation 来源:")
for src, count in sources.items():
    print(f"  {src}: {count}")
print(f"  总计: {sum(sources.values())}")

# 更直接的分解
print("\n" + "=" * 70)
print("公式化分解")
print("=" * 70)

# 统计每种 tile 类型的数量和贡献
h_only_crossing = 0
w_only_crossing = 0
both_crossing = 0
no_crossing = 0

for q in range(Q_l3):
    for p in range(P_l3):
        for s in range(S_l2):
            p_start = p * P_buf
            q_start = q * Q_buf + s
            
            h_start = p_start * stride_h
            h_end = h_start + H_per_tile
            w_start = q_start * stride_w
            w_end = w_start + W_per_tile
            
            h_cross = (h_start // block_h) != ((h_end - 1) // block_h)
            w_cross = (w_start // block_w) != ((w_end - 1) // block_w)
            
            if h_cross and w_cross:
                both_crossing += 1
            elif h_cross:
                h_only_crossing += 1
            elif w_cross:
                w_only_crossing += 1
            else:
                no_crossing += 1

total_pqs = P_l3 * Q_l3 * S_l2
print(f"\n单个 (K, C) iteration 内的 tile 类型 (P×Q×S = {total_pqs}):")
print(f"  无 crossing: {no_crossing}")
print(f"  只 H crossing: {h_only_crossing}")
print(f"  只 W crossing: {w_only_crossing}")
print(f"  Both crossing: {both_crossing}")
print(f"  总计: {no_crossing + h_only_crossing + w_only_crossing + both_crossing}")

# 每种 tile 的 block 访问数
print(f"\n每种 tile 访问的 block 数:")
print(f"  无 crossing: 1 block")
print(f"  只 H/W crossing: 2 blocks")
print(f"  Both crossing: 4 blocks")

total_blocks_per_kc = (no_crossing * 1 + 
                       h_only_crossing * 2 + 
                       w_only_crossing * 2 + 
                       both_crossing * 4)
print(f"\n每个 (K, C) iteration 访问的 block 总数: {total_blocks_per_kc}")
print(f"K_l3 × C_l3 = {K_l3} × {C_l3} = {K_l3 * C_l3}")
print(f"全部 block 访问次数: {total_blocks_per_kc} × {K_l3 * C_l3} = {total_blocks_per_kc * K_l3 * C_l3}")

print("\n但 row activation 不等于 block 访问次数！")
print("因为相邻访问如果在同一 block，不会触发新的 row activation。")

# 精确计算
print("\n" + "=" * 70)
print("精确计算 (考虑相邻 block 合并)")
print("=" * 70)

# 在一个 (K, C) iteration 内，tile 访问顺序是:
# for q in Q_l3: for p in P_l3: for s in S_l2
# 
# 相邻的 (p, s) -> (p, s+1) 可能在同一 block
# 相邻的 (p, s=6) -> (p+1, s=0) 可能在同一 block

# 让我统计实际的 block 切换
def count_block_switches_per_kc():
    """统计单个 (K, C) iteration 内的 block 切换次数"""
    last_block = None
    switches = 0
    
    for q in range(Q_l3):
        for p in range(P_l3):
            for s in range(S_l2):
                p_start = p * P_buf
                q_start = q * Q_buf + s
                
                h_start = p_start * stride_h
                h_end = min(h_start + H_per_tile, input_H)
                w_start = q_start * stride_w
                w_end = min(w_start + W_per_tile, input_W)
                
                h_blk_s = h_start // block_h
                h_blk_e = (h_end - 1) // block_h
                w_blk_s = w_start // block_w
                w_blk_e = (w_end - 1) // block_w
                
                for h_blk in range(h_blk_s, h_blk_e + 1):
                    for w_blk in range(w_blk_s, w_blk_e + 1):
                        block = (h_blk, w_blk)  # 不含 channel
                        if last_block is not None and block != last_block:
                            switches += 1
                        last_block = block
    
    return switches + 1  # +1 for first access

switches_per_kc = count_block_switches_per_kc()
print(f"\n单个 (K, C) iteration 的 row activations: {switches_per_kc}")
print(f"K_l3 × C_l3 = {K_l3 * C_l3}")
print(f"总计 (不考虑 K, C 切换): {switches_per_kc} × {K_l3 * C_l3} = {switches_per_kc * K_l3 * C_l3}")

# 考虑 C 切换 (每次 C 变化时一定切换 block)
# 考虑 K 切换 (每次 K 变化时，C 回到 0，一定切换)

# 实际计算
total_kc = K_l3 * C_l3
# 每个 (K, C) 内部: switches_per_kc - 1 次切换 (不含首次)
# 共 total_kc 个 (K, C) iteration
# (K, C) 之间: total_kc - 1 次切换 (C 或 K 变化时)

# 更精确: 
# - C 切换: K_l3 × (C_l3 - 1) = 4 × 2 = 8 次
# - K 切换: K_l3 - 1 = 3 次
# 但每次 C/K 切换都会导致 block 切换

print(f"\n精确公式:")
print(f"  每个 (K,C) 内部的首次访问 = {K_l3 * C_l3} (每个 iteration 首次都是 new activation)")
print(f"  每个 (K,C) 内部的切换 = {switches_per_kc - 1} × {K_l3 * C_l3} = {(switches_per_kc - 1) * K_l3 * C_l3}")
print(f"  总计 = {K_l3 * C_l3 + (switches_per_kc - 1) * K_l3 * C_l3}")
print(f"       = {K_l3 * C_l3} × {switches_per_kc}")
print(f"       = {K_l3 * C_l3 * switches_per_kc}")

# 验证
print(f"\n验证: {K_l3 * C_l3 * switches_per_kc} vs 模拟值 {total} vs Trace 5880")
