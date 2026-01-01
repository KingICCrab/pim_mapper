#!/usr/bin/env python3
"""精确模拟 Trace 的 row switch 计算"""

# 原始 mapping 分析
# Level 0: P=2, Q=8, K=16
# Level 1: S=7
# Level 2: R=7
# Level 3: P=28, Q=7, C=3, K=4

# Buffer tile (Level 0+1)
P_buf = 2
Q_buf = 8
R_buf = 1  # R 不在 buffer (Level 0+1)，所以是 1
S_buf = 7  # S 在 Level 1

# DRAM loop structure (Level 2+3)
P_l3 = 28
Q_l3 = 7
C_l3 = 3
K_l3 = 4
R_l2 = 7

block_h = 31
block_w = 31
H_in = 62
W_in = 62

print('=== 验证 Trace 的迭代方式 ===')
print(f'Buffer tile: P={P_buf}, Q={Q_buf}, R={R_buf}, S={S_buf}')
print(f'DRAM factors: P_l3={P_l3}, Q_l3={Q_l3}, R_l2={R_l2}, C_l3={C_l3}, K_l3={K_l3}')
print()

# Trace 的遍历顺序由 permutation 决定
# Level 3: Q -> P -> C -> K (inner to outer)
# Level 2: R
# 实际顺序 (outer to inner): K -> C -> P -> Q -> R

prev_row = None
switches = 0

for k in range(K_l3):
    for c in range(C_l3):
        for p in range(P_l3):
            for q in range(Q_l3):
                for r in range(R_l2):
                    p_start = p * P_buf  # p * 2
                    q_start = q * Q_buf  # q * 8
                    r_start = r * R_buf  # r * 1 = r
                    s_start = 0  # S in buffer, no outer loop
                    
                    h_start = p_start + r_start  # p*2 + r
                    w_start = q_start + s_start  # q*8
                    
                    # Tile 范围
                    H_per_tile = (P_buf - 1) + (R_buf - 1) + 1  # = 2
                    W_per_tile = (Q_buf - 1) + (S_buf - 1) + 1  # = 14
                    h_end = min(h_start + H_per_tile, H_in)
                    w_end = min(w_start + W_per_tile, W_in)
                    
                    # 遍历 tile 内的 blocks
                    h_block_start = h_start // block_h
                    h_block_end = (h_end - 1) // block_h
                    w_block_start = w_start // block_w
                    w_block_end = (w_end - 1) // block_w
                    
                    for h_block in range(h_block_start, h_block_end + 1):
                        for w_block in range(w_block_start, w_block_end + 1):
                            # 计算 block 内的有效范围
                            h_lo = max(h_start, h_block * block_h)
                            h_hi = min(h_end, (h_block + 1) * block_h)
                            w_lo = max(w_start, w_block * block_w)
                            w_hi = min(w_end, (w_block + 1) * block_w)
                            
                            # 遍历 block 内的元素
                            for h in range(h_lo, h_hi):
                                for w in range(w_lo, w_hi):
                                    row = (c, h_block, w_block)
                                    if prev_row is None or row != prev_row:
                                        switches += 1
                                    prev_row = row

print(f'Simulated switches: {switches}')
print(f'Expected (Trace): 5376')
print()

# 简化分析：每个 K iteration
print('=== 分析每个 K iteration ===')
prev_row = None
switches_per_k = 0

for c in range(C_l3):
    for p in range(P_l3):
        for q in range(Q_l3):
            for r in range(R_l2):
                p_start = p * P_buf
                q_start = q * Q_buf
                r_start = r * R_buf
                
                h_start = p_start + r_start
                w_start = q_start
                
                H_per_tile = 2
                W_per_tile = 14
                h_end = min(h_start + H_per_tile, H_in)
                w_end = min(w_start + W_per_tile, W_in)
                
                h_block_start = h_start // block_h
                h_block_end = (h_end - 1) // block_h
                w_block_start = w_start // block_w
                w_block_end = (w_end - 1) // block_w
                
                for h_block in range(h_block_start, h_block_end + 1):
                    for w_block in range(w_block_start, w_block_end + 1):
                        h_lo = max(h_start, h_block * block_h)
                        h_hi = min(h_end, (h_block + 1) * block_h)
                        w_lo = max(w_start, w_block * block_w)
                        w_hi = min(w_end, (w_block + 1) * block_w)
                        
                        for h in range(h_lo, h_hi):
                            for w in range(w_lo, w_hi):
                                row = (c, h_block, w_block)
                                if prev_row is None or row != prev_row:
                                    switches_per_k += 1
                                prev_row = row

print(f'Switches per K: {switches_per_k}')
print(f'× K = {switches_per_k} × {K_l3} = {switches_per_k * K_l3}')
print(f'Expected: 5376 / 4 = {5376 // 4}')
