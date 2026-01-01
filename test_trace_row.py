#!/usr/bin/env python3
"""
分析 Trace 的 row activation 计算方式

原始 Trace mapping (产生 5376):
- P_DRAM = 28, Q_DRAM = 7
- R_L2 = 7, R_DRAM = 1 (R_outer = 7)
- S_L1 = 7, S_DRAM = 1 (S 在 buffer)
- C_DRAM = 3, K_DRAM = 4
"""

# 参数
N, K, C, P, Q, R, S = 1, 64, 3, 56, 56, 7, 7
stride_h, stride_w = 1, 1
input_h = P + R - 1  # 62
input_w = Q + S - 1  # 62
element_bytes = 1
block_h, block_w = 31, 31  # from mapping tile_info

# Trace 的 tile 配置
# Level 0 (Buffer): P=2, Q=8, K=16, S=7 (implicitly)
# Level 1: S=7
# Level 2: R=7
# Level 3 (DRAM): P=28, Q=7, C=3, K=4

P_tile = 2    # P per buffer tile
Q_tile = 8    # Q per buffer tile
R_tile = 1    # R per tile (R=7 外循环)
S_tile = 7    # S in buffer

P_factor = 28  # P_DRAM
Q_factor = 7   # Q_DRAM
R_factor = 7   # R_L2 (outer R loop)
C_factor = 3   # C_DRAM
K_factor = 4   # K_DRAM

print('=== Trace Mapping 配置 ===')
print(f'P_factor={P_factor}, Q_factor={Q_factor}, R_factor={R_factor}')
print(f'C_factor={C_factor}, K_factor={K_factor}')
print(f'P_tile={P_tile}, Q_tile={Q_tile}, R_tile={R_tile}, S_tile={S_tile}')
print(f'block_h={block_h}, block_w={block_w}')
print(f'input_h={input_h}, input_w={input_w}')

# Trace 计算方式：
# Input tile 起始坐标: h = p * P_tile + r, w = q * Q_tile
# (S 在 buffer 内，不影响 tile 起始位置)

# row_id = 基于 (c, h_block, w_block)
# 其中 h_block = h_start // block_h, w_block = w_start // block_w

num_h_blocks = (input_h + block_h - 1) // block_h  # 2 
num_w_blocks = (input_w + block_w - 1) // block_w  # 2

print(f'num_h_blocks={num_h_blocks}, num_w_blocks={num_w_blocks}')

# 模拟 Trace 遍历: K -> C -> P -> Q -> R (outer to inner)
print('\n=== 模拟 Trace 遍历 (K -> C -> P -> Q -> R) ===')

prev_row = None
total_switches = 0
switches_breakdown = {'first': 0, 'k_boundary': 0, 'c_boundary': 0, 'other': 0}

for k in range(K_factor):
    for c in range(C_factor):
        for p in range(P_factor):
            for q in range(Q_factor):
                for r in range(R_factor):
                    # Input tile 起始坐标
                    h_start = p * P_tile * stride_h + r * R_tile
                    w_start = q * Q_tile * stride_w
                    
                    # 计算 h_block 和 w_block
                    h_block = h_start // block_h
                    w_block = w_start // block_w
                    
                    # row_id = c * (num_h_blocks * num_w_blocks) + h_block * num_w_blocks + w_block
                    row_id = (c * num_h_blocks + h_block) * num_w_blocks + w_block
                    
                    if prev_row is None:
                        total_switches = 1
                        switches_breakdown['first'] = 1
                    elif row_id != prev_row:
                        total_switches += 1
                        # 分类 switch 类型
                        if k > 0 and c == 0 and p == 0 and q == 0 and r == 0:
                            switches_breakdown['k_boundary'] += 1
                        elif c > 0 and p == 0 and q == 0 and r == 0:
                            switches_breakdown['c_boundary'] += 1
                        else:
                            switches_breakdown['other'] += 1
                    
                    prev_row = row_id

print(f'Total switches: {total_switches}')
print(f'Breakdown: {switches_breakdown}')
print(f'Expected (Trace): 5376')

# 分析单个 K 内部的 switches
print('\n=== 分析单个 K 内部 ===')
prev_row = None
switches_one_k = 0

for c in range(C_factor):
    for p in range(P_factor):
        for q in range(Q_factor):
            for r in range(R_factor):
                h_start = p * P_tile * stride_h + r * R_tile
                w_start = q * Q_tile * stride_w
                h_block = h_start // block_h
                w_block = w_start // block_w
                row_id = (c * num_h_blocks + h_block) * num_w_blocks + w_block
                
                if prev_row is None or row_id != prev_row:
                    switches_one_k += 1
                prev_row = row_id

print(f'Switches in one K: {switches_one_k}')
print(f'× K_factor = {switches_one_k} × {K_factor} = {switches_one_k * K_factor}')
print(f'Expected per K: 1344 (5376 / 4)')

