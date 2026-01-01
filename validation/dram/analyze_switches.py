#!/usr/bin/env python3
"""分析 row switches 的模式"""
import math

# 参数
P = 56
Q = 56
R = 7
S = 7
C = 3
K = 64
stride = 1

P_l3 = 28
Q_l3 = 7
C_l3 = 3
K_l3 = 4
R_l2 = 7
S_l2 = 1

P_buffer = P // P_l3  # = 2
Q_buffer = Q // Q_l3  # = 8

block_h = 31
block_w = 31
H_in = 62
W_in = 62

num_h_blocks = math.ceil(H_in / block_h)  # = 2
num_w_blocks = math.ceil(W_in / block_w)  # = 2

print('=== 基本参数 ===')
print(f'Buffer tile: P_buffer={P_buffer}, Q_buffer={Q_buffer}')
print(f'L3 tiles: P_l3={P_l3}, Q_l3={Q_l3}, C_l3={C_l3}, K_l3={K_l3}')
print(f'L2: R_l2={R_l2}')
print(f'Block: {block_h} × {block_w}, Num blocks: {num_h_blocks} × {num_w_blocks}')
print()

# 每个 L3 tile 访问的 Input h/w 范围
h_tile_range = P_buffer * stride + (R - 1) * 1  # = 2 + 6 = 8
w_tile_range = Q_buffer * stride + (S - 1) * 1  # = 8 + 6 = 14

print(f'每个 L3 tile 访问的 Input 范围: h={h_tile_range}, w={w_tile_range}')
print()

# 模拟访问模式
# 访问顺序: for K: for C: for P: for Q: for R
# 但根据 permutation {0: 3, 1: 2, 3: 4, 4: 5}
# 这表示 Q->P->C->K (inner to outer)

print('=== 模拟访问模式 ===')
print('顺序: K (outer) -> C -> P -> Q (inner) -> R (innermost at L2)')
print()

row_switches = 0
last_row = None
total_accesses = 0

# 模拟访问
for k in range(K_l3):
    for c in range(C_l3):
        for p in range(P_l3):
            for q in range(Q_l3):
                for r in range(R_l2):
                    # 计算当前访问的 h, w 位置
                    h = p * stride * P_buffer + r  # 简化: 假设 r 从 0 到 R-1
                    w = q * stride * Q_buffer  # Q tile 的起始 w
                    
                    # 确定 block
                    h_block = min(h // block_h, num_h_blocks - 1)
                    w_block = min(w // block_w, num_w_blocks - 1)
                    
                    # row = (c, h_block, w_block)
                    current_row = (c, h_block, w_block)
                    
                    if last_row is not None and current_row != last_row:
                        row_switches += 1
                    
                    last_row = current_row
                    total_accesses += 1

print(f'Total accesses: {total_accesses}')
print(f'Simulated row switches: {row_switches}')
print(f'Actual trace row switches: 5375')
print()

# 更精确的模拟: 考虑每个 tile 访问多个 (h, w) 点
print('=== 更精确模拟 ===')
row_switches_v2 = 0
last_row_v2 = None
total_accesses_v2 = 0

for k in range(K_l3):
    for c in range(C_l3):
        for p in range(P_l3):
            for q in range(Q_l3):
                for r in range(R_l2):
                    # 对于每个 (p, q, r)，访问的是一个 tile 的所有点
                    # 但我们只关心 row (block) 切换
                    
                    # h 范围: [p * stride * P_buffer + r, p * stride * P_buffer + r + P_buffer - 1]
                    h_start = p * stride * P_buffer + r
                    h_end = h_start  # 简化: 只看起始位置
                    
                    # w 范围: 固定在 q 对应的范围
                    w_start = q * stride * Q_buffer
                    
                    h_block = min(h_start // block_h, num_h_blocks - 1)
                    w_block = min(w_start // block_w, num_w_blocks - 1)
                    
                    current_row = (c, h_block, w_block)
                    
                    if last_row_v2 is not None and current_row != last_row_v2:
                        row_switches_v2 += 1
                    
                    last_row_v2 = current_row
                    total_accesses_v2 += 1

print(f'Simulated row switches (v2): {row_switches_v2}')
print()

# 分析 switch 原因
print('=== 分析 switch 来源 ===')

# Switch 可能来自:
# 1. R 变化导致 h 跨越 block 边界
# 2. Q 变化导致 w 跨越 block 边界  
# 3. P 变化导致 h 跨越 block 边界
# 4. C 变化

r_switches = 0
q_switches = 0
p_switches = 0
c_switches = 0
last_row_v3 = None
last_c, last_p, last_q, last_r = None, None, None, None

for k in range(K_l3):
    for c in range(C_l3):
        for p in range(P_l3):
            for q in range(Q_l3):
                for r in range(R_l2):
                    h_start = p * stride * P_buffer + r
                    w_start = q * stride * Q_buffer
                    
                    h_block = min(h_start // block_h, num_h_blocks - 1)
                    w_block = min(w_start // block_w, num_w_blocks - 1)
                    
                    current_row = (c, h_block, w_block)
                    
                    if last_row_v3 is not None and current_row != last_row_v3:
                        # Identify source of switch
                        if c != last_c:
                            c_switches += 1
                        elif p != last_p:
                            p_switches += 1
                        elif q != last_q:
                            q_switches += 1
                        else:  # r changed
                            r_switches += 1
                    
                    last_row_v3 = current_row
                    last_c, last_p, last_q, last_r = c, p, q, r

print(f'Switches from R change: {r_switches}')
print(f'Switches from Q change: {q_switches}')
print(f'Switches from P change: {p_switches}')
print(f'Switches from C change: {c_switches}')
print(f'Total: {r_switches + q_switches + p_switches + c_switches}')
print()

# 理解 R 导致的 switches
print('=== R 变化分析 ===')
# 当 r 增加时，h 增加，可能跨越 block 边界
# 边界在 h=31
# 如果 h 从 30 变到 31，就会发生 switch

boundary_crossings = 0
for p in range(P_l3):
    for r in range(R_l2 - 1):  # r to r+1 transition
        h_current = p * stride * P_buffer + r
        h_next = p * stride * P_buffer + (r + 1)
        block_current = h_current // block_h
        block_next = h_next // block_h
        if block_current != block_next:
            boundary_crossings += 1
            if boundary_crossings <= 5:
                print(f'  p={p}, r={r}->{r+1}: h={h_current}->{h_next}, block={block_current}->{block_next}')

print(f'R boundary crossings per C: {boundary_crossings}')
print(f'Total R switches (×Q×C×K): {boundary_crossings * Q_l3 * C_l3 * K_l3}')
print()

# 这解释了大部分 switches!
# 计算公式
print('=== 新公式 ===')
# R switches: 每个 (P, Q, C, K) 组合中 R 循环的 boundary crossings
# 对于固定的 P tile，h 的变化范围是 [p*2, p*2+6]
# 如果这个范围跨越 h=31，每次跨越就是一次 switch

r_switch_estimate = boundary_crossings * Q_l3 * C_l3 * K_l3
q_switch_estimate = 0  # Q 变化主要在 w 方向，block_w=31，Q_buffer=8，每个 Q tile 不太会跨越
p_switch_estimate = 0  # 类似分析
c_switch_estimate = (C_l3 - 1) * P_l3 * Q_l3 * K_l3 / (P_l3 * Q_l3)  # 每次 C 变化

print(f'R-based switches estimate: {r_switch_estimate}')
print(f'Actual R switches from simulation: {r_switches}')
print()

# 最终模型
total_estimate = r_switch_estimate + q_switches + p_switches + c_switches
print(f'总估算: {total_estimate}')
print(f'实际 Trace: 5375')
print()

# 简化公式
print('=== 简化公式 ===')
# row_switches ≈ R_boundary_crossings × Q_l3 × C_l3 × K_l3 + other_switches

# R_boundary_crossings 取决于 P_buffer 和 block_h
# 对于每个 P tile，h 范围是 [p*stride*P_buffer, p*stride*P_buffer + R - 1]
# 如果 R > block_h，会有多次 crossing

# 简单估算: 每个 P 有 ceil(R/block_h) - 1 次 crossing
crossings_per_p = 0
for p in range(P_l3):
    h_start = p * stride * P_buffer
    h_end = h_start + R - 1
    start_block = h_start // block_h
    end_block = h_end // block_h
    crossings_per_p += (end_block - start_block)

avg_crossings = crossings_per_p / P_l3
print(f'Average R crossings per P tile: {avg_crossings:.2f}')
print(f'Total R crossings: {crossings_per_p}')

# 新公式
new_estimate = crossings_per_p * Q_l3 * C_l3 * K_l3
print(f'新公式估算: {crossings_per_p} × {Q_l3} × {C_l3} × {K_l3} = {new_estimate}')
print(f'加上 C 和 Q 切换: ~{new_estimate + c_switches + q_switches}')
