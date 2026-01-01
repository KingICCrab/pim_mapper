#!/usr/bin/env python3
"""详细分析 row switches 的来源"""
import math

P_l3, Q_l3, C_l3, K_l3 = 28, 7, 3, 4
R_l2 = 7
P_buffer = 2
Q_buffer = 8
S = 7
block_h, block_w = 31, 31
H_in, W_in = 62, 62
num_h_blocks = 2
num_w_blocks = 2

row_switches = 0
last_row = None

switch_sources = {'Q': 0, 'P': 0, 'R': 0, 'C': 0, 'K': 0, 'multi_block': 0}
last_k, last_c, last_p, last_q, last_r = None, None, None, None, None

for k in range(K_l3):
    for c in range(C_l3):
        for p in range(P_l3):
            for q in range(Q_l3):
                for r in range(R_l2):
                    h = p * P_buffer + r
                    w_start = q * Q_buffer
                    w_end = w_start + Q_buffer + S - 2
                    
                    hb = min(h // block_h, num_h_blocks - 1)
                    wb_start = w_start // block_w
                    wb_end = min(w_end // block_w, num_w_blocks - 1)
                    
                    # 按 block 顺序访问
                    block_count = 0
                    for wb in range(wb_start, wb_end + 1):
                        current_row = (c, hb, wb)
                        
                        if last_row is not None and current_row != last_row:
                            row_switches += 1
                            
                            # 确定 switch 来源
                            if block_count > 0:
                                # 同一个 (p,q,r) 内的多 block 访问
                                switch_sources['multi_block'] += 1
                            elif last_k != k:
                                switch_sources['K'] += 1
                            elif last_c != c:
                                switch_sources['C'] += 1
                            elif last_p != p:
                                switch_sources['P'] += 1
                            elif last_q != q:
                                switch_sources['Q'] += 1
                            elif last_r != r:
                                switch_sources['R'] += 1
                        
                        last_row = current_row
                        block_count += 1
                    
                    last_k, last_c, last_p, last_q, last_r = k, c, p, q, r

print(f'Total row switches: {row_switches}')
print()
print('Switch sources:')
for src, count in switch_sources.items():
    print(f'  {src}: {count}')
print(f'  Sum: {sum(switch_sources.values())}')
print()

# 分析
unique_rows = num_h_blocks * num_w_blocks * C_l3
total_iterations = P_l3 * Q_l3 * R_l2 * C_l3 * K_l3

print(f'=== 简化公式参数 ===')
print(f'unique_rows = {unique_rows}')
print(f'total_iterations = {total_iterations}')
print(f'switch_rate = {row_switches / total_iterations:.4f}')
print()

# 分解
print('=== Switch 分解 ===')
# multi_block: Q tile 跨越 w_block 边界时的内部切换
# Q: Q 变化导致的切换
# P: P 变化导致的切换 (通常伴随 Q 重置)
# R: R 变化导致 h_block 变化
# C: C 变化
# K: K 变化

# 估算公式
print()
print('=== 估算公式 ===')

# multi_block 来源: 当一个 (p,q,r) 访问多个 w_blocks 时
# q=3 访问 {0,1}，每次访问时有 1 次内部切换
# 统计哪些 q 会跨越 w_block
q_with_multi = 0
for q in range(Q_l3):
    w_start = q * Q_buffer
    w_end = w_start + Q_buffer + S - 2
    wb_start = w_start // block_w
    wb_end = min(w_end // block_w, num_w_blocks - 1)
    if wb_end > wb_start:
        q_with_multi += 1
        print(f'  q={q} spans {wb_end - wb_start + 1} w_blocks')

multi_block_per_iter = q_with_multi  # 每个 (P, R, C, K) 有这么多内部切换
total_multi_block = multi_block_per_iter * P_l3 * R_l2 * C_l3 * K_l3
print(f'Expected multi_block switches: {total_multi_block}')
print(f'Actual multi_block switches: {switch_sources["multi_block"]}')
print()

# Q switches: Q 变化时的切换
# 当 Q 从 q 到 q+1 时，如果 w_block 不同
q_boundary_switches = 0
for q in range(Q_l3 - 1):
    w_curr_end = q * Q_buffer + Q_buffer + S - 2
    w_next_start = (q + 1) * Q_buffer
    
    wb_curr_last = min(w_curr_end // block_w, num_w_blocks - 1)
    wb_next_first = w_next_start // block_w
    
    if wb_curr_last != wb_next_first:
        q_boundary_switches += 1
        print(f'  q={q}->{q+1}: wb {wb_curr_last} -> {wb_next_first}')

# 每个 (P, R, C, K) 有 Q_l3-1 次 Q 变化，其中 q_boundary_switches 次会导致 w_block 变化
total_q_switches = q_boundary_switches * P_l3 * R_l2 * C_l3 * K_l3
print(f'Expected Q switches: {total_q_switches}')
print(f'Actual Q switches: {switch_sources["Q"]}')
print()

# R switches: R 变化时 h_block 变化
# 对于每个 p，r 从 0 到 R_l2-1 时 h 从 p*P_buffer 到 p*P_buffer+R_l2-1
r_boundary_crossings = 0
for p in range(P_l3):
    for r in range(R_l2 - 1):
        h_curr = p * P_buffer + r
        h_next = p * P_buffer + (r + 1)
        hb_curr = min(h_curr // block_h, num_h_blocks - 1)
        hb_next = min(h_next // block_h, num_h_blocks - 1)
        if hb_curr != hb_next:
            r_boundary_crossings += 1

# 每个 (Q, C, K) 有这么多 R 边界交叉
total_r_switches = r_boundary_crossings * Q_l3 * C_l3 * K_l3
print(f'R boundary crossings per (Q,C,K): {r_boundary_crossings}')
print(f'Expected R switches: {total_r_switches}')
print(f'Actual R switches: {switch_sources["R"]}')
print()

# P switches: P 变化时的切换
# 当 P 从 p 到 p+1 时，Q 重置到 0，R 重置到 0
# 前一个状态: (c, hb(p-1, R_l2-1), wb_last(Q_l3-1))
# 后一个状态: (c, hb(p, 0), wb_first(0))
p_switches_expected = 0
for p in range(1, P_l3):
    # 前一个 P 的最后状态
    h_prev = (p - 1) * P_buffer + R_l2 - 1
    hb_prev = min(h_prev // block_h, num_h_blocks - 1)
    
    # Q_l3-1 的最后一个 w_block
    w_prev_end = (Q_l3 - 1) * Q_buffer + Q_buffer + S - 2
    wb_prev = min(w_prev_end // block_w, num_w_blocks - 1)
    
    # 当前 P 的第一个状态
    h_curr = p * P_buffer + 0
    hb_curr = min(h_curr // block_h, num_h_blocks - 1)
    
    # Q=0 的第一个 w_block
    w_curr_start = 0 * Q_buffer
    wb_curr = w_curr_start // block_w
    
    # 检查是否切换
    if (hb_prev, wb_prev) != (hb_curr, wb_curr):
        p_switches_expected += 1

# 每个 (C, K) 有 P_l3-1 次 P 变化
total_p_switches = p_switches_expected * C_l3 * K_l3
print(f'P switches per (C,K): {p_switches_expected}')
print(f'Expected P switches: {total_p_switches}')
print(f'Actual P switches: {switch_sources["P"]}')
print()

# C switches: C 变化时必定切换 (不同 channel 在不同 row)
c_switches_expected = (C_l3 - 1) * K_l3
print(f'Expected C switches: {c_switches_expected}')
print(f'Actual C switches: {switch_sources["C"]}')
print()

# K switches: K 变化时可能切换
k_switches_expected = K_l3 - 1
print(f'Expected K switches: {k_switches_expected}')
print(f'Actual K switches: {switch_sources["K"]}')
print()

# 总估算
total_expected = (total_multi_block + total_q_switches + total_r_switches + 
                  total_p_switches + c_switches_expected + k_switches_expected)
print(f'=== 总估算 ===')
print(f'Total expected: {total_expected}')
print(f'Actual: {row_switches}')
print(f'Difference: {row_switches - total_expected}')
