#!/usr/bin/env python3
"""
分析 row switches 的来源 - 基于实际观察

观察结果:
- 460,992 Input accesses (每个 tile 访问 28 个元素，共 16464 tiles)
- 12 unique rows
- 5375 row switches

Row 分布:
- Channel 0: rows 0, 1, 7, 8
- Channel 1: rows 196, 197, 203, 204  
- Channel 2: rows 392, 393, 399, 400

每个 channel 有 4 个 blocks (2 h_blocks × 2 w_blocks)
"""
import math

# 参数
P = 56
Q = 56
R = 7
S = 7
C = 3
K = 64
H_in = 62
W_in = 62

P_l3 = 28
Q_l3 = 7
C_l3 = 3
K_l3 = 4
R_l2 = 7  # 在 Level 2

P_buffer = P // P_l3  # = 2
Q_buffer = Q // Q_l3  # = 8
R_buffer = R // R_l2  # = 1 (R 完全在 L2)

block_h = 31
block_w = 31

num_h_blocks = math.ceil(H_in / block_h)  # = 2
num_w_blocks = math.ceil(W_in / block_w)  # = 2

print("=== 基本参数 ===")
print(f"L3 loops: P_l3={P_l3}, Q_l3={Q_l3}, C_l3={C_l3}, K_l3={K_l3}")
print(f"L2 loops: R_l2={R_l2}")
print(f"Buffer tile: P={P_buffer}, Q={Q_buffer}")
print(f"Block: {block_h}×{block_w}, Blocks: {num_h_blocks}×{num_w_blocks}")
print()

# 每个 P tile 覆盖的 h 范围
# h_start = p * stride * P_buffer + r
# h_end = h_start + P_buffer - 1 + R - 1 = h_start + P_buffer + R - 2 = h_start + 2 + 6 = h_start + 8

print("=== P tile 的 h 范围 ===")
for p in range(min(P_l3, 5)):
    h_start_min = p * 1 * P_buffer
    h_start_max = h_start_min + P_buffer - 1
    h_with_r = h_start_max + R - 1  # 最大 h (when r = R-1)
    
    print(f"  p={p}: h ∈ [{h_start_min}, {h_with_r}], blocks = [{h_start_min//block_h}, {h_with_r//block_h}]")

print("  ...")
for p in range(max(0, P_l3-3), P_l3):
    h_start_min = p * 1 * P_buffer
    h_with_r = h_start_min + P_buffer - 1 + R - 1
    print(f"  p={p}: h ∈ [{h_start_min}, {h_with_r}], blocks = [{h_start_min//block_h}, {h_with_r//block_h}]")

print()

# 每个 Q tile 覆盖的 w 范围
print("=== Q tile 的 w 范围 ===")
for q in range(Q_l3):
    w_start_min = q * 1 * Q_buffer
    w_with_s = w_start_min + Q_buffer - 1 + S - 1
    print(f"  q={q}: w ∈ [{w_start_min}, {w_with_s}], blocks = [{w_start_min//block_w}, {w_with_s//block_w}]")

print()

# 访问顺序分析
# 根据 trace_generator 的实现：
# 外层循环: K -> C -> P -> Q (根据 permutation)
# 内层循环: R (at L2)
# 最内层: 元素级迭代 (by block)

print("=== 访问顺序 (Input 相关) ===")
print("根据 permutation {0: 3, 1: 2, 3: 4, 4: 5}:")
print("  L3 顺序 (外到内): K -> C -> P -> Q")
print("  L2: R")
print("  元素级: for h_block: for w_block: for h: for w")
print()

# 统计 row switches
print("=== 模拟 row switches (考虑 block 迭代) ===")

row_switches = 0
last_row = None

def get_row(c, h_block, w_block, block_h, block_w, C):
    """
    计算 (c, h_block, w_block) 对应的 row id
    
    基于观察的 row 分布:
    - C=0: rows 0, 1, 7, 8
    - C=1: rows 196, 197, 203, 204
    - C=2: rows 392, 393, 399, 400
    
    模式: row = c * 196 + h_block + w_block * 7
    """
    # 简化: 假设每个 (c, h_block, w_block) 是唯一的 row
    return (c, h_block, w_block)

# 访问顺序: K -> C -> P -> Q -> R
# 对于每个 (C, P, Q, R)，迭代所有 (h_block, w_block)

for k in range(K_l3):
    for c in range(C_l3):
        for p in range(P_l3):
            for q in range(Q_l3):
                # 计算这个 (p, q) tile 覆盖的 block 范围
                # 需要考虑 R_l2 的影响
                
                for r in range(R_l2):
                    # h 范围
                    h_start = p * P_buffer + r
                    h_end = h_start + P_buffer - 1  # 不包括 R 的扩展，因为 r 是固定的
                    
                    # w 范围  
                    # s 是在 Level 1 处理的，所以这里 s 从 0 到 S-1
                    w_start = q * Q_buffer
                    w_end = w_start + Q_buffer - 1 + S - 1
                    
                    # block 范围
                    h_block_start = h_start // block_h
                    h_block_end = h_end // block_h
                    w_block_start = w_start // block_w
                    w_block_end = w_end // block_w
                    
                    # 按 block 迭代
                    for h_block in range(h_block_start, min(h_block_end + 1, num_h_blocks)):
                        for w_block in range(w_block_start, min(w_block_end + 1, num_w_blocks)):
                            current_row = get_row(c, h_block, w_block, block_h, block_w, C)
                            
                            if last_row is not None and current_row != last_row:
                                row_switches += 1
                            
                            last_row = current_row

print(f"模拟 row switches: {row_switches}")
print(f"实际 Trace: 5375")
print()

# 检查是不是因为 S 在 Level 1
print("=== 考虑 S 在 Level 1 ===")
# 根据 loop_bounds: Level 1 temporal: {1: 7} 表示 S=7 在 Level 1
# 这意味着 S 在 buffer 内迭代，不影响 DRAM 访问

# 重新模拟：每个 (P, Q, R) 组合产生一次 tile 访问
# 但内部按 block 迭代所有 (h, w) 元素

row_switches_v2 = 0
last_row_v2 = None

for k in range(K_l3):
    for c in range(C_l3):
        for p in range(P_l3):
            for q in range(Q_l3):
                for r in range(R_l2):
                    # 这个 tile 访问的所有 h, w
                    # h: p*P_buffer + r 到 p*P_buffer + r + P_buffer - 1
                    # w: q*Q_buffer 到 q*Q_buffer + Q_buffer + S - 2 (因为 S 在 buffer 内)
                    
                    h_base = p * P_buffer + r
                    w_base = q * Q_buffer
                    
                    # 完整的 h, w 范围（考虑 buffer 内的迭代）
                    h_lo = h_base
                    h_hi = min(h_base + P_buffer, H_in)
                    w_lo = w_base
                    w_hi = min(w_base + Q_buffer + S - 1, W_in)
                    
                    # 确定涉及的 blocks
                    h_blocks = set(range(h_lo // block_h, min(h_hi // block_h + 1, num_h_blocks)))
                    w_blocks = set(range(w_lo // block_w, min(w_hi // block_w + 1, num_w_blocks)))
                    
                    # 按 block 迭代（这是 trace_generator 的实现）
                    for hb in sorted(h_blocks):
                        for wb in sorted(w_blocks):
                            current_row = (c, hb, wb)
                            if last_row_v2 is not None and current_row != last_row_v2:
                                row_switches_v2 += 1
                            last_row_v2 = current_row

print(f"模拟 row switches (v2): {row_switches_v2}")
print()

# 分析差异
print("=== 分析差异 ===")
print(f"模拟 vs Trace: {row_switches_v2} vs 5375")
print(f"差异: {5375 - row_switches_v2}")
print()

# 检查 K 循环的影响
# 每次 K 变化，Input 需要重新访问
# 所以 K 循环会导致额外的 switches

print("=== K 循环影响分析 ===")
# 如果不考虑 K，switches 是多少？
switches_no_k = row_switches_v2 // K_l3  # 假设每个 K 的 switches 一样
switches_from_k = row_switches_v2 - switches_no_k  # K 切换带来的额外 switches

print(f"每个 K 内的 switches ≈ {switches_no_k}")
print(f"K 切换带来的额外 switches ≈ {K_l3 - 1} (每次 K 变化)")
print()

# 计算平均每次访问的 switch 概率
total_accesses = P_l3 * Q_l3 * C_l3 * R_l2 * K_l3  # tile 访问次数
print(f"Tile 访问次数: {total_accesses}")
print(f"Switch 概率: {5375 / total_accesses:.4f}")
