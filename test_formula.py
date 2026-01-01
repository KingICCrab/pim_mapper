#!/usr/bin/env python3
"""验证公式是否能计算出 Trace 的 5376"""

import math
from collections import defaultdict

# ResNet-L1 参数
P, Q, R, S = 56, 56, 7, 7
C, K = 3, 64
stride_h, stride_w = 1, 1
dilation_h, dilation_w = 1, 1

# Block sizes
block_h, block_w = 31, 31

# Trace 使用的 DRAM 映射配置
P_factor = 28  # DRAM P 因子
Q_factor = 7   # DRAM Q 因子
R_factor = 7   # Level 2 的 R 因子 (不是 DRAM level!)
S_factor = 1   # S 完全在 buffer 内
C_factor = 3   # DRAM C 因子
K_factor = 4   # DRAM K 因子

# 计算 Buffer tile 大小
P_tile = P // P_factor  # = 56 / 28 = 2
Q_tile = Q // Q_factor  # = 56 / 7 = 8

print(f'Buffer tile: P_tile={P_tile}, Q_tile={Q_tile}')

# 关键理解：
# - R 在 Level 2 意味着 DRAM tile 内部有 R_factor 次迭代
# - 每次迭代，H 方向的起始位置偏移 dilation_h
# - 所以 DRAM 循环是: K × C × P × Q × R

# 计算每个 (p, r) 组合的 H 方向位置
# 总共有 P_factor × R_factor 个 H 方向 "sub-tiles"

# Input tile 尺寸 (每个 sub-tile)
# R_tile = R // R_factor = 7 // 7 = 1 (因为 R 在 Level 2)
R_tile = R // R_factor  # = 1

# 每个 sub-tile 的尺寸
H_tile = stride_h * (P_tile - 1) + dilation_h * (R_tile - 1) + 1  # = 1 + 0 + 1 = 2
W_tile = stride_w * (Q_tile - 1) + dilation_w * (S - 1) + 1  # = 7 + 6 + 1 = 14 (S 完全在 buffer)

print(f'Input sub-tile: H_tile={H_tile}, W_tile={W_tile}')
print(f'R_tile={R_tile} (R完全在Level 2)')

# 方法 1: 精确枚举计算 row switches
def count_row_switches_exact():
    """精确枚举计算 row switches"""
    prev_row = None
    row_switches = 0
    
    for k in range(K_factor):
        for c in range(C_factor):
            for p in range(P_factor):
                for q in range(Q_factor):
                    for r in range(R_factor):
                        # 计算这个 sub-tile 的起始位置
                        h_start = p * P_tile * stride_h + r * dilation_h
                        h_end = h_start + H_tile - 1
                        w_start = q * Q_tile * stride_w
                        w_end = w_start + W_tile - 1
                        
                        # 计算覆盖的 blocks
                        h_blocks = list(range(h_start // block_h, h_end // block_h + 1))
                        w_blocks = list(range(w_start // block_w, w_end // block_w + 1))
                        
                        for hb in h_blocks:
                            for wb in w_blocks:
                                row = (c, hb, wb)
                                if row != prev_row:
                                    row_switches += 1
                                    prev_row = row
    
    return row_switches

exact_result = count_row_switches_exact()
print(f'\n精确枚举结果: {exact_result}')
print(f'目标 Trace 结果: 5376')

# 方法 2: 分析循环结构
# 关键观察: 循环是 K × C × P × Q × R
# 每个 (c, p, q, r) 组合访问固定的 h_start, w_start
# 问题: 连续的迭代是否访问同一个 row?

print("\n=== 分析迭代之间的 row 连续性 ===")

def analyze_row_continuity():
    """分析迭代之间 row 是否连续"""
    total_iterations = K_factor * C_factor * P_factor * Q_factor * R_factor
    
    # 统计每种情况的数量
    same_row_r = 0  # r 变化时 row 不变
    diff_row_r = 0  # r 变化时 row 改变
    same_row_q = 0  # q 变化 (r wrap) 时 row 不变
    diff_row_q = 0  # q 变化 (r wrap) 时 row 改变
    
    prev_row = None
    
    for k in range(K_factor):
        for c in range(C_factor):
            for p in range(P_factor):
                for q in range(Q_factor):
                    for r in range(R_factor):
                        h_start = p * P_tile * stride_h + r * dilation_h
                        h_end = h_start + H_tile - 1
                        w_start = q * Q_tile * stride_w
                        w_end = w_start + W_tile - 1
                        
                        hb = h_start // block_h
                        wb = w_start // block_w
                        row = (c, hb, wb)
                        
                        if prev_row is not None:
                            if row == prev_row:
                                same_row_r += 1
                            else:
                                diff_row_r += 1
                        
                        prev_row = row
    
    print(f'Total iterations: {total_iterations}')
    print(f'Same row (consecutive): {same_row_r}')
    print(f'Different row (switch): {diff_row_r}')
    print(f'First access (no prev): 1')
    print(f'Total switches = {diff_row_r + 1}')

analyze_row_continuity()

# 方法 3: 分析每个唯一 row 的访问次数
print("\n=== 分析每个唯一 row 的访问模式 ===")

def analyze_unique_rows():
    """分析每个唯一 row 的访问模式"""
    row_accesses = defaultdict(list)  # row -> list of (k, c, p, q, r)
    
    for k in range(K_factor):
        for c in range(C_factor):
            for p in range(P_factor):
                for q in range(Q_factor):
                    for r in range(R_factor):
                        h_start = p * P_tile * stride_h + r * dilation_h
                        h_end = h_start + H_tile - 1
                        w_start = q * Q_tile * stride_w
                        w_end = w_start + W_tile - 1
                        
                        h_blocks = list(range(h_start // block_h, h_end // block_h + 1))
                        w_blocks = list(range(w_start // block_w, w_end // block_w + 1))
                        
                        for hb in h_blocks:
                            for wb in w_blocks:
                                row = (c, hb, wb)
                                row_accesses[row].append((k, c, p, q, r))
    
    unique_rows = len(row_accesses)
    print(f'Unique rows: {unique_rows}')
    
    # 分析连续访问
    total_groups = 0
    for row, accesses in row_accesses.items():
        # 计算这个 row 被访问多少个"连续组"
        groups = 1
        for i in range(1, len(accesses)):
            # 检查是否连续 (按循环顺序)
            prev_k, prev_c, prev_p, prev_q, prev_r = accesses[i-1]
            curr_k, curr_c, curr_p, curr_q, curr_r = accesses[i]
            
            # 连续意味着 iteration index 差 1
            prev_idx = ((prev_k * C_factor + prev_c) * P_factor + prev_p) * Q_factor * R_factor + prev_q * R_factor + prev_r
            curr_idx = ((curr_k * C_factor + curr_c) * P_factor + curr_p) * Q_factor * R_factor + curr_q * R_factor + curr_r
            
            if curr_idx != prev_idx + 1:
                groups += 1
        
        total_groups += groups
    
    print(f'Total access groups (= switches): {total_groups}')
    
    return unique_rows, total_groups

unique_rows, total_groups = analyze_unique_rows()

# 方法 4: 简化分析 - 只考虑 block crossing
print("\n=== 简化分析: 计算 row switches ===")

def count_per_cpq_r():
    """对于固定的 (c, p, q)，计算 R 循环内的 switches"""
    total_switches = 0
    
    prev_row = None  # 每个 CPQ 重置? 还是跨越?
    for c in range(C_factor):
        for p in range(P_factor):
            for q in range(Q_factor):
                # 注意: prev_row 跨越 CPQ!
                for r in range(R_factor):
                    h_start = p * P_tile * stride_h + r * dilation_h
                    h_end = h_start + H_tile - 1
                    w_start = q * Q_tile * stride_w
                    w_end = w_start + W_tile - 1
                    
                    h_blocks = list(range(h_start // block_h, h_end // block_h + 1))
                    w_blocks = list(range(w_start // block_w, w_end // block_w + 1))
                    
                    for hb in h_blocks:
                        for wb in w_blocks:
                            row = (c, hb, wb)
                            if row != prev_row:
                                total_switches += 1
                                prev_row = row
    
    print(f'Per CPQ×R analysis (prev_row 跨越): {total_switches} switches (no K)')
    return total_switches

switches_per_cpq = count_per_cpq_r()

# 现在验证: switches_per_cpq 应该等于每个 K 迭代的 switches
print(f'每个 K 迭代的 switches: 1344')
print(f'switches_per_cpq: {switches_per_cpq}')
print(f'匹配? {switches_per_cpq == 1344}')

# 但是 K 在外层意味着每次 K 迭代从头开始，会有额外的 switch
print("\n=== 考虑 K 边界的额外 switches ===")

def count_with_k_boundary():
    """计算包括 K 边界的 switches"""
    # 每个 K 迭代从 c=0, p=0, q=0, r=0 开始
    # 上一个 K 迭代结束时在 c=C-1, p=P-1, q=Q-1, r=R-1
    
    # 第一个 K 的起始 row
    h_start_0 = 0
    w_start_0 = 0
    row_first = (0, h_start_0 // block_h, w_start_0 // block_w)
    
    # 最后一个迭代的 row (before K wraps)
    c_last = C_factor - 1
    p_last = P_factor - 1
    q_last = Q_factor - 1
    r_last = R_factor - 1
    
    h_last = p_last * P_tile * stride_h + r_last * dilation_h + H_tile - 1
    w_last = q_last * Q_tile * stride_w + W_tile - 1
    row_last = (c_last, h_last // block_h, w_last // block_w)
    
    print(f'Row at start of K: {row_first}')
    print(f'Row at end of K: {row_last}')
    print(f'Different? {row_first != row_last}')
    
    # K 边界会产生额外的 switch 吗?
    # 如果 row_last != row_first，则每次 K 边界产生一次 switch
    # 但这已经被 switches_per_cpq × K 计算过了
    
    # 实际上，switches_per_cpq 是不考虑 K 的情况
    # 当 K 在外层时，每个 K 迭代都完全独立地计算 switches
    # 所以 total = switches_per_cpq × K
    
    return switches_per_cpq * K_factor

result_with_k = count_with_k_boundary()
print(f'Total with K: {result_with_k}')
print(f'Exact enumeration: {exact_result}')

# 验证
print("\n=== 验证 ===")
print(f'精确枚举: {exact_result}')
print(f'per_cpq × K = {switches_per_cpq} × {K_factor} = {switches_per_cpq * K_factor}')
print(f'匹配? {exact_result == switches_per_cpq * K_factor}')

# 分析 K 循环的实际行为
print("\n=== 分析 K 循环的实际行为 ===")

def analyze_k_iterations():
    """分析每个 K 迭代的 switches"""
    switches_per_k = []
    
    prev_row = None
    for k in range(K_factor):
        switches_in_k = 0
        for c in range(C_factor):
            for p in range(P_factor):
                for q in range(Q_factor):
                    for r in range(R_factor):
                        h_start = p * P_tile * stride_h + r * dilation_h
                        h_end = h_start + H_tile - 1
                        w_start = q * Q_tile * stride_w
                        w_end = w_start + W_tile - 1
                        
                        h_blocks = list(range(h_start // block_h, h_end // block_h + 1))
                        w_blocks = list(range(w_start // block_w, w_end // block_w + 1))
                        
                        for hb in h_blocks:
                            for wb in w_blocks:
                                row = (c, hb, wb)
                                if row != prev_row:
                                    switches_in_k += 1
                                    prev_row = row
        
        switches_per_k.append(switches_in_k)
        print(f'K={k}: {switches_in_k} switches')
    
    print(f'Total: {sum(switches_per_k)}')
    print(f'注意: prev_row 跨 K 边界保持!')

analyze_k_iterations()

# 关键发现: prev_row 跨 K 边界保持，所以不是简单的乘法关系
# 第一个 K 迭代从 prev_row=None 开始，后续 K 迭代从上一个 K 结束的 row 开始

# 如果我们在每个 K 迭代内独立计算 (prev_row=None 开始)
print("\n=== 如果每个 K 独立计算 ===")

def analyze_k_independent():
    """每个 K 迭代独立计算"""
    switches_per_k = []
    
    for k in range(K_factor):
        prev_row = None  # 每个 K 重置
        switches_in_k = 0
        for c in range(C_factor):
            for p in range(P_factor):
                for q in range(Q_factor):
                    for r in range(R_factor):
                        h_start = p * P_tile * stride_h + r * dilation_h
                        h_end = h_start + H_tile - 1
                        w_start = q * Q_tile * stride_w
                        w_end = w_start + W_tile - 1
                        
                        h_blocks = list(range(h_start // block_h, h_end // block_h + 1))
                        w_blocks = list(range(w_start // block_w, w_end // block_w + 1))
                        
                        for hb in h_blocks:
                            for wb in w_blocks:
                                row = (c, hb, wb)
                                if row != prev_row:
                                    switches_in_k += 1
                                    prev_row = row
        
        switches_per_k.append(switches_in_k)
        print(f'K={k}: {switches_in_k} switches (independent)')
    
    print(f'Total (K independent): {sum(switches_per_k)}')
    return sum(switches_per_k)

k_independent = analyze_k_independent()

print("\n=== 结论 ===")
print(f'精确枚举 (prev_row 跨 K): {exact_result}')
print(f'K 独立计算: {k_independent}')
print(f'per_cpq × K = {switches_per_cpq} × {K_factor} = {switches_per_cpq * K_factor}')
print()
print("关键发现: Trace 使用的是 'prev_row 跨 K 边界保持' 的方式")
print("这意味着 K 循环在外层时，row switches 不是简单的乘法关系")
print("而是只有当 K 边界处的 row 改变时才计一次 switch")

# 计算正确的公式
print("\n=== 正确的公式推导 ===")

# 不考虑 K 时的基础 switches
base_switches = switches_per_cpq  # C × P × Q × R 的 switches
print(f'基础 switches (C×P×Q×R): {base_switches}')

# K 边界产生的额外 switches
# 每次 K 边界，如果从 (c_last, h_last, w_last) 跳到 (c_first, h_first, w_first)
# 且这两个 row 不同，则产生一次 switch
# 但如果第一个 K 的第一个 row 与上一个 K 的最后一个 row 相同，则不产生 switch

c_first, p_first, q_first, r_first = 0, 0, 0, 0
h_first = p_first * P_tile * stride_h + r_first * dilation_h
w_first = q_first * Q_tile * stride_w
row_first = (c_first, h_first // block_h, w_first // block_w)

c_last, p_last, q_last, r_last = C_factor-1, P_factor-1, Q_factor-1, R_factor-1
h_last = p_last * P_tile * stride_h + r_last * dilation_h + H_tile - 1
w_last = q_last * Q_tile * stride_w + W_tile - 1
row_last = (c_last, h_last // block_h, w_last // block_w)

print(f'K 开始时的 row: {row_first}')
print(f'K 结束时的 row: {row_last}')
print(f'K 边界产生 switch? {row_first != row_last}')

# 正确公式:
# total = base_switches + (K_factor - 1) × (1 if row_first != row_last else 0)
k_boundary_switches = (K_factor - 1) * (1 if row_first != row_last else 0)
total_formula = base_switches + k_boundary_switches
print(f'\n公式: base_switches + k_boundary_switches = {base_switches} + {k_boundary_switches} = {total_formula}')
print(f'精确枚举: {exact_result}')
print(f'差异: {exact_result - total_formula}')


