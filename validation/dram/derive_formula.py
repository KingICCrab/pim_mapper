#!/usr/bin/env python3
"""
推导简化的 ILP Row Activation 公式

基于对 ResNet-L1 的详细分析，row switches 可以通过以下方式计算：
1. 统计每个 (K, C, P, Q, R) 迭代中访问的 blocks
2. 计算相邻迭代之间的 block 变化

关键洞察：
- unique_rows = num_h_blocks × num_w_blocks × C
- row_switches 取决于访问顺序和 block crossing
"""
import math

def compute_row_switches_analytical(
    P, Q, R, S, C, K,
    P_l3, Q_l3, C_l3, K_l3, R_l2,
    block_h, block_w, H_in, W_in,
    stride=1
):
    """
    分析性计算 row switches
    
    参数:
        P, Q, R, S, C, K: workload 参数
        P_l3, Q_l3, C_l3, K_l3: Level 3 (DRAM) tiling
        R_l2: Level 2 (RowBuffer) tiling for R
        block_h, block_w: 数据布局的 block 大小
        H_in, W_in: Input 维度
        stride: 卷积步长
    """
    P_buffer = P // P_l3
    Q_buffer = Q // Q_l3
    
    num_h_blocks = math.ceil(H_in / block_h)
    num_w_blocks = math.ceil(W_in / block_w)
    
    # 模拟访问顺序: K -> C -> P -> Q -> R
    row_switches = 0
    last_row = None
    
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r in range(R_l2):
                        # 计算这个迭代访问的 block 范围
                        h_base = p * stride * P_buffer + r
                        w_base = q * stride * Q_buffer
                        
                        h_lo = h_base
                        h_hi = min(h_base + P_buffer - 1, H_in - 1)
                        w_lo = w_base
                        w_hi = min(w_base + Q_buffer - 1 + S - 1, W_in - 1)
                        
                        h_blocks = set(range(h_lo // block_h, min(h_hi // block_h + 1, num_h_blocks)))
                        w_blocks = set(range(w_lo // block_w, min(w_hi // block_w + 1, num_w_blocks)))
                        
                        # 按 block 迭代
                        for hb in sorted(h_blocks):
                            for wb in sorted(w_blocks):
                                current_row = (c, hb, wb)
                                if last_row is not None and current_row != last_row:
                                    row_switches += 1
                                last_row = current_row
    
    return row_switches


def compute_row_switches_simplified(
    P_l3, Q_l3, C_l3, K_l3, R_l2,
    P_buffer, Q_buffer, S,
    block_h, block_w, H_in, W_in,
    stride=1
):
    """
    简化公式：基于 block crossing 计数
    
    关键观察：
    1. 每个 (P, Q) tile 可能访问 1 或 2 个 blocks (h 方向)
    2. 每个 (P, Q) tile 可能访问 1 或 2 个 blocks (w 方向)
    3. R_l2 增加 h 方向的 block crossing
    
    公式推导：
    - 在 K 循环内，每次 (C, h_block, w_block) 组合变化就是一次 switch
    - 需要统计访问序列中的 block 切换次数
    """
    num_h_blocks = math.ceil(H_in / block_h)
    num_w_blocks = math.ceil(W_in / block_w)
    
    # 方法 1: 基于 block crossing 计数
    # 计算 Q 循环中的 w_block switches
    q_block_switches = 0
    prev_w_blocks = None
    for q in range(Q_l3):
        w_start = q * stride * Q_buffer
        w_end = w_start + Q_buffer - 1 + S - 1
        w_blocks = set(range(w_start // block_w, min(w_end // block_w + 1, num_w_blocks)))
        if prev_w_blocks is not None and w_blocks != prev_w_blocks:
            q_block_switches += 1
        prev_w_blocks = w_blocks
    
    # 计算 P 循环中的 h_block switches (考虑 R_l2)
    p_block_switches = 0
    for r in range(R_l2):
        prev_h_blocks = None
        for p in range(P_l3):
            h_start = p * stride * P_buffer + r
            h_end = h_start + P_buffer - 1
            h_blocks = set(range(h_start // block_h, min(h_end // block_h + 1, num_h_blocks)))
            if prev_h_blocks is not None and h_blocks != prev_h_blocks:
                p_block_switches += 1
            prev_h_blocks = h_blocks
    
    # 每次 C 变化必定切换
    c_switches = C_l3 - 1
    
    # 总 switches 估算（这是一个上界）
    # 实际值需要考虑嵌套顺序
    estimated = (
        c_switches * P_l3 * Q_l3 * R_l2 +  # C 切换
        p_block_switches * Q_l3 * C_l3 +   # P 切换 (× Q × C)
        q_block_switches * P_l3 * R_l2 * C_l3  # Q 切换 (× P × R × C)
    ) * K_l3
    
    return estimated, {
        'q_block_switches': q_block_switches,
        'p_block_switches': p_block_switches,
        'c_switches': c_switches,
    }


# ResNet-L1 参数
P, Q, R, S, C, K = 56, 56, 7, 7, 3, 64
P_l3, Q_l3, C_l3, K_l3 = 28, 7, 3, 4
R_l2 = 7
block_h, block_w = 31, 31
H_in, W_in = 62, 62

print("=" * 60)
print("ResNet-L1 Row Switches 分析")
print("=" * 60)

# 精确计算
exact = compute_row_switches_analytical(
    P, Q, R, S, C, K,
    P_l3, Q_l3, C_l3, K_l3, R_l2,
    block_h, block_w, H_in, W_in
)
print(f"\n精确计算: {exact}")
print(f"实际 Trace: 5375")

# 简化公式
simplified, details = compute_row_switches_simplified(
    P_l3, Q_l3, C_l3, K_l3, R_l2,
    P // P_l3, Q // Q_l3, S,
    block_h, block_w, H_in, W_in
)
print(f"\n简化公式 (上界): {simplified}")
print(f"详情: {details}")

# 更好的简化公式
print("\n" + "=" * 60)
print("更好的简化公式推导")
print("=" * 60)

# 观察：row switches 主要来自三个来源
# 1. C channel 切换: 每次 C 变化
# 2. h_block 切换: P 迭代中跨越 h_block 边界
# 3. w_block 切换: Q 迭代中跨越 w_block 边界

# 让我们分析每种来源

num_h_blocks = math.ceil(H_in / block_h)
num_w_blocks = math.ceil(W_in / block_w)
P_buffer = P // P_l3
Q_buffer = Q // Q_l3

print(f"\nBlock 结构:")
print(f"  num_h_blocks = {num_h_blocks}")
print(f"  num_w_blocks = {num_w_blocks}")
print(f"  P_buffer = {P_buffer}")
print(f"  Q_buffer = {Q_buffer}")

# 分析 P 迭代中的 h_block 边界
print(f"\nP 迭代分析:")
h_crossings_by_r = []
for r in range(R_l2):
    crossings = 0
    prev_hb = None
    for p in range(P_l3):
        h = p * P_buffer + r
        hb = min(h // block_h, num_h_blocks - 1)
        if prev_hb is not None and hb != prev_hb:
            crossings += 1
        prev_hb = hb
    h_crossings_by_r.append(crossings)
    print(f"  r={r}: h_block crossings = {crossings}")

total_h_crossings = sum(h_crossings_by_r)
print(f"  Total h crossings (all r): {total_h_crossings}")

# 分析 Q 迭代中的 w_block 边界
print(f"\nQ 迭代分析:")
w_crossings = 0
prev_wb_set = None
for q in range(Q_l3):
    w_start = q * Q_buffer
    w_end = w_start + Q_buffer - 1 + S - 1
    wb_set = set(range(w_start // block_w, min(w_end // block_w + 1, num_w_blocks)))
    if prev_wb_set is not None and wb_set != prev_wb_set:
        w_crossings += 1
    prev_wb_set = wb_set
    print(f"  q={q}: w ∈ [{w_start}, {w_end}], w_blocks = {wb_set}")

print(f"  W crossings = {w_crossings}")

# 推导公式
print(f"\n" + "=" * 60)
print("ILP 公式推导")
print("=" * 60)

# 访问顺序: K -> C -> P -> Q -> R -> (per-element by block)
# 
# 对于每个 K:
#   对于每个 C:
#     对于每个 P:
#       对于每个 Q:
#         对于每个 R:
#           访问涉及的 blocks (按 h_block, w_block 顺序)

# 每次进入新的 (C, h_block, w_block) 组合时发生 switch

# 简化观察：
# 1. unique_rows = num_h_blocks × num_w_blocks × C_l3 = 2 × 2 × 3 = 12
# 2. 每个 (P, Q, R) 组合访问的 blocks 是固定的
# 3. block 切换发生在 Q 循环跨越 w_block 边界，或 P/R 循环跨越 h_block 边界

# 精确公式需要模拟，但可以用上界估算
# switches ≤ unique_rows × (访问次数 / unique_rows - 1) × K_l3
#           ≈ 访问次数 - K_l3

# 更精确：考虑每轮 Q 循环的 switches
# 每轮 Q 循环: w_crossings 次 w_block 切换
# 每轮 P 循环: P_l3 × (每个 P 的 h_block 切换)
# 每轮 C 循环: 进入时 1 次切换 (如果从不同 row)

# 让我计算更精确的值
# 对于固定的 (K, C):
#   switches = Σ_{p,q,r} (当前 block set ≠ 上一个 block set)

per_k_c_switches = 0
last_blocks = None
for p in range(P_l3):
    for q in range(Q_l3):
        for r in range(R_l2):
            h = p * P_buffer + r
            w_start = q * Q_buffer
            w_end = w_start + Q_buffer - 1 + S - 1
            
            hb = min(h // block_h, num_h_blocks - 1)
            wb_set = set(range(w_start // block_w, min(w_end // block_w + 1, num_w_blocks)))
            
            current_blocks = set()
            for wb in wb_set:
                current_blocks.add((hb, wb))
            
            if last_blocks is not None:
                if current_blocks != last_blocks:
                    per_k_c_switches += 1
            last_blocks = current_blocks

print(f"Per (K, C) switches: {per_k_c_switches}")

# 每次 C 变化: 1 次额外 switch
# 每次 K 变化: 重新访问，需要计算首次进入的 switch

# 总 switches
total_kc = K_l3 * C_l3
switches_within_kc = per_k_c_switches * total_kc  # 每个 (K,C) 内的 switches
switches_between_c = (C_l3 - 1) * K_l3  # C 切换
switches_between_k = K_l3 - 1  # K 切换

total_estimate = switches_within_kc + switches_between_c + switches_between_k

print(f"\n估算分解:")
print(f"  Within (K,C): {per_k_c_switches} × {total_kc} = {switches_within_kc}")
print(f"  Between C: {C_l3 - 1} × {K_l3} = {switches_between_c}")
print(f"  Between K: {K_l3 - 1}")
print(f"  Total: {total_estimate}")

# 这个估算太大了，因为我们重复计算了 C 和 K 切换
# 实际上，per_k_c_switches 已经包含了所有切换

print(f"\n正确估算: per_k_c × K × C = {per_k_c_switches * K_l3 * C_l3}")

# 实际公式
print(f"\n" + "=" * 60)
print("最终 ILP 公式建议")
print("=" * 60)

print("""
基于分析，row_switches 可以用以下方式估算:

1. 精确方法 (需要模拟):
   - 模拟访问顺序，统计 block 切换次数
   - 这在 ILP 中不可行

2. 简化方法 A:
   row_acts_input = (P_l3 × Q_l3 × R_l2 × C_l3 + Block_Crossing) × K_l3
   
   其中 Block_Crossing 估算:
   - h_crossings: P 迭代中跨越 h_block 边界的次数 (需要考虑 R_l2)
   - w_crossings: Q 迭代中跨越 w_block 边界的次数
   
3. 简化方法 B (更简单但不那么准确):
   row_acts_input ≈ P_l3 × Q_l3 × C_l3 × max(1, R_l2 × (num_h_blocks - 1) / P_l3) × K_l3

   这假设 R_l2 增加 h 方向的 block crossing 概率

4. 当前 ILP 方法:
   row_acts_input = (P_l3 × Q_l3 × C_l3 + BC) × K_l3
   
   问题: 没有考虑 R_l2 的影响

建议的修正:
   row_acts_input = (P_l3 × Q_l3 × C_l3 × (1 + h_crossing_factor) + BC) × K_l3
   
   其中 h_crossing_factor 与 R_l2 和 block_h 相关
""")

# 计算 h_crossing_factor
h_crossing_per_pqc = total_h_crossings / (P_l3 * Q_l3 * C_l3)
print(f"h_crossing_factor ≈ {h_crossing_per_pqc:.4f}")

# 验证
adjusted_estimate = int((P_l3 * Q_l3 * C_l3 * (1 + h_crossing_per_pqc)) * K_l3)
print(f"\n调整后的估算: {adjusted_estimate}")
print(f"实际: {exact}")
print(f"误差: {abs(adjusted_estimate - exact) / exact * 100:.1f}%")
