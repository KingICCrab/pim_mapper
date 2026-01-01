#!/usr/bin/env python3
"""
设计适合 ILP 的简化 Row Activation 公式

基于详细分析，row switches 来自以下来源:
1. multi_block: Q tile 跨越 w_block 边界时的内部切换
2. R: R 迭代导致 h_block 变化
3. P: P 变化导致的切换
4. Q: Q 变化导致 w_block 变化
5. C: channel 切换
6. K: K 切换

ILP 需要一个可以用线性约束表达的公式。
"""

import math

def compute_exact_switches(
    P_l3, Q_l3, C_l3, K_l3, R_l2,
    P_buffer, Q_buffer, S,
    block_h, block_w, H_in, W_in
):
    """精确计算 row switches (模拟方法)"""
    num_h_blocks = math.ceil(H_in / block_h)
    num_w_blocks = math.ceil(W_in / block_w)
    
    row_switches = 0
    last_row = None
    
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
                        
                        for wb in range(wb_start, wb_end + 1):
                            current_row = (c, hb, wb)
                            if last_row is not None and current_row != last_row:
                                row_switches += 1
                            last_row = current_row
    
    return row_switches


def compute_simplified_formula_v1(
    P_l3, Q_l3, C_l3, K_l3, R_l2,
    P_buffer, Q_buffer, S,
    block_h, block_w, H_in, W_in
):
    """
    简化公式 V1: 基于分量估算
    
    row_acts = (multi_block + R_switches + P_switches + Q_switches + C_switches) × K_l3
    """
    num_h_blocks = math.ceil(H_in / block_h)
    num_w_blocks = math.ceil(W_in / block_w)
    
    # 1. multi_block: Q tiles 跨越 w_block 边界的次数
    q_multi_count = 0
    for q in range(Q_l3):
        w_start = q * Q_buffer
        w_end = w_start + Q_buffer + S - 2
        wb_start = w_start // block_w
        wb_end = min(w_end // block_w, num_w_blocks - 1)
        if wb_end > wb_start:
            q_multi_count += 1
    
    multi_block = q_multi_count * P_l3 * R_l2 * C_l3 * K_l3
    
    # 2. R_switches: R 变化导致 h_block 变化
    # 对于每个 P，计算 R 循环中的 h_block 边界交叉
    r_crossings_per_p = []
    for p in range(P_l3):
        crossings = 0
        for r in range(R_l2 - 1):
            h_curr = p * P_buffer + r
            h_next = p * P_buffer + (r + 1)
            if h_curr // block_h != h_next // block_h:
                crossings += 1
        r_crossings_per_p.append(crossings)
    
    total_r_crossings = sum(r_crossings_per_p)
    # 但是！每次 R crossing 会影响该 P 之后的所有 Q iterations
    # 所以需要乘以 Q_l3
    r_switches = total_r_crossings * Q_l3 * C_l3 * K_l3
    
    # 3. P_switches: P 变化导致的切换
    # 每次 P 变化，Q 和 R 重置，所以状态从 (p-1, Q_l3-1, R_l2-1) 到 (p, 0, 0)
    p_switches = (P_l3 - 1) * C_l3 * K_l3  # 简化：假设每次 P 变化都切换
    
    # 4. Q_switches: Q 变化导致 w_block 变化
    q_switches_per_p_r = 0
    for q in range(Q_l3 - 1):
        w_curr_last = q * Q_buffer + Q_buffer + S - 2
        w_next_first = (q + 1) * Q_buffer
        if w_curr_last // block_w != w_next_first // block_w:
            q_switches_per_p_r += 1
    
    q_switches = q_switches_per_p_r * P_l3 * R_l2 * C_l3 * K_l3
    
    # 5. C_switches
    c_switches = (C_l3 - 1) * K_l3
    
    # 6. K_switches
    k_switches = K_l3 - 1
    
    total = multi_block + r_switches + p_switches + q_switches + c_switches + k_switches
    
    return total, {
        'multi_block': multi_block,
        'R_switches': r_switches,
        'P_switches': p_switches,
        'Q_switches': q_switches,
        'C_switches': c_switches,
        'K_switches': k_switches,
    }


def compute_ilp_compatible_formula(
    P_l3, Q_l3, C_l3, K_l3, R_l2,
    block_h, H_in, num_w_blocks,
    block_crossing_count  # ILP 计算的 block crossing count
):
    """
    ILP 兼容公式
    
    这个公式可以在 ILP 中用线性约束表达：
    
    row_acts_input = (P_l3 × Q_l3 × C_l3 × R_l2_factor + block_crossing_count) × K_l3
    
    其中:
    - R_l2_factor 表示 R_l2 对 h_block 切换的影响
    - block_crossing_count 是 ILP 现有的计算
    """
    num_h_blocks = math.ceil(H_in / block_h)
    
    # R_l2 导致的 h_block 切换因子
    # 当 R_l2 > block_h 时，R 循环会跨越 block 边界
    # 简化假设: 每个 P tile 有 (R_l2 - 1) / block_h 次 h_block 切换
    if R_l2 > 1:
        r_factor = min(R_l2 - 1, num_h_blocks - 1) / R_l2
    else:
        r_factor = 0
    
    # 基础访问次数 (考虑 R_l2 的影响)
    base_accesses = P_l3 * Q_l3 * C_l3 * R_l2
    
    # 调整后的公式
    row_acts = (base_accesses * (1 + r_factor) + block_crossing_count) * K_l3
    
    return row_acts


# 测试
print("=" * 60)
print("ResNet-L1 测试")
print("=" * 60)

# 参数
P, Q, R, S, C, K = 56, 56, 7, 7, 3, 64
P_l3, Q_l3, C_l3, K_l3 = 28, 7, 3, 4
R_l2 = 7
block_h, block_w = 31, 31
H_in, W_in = 62, 62
P_buffer = P // P_l3
Q_buffer = Q // Q_l3

print(f"\n参数:")
print(f"  P_l3={P_l3}, Q_l3={Q_l3}, C_l3={C_l3}, K_l3={K_l3}")
print(f"  R_l2={R_l2}")
print(f"  P_buffer={P_buffer}, Q_buffer={Q_buffer}")
print(f"  block_h={block_h}, block_w={block_w}")
print()

# 精确计算
exact = compute_exact_switches(
    P_l3, Q_l3, C_l3, K_l3, R_l2,
    P_buffer, Q_buffer, S,
    block_h, block_w, H_in, W_in
)
print(f"精确计算: {exact}")

# 简化公式 V1
simplified, details = compute_simplified_formula_v1(
    P_l3, Q_l3, C_l3, K_l3, R_l2,
    P_buffer, Q_buffer, S,
    block_h, block_w, H_in, W_in
)
print(f"\n简化公式 V1: {simplified}")
for k, v in details.items():
    print(f"  {k}: {v}")

# 当前 ILP
current_ilp = (P_l3 * Q_l3 * C_l3 + 10) * K_l3  # 假设 BC=10
print(f"\n当前 ILP (无 R): {current_ilp}")

# 建议的 ILP 公式
num_h_blocks = math.ceil(H_in / block_h)
num_w_blocks = math.ceil(W_in / block_w)

print(f"\n" + "=" * 60)
print("建议的 ILP 修改")
print("=" * 60)

print("""
问题: 当前 ILP 公式 (P × Q × C + BC) × K 忽略了 R_l2 的影响

建议修改:

方案 1: 简单乘以 R_l2
  row_acts = (P_l3 × Q_l3 × C_l3 × R_l2 + BC × R_l2) × K_l3
  
  对于 ResNet-L1: (28 × 7 × 3 × 7 + 10 × 7) × 4 = 16744
  
  问题: 过度估算，因为不是每次 R 变化都会导致 block 切换

方案 2: 使用 h_block crossing 因子
  crossing_factor = min(R_l2 - 1, num_h_blocks - 1)
  row_acts = (P_l3 × Q_l3 × C_l3 + BC) × K_l3 + R_crossings × Q_l3 × C_l3 × K_l3
  
  其中 R_crossings = Σ_p (p * P_buffer + R_l2 - 1) // block_h - (p * P_buffer) // block_h)

方案 3: 基于 unique_rows 的简化
  unique_rows = num_h_blocks × num_w_blocks × C_l3
  base_switches = (P_l3 × Q_l3 × C_l3 × R_l2 - unique_rows) × switch_prob
  row_acts = (unique_rows + base_switches) × K_l3
  
  其中 switch_prob ≈ 0.3 (基于观察)

推荐: 方案 2，因为它:
1. 可以在 ILP 中线性表达
2. 考虑了 R_l2 和 block_h 的关系
3. 不需要精确模拟
""")

# 计算方案 2
r_crossings = 0
for p in range(P_l3):
    h_start = p * P_buffer
    h_end = h_start + R_l2 - 1
    r_crossings += (h_end // block_h) - (h_start // block_h)

plan2_estimate = ((P_l3 * Q_l3 * C_l3 + 10) + r_crossings * Q_l3) * K_l3
print(f"\n方案 2 估算:")
print(f"  R_crossings = {r_crossings}")
print(f"  公式: ((P×Q×C + BC) + R_crossings × Q) × K")
print(f"  = (({P_l3}×{Q_l3}×{C_l3} + 10) + {r_crossings} × {Q_l3}) × {K_l3}")
print(f"  = {plan2_estimate}")
print(f"  精确值: {exact}")
print(f"  误差: {abs(plan2_estimate - exact) / exact * 100:.1f}%")

# 改进方案 2: 考虑 multi_block
q_multi = sum(1 for q in range(Q_l3) 
              if (q * Q_buffer + Q_buffer + S - 2) // block_w > (q * Q_buffer) // block_w)
multi_block_term = q_multi * P_l3 * R_l2 * C_l3

plan2_improved = (P_l3 * Q_l3 * C_l3 + 10 + r_crossings * Q_l3 + multi_block_term // K_l3) * K_l3
print(f"\n方案 2 改进 (加入 multi_block):")
print(f"  multi_block_term = {multi_block_term}")
print(f"  估算: {plan2_improved}")
print(f"  误差: {abs(plan2_improved - exact) / exact * 100:.1f}%")
