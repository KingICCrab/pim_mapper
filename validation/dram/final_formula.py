#!/usr/bin/env python3
"""
最终的 ILP Row Activation 公式设计

row_switches = multi_block + r_switches + p_switches + q_switches + c_switches + k_switches
"""

import math

def compute_exact(P_l3, Q_l3, C_l3, K_l3, R_l2, P_buffer, Q_buffer, S, block_h, block_w, H_in, W_in):
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
                        hb = min(h // block_h, num_h_blocks - 1)
                        
                        w_start = q * Q_buffer
                        w_end = w_start + Q_buffer + S - 2
                        wb_start = w_start // block_w
                        wb_end = min(w_end // block_w, num_w_blocks - 1)
                        
                        for wb in range(wb_start, wb_end + 1):
                            current_row = (c, hb, wb)
                            if last_row is not None and current_row != last_row:
                                row_switches += 1
                            last_row = current_row
    
    return row_switches


def compute_formula(P_l3, Q_l3, C_l3, K_l3, R_l2, P_buffer, Q_buffer, S, block_h, block_w, H_in, W_in):
    num_h_blocks = math.ceil(H_in / block_h)
    num_w_blocks = math.ceil(W_in / block_w)
    
    # 1. multi_block
    multi_block = 0
    for q in range(Q_l3):
        w_start = q * Q_buffer
        w_end = w_start + Q_buffer + S - 2
        wb_start = w_start // block_w
        wb_end = min(w_end // block_w, num_w_blocks - 1)
        multi_block += (wb_end - wb_start)
    multi_block *= P_l3 * R_l2 * C_l3 * K_l3
    
    # 2. R switches
    r_switches_per_ck = 0
    for p in range(P_l3):
        for q in range(Q_l3):
            w_end = q * Q_buffer + Q_buffer + S - 2
            wb_end = min(w_end // block_w, num_w_blocks - 1)
            wb_start = q * Q_buffer // block_w
            
            last_hb, last_wb = None, None
            for r in range(R_l2):
                h = p * P_buffer + r
                hb = min(h // block_h, num_h_blocks - 1)
                if last_hb is not None and (hb, wb_start) != (last_hb, last_wb):
                    r_switches_per_ck += 1
                last_hb, last_wb = hb, wb_end
    r_switches = r_switches_per_ck * C_l3 * K_l3
    
    # 3. P switches
    p_switches_per_ck = 0
    for p in range(1, P_l3):
        h_prev = (p - 1) * P_buffer + R_l2 - 1
        hb_prev = min(h_prev // block_h, num_h_blocks - 1)
        w_end_prev = (Q_l3 - 1) * Q_buffer + Q_buffer + S - 2
        wb_prev = min(w_end_prev // block_w, num_w_blocks - 1)
        
        h_curr = p * P_buffer
        hb_curr = min(h_curr // block_h, num_h_blocks - 1)
        wb_curr = 0
        
        if (hb_prev, wb_prev) != (hb_curr, wb_curr):
            p_switches_per_ck += 1
    p_switches = p_switches_per_ck * C_l3 * K_l3
    
    # 4. Q switches
    q_switches_per_prck = 0
    for q in range(1, Q_l3):
        w_end_prev = (q - 1) * Q_buffer + Q_buffer + S - 2
        wb_prev = min(w_end_prev // block_w, num_w_blocks - 1)
        wb_curr = q * Q_buffer // block_w
        if wb_prev != wb_curr:
            q_switches_per_prck += 1
    q_switches = q_switches_per_prck * P_l3 * R_l2 * C_l3 * K_l3
    
    # 5. C switches
    c_switches = (C_l3 - 1) * K_l3
    
    # 6. K switches
    k_switches = K_l3 - 1
    
    return multi_block + r_switches + p_switches + q_switches + c_switches + k_switches


# Test
P, Q, R, S, C, K = 56, 56, 7, 7, 3, 64
P_l3, Q_l3, C_l3, K_l3 = 28, 7, 3, 4
R_l2 = 7
P_buffer, Q_buffer = 2, 8
block_h, block_w = 31, 31
H_in, W_in = 62, 62

exact = compute_exact(P_l3, Q_l3, C_l3, K_l3, R_l2, P_buffer, Q_buffer, S, block_h, block_w, H_in, W_in)
formula = compute_formula(P_l3, Q_l3, C_l3, K_l3, R_l2, P_buffer, Q_buffer, S, block_h, block_w, H_in, W_in)
current_ilp = (P_l3 * Q_l3 * C_l3 + 10) * K_l3

print(f"ResNet-L1 Input Row Switches:")
print(f"  Exact (simulation): {exact}")
print(f"  Formula: {formula}")
print(f"  Current ILP (no R): {current_ilp}")
print(f"  Error: {abs(formula - exact) / exact * 100:.1f}%")
print()

# 简化公式推导
print("=" * 60)
print("简化公式推导")
print("=" * 60)

# 关键洞察:
# row_acts ≈ (P × Q × C × R + multi_block + r_penalty) × K
# 其中 multi_block 和 r_penalty 主要取决于有多少 Q tiles 跨越 w_block

# 计算跨越 w_block 的 Q tiles 数量
q_crossing = sum(1 for q in range(Q_l3) 
                  if (q * Q_buffer + Q_buffer + S - 2) // block_w > q * Q_buffer // block_w)

print(f"Q tiles crossing w_block: {q_crossing} / {Q_l3}")

# 简化公式
# row_acts ≈ P × Q × C × (R + q_crossing × (R - 1 + R)) × K
#          ≈ P × Q × C × (R + q_crossing × (2R - 1)) × K
#          ≈ P × Q × C × R × (1 + q_crossing × (2 - 1/R)) × K

adjustment_factor = 1 + q_crossing / Q_l3 * (2 - 1/R_l2)
simplified = int(P_l3 * Q_l3 * C_l3 * R_l2 * adjustment_factor * K_l3)

print(f"\n简化公式:")
print(f"  row_acts = P × Q × C × R × adjustment × K")
print(f"  adjustment = 1 + (q_crossing / Q) × (2 - 1/R)")
print(f"  adjustment = 1 + ({q_crossing} / {Q_l3}) × (2 - 1/{R_l2}) = {adjustment_factor:.3f}")
print(f"  简化估算: {simplified}")
print(f"  误差: {abs(simplified - exact) / exact * 100:.1f}%")

print("\n" + "=" * 60)
print("ILP 实现建议")
print("=" * 60)
print("""
方案 A: 保守估算 (简单但可能过估)
  row_acts_input = P_l3 × Q_l3 × C_l3 × R_l2 × 1.5 × K_l3
  
  优点: 简单，不需要新变量
  缺点: 可能过度惩罚某些配置

方案 B: 精确估算 (需要新约束)
  1. 添加辅助变量 q_crossing 表示跨越 w_block 的 Q tiles
  2. row_acts = (P × Q × C × R + q_crossing × P × (R - 1 + R) × C) × K
  
  优点: 更准确
  缺点: 需要添加 ILP 约束来计算 q_crossing

方案 C: 使用现有的 Block Crossing 变量
  如果 ILP 已经有 block_crossing_count 变量，可以:
  row_acts = (P × Q × C × R + block_crossing × R) × K
  
  需要验证 block_crossing 的定义是否匹配
""")
