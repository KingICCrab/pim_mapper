#!/usr/bin/env python3
"""Test row switch calculation."""

import math

def compute_row_switches_full(
    block_h: int, block_w: int,
    P_factor: int, Q_factor: int,
    R_factor: int, S_factor: int,
    C_factor: int, K_factor: int,
    total_P: int, total_Q: int,
    total_R: int, total_S: int,
    stride_h: int, stride_w: int,
    dilation_h: int, dilation_w: int,
) -> dict:
    """
    计算完整的 row switches，包括 K, C 循环。
    
    遍历顺序：K -> C -> P -> Q -> R (with S in buffer)
    """
    P_tile = total_P // P_factor
    Q_tile = total_Q // Q_factor
    R_tile = total_R // R_factor
    S_tile = total_S // S_factor if S_factor > 0 else total_S
    
    switches = {
        'first': 0,
        'c_change': 0,  # C 变化导致的 switch (重新开始)
        'h_only': 0,
        'w_only': 0,
        'h_and_w': 0,
    }
    
    prev_row_h = None
    prev_row_w = None
    prev_c = None
    
    for k in range(K_factor):
        for c in range(C_factor):
            for p in range(P_factor):
                for q in range(Q_factor):
                    for r in range(R_factor):
                        # S is in buffer, so we don't iterate over S here
                        # 计算 Input 坐标 (基于 P, Q, R 的位置，与 C 无关)
                        h_start = p * P_tile * stride_h + r * R_tile * dilation_h
                        w_start = q * Q_tile * stride_w  # R direction, no S here
                        
                        # 计算所在的 row (block)
                        current_row_h = h_start // block_h
                        current_row_w = w_start // block_w
                        
                        if prev_row_h is None:
                            switches['first'] = 1
                        elif prev_c != c:
                            # C changed, this is a new channel, count as switch
                            switches['c_change'] += 1
                        else:
                            h_changed = (current_row_h != prev_row_h)
                            w_changed = (current_row_w != prev_row_w)
                            
                            if h_changed and w_changed:
                                switches['h_and_w'] += 1
                            elif h_changed:
                                switches['h_only'] += 1
                            elif w_changed:
                                switches['w_only'] += 1
                            # else: no change, no switch
                        
                        prev_row_h = current_row_h
                        prev_row_w = current_row_w
                        prev_c = c
    
    total = sum(switches.values())
    return switches, total


# 测试 Trace 配置
block_h, block_w = 31, 31
P_factor, Q_factor = 28, 7
R_factor, S_factor = 7, 1  # S in buffer means S_factor=1 for this calculation
C_factor, K_factor = 3, 4
total_P, total_Q = 56, 56
total_R, total_S = 7, 7
stride_h, stride_w = 1, 1
dilation_h, dilation_w = 1, 1

switches, total = compute_row_switches_full(
    block_h, block_w,
    P_factor, Q_factor,
    R_factor, S_factor,
    C_factor, K_factor,
    total_P, total_Q,
    total_R, total_S,
    stride_h, stride_w,
    dilation_h, dilation_w,
)
print(f'Full switches (K={K_factor}, C={C_factor}, P={P_factor}, Q={Q_factor}, R={R_factor}, S={S_factor}):')
print(f'  {switches}')
print(f'  Total: {total}')
print(f'  Trace value: 5376')

# Trace breakdown:
print(f'\nTrace breakdown: first=1, h_only=144*K=576, w_only=1176*K=4704, h_and_w=21*K=84, c_h_w=2*K=8?')
print(f'Or maybe: 1344 * 4 = 5376')

# 检查单个 K 迭代
print('\n=== 单个 K 迭代 ===')
switches_1k, total_1k = compute_row_switches_full(
    block_h, block_w,
    P_factor, Q_factor,
    R_factor, S_factor,
    C_factor, 1,  # K_factor = 1
    total_P, total_Q,
    total_R, total_S,
    stride_h, stride_w,
    dilation_h, dilation_w,
)
print(f'Switches for K=1: {switches_1k}, total={total_1k}')
print(f'Trace value per K: 1344')
