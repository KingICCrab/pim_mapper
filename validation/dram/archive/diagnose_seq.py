#!/usr/bin/env python3
"""诊断 Sequential 模式计算差异"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')
from pim_optimizer.model.row_activation import (
    compute_input_sequential_row_activation, 
    compute_input_block_crossing_ratio
)

row_bytes = 1024
element_bytes = 2

# H=W=6 的所有 divisors
h_divisors = [1, 2, 3, 6]
w_divisors = [1, 2, 3, 6]

# Q=4, P=4, R=3, S=3
# input_tile_h = Q_tile + S - 1 = Q_tile + 2
# Q_tile 可能是 1, 2, 4
yh_values = [3, 4, 6]  # Q_tile + 2
yw_values = [3, 4, 6]
yh_q_values = [1, 2, 4]
yw_p_values = [1, 2, 4]

print('=== 检查各种 (block, tile) 组合的 seq_factor ===')
print()

all_factors = []
for bh in h_divisors:
    for bw in w_divisors:
        for i, yh in enumerate(yh_values):
            for j, yw in enumerate(yw_values):
                step_h = yh_q_values[i]
                step_w = yw_p_values[j]
                
                seq_factor = compute_input_sequential_row_activation(
                    block_h=bh, block_w=bw,
                    tile_h=yh, tile_w=yw,
                    step_h=step_h, step_w=step_w,
                    row_bytes=row_bytes,
                    element_bytes=element_bytes,
                    total_S=3, total_R=3,
                    dilation_h=1, dilation_w=1
                )
                all_factors.append(seq_factor)

avg = sum(all_factors) / len(all_factors)
print(f'Total combinations: {len(all_factors)}')
print(f'Average seq_factor: {avg:.4f}')
print(f'If mem_per_bank = 76.59, row_act_seq = {76.59 * avg:.2f}')
print()

# 目标值: 2.235
target = 171.18 / 76.59
print(f'Target avg_seq_factor: {target:.4f}')
print(f'Difference: {abs(avg - target):.4f}')
print()

# 只看 block=(1,3) 的情况
print('=== 只看 block=(1,3) 的组合 ===')
bh, bw = 1, 3
factors_13 = []
for i, yh in enumerate(yh_values):
    for j, yw in enumerate(yw_values):
        step_h = yh_q_values[i]
        step_w = yw_p_values[j]
        
        seq_factor = compute_input_sequential_row_activation(
            block_h=bh, block_w=bw,
            tile_h=yh, tile_w=yw,
            step_h=step_h, step_w=step_w,
            row_bytes=row_bytes, element_bytes=element_bytes,
            total_S=3, total_R=3, dilation_h=1, dilation_w=1
        )
        factors_13.append(seq_factor)
        
        cr_h = compute_input_block_crossing_ratio(bh, yh, step_h, 3, 3, 1)
        cr_w = compute_input_block_crossing_ratio(bw, yw, step_w, 3, 3, 1)
        
        print(f'tile={yh}×{yw}, step={step_h}×{step_w}: cr_h={cr_h:.2f}, cr_w={cr_w:.2f}, seq={seq_factor:.4f}')

print(f'Average for block=(1,3): {sum(factors_13)/len(factors_13):.4f}')
