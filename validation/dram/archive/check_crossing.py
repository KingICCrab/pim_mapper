import sys
sys.path.insert(0, 'src')

from pim_optimizer.workload.conv import ConvWorkload
import math

wl = ConvWorkload(name='tiny', N=1, K=8, C=8, P=4, Q=4, R=3, S=3)

stride_h = wl.stride[1]
dilation_h = wl.dilation[1]
total_S = wl.bounds[1]  # S = 3
total_Q = wl.bounds[3]  # Q = 4

def compute_unique_input_size(stride, dilation, tile_output, tile_kernel):
    if tile_output <= 0 or tile_kernel <= 0:
        return 0
    return stride * (tile_output - 1) + dilation * (tile_kernel - 1) + 1

# 枚举所有 Q × S 组合
yh_list = []
yh_q_list = []
yh_s_list = []

for q in wl.divisors[3]:  # Q divisors
    for s in wl.divisors[1]:  # S divisors
        tile_h = compute_unique_input_size(stride_h, dilation_h, q, s)
        yh_list.append(tile_h)
        yh_q_list.append(q)
        yh_s_list.append(s)
        print(f'Q={q}, S={s} => tile_h={tile_h}')

print(f'\nyh_list: {yh_list}')
print(f'yh_q_list: {yh_q_list}')
print(f'yh_s_list: {yh_s_list}')

# crossing ratio 计算
print('\n=== 计算 crossing ratio ===')
h_divisors = [d for d in range(1, 7) if 6 % d == 0]
print(f'h_divisors: {h_divisors}')

input_h_total = (total_Q - 1) * stride_h + (total_S - 1) * dilation_h + 1

def compute_input_crossing_ratio(block_h, tile_h, step, tiler_s, total_S, dilation, input_h):
    if block_h <= 0 or tile_h <= 0 or step <= 0:
        return 0.0
    if tile_h > block_h:
        return 1.0
    
    g = math.gcd(int(step), int(block_h))
    period = int(block_h) // g
    
    if tiler_s >= total_S:
        if input_h is not None and input_h > 0:
            if tile_h >= input_h:
                num_tiles = 1
            else:
                num_tiles = (input_h - tile_h) // step + 1
            
            crossing_positions = set()
            for k in range(period):
                pos_mod = (k * step) % block_h
                if pos_mod + tile_h > block_h:
                    crossing_positions.add(k)
            
            num_complete_periods = num_tiles // period
            remainder_tiles = num_tiles % period
            crossing_in_remainder = sum(1 for k in range(remainder_tiles) if k in crossing_positions)
            
            if num_tiles > 0:
                last_tile_pos = (num_tiles - 1) * step
                last_tile_actual_size = min(tile_h, input_h - last_tile_pos)
                last_tile_k = (num_tiles - 1) % period
                
                last_tile_pos_mod = (last_tile_k * step) % block_h
                crosses_if_full = last_tile_pos_mod + tile_h > block_h
                crosses_actual = last_tile_pos_mod + last_tile_actual_size > block_h
                
                if crosses_if_full and not crosses_actual:
                    if last_tile_k < remainder_tiles:
                        crossing_in_remainder -= 1
            
            total_crossing = num_complete_periods * len(crossing_positions) + crossing_in_remainder
            return total_crossing / num_tiles if num_tiles > 0 else 0.0
        else:
            crossing_count = 0
            for k in range(period):
                pos_mod = (k * step) % block_h
                if pos_mod + tile_h > block_h:
                    crossing_count += 1
            return crossing_count / period if period > 0 else 0.0
    else:
        return 0.0

# 计算所有组合
seq_cr_h = []
for block_h in h_divisors:
    for j_idx, tile_h in enumerate(yh_list):
        q_factor = yh_q_list[j_idx]
        s_factor = yh_s_list[j_idx]
        step_h = q_factor * stride_h
        
        cr_h = compute_input_crossing_ratio(
            block_h=block_h,
            tile_h=tile_h,
            step=step_h,
            tiler_s=s_factor,
            total_S=total_S,
            dilation=dilation_h,
            input_h=input_h_total
        )
        seq_cr_h.append(cr_h)
        print(f'block={block_h}, tile_h={tile_h} (Q={q_factor}, S={s_factor}), step={step_h} => cr={cr_h:.4f}')

avg_cr_h = sum(seq_cr_h) / len(seq_cr_h)
print(f'\n平均 cr_h: {avg_cr_h:.4f}')
