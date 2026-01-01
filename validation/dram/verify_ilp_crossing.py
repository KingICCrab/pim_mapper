#!/usr/bin/env python3
"""
验证 ILP 模型的 crossing 计算方式，找出 2384 vs 2736 的差异
"""
import math
from math import gcd, ceil

# ResNet-L1 参数
block_h, block_w = 31, 31
stride_h, stride_w = 1, 1
dilation_h, dilation_w = 1, 1

# Workload bounds
total_Q = 56
total_S = 7
total_P = 56
total_R = 7

# 选中的 factors (从 mapping 中)
P_l3_factor = 7   # P_l3 = 7 means total_P / 7 = 8 = P_buf
Q_l3_factor = 28  # Q_l3 = 28 means total_Q / 28 = 2 = Q_buf
S_l2_factor = 7   # S_l2 = 7 means total_S / 7 = 1 = S_buf

# 计算 tile sizes
Q_rb = total_Q // Q_l3_factor  # = 2
S_rb = total_S // S_l2_factor  # = 1
P_rb = total_P // P_l3_factor  # = 8

print("=" * 80)
print("ILP 模型 Input Row Activation 计算分析")
print("=" * 80)

print(f"\n选中的参数:")
print(f"  block_h = {block_h}, block_w = {block_w}")
print(f"  Q_rb (tile size) = {Q_rb}")
print(f"  S_rb (tile size) = {S_rb}")
print(f"  P_rb (tile size) = {P_rb}")

# ILP 计算 tile_h 使用的公式
# tile_h = stride × Q_rb + dilation × S_rb - stride - dilation + 1
tile_h_ilp = stride_h * Q_rb + dilation_h * S_rb - stride_h - dilation_h + 1
print(f"\nILP 计算的 tile_h:")
print(f"  tile_h = stride × Q_rb + dilation × S_rb - stride - dilation + 1")
print(f"         = {stride_h} × {Q_rb} + {dilation_h} × {S_rb} - {stride_h} - {dilation_h} + 1")
print(f"         = {tile_h_ilp}")

# 实际的 H_per_tile (trace 使用的)
H_per_tile = (P_rb - 1) * stride_h + (total_R - 1) * dilation_h + 1
print(f"\n实际的 H_per_tile (trace 计算):")
print(f"  H_per_tile = (P_rb - 1) × stride + (R - 1) × dilation + 1")
print(f"             = ({P_rb} - 1) × {stride_h} + ({total_R} - 1) × {dilation_h} + 1")
print(f"             = {H_per_tile}")

print(f"\n*** 关键发现: ILP 使用 tile_h = {tile_h_ilp}, trace 使用 H_per_tile = {H_per_tile} ***")

# Input H
input_h = stride_h * (total_Q - 1) + dilation_h * (total_S - 1) + 1
print(f"\ninput_h = {input_h}")

print("\n" + "=" * 80)
print("ILP crossing 计算 (tile_h = 2)")
print("=" * 80)

# ILP 使用的 step = Q_rb × stride
step_ilp = Q_rb * stride_h
print(f"\nstep = Q_rb × stride = {Q_rb} × {stride_h} = {step_ilp}")

# 计算 num_tiles
num_tiles_ilp = (input_h - tile_h_ilp) // step_ilp + 1
print(f"num_tiles = (input_h - tile_h) // step + 1 = ({input_h} - {tile_h_ilp}) // {step_ilp} + 1 = {num_tiles_ilp}")

# GCD 分析
g = gcd(step_ilp, block_h)
period = block_h // g
print(f"gcd({step_ilp}, {block_h}) = {g}")
print(f"period = {block_h} / {g} = {period}")

# 计算 crossing positions
crossing_positions = []
for k in range(period):
    pos_mod = (k * step_ilp) % block_h
    if pos_mod + tile_h_ilp > block_h:
        crossing_positions.append(k)
print(f"crossing positions: {crossing_positions}")
print(f"cross_count_per_period = {len(crossing_positions)}")

# 计算总 crossing
num_complete_periods = num_tiles_ilp // period
remainder_tiles = num_tiles_ilp % period
crossings_in_remainder = sum(1 for k in range(remainder_tiles) if (k * step_ilp) % block_h + tile_h_ilp > block_h)

ilp_crossing = num_complete_periods * len(crossing_positions) + crossings_in_remainder
print(f"\n完整周期数: {num_complete_periods}")
print(f"余数 tiles: {remainder_tiles}")
print(f"余数中的 crossing: {crossings_in_remainder}")
print(f"ILP crossing_count = {num_complete_periods} × {len(crossing_positions)} + {crossings_in_remainder} = {ilp_crossing}")

print("\n" + "=" * 80)
print("Trace crossing 计算 (H_per_tile = 14)")
print("=" * 80)

# Trace 使用的参数
step_trace = P_rb * stride_h  # = 8 (P 方向的跳跃)
print(f"\nstep = P_rb × stride = {P_rb} × {stride_h} = {step_trace}")

# 计算 num_tiles
num_tiles_trace = (input_h - H_per_tile) // step_trace + 1 if H_per_tile < input_h else 1
print(f"num_tiles = (input_h - H_per_tile) // step + 1 = ({input_h} - {H_per_tile}) // {step_trace} + 1 = {num_tiles_trace}")

# GCD 分析
g_trace = gcd(step_trace, block_h)
period_trace = block_h // g_trace
print(f"gcd({step_trace}, {block_h}) = {g_trace}")
print(f"period = {block_h} / {g_trace} = {period_trace}")

# 计算 crossing positions
crossing_positions_trace = []
for k in range(period_trace):
    pos_mod = (k * step_trace) % block_h
    if pos_mod + H_per_tile > block_h:
        crossing_positions_trace.append(k)
print(f"crossing positions: {crossing_positions_trace}")
print(f"cross_count_per_period = {len(crossing_positions_trace)}")

# 直接枚举
h_cross_count = 0
for p in range(P_l3_factor):  # P = 0..6
    h_start = p * P_rb * stride_h  # h = 0, 8, 16, 24, 32, 40, 48
    h_end = h_start + H_per_tile - 1
    if (h_start // block_h) != (h_end // block_h):
        h_cross_count += 1
        print(f"  P={p}: h=[{h_start}, {h_end}], crosses block {h_start // block_h} -> {h_end // block_h}")

print(f"\nH crossing 的 P tile 数: {h_cross_count}")

# 乘以循环次数
other_loops = Q_l3_factor * S_l2_factor * 3 * 4  # Q × S × C × K
total_h_crossing = h_cross_count * other_loops
print(f"其他循环次数: {Q_l3_factor} × {S_l2_factor} × 3 × 4 = {other_loops}")
print(f"Total H crossing tiles = {h_cross_count} × {other_loops} = {total_h_crossing}")

print("\n" + "=" * 80)
print("ILP 2384 的来源分析")
print("=" * 80)

# 查看 ILP 输出
print(f"\nILP 预测: 2384")
print(f"Trace 统计: 5880")

# 2384 可能的组成
print(f"\n可能的组成分析:")
print(f"  H crossing (ILP tile_h=2): {ilp_crossing} × DRAM_iterations")
print(f"  H crossing (实际 H_per_tile=14): {total_h_crossing}")

# ILP 可能乘以的循环次数
# crossing_acts = 2 × crossing_count × reuse_penalty
# 假设 reuse_penalty 包含了循环次数

# 实际计算
# 从 row_activation.py:
# crossing_acts = 2 * aux = 2 * selected_count * reuse_penalty

# selected_count = crossing_expr = 来自 precompute_input_block_crossing_table
# reuse_penalty 是另一个变量

print(f"\n从 ILP 代码看:")
print(f"  crossing_acts = 2 × crossing_count × reuse_penalty")
print(f"  如果 crossing_count = {ilp_crossing} (用 tile_h=2 计算)")
print(f"  那么 2384 = 2 × {ilp_crossing} × reuse_penalty")
print(f"  reuse_penalty = 2384 / (2 × {ilp_crossing}) = {2384 / (2 * ilp_crossing) if ilp_crossing > 0 else 'N/A'}")

# 或者 ILP 只计算了一次 crossing
print(f"\n或者 ILP 计算方式不同:")
print(f"  2384 = H_crossing × 1 + W_crossing × 1 + base")
print(f"  2384 = 2352 + 32 ???")
print(f"  其中 2352 = H crossing tiles")
print(f"  32 可能是某种 unique row 计算")

# 检查 unique rows
unique_rows = 12  # from trace
print(f"\nunique rows = {unique_rows}")
print(f"2352 + 32 = 2384 ✓")
print(f"32 可能来自 W crossing (336) 的某种处理")
print(f"336 / 10.5 ≈ 32")

# 更可能的解释
print(f"\n更可能的解释:")
print(f"  ILP Input row_acts = H_crossing_unique + W_crossing_unique")
print(f"  H_crossing_unique = 2352")
print(f"  W_crossing_unique = 32 (336 / 某个因子)")
print(f"  Total = 2352 + 32 = 2384")
