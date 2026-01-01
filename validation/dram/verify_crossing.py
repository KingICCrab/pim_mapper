#!/usr/bin/env python3
"""
验证 ILP 模型的 crossing tile 计算是否与理论分析一致
"""
import math
from math import gcd, ceil

# ResNet-L1 参数
block_h, block_w = 31, 31
P_buf, Q_buf = 8, 2
R_buf, S_buf = 7, 1
stride_h, stride_w = 1, 1
dilation_h, dilation_w = 1, 1

# DRAM 循环因子
P_l3, Q_l3, C_l3, K_l3 = 7, 28, 3, 4
S_l2 = 7

# Input 尺寸
input_H, input_W = 62, 62

# 计算 access tile size
H_per_tile = (P_buf - 1) * stride_h + (R_buf - 1) * dilation_h + 1  # = 14
W_per_tile = (Q_buf - 1) * stride_w + (S_buf - 1) * dilation_w + 1  # = 2

print("=" * 80)
print("ILP 模型 Crossing 计算验证")
print("=" * 80)

print(f"\n参数:")
print(f"  block_h = {block_h}, block_w = {block_w}")
print(f"  H_per_tile = {H_per_tile}, W_per_tile = {W_per_tile}")
print(f"  P_l3 = {P_l3}, Q_l3 = {Q_l3}, C_l3 = {C_l3}, K_l3 = {K_l3}")
print(f"  S_l2 = {S_l2}")

# ======================================================================
# 方法 1: 直接枚举 (理论分析)
# ======================================================================
print("\n" + "=" * 80)
print("方法 1: 直接枚举所有 tile")
print("=" * 80)

h_crossing_count = 0
w_crossing_count = 0
total_tiles = 0

for k in range(K_l3):
    for c in range(C_l3):
        for q in range(Q_l3):
            for p in range(P_l3):
                for s in range(S_l2):
                    # 计算 H, W 起始和结束位置
                    p_start = p * P_buf
                    q_start = q * Q_buf + s
                    
                    h_start = p_start * stride_h
                    h_end = h_start + H_per_tile
                    w_start = q_start * stride_w
                    w_end = w_start + W_per_tile
                    
                    # 检查是否跨越边界
                    h_block_start = h_start // block_h
                    h_block_end = (h_end - 1) // block_h
                    w_block_start = w_start // block_w
                    w_block_end = (w_end - 1) // block_w
                    
                    if h_block_start != h_block_end:
                        h_crossing_count += 1
                    if w_block_start != w_block_end:
                        w_crossing_count += 1
                    
                    total_tiles += 1

print(f"  总 tile 数: {total_tiles}")
print(f"  H crossing tiles: {h_crossing_count}")
print(f"  W crossing tiles: {w_crossing_count}")

# ======================================================================
# 方法 2: 使用 ILP 模型的 GCD 公式
# ======================================================================
print("\n" + "=" * 80)
print("方法 2: ILP 模型的 GCD 周期分析")
print("=" * 80)

# H 方向分析
step_h = P_buf * stride_h  # = 8
tile_h = H_per_tile  # = 14

g_h = gcd(step_h, block_h)
period_h = block_h // g_h
safe_zone_h = max(0, block_h - tile_h + 1)
safe_positions_h = ceil(safe_zone_h / g_h) if safe_zone_h > 0 else 0
cross_count_h = max(0, period_h - safe_positions_h)

print(f"\nH 方向:")
print(f"  tile_h = {tile_h}, block_h = {block_h}, step_h = {step_h}")
print(f"  gcd({step_h}, {block_h}) = {g_h}")
print(f"  period = {block_h} / {g_h} = {period_h}")
print(f"  safe_zone = max(0, {block_h} - {tile_h} + 1) = {safe_zone_h}")
print(f"  safe_positions = ceil({safe_zone_h} / {g_h}) = {safe_positions_h}")
print(f"  cross_count_per_period = {period_h} - {safe_positions_h} = {cross_count_h}")
print(f"  crossing_ratio = {cross_count_h}/{period_h} = {cross_count_h / period_h:.4f}")

# 计算总的 H crossing
num_p_tiles = P_l3
h_positions = [p * P_buf * stride_h for p in range(num_p_tiles)]
h_crossings_list = []
for pos in h_positions:
    end = pos + tile_h - 1
    crosses = (pos // block_h) != (end // block_h)
    h_crossings_list.append(crosses)
    
h_crossing_from_period = sum(h_crossings_list)
print(f"  P 方向 {num_p_tiles} 个 tile 的起始位置: {h_positions}")
print(f"  各位置是否 crossing: {h_crossings_list}")
print(f"  H crossing 的 P tile 数: {h_crossing_from_period}")

other_loops = Q_l3 * S_l2 * C_l3 * K_l3
total_h_crossing_from_gcd = h_crossing_from_period * other_loops
print(f"  其他循环次数 = {Q_l3} × {S_l2} × {C_l3} × {K_l3} = {other_loops}")
print(f"  预测 H crossing tiles = {h_crossing_from_period} × {other_loops} = {total_h_crossing_from_gcd}")

# W 方向分析
print(f"\nW 方向:")
step_w = Q_buf * stride_w  # = 2
tile_w = W_per_tile  # = 2

g_w = gcd(step_w, block_w)
period_w = block_w // g_w
safe_zone_w = max(0, block_w - tile_w + 1)
safe_positions_w = ceil(safe_zone_w / g_w) if safe_zone_w > 0 else 0
cross_count_w = max(0, period_w - safe_positions_w)

print(f"  tile_w = {tile_w}, block_w = {block_w}, step_w = {step_w}")
print(f"  gcd({step_w}, {block_w}) = {g_w}")
print(f"  period = {block_w} / {g_w} = {period_w}")
print(f"  safe_zone = max(0, {block_w} - {tile_w} + 1) = {safe_zone_w}")
print(f"  safe_positions = ceil({safe_zone_w} / {g_w}) = {safe_positions_w}")
print(f"  cross_count_per_period = {period_w} - {safe_positions_w} = {cross_count_w}")
print(f"  crossing_ratio = {cross_count_w}/{period_w} = {cross_count_w / period_w:.4f}")

# 计算实际的 W crossing - 考虑 S_l2 的偏移
w_crossing_qs = set()
for q in range(Q_l3):
    for s in range(S_l2):
        w_start = q * Q_buf + s
        w_end = w_start + W_per_tile - 1
        if (w_start // block_w) != (w_end // block_w):
            w_crossing_qs.add((q, s))

print(f"  (Q, S) 组合导致 W crossing 的数量: {len(w_crossing_qs)}")
print(f"  具体组合: {sorted(w_crossing_qs)}")

other_loops_w = P_l3 * C_l3 * K_l3
total_w_crossing_from_gcd = len(w_crossing_qs) * other_loops_w
print(f"  其他循环次数 = {P_l3} × {C_l3} × {K_l3} = {other_loops_w}")
print(f"  预测 W crossing tiles = {len(w_crossing_qs)} × {other_loops_w} = {total_w_crossing_from_gcd}")

# ======================================================================
# 总结对比
# ======================================================================
print("\n" + "=" * 80)
print("总结对比")
print("=" * 80)

print(f"\n  | 指标 | 直接枚举 | GCD 分析 |")
print(f"  |------|----------|----------|")
print(f"  | H crossing tiles | {h_crossing_count} | {total_h_crossing_from_gcd} |")
print(f"  | W crossing tiles | {w_crossing_count} | {total_w_crossing_from_gcd} |")

# ILP 预测的 row activation
print(f"\n根据这些 crossing tiles 估算 row activations:")
print(f"  每个 crossing tile 产生 1 次额外 row switch")
print(f"  H crossing 贡献: {h_crossing_count}")
print(f"  W crossing 贡献: {w_crossing_count}")
print(f"  总 crossing 贡献: {h_crossing_count + w_crossing_count}")
print(f"  (实际可能有重叠，即同时 H 和 W crossing)")

# 检查重叠
both_crossing = 0
for k in range(K_l3):
    for c in range(C_l3):
        for q in range(Q_l3):
            for p in range(P_l3):
                for s in range(S_l2):
                    p_start = p * P_buf
                    q_start = q * Q_buf + s
                    h_start = p_start * stride_h
                    h_end = h_start + H_per_tile
                    w_start = q_start * stride_w
                    w_end = w_start + W_per_tile
                    
                    h_cross = (h_start // block_h) != ((h_end - 1) // block_h)
                    w_cross = (w_start // block_w) != ((w_end - 1) // block_w)
                    
                    if h_cross and w_cross:
                        both_crossing += 1

print(f"\n  同时 H 和 W crossing 的 tiles: {both_crossing}")

# ======================================================================
# ILP 预测值对比
# ======================================================================
print("\n" + "=" * 80)
print("ILP 预测 Input Row Activations 分析")
print("=" * 80)

print(f"\n  ILP 预测 Input row acts = 2384")
print(f"  Trace 统计 Input row acts = 5880")
print(f"\n  理论 crossing tiles:")
print(f"    H crossing: {h_crossing_count}")
print(f"    W crossing: {w_crossing_count}")
print(f"    Both crossing: {both_crossing}")
print(f"    只 H crossing: {h_crossing_count - both_crossing}")
print(f"    只 W crossing: {w_crossing_count - both_crossing}")
print(f"\n  如果每个 crossing tile 贡献 1 次 row act:")
print(f"    H crossing 贡献: {h_crossing_count}")
print(f"    W crossing 贡献: {w_crossing_count}")
print(f"    (重叠部分 {both_crossing} 可能贡献 2 次)")

# 假设基础 row activation
non_crossing_tiles = total_tiles - (h_crossing_count + w_crossing_count - both_crossing)
print(f"\n  Non-crossing tiles: {non_crossing_tiles}")
print(f"  只 crossing 一个方向的 tiles: {h_crossing_count + w_crossing_count - 2*both_crossing}")
print(f"  Both crossing tiles: {both_crossing}")
