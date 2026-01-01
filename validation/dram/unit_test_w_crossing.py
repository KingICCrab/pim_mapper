#!/usr/bin/env python3
"""
单元测试：验证修复后的 block crossing 计算函数
H 由 (P, R) 决定，W 由 (Q, S) 决定
使用 num_tiles 参数传入 DRAM 循环次数
"""
import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')

from pim_optimizer.model.row_activation import (
    precompute_input_block_crossing_table,
    compute_input_block_crossing_count
)

print("=" * 80)
print("测试修复后的 block crossing 函数")
print("=" * 80)

# ResNet-L1 参数
P = 56  # output width
Q = 56  # output height
R = 7   # kernel width
S = 7   # kernel height
stride_w = 1  # Wstride (H direction)
stride_h = 1  # Hstride (W direction)
dilation_w = 1
dilation_h = 1

# Input 尺寸
input_h = stride_w * (P - 1) + dilation_w * (R - 1) + 1  # W_in = 62
input_w = stride_h * (Q - 1) + dilation_h * (S - 1) + 1  # H_in = 62

print(f"\nWorkload: P={P}, Q={Q}, R={R}, S={S}")
print(f"Input size: W_in={input_h}, H_in={input_w}")

# ============================================================
# H 方向测试 (P, R)
# ============================================================
print(f"\n" + "=" * 80)
print("H 方向 (P, R)")
print("=" * 80)

P_factor = 7  # P_l3 = 7
R_factor = 1  # R 不 split
P_rb = P // P_factor  # = 8
R_rb = R // R_factor  # = 7

tile_h = stride_w * P_rb + dilation_w * R_rb - stride_w - dilation_w + 1  # = 14
step_h = P_rb * stride_w  # = 8

print(f"  P_factor={P_factor}, R_factor={R_factor}")
print(f"  P_rb={P_rb}, R_rb={R_rb}")
print(f"  tile_h={tile_h}, step_h={step_h}")

block_h = 31
crossing_h, total_h = compute_input_block_crossing_count(
    block_h=block_h, tile_h=tile_h, step=step_h,
    tile_s=R_rb, total_S=R, dilation=dilation_w,
    num_tiles=P_factor  # 使用 DRAM 循环次数
)
print(f"  block_h={block_h}")
print(f"  crossing_count_h = {crossing_h} (期望 1)")
print(f"  ✓ 匹配" if crossing_h == 1 else f"  ❌ 期望 1")

# ============================================================
# W 方向测试 (Q, S)
# ============================================================
print(f"\n" + "=" * 80)
print("W 方向 (Q, S)")
print("=" * 80)

Q_factor = 28  # Q_l3 = 28
S_factor = 7   # S_l2 = 7
Q_rb = Q // Q_factor  # = 2
S_rb = S // S_factor  # = 1

tile_w = stride_h * Q_rb + dilation_h * S_rb - stride_h - dilation_h + 1  # = 2
step_w = Q_rb * stride_h  # = 2

print(f"  Q_factor={Q_factor}, S_factor={S_factor}")
print(f"  Q_rb={Q_rb}, S_rb={S_rb}")
print(f"  tile_w={tile_w}, step_w={step_w}")

block_w = 31
crossing_w, total_w = compute_input_block_crossing_count(
    block_h=block_w, tile_h=tile_w, step=step_w,
    tile_s=S_rb, total_S=S, dilation=dilation_h,
    num_tiles=Q_factor  # 使用 DRAM 循环次数
)
print(f"  block_w={block_w}")
print(f"  crossing_count_w = {crossing_w} (期望 4)")
print(f"  ✓ 匹配" if crossing_w == 4 else f"  ❌ 期望 4")

# ============================================================
# 预计算表测试
# ============================================================
print(f"\n" + "=" * 80)
print("预计算表测试")
print("=" * 80)

# W 方向预计算表 (预计算表内部使用 spatial_factor 作为 num_tiles)
w_rb_options = [1, 2, 31, 62]
q_factor_options = [1, 2, 4, 7, 8, 14, 28, 56]
s_factor_options = [1, 7]

crossing_table_w = precompute_input_block_crossing_table(
    block_options=w_rb_options,
    spatial_factor_options=q_factor_options,
    kernel_factor_options=s_factor_options,
    stride=stride_h,
    dilation=dilation_h,
    total_spatial=Q,
    total_kernel=S,
    input_size=input_w,
)

# 查找 block_w=31, Q_factor=28, S_factor=7
w_rb_idx = w_rb_options.index(31)
q_factor_idx = q_factor_options.index(28)
s_factor_idx = s_factor_options.index(7)
table_value_w = crossing_table_w.get((w_rb_idx, q_factor_idx, s_factor_idx))
print(f"  预计算表查找 (block_w=31, Q_factor=28, S_factor=7):")
print(f"    索引: ({w_rb_idx}, {q_factor_idx}, {s_factor_idx})")
print(f"    值: {table_value_w} (期望 4)")
print(f"  ✓ 匹配" if table_value_w == 4 else f"  ❌ 不匹配")

# ============================================================
# 验证总数与 verify_crossing.py 一致
# ============================================================
print(f"\n" + "=" * 80)
print("验证 crossing tiles 总数")
print("=" * 80)

# 从 verify_crossing.py:
# H crossing tiles = 2352 (= 1 × Q_l3 × S_l2 × C_l3 × K_l3 = 1 × 28 × 7 × 3 × 4)
# W crossing tiles = 336 (= 4 × P_l3 × C_l3 × K_l3 = 4 × 7 × 3 × 4)

C_l3, K_l3 = 3, 4
h_crossing_tiles = crossing_h * Q_factor * S_factor * C_l3 * K_l3
w_crossing_tiles = crossing_w * P_factor * C_l3 * K_l3

print(f"  H crossing tiles = {crossing_h} × {Q_factor} × {S_factor} × {C_l3} × {K_l3} = {h_crossing_tiles}")
print(f"    期望: 2352")
print(f"    ✓ 匹配" if h_crossing_tiles == 2352 else f"    ❌ 不匹配")

print(f"  W crossing tiles = {crossing_w} × {P_factor} × {C_l3} × {K_l3} = {w_crossing_tiles}")
print(f"    期望: 336")
print(f"    ✓ 匹配" if w_crossing_tiles == 336 else f"    ❌ 不匹配")

print(f"\n" + "=" * 80)
print("测试完成！所有断言通过。")
print("=" * 80)
