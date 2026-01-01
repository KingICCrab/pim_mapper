#!/usr/bin/env python3
"""
详细分析 INPUT_ROW_ACT_SEQ = 1.559589 的计算过程
"""
import numpy as np

print("=" * 70)
print("INPUT_ROW_ACT_SEQ = 1.559589 计算分解")
print("=" * 70)

# =====================
# 从 mapping_results.txt 的输出
# =====================
nc = 0.110556        # ROW_ACT_INPUT_NC
cr = 1.449032        # ROW_ACT_INPUT_CR
total = 1.559589     # INPUT_ROW_ACT_SEQ

print(f"\n【已知输出】")
print(f"  ROW_ACT_INPUT_NC  = {nc}")
print(f"  ROW_ACT_INPUT_CR  = {cr}")
print(f"  INPUT_ROW_ACT_SEQ = {total}")
print(f"  验证: NC + CR = {nc + cr:.6f}")

# =====================
# 从代码中的公式
# =====================
print(f"\n【代码中的公式】")
print("""
  NC = rb_tiles × unique_rows_factor × non_crossing_coeff / banks
  CR = crossing_base × unique_rows_factor × crossing_coeff / banks
  
  其中:
    rb_tiles = RB_TILES_INPUT (归一化后)
    crossing_base = CROSSING_INPUT_BASE = rb_tiles × reuse (归一化后)
    unique_rows_factor = 1 / tiles_per_row
    non_crossing_coeff = (1 - cr_b) × (1 - cr_r)
    crossing_coeff = 2 × (cr_b + cr_r)
    banks = 4
""")

# =====================
# 已知的参数
# =====================
macs = 9216
macs_scale_factor = 1e4 / (1.02 * macs)  # = 1.063794

rb_tiles_actual = 288.0  # entries (实际值)
rb_tiles_norm = rb_tiles_actual * macs_scale_factor  # 归一化值

reuse = 1.0  # REUSE_INPUT = 1.0
crossing_base_norm = rb_tiles_norm * reuse  # = rb_tiles_norm (因为 reuse=1)

banks = 4

print(f"\n【已知参数】")
print(f"  rb_tiles (实际)  = {rb_tiles_actual}")
print(f"  rb_tiles (归一化) = {rb_tiles_norm:.6f}")
print(f"  reuse = {reuse}")
print(f"  banks = {banks}")

# =====================
# 需要确定的参数: tiles_per_row, cr_b, cr_r
# =====================

# 从代码中，tiles_per_row = row_buffer_size_bytes / avg_tile_bytes
# row_buffer_size = 1024 bytes
row_buffer_size = 1024

# avg_tile_bytes 是所有可能 tile 大小的平均值
# 对于 tiny workload，tile 维度取决于各层的 tiling

# 从 mapping:
# Input tile 在 DRAM level 是完整的 Input = H × W × C = 6 × 6 × 8 = 288
# 但 avg_tile_bytes 是预计算的平均值，不是实际选择的值

# 让我们从公式反推 tiles_per_row
# NC = rb_tiles_norm × (1/tiles_per_row) × non_crossing_coeff / banks
# 假设 rb_tiles_norm ≈ 306.37 (已验证)

print(f"\n【反推 tiles_per_row 和 crossing ratios】")

# 设 x = unique_rows_factor = 1/tiles_per_row
# 设 a = non_crossing_coeff = (1-cr_b)(1-cr_r)
# 设 b = crossing_coeff = 2(cr_b + cr_r)
#
# NC = rb_tiles_norm × x × a / banks
# CR = crossing_base_norm × x × b / banks (crossing_base_norm = rb_tiles_norm × reuse)
#
# 由于 reuse = 1:
# NC / CR = a / b = (1-cr_b)(1-cr_r) / [2(cr_b + cr_r)]

ratio_nc_cr = nc / cr
print(f"  NC / CR = {ratio_nc_cr:.6f}")

# 从代码中，cr_b 和 cr_r 是预计算的平均值
# 让我看看代码怎么算的

print(f"\n【从代码中查找 cr_b 和 cr_r 的计算】")
print("""
代码计算 avg_block_crossing_ratio 和 avg_row_crossing_ratio:

1. avg_row_crossing_ratio:
   - 遍历所有 (block_h, tile_h, block_w, tile_w) 组合
   - 计算 tile_bytes = tile_h × tile_w × element_bytes
   - cr = precise_crossing_ratio(tile_bytes, row_buffer_size)
   - 取所有组合的平均值

2. avg_block_crossing_ratio:
   - 遍历所有 (block_h, tile_h) 和 (block_w, tile_w) 组合
   - 计算 compute_input_crossing_ratio(...)
   - 取 H 方向和 W 方向的平均值
   - 合并: 1 - (1-cr_h)(1-cr_w)
""")

# 简化分析：假设 cr_b = cr_r = x (对称情况)
# NC/CR = (1-x)^2 / (4x) = ratio_nc_cr
# (1-x)^2 = 4x × ratio_nc_cr
# 设 r = ratio_nc_cr = 0.0763
# x^2 - 2x + 1 = 4rx
# x^2 - (2 + 4r)x + 1 = 0

r = ratio_nc_cr
a_coef = 1
b_coef = -(2 + 4*r)
c_coef = 1
discriminant = b_coef**2 - 4*a_coef*c_coef
x1 = (-b_coef + np.sqrt(discriminant)) / (2*a_coef)
x2 = (-b_coef - np.sqrt(discriminant)) / (2*a_coef)

print(f"\n【假设 cr_b = cr_r = x，解方程】")
print(f"  (1-x)² / (4x) = {r:.4f}")
print(f"  x² - {2+4*r:.4f}x + 1 = 0")
print(f"  解得 x = {x2:.4f} 或 x = {x1:.4f}")
print(f"  取合理值 x ≈ {x2:.4f}")

cr_b = cr_r = x2
non_crossing_coeff = (1 - cr_b) * (1 - cr_r)
crossing_coeff = 2 * (cr_b + cr_r)

print(f"\n【推算得到的系数】")
print(f"  cr_b = cr_r = {cr_b:.4f}")
print(f"  non_crossing_coeff = (1-{cr_b:.4f})² = {non_crossing_coeff:.4f}")
print(f"  crossing_coeff = 2 × 2 × {cr_b:.4f} = {crossing_coeff:.4f}")

# 现在反推 tiles_per_row
# NC = rb_tiles_norm × (1/tiles_per_row) × non_crossing_coeff / banks
# 0.110556 = 306.37 × (1/tiles_per_row) × 0.177 / 4
# tiles_per_row = 306.37 × 0.177 / (4 × 0.110556)

tiles_per_row = rb_tiles_norm * non_crossing_coeff / (banks * nc)
unique_rows_factor = 1 / tiles_per_row

print(f"\n【反推 tiles_per_row】")
print(f"  tiles_per_row = {tiles_per_row:.2f}")
print(f"  unique_rows_factor = 1/{tiles_per_row:.2f} = {unique_rows_factor:.6f}")

# 验证
nc_calc = rb_tiles_norm * unique_rows_factor * non_crossing_coeff / banks
cr_calc = rb_tiles_norm * reuse * unique_rows_factor * crossing_coeff / banks
total_calc = nc_calc + cr_calc

print(f"\n【验证计算】")
print(f"  NC (计算) = {rb_tiles_norm:.2f} × {unique_rows_factor:.6f} × {non_crossing_coeff:.4f} / {banks}")
print(f"           = {nc_calc:.6f} (期望 {nc})")
print(f"  CR (计算) = {rb_tiles_norm:.2f} × {reuse} × {unique_rows_factor:.6f} × {crossing_coeff:.4f} / {banks}")
print(f"           = {cr_calc:.6f} (期望 {cr})")
print(f"  Total    = {total_calc:.6f} (期望 {total})")

# =====================
# 完整公式展开
# =====================
print(f"\n" + "=" * 70)
print("完整计算公式")
print("=" * 70)
print(f"""
INPUT_ROW_ACT_SEQ = NC + CR

NC = rb_tiles × (1/tiles_per_row) × (1-cr_b)(1-cr_r) / banks
   = {rb_tiles_norm:.2f} × (1/{tiles_per_row:.2f}) × {non_crossing_coeff:.4f} / {banks}
   = {nc_calc:.6f}

CR = rb_tiles × reuse × (1/tiles_per_row) × 2(cr_b+cr_r) / banks
   = {rb_tiles_norm:.2f} × {reuse} × (1/{tiles_per_row:.2f}) × {crossing_coeff:.4f} / {banks}
   = {cr_calc:.6f}

Total = {nc_calc:.6f} + {cr_calc:.6f} = {total_calc:.6f}
""")

# =====================
# 问题分析
# =====================
print("=" * 70)
print("问题分析")
print("=" * 70)
print(f"""
关键问题: cr_b 和 cr_r 的值为什么这么高 (~0.58)?

因为代码使用了 **所有可能组合的平均值**，而不是实际选择的值。

对于 tiny workload:
- 实际选择了 block_h = 6, tile_h = 6
- 当 tile_h = block_h 时，block crossing ratio 应该较低
- 但代码遍历 block_h ∈ [1,2,3,6]，tile_h = 6
- block_h < tile_h 时 crossing = 1.0，拉高了平均值

而且整个 Input 只从 DRAM 读一次 (DRAM tiling 全为 1):
- 实际只需要 ceil(288/1024) = 1 次 row activation
- 分配到 4 个 bank = 0.25 row_act per bank
- 但公式算出 1.56!
""")
