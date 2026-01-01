#!/usr/bin/env python3
"""解释为什么 tiny workload 的 crossing ratio 不是 0"""

import math

# Tiny workload 参数
block_h = 6  # 选择的 block height
block_w = 6  # 选择的 block width
tile_h = 6   # Input tile height = (P-1)*stride + (R-1)*dilation + 1 = 3 + 2 + 1 = 6
tile_w = 6   # Input tile width = (Q-1)*stride + (S-1)*dilation + 1 = 3 + 2 + 1 = 6

# 从 mapping_results.txt 中可以看到:
# PE level: P=2, Q=2 (temporal)
# RowBuffer level: R=3 (temporal)
# step_h = Q_factor * stride = 2 * 1 = 2
# step_w = P_factor * stride = 2 * 1 = 2
step_h = 2
step_w = 2

print("=" * 70)
print("为什么 Row-Aligned 模式的 Crossing Ratio = 0.666667？")
print("=" * 70)

print(f"\n【参数】")
print(f"  block_h = {block_h}, tile_h = {tile_h}")
print(f"  block_w = {block_w}, tile_w = {tile_w}")
print(f"  step_h = {step_h}, step_w = {step_w}")

print(f"\n【Crossing Ratio 计算公式】")
print("  对于 Row-Aligned 布局，检查滑动窗口是否跨越 block 边界:")
print("  crossing 发生条件: (k × step) mod block + tile > block")

print(f"\n【H 维度分析】")
g_h = math.gcd(step_h, block_h)
period_h = block_h // g_h
print(f"  g = gcd({step_h}, {block_h}) = {g_h}")
print(f"  period = block_h / g = {block_h} / {g_h} = {period_h}")
print()

crossing_count_h = 0
for k in range(period_h):
    pos_mod = (k * step_h) % block_h
    crosses = pos_mod + tile_h > block_h
    status = "✗ CROSS" if crosses else "✓ no cross"
    print(f"  k={k}: pos = ({k}×{step_h}) mod {block_h} = {pos_mod}")
    print(f"        pos + tile_h = {pos_mod} + {tile_h} = {pos_mod + tile_h}")
    print(f"        {pos_mod + tile_h} > {block_h}? {status}")
    if crosses:
        crossing_count_h += 1
    print()

cr_h = crossing_count_h / period_h
print(f"  Crossing ratio H = {crossing_count_h} / {period_h} = {cr_h:.6f}")

print(f"\n【W 维度分析】(与 H 相同)")
g_w = math.gcd(step_w, block_w)
period_w = block_w // g_w
cr_w = crossing_count_h / period_w  # 参数相同，结果相同
print(f"  Crossing ratio W = {cr_w:.6f}")

print(f"\n【结果】")
print(f"  INPUT_CROSSING_RATIO_H = {cr_h:.6f}")
print(f"  INPUT_CROSSING_RATIO_W = {cr_w:.6f}")
print(f"  TOTAL_INPUT_CR = {cr_h + cr_w:.6f}")

print("\n" + "=" * 70)
print("关键理解：为什么 tile_h == block_h 时 crossing ratio 仍不为 0？")
print("=" * 70)

print("""
【原因】

Crossing ratio 不是看 tile 能否 "装进" block，
而是看滑动窗口访问时是否会跨越 block 边界！

对于 Input 数据的滑动窗口访问模式:
  - tile 0: 访问行 [0, tile_h) = [0, 6)  → 在 block 0 内
  - tile 1: 访问行 [step, step+tile_h) = [2, 8) → 跨越 block 0 和 block 1!
  - tile 2: 访问行 [2×step, 2×step+tile_h) = [4, 10) → 跨越 block 0 和 block 1!

即使 tile_h == block_h == 6:
  - 当 tile 起始位置在 block 内部 (不是 block 开头) 时
  - tile 结尾会超出 block 边界，造成 crossing

可视化 (block_h = 6, tile_h = 6, step = 2):

Block 0: [row 0, 1, 2, 3, 4, 5]
Block 1: [row 6, 7, 8, 9, 10, 11]

tile 0: [0-5] ✓ 全在 Block 0
tile 1: [2-7] ✗ Block 0 的 [2-5] + Block 1 的 [6-7]
tile 2: [4-9] ✗ Block 0 的 [4-5] + Block 1 的 [6-9]

所以 3 个 tile 中有 2 个 crossing → crossing_ratio = 2/3 = 0.666667
""")

print("\n" + "=" * 70)
print("如果要让 crossing ratio = 0，需要什么条件？")
print("=" * 70)

print("""
【条件】: step 能整除 block_h，且 tile_h <= block_h

例如: block_h = 6, step_h = 6 (或 3, 或 2 配合更大的 block)

当 step = block_h 时:
  tile 0: [0, 6) → Block 0
  tile 1: [6, 12) → Block 1  
  tile 2: [12, 18) → Block 2
  
每个 tile 都从 block 开头开始，不会跨越边界！

但当前 tiny workload:
  step_h = 2 (因为 Q_factor=2, stride=1)
  step 不能整除 block_h=6 能整除，但 step < tile_h
  
导致窗口滑动时必然跨越 block 边界。
""")

# 额外验证：如果 step 整除 block 会怎样
print("\n【验证：如果 step = 6 会怎样？】")
step_test = 6
g_test = math.gcd(step_test, block_h)
period_test = block_h // g_test
print(f"  g = gcd({step_test}, {block_h}) = {g_test}")
print(f"  period = {block_h} / {g_test} = {period_test}")

crossing_test = 0
for k in range(period_test):
    pos_mod = (k * step_test) % block_h
    crosses = pos_mod + tile_h > block_h
    print(f"  k={k}: pos = {pos_mod}, pos + tile = {pos_mod + tile_h}, crosses = {crosses}")
    if crosses:
        crossing_test += 1
        
print(f"  Crossing ratio = {crossing_test} / {period_test} = {crossing_test/period_test}")
