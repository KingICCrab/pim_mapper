#!/usr/bin/env python3
"""解释为什么 tile_h == block_h 时 crossing ratio 不为 0"""

print("=" * 70)
print("tile_h == block_h 时为什么还有 crossing？")
print("=" * 70)

print("""
【你的直觉】

如果 tile_h == block_h == 6:
  - 一个 tile 正好是一个 block 的大小
  - 应该只需要 1 次 row activation
  - crossing ratio 应该是 0

【为什么直觉不对？】

crossing ratio 计算的不是 "tile 能否装进 block"，
而是 "滑动窗口访问时，tile 是否跨越 block 边界"。

关键在于：tile 的起始位置不一定对齐到 block 边界！
""")

print("=" * 70)
print("滑动窗口访问模式")
print("=" * 70)

block_h = 6
tile_h = 6
step = 4  # q_factor × stride = 4 × 1

print(f"""
【参数】
  block_h = {block_h}
  tile_h = {tile_h}  
  step = {step} (滑动步长 = Q_factor × stride)

【滑动窗口访问序列】

在 H 方向，相邻 output tile 对应的 input tile 起始位置相差 step = {step}

  output tile 0 → input tile 从位置 0 开始
  output tile 1 → input tile 从位置 {step} 开始  
  output tile 2 → input tile 从位置 {2*step} 开始
  ...
""")

print("【可视化】")
print()
print("内存布局 (每行是一个 block):")
print("-" * 40)

num_blocks = 4
for b in range(num_blocks):
    start = b * block_h
    end = start + block_h
    print(f"Block {b}: positions [{start:2d}, {end:2d})")

print("-" * 40)
print()

print("滑动窗口访问:")
print()
for i in range(4):
    start = i * step
    end = start + tile_h
    
    # 判断是否 crossing
    start_block = start // block_h
    end_block = (end - 1) // block_h  # -1 因为是半开区间
    
    if start_block == end_block:
        status = "✓ 在单个 block 内"
        blocks = f"Block {start_block}"
    else:
        status = "✗ CROSSING！"
        blocks = f"Block {start_block} + Block {end_block}"
    
    print(f"  Tile {i}: positions [{start:2d}, {end:2d}) → {blocks} {status}")

print()

print("=" * 70)
print("核心问题")
print("=" * 70)

print(f"""
【问题的本质】

即使 tile_h == block_h == {block_h}:

  Tile 0: [{0}, {tile_h}) → 正好在 Block 0 内 ✓
  Tile 1: [{step}, {step+tile_h}) → 起始在 Block 0，结尾在 Block 1 ✗
  
因为 step = {step} 不等于 block_h = {block_h}，
tile 的起始位置不是对齐到 block 边界的！

当 tile 起始位置在 block 内部（不是开头）时，
tile 的结尾就会超出 block 边界。
""")

print("=" * 70)
print("什么条件下 crossing = 0？")
print("=" * 70)

print("""
【条件】

只有当 tile 的起始位置总是对齐到 block 边界时，才不会 crossing。

这要求：
  step mod block_h == 0  
  即 step 是 block_h 的倍数

或者另一种情况：
  所有 tile 都从 block 开头开始（比如没有滑动窗口）
""")

print("\n【验证】")
print()

import math

for step_test in [1, 2, 3, 4, 6, 12]:
    g = math.gcd(step_test, block_h)
    period = block_h // g
    crossing_count = 0
    
    for k in range(period):
        pos_mod = (k * step_test) % block_h
        if pos_mod + tile_h > block_h:
            crossing_count += 1
    
    cr = crossing_count / period if period > 0 else 0
    
    is_multiple = step_test % block_h == 0
    note = "← step 是 block 的倍数，所以 cr = 0" if is_multiple else ""
    
    print(f"step = {step_test:2d}: crossing_ratio = {cr:.4f} {note}")

print("""
\n【结论】

tile_h == block_h 只能保证 "tile 大小等于 block 大小"，
但不能保证 "tile 对齐到 block 边界"。

只有当 step 是 block_h 的倍数时，
每个 tile 的起始位置才会对齐到 block 边界，
此时 crossing_ratio = 0。

对于 tiny workload:
  step = 4, block_h = 6
  4 不是 6 的倍数
  所以即使 tile_h == block_h == 6，
  仍然有 crossing_ratio = 0.667
""")
