#!/usr/bin/env python3
"""详细解释 crossing_ratio 的计算公式"""

print("=" * 70)
print("Crossing Ratio 公式解释")
print("=" * 70)

print("""
【公式】

g = gcd(step, block_h)
period = block_h / g
crossing_count = 0
for k in range(period):
    pos_mod = (k * step) % block_h
    if pos_mod + tile_h > block_h:
        crossing_count += 1
crossing_ratio = crossing_count / period
""")

print("=" * 70)
print("1. 为什么用周期 (period)？")
print("=" * 70)

print("""
【滑动窗口的起始位置序列】

tile 0: pos = 0
tile 1: pos = step
tile 2: pos = 2 * step
tile 3: pos = 3 * step
...

【对 block_h 取模后的序列】

pos_mod = (k * step) % block_h

这个序列是周期性的！
周期 = block_h / gcd(step, block_h)

【为什么？】

根据数论：
- 序列 {0, step, 2*step, ...} mod block_h 的周期是 block_h / gcd(step, block_h)
- 因为 step 和 block_h 的最小公倍数 lcm = block_h * step / gcd(step, block_h)
- 需要 lcm / step = block_h / gcd 步才能回到起点
""")

print("=" * 70)
print("2. 具体例子：step=4, block_h=6")
print("=" * 70)

import math

step = 4
block_h = 6
tile_h = 6

g = math.gcd(step, block_h)
period = block_h // g

print(f"step = {step}")
print(f"block_h = {block_h}")
print(f"gcd = {g}")
print(f"period = {block_h} / {g} = {period}")
print()

print("【位置序列 (对 block_h 取模)】")
print()
for k in range(period * 2):  # 显示两个周期
    pos = k * step
    pos_mod = pos % block_h
    cycle_marker = "← 周期开始" if k % period == 0 else ""
    print(f"  k={k}: pos={pos:2d}, pos mod {block_h} = {pos_mod} {cycle_marker}")

print(f"""
\n可以看到：位置序列 mod {block_h} 是 [0, 4, 2, 0, 4, 2, ...]
周期 = {period}
""")

print("=" * 70)
print("3. 判断 crossing")
print("=" * 70)

print(f"""
【Crossing 条件】

一个 tile 从位置 pos_mod 开始，大小为 tile_h
如果 pos_mod + tile_h > block_h，则跨越了 block 边界

例如 tile_h = {tile_h}, block_h = {block_h}:
""")

crossing_count = 0
for k in range(period):
    pos_mod = (k * step) % block_h
    crosses = pos_mod + tile_h > block_h
    status = "CROSSING!" if crosses else "no cross"
    print(f"  k={k}: pos_mod={pos_mod}, pos_mod + tile_h = {pos_mod} + {tile_h} = {pos_mod + tile_h}")
    print(f"        {pos_mod + tile_h} > {block_h}? → {status}")
    if crosses:
        crossing_count += 1
    print()

crossing_ratio = crossing_count / period
print(f"crossing_count = {crossing_count}")
print(f"period = {period}")
print(f"crossing_ratio = {crossing_count} / {period} = {crossing_ratio:.4f}")

print("=" * 70)
print("4. 假设：无限周期性访问")
print("=" * 70)

print(f"""
【公式的核心假设】

这个公式假设：
1. 有无限多个 tile 需要访问
2. tile 大小固定为 tile_h
3. 访问模式是周期性的

然后计算：在一个周期内，有多少比例的 tile 会 crossing

【问题】

这个假设不考虑：
1. 实际有多少个 tile（边界效应）
2. 最后一个 tile 可能取不到完整的 tile_h
3. 数据的实际范围（input_h）

对于 tiny workload:
- input_h = 6
- tile_h = 6
- 实际只需要 1 个 tile 就能覆盖全部数据
- 但公式假设有无限多个周期性访问
""")

print("=" * 70)
print("5. 可视化")
print("=" * 70)

print("""
Block 布局 (每个 block 大小 = 6):

    Block 0         Block 1         Block 2
[0, 1, 2, 3, 4, 5][6, 7, 8, 9,10,11][12,13,14,15,16,17]

滑动窗口访问 (step = 4, tile_h = 6):

Tile 0: [0-5]       ████████████████
        pos_mod = 0, 不 crossing

Tile 1: [4-9]             ████████████████
        pos_mod = 4, 4+6=10 > 6, CROSSING!

Tile 2: [8-13]                  ████████████████  
        pos_mod = 2, 2+6=8 > 6, CROSSING!

Tile 3: [12-17]                       ████████████████
        pos_mod = 0, 不 crossing (周期重复)

周期 = 3, 其中 2 个 crossing
crossing_ratio = 2/3 = 0.667
""")
