#!/usr/bin/env python3
"""理解用户指出的问题：最后一个 tile 取不到完整的 6"""

print("=" * 70)
print("问题的核心：最后一个 tile 取不到完整大小")
print("=" * 70)

print("""
【Tiny Workload 的参数】
  Input H = 6  (只有 6 行)
  tile_h = 6
  step = 4

【滑动窗口访问的理论位置】
  Tile 0: 起始位置 0 → 访问 [0, 6)
  Tile 1: 起始位置 4 → 访问 [4, 10) ← 但 input 只有 6 行！

【实际情况】
  Tile 0: [0, 6) → 读取全部 6 行 ✓
  Tile 1: [4, 6) → 只能读到 2 行（因为 input 边界是 6）
  
  而且 [4, 6) 这 2 行已经被 Tile 0 读过了！
  Tile 1 完全是多余的！
""")

print("=" * 70)
print("Crossing Ratio 模型的错误")
print("=" * 70)

print("""
【模型的假设】

模型计算 crossing ratio 时，假设：
  - 每个 tile 都是完整的 tile_h = 6
  - 有多个这样的 tile 需要访问
  - 计算每个 tile 跨越边界的概率

【但实际上】

1. 第一个 tile [0, 6) 已经覆盖了整个 input
2. 后续的 "tile" 要么：
   - 不存在（因为 output 已经处理完了）
   - 取不到完整大小（因为超出 input 边界）
   - 访问的数据已经被读过（重叠部分）

【问题的本质】

Crossing ratio 计算了一个"无限循环"的滑动窗口：
  Tile 0: [0, 6)
  Tile 1: [4, 10) → crossing！
  Tile 2: [8, 14) → crossing！
  ...
  
但实际上 input 只有 6 行，只有 Tile 0 有意义！
""")

print("=" * 70)
print("更精确的问题描述")
print("=" * 70)

input_h = 6
tile_h = 6
step = 4

print(f"Input H = {input_h}")
print(f"tile_h = {tile_h}")  
print(f"step = {step}")
print()

# 计算实际需要多少个 tile 来覆盖整个 input
num_tiles = 0
current_end = 0
tile_start = 0

print("【实际需要的 tile 访问】")
while current_end < input_h:
    tile_end = min(tile_start + tile_h, input_h)
    actual_tile_size = tile_end - tile_start
    new_data = max(0, tile_end - current_end)
    
    print(f"  Tile {num_tiles}: [{tile_start}, {tile_end}) → 大小 {actual_tile_size}, 新数据 {new_data} 行")
    
    if new_data > 0:
        current_end = tile_end
        num_tiles += 1
    else:
        print(f"    ↳ 这个 tile 没有新数据，不需要！")
        break
    
    tile_start += step
    
    if tile_start >= input_h:
        break

print(f"\n只需要 {num_tiles} 个 tile 就能覆盖整个 input！")

print("""
\n【结论】

对于 tiny workload:
  - 只需要 1 个 tile 就能覆盖全部 input (6 行)
  - 不存在 "第二个 tile"
  - 不存在 crossing 问题
  
Crossing ratio 模型错误地假设有多个完整的 tile，
并计算每个 tile 跨边界的概率，
但实际上根本没有多个 tile！
""")

print("=" * 70)
print("正确的 Row Activation")
print("=" * 70)

print("""
【正确的计算】

Row activation 应该基于：
  1. 实际需要从 DRAM 读取的 unique 数据量
  2. 这些数据分布在多少个 DRAM row 上

对于 tiny workload:
  - unique 数据 = 整个 input = 288 entries = 576 bytes
  - 576 bytes < 1024 bytes (row buffer)
  - 分布在 4 个 bank
  
  row_act = 576 / 4 = 144 bytes/bank = 72 entries/bank
  
  或者按 row activation 次数：
  row_act = 1 次/bank (因为数据能装进一个 row)

【模型的错误计算】
  
  模型计算了 168 entries/bank (包含了不存在的 crossing cost)
  多算了 96 entries/bank！
""")
