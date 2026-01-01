#!/usr/bin/env python3
"""
5880 Row Activations 精确来源分析

验证结果：
- Trace 统计的 Input row switches = 5880 ✓
- 分布：
  - Channel 0: Row 0 (532), Row 1 (532), Row 7 (448), Row 8 (448) = 1960
  - Channel 1: Row 196 (532), Row 197 (532), Row 203 (448), Row 204 (448) = 1960
  - Channel 2: Row 392 (532), Row 393 (532), Row 399 (448), Row 400 (448) = 1960
  - 总计: 1960 × 3 = 5880 ✓
"""

print("=" * 70)
print("5880 Row Activations 来源分析")
print("=" * 70)

# 关键发现：row switches 来自元素级别的访问模式
# 而不是 tile 级别

# 从 trace 输出可以看到：
# - 访问模式在 row 0 和 row 1 之间频繁切换
# - 每次切换大约访问 14 个元素 (7 次访问每个方向)

# 这是因为 tile 跨越了 row 边界!
# tile 访问 H 范围 [h_start, h_start + H_per_tile)
# 如果这个范围跨越了 row 边界，则会产生多次 row switch

# Row 边界分析:
# - Row 0: bytes [0, 1024)
# - Row 1: bytes [1024, 2048)
# - ...

# Block 边界分析:
# - Block (0,0): bytes [0, 1922)  -> rows [0, 1]
# - Block (0,1): bytes [1922, 3844) -> rows [1, 3]
# - Block (1,0): bytes [3844, 5766) -> rows [3, 5]
# - Block (1,1): bytes [5766, 7688) -> rows [5, 7]

# 关键问题：为什么会频繁切换?

# 从前 50 次 switch 的模式看：
# Access 1: row=0 -> 访问第一个元素
# Access 2: row=1 -> 切换到 row 1
# Access 3: row=0 -> 切回 row 0
# ...

# 这说明 tile 内部的访问顺序导致了 row thrashing!
# 而不是 tile 之间的切换

# 让我分析一个具体的 tile 访问

print("\nTile 内部的 row 访问分析:")
print("=" * 70)

# 假设 tile 访问 H 范围 [0, 14), W 范围 [0, 2)
# 使用 block-wise 访问: for h_block: for w_block: for h: for w

# 但从 trace 看，地址是这样的：
# 0x00000000 (h=0, w=0?)
# 0x00000400 (h=8, w=0?) row 1 开始于 offset 512 elements = 1024 bytes

# 让我反推地址计算
row_buffer_bytes = 1024
element_bytes = 2

# 地址 0x00000018 = 24 bytes = 12 elements
# 12 = h * W * C + w * C + c
# 如果 W=62, C=3: 12 = h * 186 + w * 3 + c
# h=0, w=4, c=0: 0*186 + 4*3 + 0 = 12 ✓

# 地址 0x00000400 = 1024 bytes = 512 elements  
# 512 = h * 186 + w * 3 + c
# h=2, w=140/3 不对...

# 也许是 block-wise layout?
# 假设每个 block = 31 × 31 = 961 elements
# Block (0,0) 包含 elements 0-960, bytes 0-1921
# Row 0: bytes 0-1023, elements 0-511
# Row 1: bytes 1024-2047, elements 512-1023

# 但 block 只有 961 个 element, 所以:
# Row 0: block (0,0) 的前 512 elements
# Row 1: block (0,0) 的后 449 elements + block (0,1) 的一部分?

# 这个布局很复杂，让我用 trace 的实际数据分析

print("从 trace 数据分析:")
print("  Row 0: 532 visits (per channel)")
print("  Row 1: 532 visits")
print("  Row 7: 448 visits")
print("  Row 8: 448 visits")
print()
print("  注意: Row 7 = 7 × 1024 = 7168 bytes")
print("  这对应 stride_Q_l3 = 7168!")
print()
print("  所以 Row 7 是 Q tile 1 的起始 row")

# 分析:
# stride_P_l3 = 1024 bytes = 1 row
# stride_Q_l3 = 7168 bytes = 7 rows
# stride_C_l3 = 200704 bytes = 196 rows

# P 方向: 28 个 tile, 每个占 1 row
# Q 方向: 7 个 tile, 每个占 7 rows
# C 方向: 3 个 tile, 每个占 196 rows

print("\n地址空间布局:")
print("  stride_P_l3 = 1024 bytes = 1 row")
print("  stride_Q_l3 = 7168 bytes = 7 rows")
print("  stride_C_l3 = 200704 bytes = 196 rows")
print()
print("  P tiles: 28, 每个占 1 row -> rows 0-27 per Q tile")
print("  Q tiles: 7, 每个占 7 rows -> Q tile 0: rows 0-6, Q tile 1: rows 7-13, ...")
print("  C tiles: 3, 每个占 196 rows -> C0: rows 0-195, C1: rows 196-391, C2: rows 392-587")

print()
print("=" * 70)
print("Row 访问分布分析")
print("=" * 70)

# 每个 channel 的 row 分布:
# Row 0, 1: 532 visits each (属于 Q tile 0, P tiles 0-1)
# Row 7, 8: 448 visits each (属于 Q tile 1, P tiles 0-1)

# 为什么 532 vs 448?

# 假设 DRAM 循环: K -> C -> Q -> P -> S
# 每个 (K, C, Q) 组合, P 循环 28 次, S 循环 7 次
# 28 × 7 = 196 个 tile 访问 per (K, C, Q)

# K=4, C=3, Q=7: total = 4 × 3 × 7 × 196 = 16464 (与 tile 数一致)

# Row 0 的访问:
# - P tile 0 在 row 0 (stride_P_l3 = 1024, P=0 -> offset=0)
# - P tile 1 也在 row 0? 不对, P=1 -> offset=1024 = row 1

# 让我检查 aligned_tile_size = 1024
# 如果每个 P tile 恰好是 1024 bytes (一个 row)
# 那么 P tile 0 在 row 0, P tile 1 在 row 1, ...

# 但为什么 row 0 和 row 1 会频繁切换?

print("\n关键发现: Tile 跨越 row 边界!")
print()
print("  从 trace 前 50 次 switch 可以看到:")
print("  - 在访问单个 tile 时，地址在 row 0 和 row 1 之间切换")
print("  - 这说明 tile 内部的元素跨越了 row 边界")
print()
print("  虽然 stride_P_l3 = 1024 (aligned to row)")
print("  但 tile 内部按 (h, w) 顺序访问元素")
print("  如果 tile 的 H 范围跨越了 block 边界")
print("  就会访问到不同的 row")

# 核心问题: H_per_tile = 14
# 访问 h = [0, 14), 跨越 block_h = 31 吗?
# h=0..13 都在 block 0 (h < 31)
# 不跨越 block!

# 但为什么还是切换 row?

# 答案: 数据布局!
# 数据按 block-wise 存储: [C][H_block][W_block][h_in_block][w_in_block]
# 每个 element 的地址 = block_base + h_in_block * block_w + w_in_block

# 但 trace 访问顺序是按 tile, 而 tile 内按 (h, w) 顺序
# 如果 h 从 0 增加到 13, w 从 0 到 1
# 地址增量 = h * block_w + w = h * 31 + w

# h=0, w=0: offset = 0
# h=0, w=1: offset = 1
# h=1, w=0: offset = 31
# h=1, w=1: offset = 32
# ...

# 元素是连续的 (以 h * block_w + w 计)
# 14 × 2 = 28 个元素, 最大 offset = 13 * 31 + 1 = 404

# 404 elements × 2 bytes = 808 bytes < 1024 (一个 row)
# 应该不会跨越 row!

# 但实际 trace 显示跨越了...让我重新检查

print()
print("=" * 70)
print("重新分析地址计算")
print("=" * 70)

# 从 trace: 
# Access 1: addr=0x00000000, row=0 -> 第一个元素
# Access 2: addr=0x00000400, row=1 -> offset 1024

# 0x400 = 1024 bytes, 正好是 row 1 的起始
# 这说明第一个 tile 的某些元素在 row 1!

# 为什么?
# 让我检查 input_aligned_tile_size = 1024
# 这意味着每个 "aligned tile" 是 1024 bytes
# 但实际 tile 可能更小 (H_per_tile × W_per_tile × C_per_tile × elem_size)

# 从 trace generator 看:
# L3 tile stride = 1024 bytes
# 但 tile 实际大小 = H_per_tile × W_per_tile = 14 × 2 = 28 elements = 56 bytes

# 所以 28 个元素被 pad 到 1024 bytes!
# 但 trace generator 可能在访问时用了不同的地址计算...

# 让我检查地址 0x18 = 24 bytes = 12 elements
# 和地址 0x400 = 1024 bytes

# 如果 tile 内部按 h, w 遍历:
# 地址 = tile_base + (h * W + w) * elem_size?
# 不对, layout 是 block-wise

# 地址 0x18 = 24: 这可能是 h=0, w=12 (if W stride = 2)
# 或 h=12/2=6, w=0

# 让我从 stride 推断:
# L2 strides: stride_P_l2, stride_Q_l2

# 从 debug 输出没看到 L2 strides...

# 关键: row 0 和 row 1 之间的切换说明访问跨越了 1024 byte 边界
# 这可能是因为 H 方向的访问跨越了多个 "P tiles"

# 让我重新理解:
# P_l3 = 28 表示 P 方向有 28 个 DRAM tiles
# P 范围 = 56, 所以每个 P tile 包含 56/28 = 2 个 P 值

# 同理 Q_l3 = 7, Q 范围 = 56, 每个 Q tile 包含 8 个 Q 值

# 所以实际的 tile 大小:
# P_per_tile = 2 (P 方向)
# Q_per_tile = 8 (Q 方向)

# 但 Input 不是按 P, Q 索引的，是按 H, W!
# H_per_tile = P_per_tile + R - 1 = 2 + 7 - 1 = 8
# W_per_tile = Q_per_tile + S - 1 = 8 + 7 - 1 = 14

# 等等，这和 verify_crossing.py 的 H_per_tile=14, W_per_tile=2 不同!

print("Tile 尺寸计算:")
print("  P_l3=28, Q_l3=7, R=7, S=7, P=Q=56")
print()
print("  P_per_tile = P / P_l3 = 56 / 28 = 2")
print("  Q_per_tile = Q / Q_l3 = 56 / 7 = 8")
print()
print("  H_per_tile = P_per_tile + R - 1 = 2 + 7 - 1 = 8")
print("  W_per_tile = Q_per_tile + S - 1 = 8 + 7 - 1 = 14")
print()
print("  实际 tile: 8 × 14 = 112 elements = 224 bytes")

# 这与 input_aligned_tile_size = 1024 不同!
# 224 bytes 被 pad 到 1024 bytes

# 但 trace 访问 224 个元素需要地址范围 0 到 224 bytes
# 不应该跨越 1024 byte 边界...

# 除非 L2 层也有循环!

# 从 loop bounds:
# Level 2 temporal: R=7
# 这意味着 R 方向在 Level 2 循环 7 次!

# 所以在 Level 2 内:
# 访问 H 范围 = P_per_tile + (r_l2) × dilation = 2 + r × 1 (r = 0..6)
# 当 r=0: h=[0, 2)
# 当 r=1: h=[1, 3)
# ...
# 当 r=6: h=[6, 8)

# 综合: h 范围 [0, 8) 与上面计算一致

# 关键问题: Level 2 的每次迭代是否独立访问 DRAM?
# 如果是, 那么 S_l2=7 意味着 7 次独立的 DRAM 访问
# 每次访问 P_per_tile × Q_per_tile + S_per_iter = 2 × 8 = 16 elements? 不对

print()
print("=" * 70)
print("L2 循环分析")  
print("=" * 70)

# 从 loop bounds:
# Level 1 temporal: S=7 (buffer level)
# Level 2 temporal: R=7

# 这意味着:
# Level 2: R 循环 7 次
# Level 1: S 循环 7 次

# 每次 L2 迭代:
# - 固定一个 r 值
# - L1 内 S 循环 7 次
# - 访问 H = [p_start + r, p_start + r + P_per_tile], W = [q_start, q_start + Q_per_tile + S - 1]

# 实际访问范围:
# 单次 L2 迭代: H = 2, W = 8 + 6 = 14
# 7 次 L2 迭代覆盖: H = 2 + 6 = 8, W = 14

# 但每次 L2 迭代会访问不同的 H 位置!
# r=0: h=[p_start, p_start+2)
# r=1: h=[p_start+1, p_start+3)
# ...

# 地址计算: addr = tile_base + h * stride_h + w * stride_w

# 如果 stride_h 跨越 row 边界...

# 让我检查实际的 stride

# 从 trace:
# 0x00000000 -> 0x00000400 的跳跃
# 0x400 = 1024, 这是一整个 row!

# 可能的原因:
# 1. stride_h = 1024 bytes (每 h 增加一个 row)
# 2. 某种对齐导致了 padding

# 让我检查 Input 的实际地址公式

print("地址公式分析:")
print("  addr = tile_base + offset")
print()
print("  对于 row_aligned layout:")
print("  tile_base = (p_l3 * stride_p_l3 + q_l3 * stride_q_l3 + c_l3 * stride_c_l3)")
print("  offset = (h_in_block * stride_h + w_in_block * stride_w)")
print()
print("  stride_p_l3 = 1024")
print("  stride_q_l3 = 7168 = 7 * 1024")
print()
print("  这意味着每个 P tile 占 1024 bytes (一个 row)")
print("  但 P_per_tile = 2, 实际只用 2 * W_stride bytes")
print()
print("  关键: h_in_block 的 stride 是什么?")

# 从 trace 数据推断:
# addr 0x00 -> h=0
# addr 0x18 = 24 -> 可能是 w 方向的某个位置
# addr 0x400 = 1024 -> 下一个 row, 可能是 h 增加后的位置

# 如果 stride_h = 1024, 那么:
# h=0: addr = 0
# h=1: addr = 1024 = 0x400

# 但这不对, H_per_tile = 8 意味着需要 8 * 1024 = 8 rows 每个 tile
# 与 stride_p_l3 = 1024 矛盾

# 更可能的解释:
# P tile 内部的 h 使用 L2 stride (未对齐)
# P tile 之间使用 L3 stride (对齐到 1024)

# 但 P tile 只有 P_per_tile = 2 个 P 值
# 对应 H = P_per_tile + R - 1 = 2 + 6 = 8? (如果 R_buf = 7, R_l2 = 1)

# 我需要重新理解 level 结构...

print()
print("关键发现:")
print("  trace 显示元素访问在 row 0 和 row 1 之间频繁切换")
print("  这说明 tile 内部的 H 方向访问跨越了 row 边界")
print()
print("  可能原因:")
print("  1. L2 stride 跨越了 row (stride_h_l2 >= 512 elements)")
print("  2. block 布局导致 H 方向不连续")
print()
print("  验证: 532 + 532 + 448 + 448 = 1960 per channel")
print("  1960 / 4 rows = 490 per row (平均)")
print("  但实际是 532 (rows 0,1) vs 448 (rows 7,8)")
print()  
print("  532 - 448 = 84 差异")
print("  可能与 crossing tiles 数量相关")
