#!/usr/bin/env python3
"""
重新理解 5880 的来源:
- verify_crossing.py 计算的是 "crossing tiles" 的数量
- Trace generator 的 row activation 是什么意思?

关键区别:
- 一个 DRAM tile = P_l3 × Q_l3 × C_l3 × K_l3 个元素
- 每个 tile 访问时，只要 row 变化了就计 1 次 row activation
- 16464 个 tiles，每个最多产生 1 次 row activation?

让我重新分析 verify_crossing.py 的输出:
- H crossing tiles: 2352
- W crossing tiles: 336  
- Both crossing: 48
- Non-crossing: 13824
- Total: 16464 tiles

但 5880 不等于这些数字...让我回顾 Trace 的真正含义。
"""

print("=" * 70)
print("分析 5880 的真正来源")
print("=" * 70)

# 从 Trace 日志可知：
# - Row 0: 532 visits per channel
# - Row 1: 532 visits per channel  
# - Row 7: 448 visits per channel
# - Row 8: 448 visits per channel

# 每个 channel 的 row visits:
per_channel = 532 + 532 + 448 + 448  # = 1960
total_3ch = per_channel * 3  # = 5880

print(f"每个 channel 的 row visits: {per_channel}")
print(f"3 个 channel 总计: {total_3ch}")
print()

# 关键问题: 为什么每个 row 是 532 或 448 次访问?

# 分析:
# - Block h=0 包含 rows 0-30, block h=1 包含 rows 31-61
# - Row 0 是 block h=0 的第一个 row
# - Row 31 是 block h=1 的第一个 row

# 在 DRAM 中，每个 row = 1024 bytes
# Input layout: (H, W, C) = (62, 62, 3)
# 一个 row 包含多少 input?
row_bytes = 1024
element_bytes = 2  # FP16
elements_per_row = row_bytes // element_bytes  # = 512 elements

# Input 大小: 62 × 62 × 3 = 11532 elements = 23064 bytes
# 需要 23064 / 1024 = 22.5 → 23 rows

# Row 地址分布:
# Row 0: offset 0-511 (元素 0-255) -> input[0:4][0:62][0:3] 约前 4.x 行
# Row 1: offset 512-1023 -> 接下来 ~4 行
# ...

print("Input layout 分析:")
print(f"  Elements per row: {elements_per_row}")
print(f"  Input shape: (62, 62, 3)")
print(f"  Elements per H row: 62 * 3 = {62 * 3}")
print(f"  H rows per DRAM row: {elements_per_row / 186:.2f}")
print()

# 每个 DRAM row 包含约 2.75 行 input (512/186 = 2.75)
# 所以:
# - Row 0: H=[0, 2], partial H=3
# - Row 1: H=[3, 5], partial 
# ...

# 那 "Row visits" 是什么意思?
# 回顾 Trace 的分析:
# - 它统计的是 "row switch" 次数
# - 当访问一个新的 DRAM row 时，计数 +1

# 关键: 在 verify_crossing.py 中，统计的是 "block switch"
# 在 Trace 中，统计的是 "row switch"
# 一个 block 可能包含多个 rows!

print("Block vs Row 分析:")
print(f"  Block H size: 31 elements")
print(f"  Block W size: 31 elements") 
print(f"  One block covers: 31 × 31 × 3 = {31*31*3} elements = {31*31*3*2} bytes")
print(f"  One block spans: {31*31*3*2/1024:.1f} DRAM rows")
print()

# 等等，我误解了 block_h, block_w 的定义
# 让我重新理解:
# block_h = 31 表示 一个 DRAM row 在 H 维度上跨越 31 个 input pixels
# block_w = 31 表示 ... 在 W 维度

# 检查: row_buffer = 1024 bytes = 512 FP16 elements
# Input shape (H, W, C) with C innermost
# 一个 row 包含连续的 512 个元素

# 如果 W × C = 62 × 3 = 186 个元素是一个 H slice
# 512 / 186 = 2.75 个 H slices per row
# 不对，这不能解释 block_h = 31

# 让我从 verify_crossing.py 理解 block 的真正含义
print("=" * 70)
print("从 row_buffer_bytes 推导 block_h, block_w")
print("=" * 70)

row_buffer_bytes = 1024
bytes_per_element = 2
input_C = 3
input_H = 62
input_W = 62

# row_buffer_elements = 1024 / 2 = 512
# block_h × block_w × C = 512?
# 31 × 31 × 3 = 2883 ≠ 512

# 不对。让我读 verify_crossing.py 的定义
print(f"row_buffer_elements = {row_buffer_bytes // bytes_per_element}")
print(f"block_h × block_w = 31 × 31 = {31 * 31}")
print()
print("需要查看 crossing 分析中 block 的精确定义...")
