#!/usr/bin/env python3
"""
最终分析: 5880 vs 5376 的差值 504 来自哪里？

关键发现:
- Trace 实际: 5880
- 模拟 (无 R_l2): 5376  
- 差值: 504

可能原因:
1. Trace 每个 channel 的访问模式不同 (532 vs 448)
2. R_l2=7 没有在模拟中正确考虑
3. 地址计算方式不同

让我从 Trace 的实际数据反推。
"""

print("=" * 70)
print("5880 vs 5376 差值分析")
print("=" * 70)

# 从 Trace 日志得到的数据:
# - Row 0: 532 visits (channel 0-2)
# - Row 1: 532 visits
# - Row 7: 448 visits  
# - Row 8: 448 visits
# 每个 channel: 532 + 532 + 448 + 448 = 1960
# 3 channels: 1960 × 3 = 5880

print("\nTrace 统计数据:")
print("  每个 channel 访问模式:")
print("    Row 0: 532 visits")
print("    Row 1: 532 visits")
print("    Row 7: 448 visits")
print("    Row 8: 448 visits")
per_channel = 532 + 532 + 448 + 448
print(f"  每个 channel 总计: {per_channel}")
print(f"  3 channels 总计: {per_channel * 3}")

print()
print("模拟数据:")
print("  所有 blocks 都是 448 visits")
print("  4 blocks × 3 channels × 448 visits / 4 rows = ?")
# 不对，让我重新理解

# 在模拟中，我们跟踪的是 (c, h_block, w_block)
# 在 Trace 中，跟踪的是 row (地址 // row_buffer_bytes)

# 关键区别：
# - 模拟中 block = (c, h_block, w_block)
# - Trace 中 row = (c * input_size + h_block * block_stride + ...) // 1024

# 让我计算每个 block 对应哪些 DRAM rows

print()
print("=" * 70)
print("Block 到 DRAM Row 的映射")
print("=" * 70)

block_h, block_w = 31, 31
row_buffer_bytes = 1024
element_bytes = 2  # FP16
input_H, input_W, input_C = 62, 62, 3

# Input layout: (H, W, C) 或 (C, H, W)?
# 从 trace_generator 看，是 block-wise layout

# 假设 row-major with blocks:
# Input[c][h_block][w_block][h_in_block][w_in_block]
# block_size = block_h × block_w × element_bytes = 31 × 31 × 2 = 1922 bytes

block_size_bytes = block_h * block_w * element_bytes
print(f"\n每个 block 的大小: {block_h} × {block_w} × {element_bytes} = {block_size_bytes} bytes")
print(f"Row buffer: {row_buffer_bytes} bytes")
print(f"每个 block 跨越: {block_size_bytes / row_buffer_bytes:.2f} rows")

# 所以一个 block (31×31) = 1922 bytes 约占 1.88 rows
# 也就是说，访问一个 block 可能需要访问 2 个不同的 DRAM rows

# 但模拟中我们把整个 block 当作一个单位
# Trace 中如果 block 跨越 row 边界，会产生额外的 row activation

# 差值 504 的来源:
# 如果每个 block 的首次访问需要访问 2 个 rows (因为 block 跨越 row 边界)
# 但后续访问同一 block 不再切换 row...

# 不对，让我重新思考

# Trace 的 row activation 定义:
# 当访问的 DRAM row 与上一次不同时，row activation += 1

# 问题是：在一个 tile 访问期间，访问多个元素时，
# 这些元素可能跨越多个 DRAM rows

print()
print("=" * 70)
print("重新理解 Trace 的计数方式")
print("=" * 70)

# 在 trace_generator 中，每个元素都会产生一条 LD 指令
# 分析器会跟踪每条指令的 row number
# 当 row number 变化时，row_activation += 1

# 所以 Trace 的 5880 是**元素级别**的 row switch 计数
# 而我的模拟是 **tile 级别** 的 block switch 计数

# 这就解释了为什么数字不同！

# 让我验证：一个 tile (14 × 2 = 28 元素) 内部有多少 row switches?

P_buf, Q_buf = 8, 2
R_buf, S_buf = 7, 1
H_per_tile = 14  # (P_buf - 1) + (R_buf - 1) + 1
W_per_tile = 2   # (Q_buf - 1) + (S_buf - 1) + 1

print(f"\n每个 tile 的元素数: {H_per_tile} × {W_per_tile} = {H_per_tile * W_per_tile}")

# 元素 layout 假设: (h, w, c) 在内存中
# 每个 input 元素的地址 = base + h * W * C * elem_size + w * C * elem_size + c * elem_size
# 但这是原始 layout，trace_generator 使用的是 block layout

# 让我检查 block layout 的地址计算

print()
print("=" * 70)
print("Block Layout 地址计算")
print("=" * 70)

# Block layout: [N][C][H_block][W_block][h_in_block][w_in_block]
# 对于 Input (N=1):
#   base_c = c × (num_h_blocks × num_w_blocks × block_h × block_w)
#   base_h_block = h_block × (num_w_blocks × block_h × block_w)
#   base_w_block = w_block × (block_h × block_w)
#   offset = h_in_block × block_w + w_in_block
#   addr = base_c + base_h_block + base_w_block + offset

num_h_blocks = (input_H + block_h - 1) // block_h  # = 2
num_w_blocks = (input_W + block_w - 1) // block_w  # = 2
block_elements = block_h * block_w

print(f"Input shape: ({input_H}, {input_W}, {input_C})")
print(f"Block shape: ({block_h}, {block_w})")
print(f"Num blocks: ({num_h_blocks}, {num_w_blocks})")
print(f"Elements per block: {block_elements}")

# 检查第一个 tile 内部的 row switches
print()
print("第一个 tile (h=[0,14), w=[0,2)) 的元素地址分析:")

def calc_addr(c, h, w):
    """计算 block layout 下的地址"""
    h_block = h // block_h
    w_block = w // block_w
    h_in_block = h % block_h
    w_in_block = w % block_w
    
    base_c = c * num_h_blocks * num_w_blocks * block_elements
    base_h_block = h_block * num_w_blocks * block_elements
    base_w_block = w_block * block_elements
    offset = h_in_block * block_w + w_in_block
    
    addr = base_c + base_h_block + base_w_block + offset
    return addr * element_bytes  # Convert to bytes

def calc_row(addr):
    return addr // row_buffer_bytes

# 分析第一个 tile
print("\nh, w, addr (bytes), row:")
prev_row = None
row_switches_in_tile = 0
for h in range(14):
    for w in range(2):
        addr = calc_addr(0, h, w)
        row = calc_row(addr)
        marker = ""
        if prev_row is not None and row != prev_row:
            row_switches_in_tile += 1
            marker = " <-- ROW SWITCH"
        if h < 5 or (h >= 10 and h < 14):  # 只打印部分
            print(f"  h={h}, w={w}: addr={addr}, row={row}{marker}")
        prev_row = row

print(f"  ...")
print(f"  第一个 tile 内的 row switches: {row_switches_in_tile}")

# 这才是真正的问题！
# Trace 统计的是元素级别的 row switch
# 而我之前的模拟是 block 级别的

print()
print("=" * 70)
print("关键发现")
print("=" * 70)
print("""
问题原因：
1. 我之前的模拟假设同一 block 内不会有 row switch
2. 但实际上，一个 block (31×31) = 1922 bytes 跨越约 1.88 个 DRAM rows
3. 在 block 内部遍历时，也会触发 row switch!

但这还是不对...因为 Trace 的 block-wise 访问应该已经考虑了这点。

让我检查 Trace generator 的访问顺序。
""")

# 在 trace_generator 中，访问顺序是：
# for h_block: for w_block: for h_in_block: for w_in_block
# 这是 block-wise 访问

# 在这种访问模式下，在一个 block 内部：
# h_in_block = 0..30, w_in_block = 0..30 (for full block)
# 地址是连续的: 0, 1, 2, ..., 960 (first row of block)
# 然后 961, 962, ..., 1921 (second row of block)

# 所以一个 full block 内部会有 1 次 row switch (从 row N 到 row N+1)

print("重新计算一个 full block 内的 row switches:")
prev_row = None
switches = 0
for h_in in range(block_h):
    for w_in in range(block_w):
        offset = h_in * block_w + w_in
        addr_bytes = offset * element_bytes
        row = addr_bytes // row_buffer_bytes
        if prev_row is not None and row != prev_row:
            switches += 1
        prev_row = row

print(f"  一个 full block ({block_h}×{block_w}) 内的 row switches: {switches}")

# 元素数 = 31 × 31 = 961
# 字节数 = 961 × 2 = 1922
# Rows = ceil(1922 / 1024) = 2
# So there's 1 switch inside a full block

# 但 Trace 的统计可能是按 tile 级别，不是按元素级别...
# 让我重新检查
