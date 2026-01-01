#!/usr/bin/env python3
"""重新理解 crossing 问题 - 重叠部分确实需要重复读取"""

print("=" * 70)
print("重新理解：重叠部分确实需要重复读取")
print("=" * 70)

print("""
【正确的理解】

重叠部分需要重复读取，这就是 crossing 问题的来源。

假设一个更大的例子：Input H = 14, tile_h = 6, step = 4

Tile 0: [0, 6)
Tile 1: [4, 10)  ← 需要重新读取 [4, 6)（重叠）+ 新读 [6, 10)
Tile 2: [8, 14)  ← 需要重新读取 [8, 10)（重叠）+ 新读 [10, 14)

[4, 6) 被读取了 2 次 - 这是 crossing 的 cost
[8, 10) 被读取了 2 次 - 这也是 crossing 的 cost

这是正确的！重叠部分确实需要重复读取。
""")

print("=" * 70)
print("那么问题在哪里？—— 最后一个 tile 取不到完整的 6")
print("=" * 70)

print("""
【周期性 crossing ratio 的计算】

模型按周期计算 crossing ratio:
  step = 4, block_h = 6
  period = block_h / gcd(step, block_h) = 6 / 2 = 3
  
周期内的位置 (mod block_h):
  k=0: pos = 0      → [0, 6)   → 不 crossing (0+6 ≤ 6)
  k=1: pos = 4      → [4, 10)  → crossing! (4+6 > 6)
  k=2: pos = 8%6=2  → [2, 8)   → crossing! (2+6 > 6)
  
crossing_ratio = 2/3 = 0.667

【问题：假设每个 tile 都是完整的 tile_h = 6】

但实际上，边界处的 tile 可能取不到完整的 tile_h!
""")

print("=" * 70)
print("具体例子")
print("=" * 70)

# 例子 1: Input H = 14
print("【例子 1: Input H = 14】")
input_h = 14
tile_h = 6
step = 4
block_h = 6

print(f"Input H = {input_h}, tile_h = {tile_h}, step = {step}")
print()

pos = 0
tile_num = 0
while pos < input_h:
    tile_end = pos + tile_h
    actual_end = min(tile_end, input_h)
    actual_size = actual_end - pos
    
    # 检查是否 crossing（基于完整 tile）
    pos_in_block = pos % block_h
    would_cross_if_full = (pos_in_block + tile_h > block_h)
    actually_crosses = (pos_in_block + actual_size > block_h)
    
    status = ""
    if would_cross_if_full and not actually_crosses:
        status = "← 模型认为 crossing，但实际不 crossing！"
    elif actually_crosses:
        status = "← 真正的 crossing"
    
    print(f"  Tile {tile_num}: pos={pos}, 理论 [{pos}, {tile_end}), 实际 [{pos}, {actual_end})")
    print(f"           pos_in_block={pos_in_block}, actual_size={actual_size}")
    print(f"           模型预测 crossing={would_cross_if_full}, 实际 crossing={actually_crosses} {status}")
    print()
    
    pos += step
    tile_num += 1
    
    if pos >= input_h:
        break

# 例子 2: Input H = 6 (tiny workload)
print("\n【例子 2: Input H = 6 (tiny workload)】")
input_h = 6
print(f"Input H = {input_h}, tile_h = {tile_h}, step = {step}")
print()

pos = 0
tile_num = 0
while pos < input_h:
    tile_end = pos + tile_h
    actual_end = min(tile_end, input_h)
    actual_size = actual_end - pos
    
    pos_in_block = pos % block_h
    would_cross_if_full = (pos_in_block + tile_h > block_h)
    actually_crosses = (pos_in_block + actual_size > block_h)
    
    status = ""
    if would_cross_if_full and not actually_crosses:
        status = "← 模型认为 crossing，但实际不 crossing！多算了！"
    elif actually_crosses:
        status = "← 真正的 crossing"
    
    print(f"  Tile {tile_num}: pos={pos}, 理论 [{pos}, {tile_end}), 实际 [{pos}, {actual_end})")
    print(f"           pos_in_block={pos_in_block}, actual_size={actual_size}")
    print(f"           模型预测 crossing={would_cross_if_full}, 实际 crossing={actually_crosses} {status}")
    print()
    
    pos += step
    tile_num += 1
    
    # 检查下一个 tile 的起始位置是否还有新数据
    if pos >= input_h or (pos < input_h and pos + tile_h <= actual_end + step):
        # 没有新数据了
        pass

print("=" * 70)
print("核心问题总结")
print("=" * 70)

print("""
【Crossing Ratio 模型的错误】

模型假设：每个 tile 都访问完整的 tile_h 行
实际情况：边界处的 tile 只能访问 min(tile_h, input_h - pos) 行

当 tile 取不到完整的 tile_h 时：
  - 模型预测它会 crossing（因为 pos + tile_h > block_h）
  - 但实际上它可能不会 crossing（因为实际大小 < tile_h）
  
这导致模型高估了 crossing 次数！

【修复方向】

计算 crossing 时，需要考虑：
1. 实际的 tile 数量（不是无限周期）
2. 每个 tile 的实际大小（边界效应）
3. 只有当实际访问范围跨越 block 边界时才算 crossing
""")
