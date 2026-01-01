#!/usr/bin/env python3
"""
分析 Sequential 模式 crossing ratio 计算的问题
"""
import math

# tiny workload 参数
h_divisors = [1, 2, 3, 6]  # block_h 可选值
total_H = 6  # Input height = R + P - 1 = 3 + 4 - 1 = 6
total_S = 3  # kernel height
dilation_h = 1

def compute_input_crossing_ratio(block_h, tile_h, step, tiler_s, total_S, dilation):
    if block_h <= 0 or tile_h <= 0 or step <= 0:
        return 0.0
    if tile_h > block_h:
        return 1.0
    
    g = math.gcd(int(step), int(block_h))
    period = int(block_h) // g
    
    if tiler_s >= total_S:
        crossing_count = 0
        for k in range(period):
            pos_mod = (k * step) % block_h
            if pos_mod + tile_h > block_h:
                crossing_count += 1
        return crossing_count / period if period > 0 else 0.0
    else:
        num_kernel_groups = (total_S + tiler_s - 1) // tiler_s
        total_crossing_ratio = 0.0
        for group_idx in range(num_kernel_groups):
            base_kernel_row = group_idx * tiler_s
            offset = base_kernel_row * dilation
            crossing_count = 0
            for k in range(period):
                pos_mod = (k * step + offset) % block_h
                if pos_mod + tile_h > block_h:
                    crossing_count += 1
            crossing_ratio_group = crossing_count / period if period > 0 else 0.0
            total_crossing_ratio += crossing_ratio_group
        return total_crossing_ratio / num_kernel_groups if num_kernel_groups > 0 else 0.0


print("=" * 60)
print("问题分析: Sequential 模式 Crossing Ratio 的计算")
print("=" * 60)

print("\n【问题 1: 使用平均值而非实际选择的值】\n")

tile_h = 6  # 完整高度
step = 1
tiler_s = total_S  # kernel 不 split

print(f"tiny workload: tile_h = {tile_h}, step = {step}")
print()

all_cr = []
for block_h in h_divisors:
    cr = compute_input_crossing_ratio(block_h, tile_h, step, tiler_s, total_S, dilation_h)
    all_cr.append(cr)
    marker = " ← 实际选择的 block_h" if block_h == 6 else ""
    print(f"  block_h = {block_h}: crossing_ratio = {cr:.4f}{marker}")

avg_cr = sum(all_cr) / len(all_cr)
print()
print(f"  代码使用的平均值: {avg_cr:.4f}")
print(f"  实际应该是:       {all_cr[-1]:.4f} (block_h=6 时)")
print()

print("【原因】")
print("  - block_h=1: tile_h(6) > block_h(1) → crossing = 1.0")
print("  - block_h=2: tile_h(6) > block_h(2) → crossing = 1.0")
print("  - block_h=3: tile_h(6) > block_h(3) → crossing = 1.0")
print("  - block_h=6: tile_h(6) = block_h(6) → crossing = 0.0")
print("  平均值被前三个 1.0 拉高到 0.75!")

print("\n" + "=" * 60)
print("【问题 2: 公式不适用于 DRAM tiling 全为 1 的情况】")
print("=" * 60)

print("""
当 DRAM tiling 全为 1 时 (即数据只从 DRAM 读一次):

公式假设:
  - 数据被多次以 tile 为单位访问
  - 每次访问可能跨越 block/row 边界
  - 跨越时需要额外的 row activation × reuse

实际情况:
  - 只有 1 次访问 (不是多次小 tile 访问)
  - crossing 开销是一次性的，不会 × reuse
  - 应该直接计算: ceil(total_bytes / row_buffer_size) / banks

对于 tiny workload:
  - Input = 6 × 6 × 8 = 288 bytes
  - row_buffer_size = 1024 bytes
  - 288 < 1024，完全 fit 在一个 row
  - 理论上 row_act = 1 / 4 banks = 0.25
  - 但代码计算出 1.56!
""")

print("=" * 60)
print("【总结: Sequential 模式的两个 BUG】")
print("=" * 60)

print("""
BUG 1: crossing_ratio 使用所有可能组合的平均值
  - 应该使用 ILP 变量，根据实际选择的 (block, tile) 计算
  - Row-Aligned 模式已经这样做了 (用 AND 变量)
  - Sequential 模式没有

BUG 2: 没有处理"只访问一次"的特殊情况
  - 当 DRAM tiling 全为 1 时，数据只读一次
  - crossing 惩罚不应该 × reuse
  - 应该有一个简化路径
""")
