#!/usr/bin/env python3
"""分析 Sequential 和 Row-Aligned 两种模式的 crossing 问题"""

print("=" * 70)
print("两种模式的 Crossing 计算对比")
print("=" * 70)

print("""
【Row-Aligned 模式】

使用 compute_input_crossing_ratio() 计算每个 (block, tile) 组合的 crossing ratio
然后用 AND 变量选择实际的组合

公式:
  cr_h = compute_input_crossing_ratio(block_h, tile_h, step_h, ...)
  cr_w = compute_input_crossing_ratio(block_w, tile_w, step_w, ...)
  input_row_act = rb_tiles / banks × (1 + cr_h + cr_w)

问题:
  compute_input_crossing_ratio 使用周期性假设
  → 不考虑实际 tile 数量
  → 最后一个 tile 可能取不到完整大小


【Sequential 模式】

也使用 compute_input_crossing_ratio() 来计算 block crossing ratio
额外还有一个 row crossing ratio（基于 tile_bytes vs row_buffer_size）

公式:
  cr_b = avg_block_crossing_ratio (所有 block/tile 组合的平均)
  cr_r = avg_row_crossing_ratio (基于 tile_bytes / row_buffer_size)
  
  input_row_act = rb_tiles / tiles_per_row / banks × 
                  [(1-cr_b)(1-cr_r) + 2 × reuse × (cr_b + cr_r)]

问题:
  1. compute_input_crossing_ratio 同样使用周期性假设
  2. 使用"所有组合的平均"而不是"实际选择的组合"
  3. 同样不考虑实际 tile 数量和边界效应
""")

print("=" * 70)
print("共同的根本问题")
print("=" * 70)

print("""
【问题的根源】

compute_input_crossing_ratio() 的计算方式：

g = gcd(step, block_h)
period = block_h / g
crossing_count = 0
for k in range(period):
    pos_mod = (k * step) % block_h
    if pos_mod + tile_h > block_h:
        crossing_count += 1
crossing_ratio = crossing_count / period

这个公式假设：
1. 有无限多个 tile 周期性访问
2. 每个 tile 都是完整的 tile_h 大小
3. 不考虑实际数据范围 (input_h)

【实际情况】

1. 实际 tile 数量 = ceil((input_h - tile_h) / step) + 1
2. 最后一个 tile 可能只访问 min(tile_h, input_h - pos) 行
3. 边界处的 tile 可能不 crossing（因为实际大小 < tile_h）
""")

print("=" * 70)
print("修复建议（两种模式通用）")
print("=" * 70)

print("""
【方案 A: 修改 compute_input_crossing_ratio() 函数】

添加 input_h 参数，精确计算而不是用周期公式：

def compute_input_crossing_ratio_exact(
    block_h, tile_h, step, input_h
):
    num_tiles = 0
    crossing_count = 0
    pos = 0
    
    while pos < input_h:
        actual_size = min(tile_h, input_h - pos)
        pos_in_block = pos % block_h
        
        if pos_in_block + actual_size > block_h:
            crossing_count += 1
        
        num_tiles += 1
        pos += step
        if pos >= input_h:
            break
    
    return crossing_count / num_tiles if num_tiles > 0 else 0.0

问题：input_h 在 ILP 中是变量（依赖于 tiling 决策）


【方案 B: 保守估计 + 边界修正】

如果 num_tiles <= period:
    # 精确计算前 num_tiles 个 tile 的 crossing
    crossing = 枚举计算
else:
    # 大部分 tile 用周期公式
    # 最后 (num_tiles % period) 个 tile 可能需要修正
    crossing ≈ crossing_ratio × (num_tiles - 1) / num_tiles


【方案 C: 特殊情况跳过（最简单）】

检测以下情况，不计算 crossing：

1. tile_h >= input_h (一个 tile 覆盖全部)
   → num_tiles = 1, crossing 看第一个 tile 是否跨边界

2. DRAM tiling 全为 1 (数据只读一次)
   → 不需要 crossing

3. rb_tiles <= tiles_per_row (所有数据在一个 DRAM row 内)
   → 可能不需要 crossing

这些情况可以在 ILP 中用指示约束处理。
""")

print("=" * 70)
print("具体建议")
print("=" * 70)

print("""
【建议优先级】

1. 首先实现 "方案 C" 的特殊情况检测
   - 简单有效
   - 能修复 tiny workload 这类极端情况
   - 对 ILP 复杂度影响小

2. 对于 Sequential 模式：
   - 不使用"所有组合的平均"
   - 而是像 Row-Aligned 一样用 AND 变量选择实际组合

3. 长期：考虑是否需要重新设计 Row Activation 模型
   - 当前模型对小 workload 不准确
   - 可能需要考虑更精细的边界效应
""")
