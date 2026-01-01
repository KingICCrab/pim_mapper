#!/usr/bin/env python3
"""处理 crossing ratio 问题的几种方案"""

print("=" * 70)
print("问题总结")
print("=" * 70)

print("""
【当前公式的问题】

1. 假设无限周期性访问，不考虑实际 tile 数量
2. 最后一个 tile 可能取不到完整的 tile_h
3. 可能多算了不存在的 crossing

【影响】

- 对于小 workload（如 tiny），误差很大
- 对于大 workload，周期性假设近似成立，误差较小
""")

print("=" * 70)
print("方案 1: 精确计算（适合后处理验证）")
print("=" * 70)

def exact_crossing_count(input_h, tile_h, step, block_h):
    """精确计算实际的 crossing 次数"""
    num_tiles = 0
    crossing_count = 0
    pos = 0
    
    while pos < input_h:
        actual_end = min(pos + tile_h, input_h)
        actual_size = actual_end - pos
        
        pos_in_block = pos % block_h
        if pos_in_block + actual_size > block_h:
            crossing_count += 1
        
        num_tiles += 1
        pos += step
        
        # 如果下一个 tile 的起始位置已经没有新数据，停止
        if pos >= input_h:
            break
    
    return num_tiles, crossing_count

# Tiny workload 例子
input_h = 6
tile_h = 6
step = 4
block_h = 6

num_tiles, crossing = exact_crossing_count(input_h, tile_h, step, block_h)
print(f"Tiny workload (input_h={input_h}, tile_h={tile_h}, step={step}, block_h={block_h}):")
print(f"  实际 tile 数量: {num_tiles}")
print(f"  实际 crossing 次数: {crossing}")
print(f"  精确 crossing_ratio: {crossing / num_tiles if num_tiles > 0 else 0:.4f}")
print(f"  当前公式 crossing_ratio: 0.6667")

# 较大的 workload 例子
input_h = 100
num_tiles, crossing = exact_crossing_count(input_h, tile_h, step, block_h)
print(f"\n较大 workload (input_h={input_h}):")
print(f"  实际 tile 数量: {num_tiles}")
print(f"  实际 crossing 次数: {crossing}")
print(f"  精确 crossing_ratio: {crossing / num_tiles if num_tiles > 0 else 0:.4f}")
print(f"  当前公式 crossing_ratio: 0.6667 (接近！)")

print("\n" + "=" * 70)
print("方案 2: 在 ILP 中添加边界修正项")
print("=" * 70)

print("""
【思路】

在 ILP 中，可以添加一个修正项来处理边界效应：

num_complete_periods = floor((num_tiles - 1) / period)
boundary_tiles = (num_tiles - 1) % period + 1

# 完整周期使用周期性 crossing ratio
complete_crossing = num_complete_periods × period × crossing_ratio

# 边界 tiles 单独计算（或保守估计）
boundary_crossing = 精确计算或保守估计

total_crossing = complete_crossing + boundary_crossing

【问题】

num_tiles 是 ILP 变量的函数，难以直接计算
可能需要用指示变量或分段线性近似
""")

print("=" * 70)
print("方案 3: 简化处理 - 特殊情况检测（推荐）")
print("=" * 70)

print("""
【思路】

检测特殊情况，对于这些情况不使用周期性公式：

情况 1: 数据只读取一次（DRAM tiling 全为 1 或 reuse = 1）
  → crossing_ratio = 0 或直接用 rb_tiles / banks

情况 2: tile_h >= input_h（一个 tile 覆盖全部数据）
  → num_tiles = 1
  → crossing = 0（如果 tile_h <= block_h）或 1（如果 tile_h > block_h）

情况 3: 只有少量 tiles（比如 num_tiles <= period）
  → 精确计算而不是用周期公式

【在 ILP 中实现】

可以用指示约束（indicator constraints）来处理：

if all_dram_tiling_is_one:
    input_row_act = rb_tiles / banks  # 不加 crossing
else:
    input_row_act = rb_tiles / banks × (1 + crossing_ratio)
""")

print("=" * 70)
print("方案 4: 使用 unique reads 而不是 tile-based 计算")
print("=" * 70)

print("""
【思路】

不按 tile 计算 crossing，而是按实际的 unique 数据量计算：

row_activation = ceil(unique_data_bytes / row_buffer_size) × row_buffer_size

【对于 Row-Aligned 布局】

unique_data_per_bank = rb_tiles / banks
rows_activated = ceil(unique_data_per_bank × element_bytes / row_buffer_size)

这种方法完全不需要 crossing_ratio！
只需要知道总数据量和 row buffer 大小。

【问题】

Row-Aligned 布局的意义是什么？
- 如果 block 对齐到 DRAM row，一个 block 内的访问只需要 1 次 row activation
- 跨越 block 意味着跨越 DRAM row

但如果我们只看 unique 数据量，就失去了这个优化的意义。
""")

print("=" * 70)
print("推荐方案")
print("=" * 70)

print("""
【建议】

1. **短期修复**：添加特殊情况检测
   - 如果 DRAM 所有相关维度 tiling == 1，设 crossing_ratio = 0
   - 这能修复 tiny workload 这类小问题

2. **中期改进**：考虑 tile 数量的上限
   - 如果能估计 num_tiles 的上界，可以修正周期公式
   - crossing_real ≈ min(crossing_count, num_tiles) / num_tiles

3. **长期改进**：重新审视 Row Activation 模型
   - 考虑是否需要区分 Sequential 和 Row-Aligned
   - 对于 Sequential，直接用 ceil(data / row_buffer) 更准确
   - 对于 Row-Aligned，crossing 概念才有意义

【实现优先级】

先实现方案 3 的 "情况 1" 检测：
当 DRAM tiling 全为 1 时，跳过 crossing 计算
这是最简单且影响最大的修复
""")
