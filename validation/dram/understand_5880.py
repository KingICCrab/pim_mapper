#!/usr/bin/env python3
"""
精确分析 Trace 的 5880 vs ILP 的 2392

从运行日志可以得到关键参数：
- P_l3=28, Q_l3=7, C_l3=3 (注意：与 verify_crossing.py 不同!)
- block_h=31, block_w=31
- input_aligned_tile_size = 1024

关键发现：P_l3 和 Q_l3 的值是交换的！
- verify_crossing.py 用的是 P_l3=7, Q_l3=28
- 实际 mapping 是 P_l3=28, Q_l3=7 (从 permutation 顺序)

这会影响 H 和 W 方向的访问模式和 crossing 计算！
"""

print("=" * 70)
print("ResNet-L1 实际 Mapping 参数分析")
print("=" * 70)

# 实际参数 (从运行日志)
P_l3, Q_l3, C_l3 = 28, 7, 3
K_l3 = 4
block_h, block_w = 31, 31

# Buffer 级别参数 (Level 1)
P_buf = 1  # H 方向 buffer tile
Q_buf = 1  # W 方向 buffer tile  
# 但 input tile size = 1024 bytes = 512 elements

# L2 级别参数
R_l2 = 7  # from Level 2 temporal
S_l2 = 7  # from Level 1 temporal (注意这里!)

# 访问 tile 尺寸
# H_per_tile = (P_buf - 1) + R_buf + 1 = 0 + 7 + 1 = 8? 不对
# 让我重新理解

print("\n从 Mapping 推断访问参数:")
print(f"  P_l3={P_l3}, Q_l3={Q_l3}, C_l3={C_l3}")
print(f"  R_l2={R_l2}, S_l2={S_l2}")

# 分析 input aligned tile size = 1024 bytes
# 1024 bytes / 2 bytes per element = 512 elements per tile
# 如果是 H × W 的 tile: √512 ≈ 22.6
# 或者: 8 × 2 = 16 (太小), 14 × 2 = 28, 等等

# 检查 loop bounds
# Level 1: H.0=1, H.1=1, H.2=2, H.3=8, ...
# H 的因子: 1×1×2×8 = 16, 但 input H = 62
# W 的因子: 1×1×1×1×16 = 16? 

# 需要从 trace 的 L3 strides 理解:
# stride_P_l3 = 1024 (permutation dim=3)
# stride_Q_l3 = 7168 (permutation dim=2)  
# stride_C_l3 = 200704 (permutation dim=4)

print("\nInput 地址 strides:")
print(f"  stride_P_l3 = 1024 bytes")
print(f"  stride_Q_l3 = 7168 bytes")
print(f"  stride_C_l3 = 200704 bytes")

# 推断:
# stride_Q_l3 = 7168 = 7 × 1024 = Q_l3 × stride_P_l3
# 所以一个 P tile 是 1024 bytes
# stride_C_l3 = 200704 = 28 × 7168 = P_l3 × stride_Q_l3

# 这意味着 input layout 是:
# [C][P_tile][Q_tile][intra_tile]
# 每个 intra_tile = 1024 bytes = 512 elements

# 那 block_h, block_w 是什么?
# block_h = block_w = 31
# 一个 block = 31 × 31 = 961 elements = 1922 bytes ≈ 1.88 rows

print("\nBlock 和 Row 关系:")
print(f"  block_h × block_w = {block_h} × {block_w} = {block_h * block_w} elements")
print(f"  = {block_h * block_w * 2} bytes = {block_h * block_w * 2 / 1024:.2f} rows")

# 关键问题: Trace 怎么计算 row activation?
# 答案: 对于每个访问的地址，检查是否切换到了新的 row

# 现在让我用正确的参数模拟

input_H, input_W, input_C = 62, 62, 3
num_h_blocks = (input_H + block_h - 1) // block_h  # = 2
num_w_blocks = (input_W + block_w - 1) // block_w  # = 2

print(f"\nInput 维度:")
print(f"  H={input_H}, W={input_W}, C={input_C}")
print(f"  num_h_blocks={num_h_blocks}, num_w_blocks={num_w_blocks}")

# 推算 tile 尺寸
# 如果 P_l3=28 是 P 方向的 DRAM 循环因子
# 那么 P_buf = P / P_l3 = 56 / 28 = 2? 不对，Output P=56
# 但 Input H = P + R - 1 = 56 + 7 - 1 = 62

# 从 loop bounds 看:
# H level 3: {0: 1, 1: 1, 2: 2, 3: 8, ...}
# 所以 H.3 = 8 是 DRAM level 的因子?

# 我需要重新理解 mapping 的结构...

# 让我直接用 crossing 公式计算
print("\n" + "=" * 70)
print("使用正确参数重新计算 Crossing")
print("=" * 70)

# 假设:
# - 每个 DRAM tile 访问的 H 范围: H_per_tile = ?
# - 每个 DRAM tile 访问的 W 范围: W_per_tile = ?

# 从 input_aligned_tile_size = 1024 bytes
# 如果 layout 是 block-wise: tile_elements = 512
# 可能是 H_per_tile × W_per_tile × C_per_tile = 512

# 检查: 14 × 2 × 1 = 28? 不对
# 检查: 8 × 2 × 1 = 16? 太小
# 检查: 512 个连续 elements in one block

# 关键洞察: input_aligned_tile_size = 1024 = row_buffer_bytes
# 所以每个 aligned tile 刚好是一个 DRAM row

# 对于 row_aligned layout:
# tile 被对齐到 row 边界
# 所以非 crossing tiles 恰好占一个 row
# crossing tiles 占两个 rows

# 因此:
# ILP 的计算方式:
# - 非 crossing tiles: 1 row activation each
# - crossing tiles: 2 row activations each (或者只计算额外的 1?)

# 让我计算 crossing tiles 数量

# H crossing: tile 跨越 H block 边界
# W crossing: tile 跨越 W block 边界

# 问题: tile 的 H, W 范围是什么?

# 从 verify_crossing.py 的参数:
# H_per_tile = 14, W_per_tile = 2
# 但这是基于 P_buf=8, Q_buf=2, R_buf=7, S_buf=1

# 让我检查实际的 loop bounds
print("\nMapping loop bounds 分析:")
print("  Level 1 (buffer):")
print("    H: 1×1×2×8 = 16 (spatial? probably H direction)")
print("    W: 1×1×1×1×16 = 16 (W direction)")
print("  Level 2 (local):")
print("    temporal: R=7")
print("  Level 3 (DRAM):")  
print("    temporal: Q=28, P=7, C=3, K=4")

# 这里有些混淆，让我重新整理

# 从 mapping 字符串:
# loop_bounds = {0: {'H': {...}, 'W': {...}, ...}}
# level 0 是最内层

# H 在各层的因子:
# Level 0 (compute): H.0=1
# Level 1 (buffer): H.1=1, H.2=2, H.3=8 (这是空间因子?)
# 等等，H.2 和 H.3 都在 level 1 吗?

# 我需要重新理解 mapping 格式...

# 让我直接从 trace 统计来分析

print("\n" + "=" * 70)
print("从 Trace 结果反推")
print("=" * 70)

# Trace Input: 5880
# ILP Input: 2392
# 比率: 5880 / 2392 ≈ 2.46

# 如果 ILP 只计算 "unique tiles" (每个 tile 1 次)
# 而 Trace 计算 "row switches" (包括 tile 内部的 row switches)

# 假设 ILP = 非crossing tiles + crossing tiles = total tiles
# 假设 Trace = 非crossing tiles + 2 × crossing tiles

# 设 x = 非 crossing tiles, y = crossing tiles
# ILP = x + y = 2392
# Trace = x + 2y = 5880?

# 那么: y = 5880 - 2392 = 3488
# x = 2392 - 3488 = -1096? 负数不可能

# 所以这个假设不对

# 让我尝试另一种假设:
# ILP 可能只计算 unique rows (去重后的行数)
# Trace 计算每次访问新 row 的次数 (不去重)

# 或者 ILP 使用了某种 reuse 模型

# 从 Trace 数据:
# - 每个 channel 有 4 个 rows (0, 1 for block 0; 7, 8 for block 1? 或其他)
# - Row 访问模式可能是 532, 532, 448, 448 per channel
# - 3 channels: (532 + 532 + 448 + 448) × 3 = 1960 × 3 = 5880

# 这正好匹配!

print("从 row 访问模式分析:")
print("  每个 channel: 532 + 532 + 448 + 448 = 1960")
print("  3 channels: 1960 × 3 = 5880 ✓")
print()
print("  532 次访问可能来自 h_block=0 的 rows")
print("  448 次访问可能来自 h_block=1 的 rows")
print()
print("  差异: 532 - 448 = 84 per pair of rows")
print("  84 × 3 channels × 2 pairs = 504")
print("  或者: 84 × 6 = 504 (如果 6 是某个因子)")

# 现在分析为什么是 532 vs 448

# 在 H 方向:
# h_block=0 包含 h=[0, 30]
# h_block=1 包含 h=[31, 61]

# 每次访问 tile 时，可能访问 h_block=0 或 h_block=1

# 如果 tile 跨越边界 (h=31 附近)，会同时访问两个 blocks
# 非跨越 tiles 只访问一个 block

# h_block=0 的访问次数更多，因为:
# 1. 大多数 tiles 从 h=0 开始
# 2. crossing tiles 会额外访问 h_block=0 的部分

# 让我计算具体数字

# P_l3 × Q_l3 × S_l2 × K_l3 × C_l3 = ?
# 但 S_l2 是在 level 2, 所以...

# 实际的 DRAM 迭代次数:
# K_l3 × C_l3 × P_l3 × Q_l3 = 4 × 3 × 28 × 7 = 2352

print(f"\nDRAM 迭代次数 (不含 S_l2):")
print(f"  K_l3 × C_l3 × P_l3 × Q_l3 = 4 × 3 × 28 × 7 = {4*3*28*7}")

# 但每次 DRAM 迭代，S 循环在 Level 2 重复 7 次
# 所以总 tiles = 2352 × 7 = 16464
print(f"  含 S_l2: 2352 × 7 = {2352*7}")

# 这与 verify_crossing.py 一致

# ILP = 2392
# 如果 unique rows = 总 tiles / (tiles per unique row)
# 2392 ≈ 16464 / 6.88

# 这不太对，让我看 ILP 的计算公式

print("\n" + "=" * 70)
print("ILP 公式分析")
print("=" * 70)

# ILP row_activations_input = 2392
# 这可能是:
# 1. 基础行数 + crossing 额外贡献
# 2. 或者某种 aligned 计算

# 从 verify_crossing.py:
# H crossing tiles: 2352
# W crossing tiles: 336
# Both crossing: 48

# 不含 crossing 的 tiles: 16464 - 2352 - 336 + 48 = 13824

# 如果 ILP 只计算 "有效行激活":
# aligned_tiles / tiles_per_row + crossing_extra

# 假设每 row 可以容纳 7 个 tiles (因为 aligned_tile_size = row_buffer_size)
# 那么 unique rows = 16464 / 7 ≈ 2352

# 但 ILP = 2392 ≈ 2352 + 40
# 40 可能是 crossing 的额外贡献?

# 让我检查 verify_crossing.py 的输出
print("\nverify_crossing.py 结果:")
print("  H crossing: 2352")
print("  W crossing: 336")
print("  Both crossing: 48")
print("  Non-crossing: 13824")
print("  Total: 16464")

# ILP 可能的计算:
# row_acts = tiles / tiles_per_row + crossing_factor
# 2392 = 16464 / 7 + ?
# 16464 / 7 = 2352
# 2392 - 2352 = 40

# 40 来自哪里?
# 可能是 W crossing 的一部分: 336 / 8 = 42 ≈ 40?

print("\n假设 ILP 公式:")
print("  base_rows = 16464 / 7 = 2352")
print("  crossing_extra = 336 / 8 = 42 (W crossing / 8?)")
print("  total = 2352 + 42 ≈ 2394 ≈ ILP 的 2392")

# 但 Trace = 5880 = 2.46 × ILP
# 这意味着 Trace 计算的是每次 row switch，而不是 unique rows

# 5880 = 16464 / 2.8?
# 不对

# 让我想想...
# 5880 / 3 = 1960 per channel
# 1960 / 4 blocks = 490 per block? 不对，532+532+448+448=1960

# 分析 K 循环的影响:
# K_l3 = 4, 每次 K 循环访问同一 Input
# 所以 Input 被访问 K_l3 = 4 次?
# 2392 × 4 = 9568 ≠ 5880

# 不对，K 是 Output/Weight 的维度，Input 不依赖 K

# 让我重新整理:
# Input 依赖: R, S, P, Q, C, N (不依赖 K)
# Output 依赖: P, Q, K, N
# Weight 依赖: R, S, C, K

# 在 DRAM 循环顺序中: K -> C -> Q -> P (或类似)
# 每次 K iteration, 访问不同的 Output/Weight, 但重用 Input
# 所以 K_l3=4 次循环访问相同的 Input 部分

# 但 Trace 可能没有利用这个 reuse?
# 或者 Trace 确实重复访问了 4 次?

# 如果 Trace 不做 reuse:
# base_row_acts = 2392 (每个 K 迭代)
# total = 2392 × ? (重复访问次数)

# 但 5880 / 2392 = 2.46, 不是整数

# 让我换个角度: 直接计算 row switches

print("\n" + "=" * 70)
print("直接计算 row switches")
print("=" * 70)

# 使用 trace_generator 的实际逻辑
# 对于每个 DRAM tile, trace_generator 会:
# 1. 遍历 tile 内的元素
# 2. 每个元素有一个地址
# 3. 当地址的 row 变化时, row_activation += 1

# 每个 tile 内有多少元素?
# tile_elements = H_per_tile × W_per_tile = 14 × 2 = 28 (from verify_crossing.py)

# 每个 tile 内可能的 row switches:
# 0 (如果所有元素在同一 row)
# 或 1+ (如果元素跨越 row 边界)

# 但 aligned layout 应该保证每个 tile 在一个 row 内
# 除非 tile 跨越 block 边界

# 问题可能在于: trace_generator 访问元素的顺序
# 如果按 (h, w) 顺序遍历，而不是按 block 顺序
# 可能会导致更多的 row switches

print("假设 tile 内部的 row switches 导致了 5880:")
print("  如果每个 tile 平均产生 5880/16464 = 0.357 次 row switch")
print("  这不太合理，应该是整数")

print("\n  如果按 (K,C) 级别分析:")
print(f"    total (K,C) iterations = {K_l3} × {C_l3} = {K_l3 * C_l3}")
print(f"    5880 / 12 = 490 per (K,C)")
print(f"    这与单个 (K,C) 内的 block switches 相关")

# 从 explain_5880.py:
# 单个 (K,C) iteration 的 row activations: 448
# 12 × 448 = 5376 ≠ 5880

# 差值 = 5880 - 5376 = 504

print("\n差值分析:")
print("  explain_5880.py 模拟: 5376")
print("  Trace 实际: 5880")
print("  差值: 504")
print()
print("  504 = 84 × 6")
print("  84 = 532 - 448 (h_block=0 vs h_block=1 的访问差)")
print("  6 = ? (可能是某个因子)")
