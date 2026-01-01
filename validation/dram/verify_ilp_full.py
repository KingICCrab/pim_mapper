#!/usr/bin/env python3
"""
完整验证 ILP 的 Input row activation 计算
"""

# ResNet-L1 参数
P_l3, Q_l3, C_l3, K_l3 = 7, 28, 3, 4
S_l2 = 7
R_l0 = 7

print("=" * 80)
print("ILP Input Row Activation 完整计算")
print("=" * 80)

# 对于 row_aligned 模式:
# total_acts = row_acts_aligned + block_crossing_acts

# 1. row_acts_aligned = Π_{所有 DRAM 相关维度} bound^x
# 对于 Input，relevant dims = [R, S, P, Q, C, N]

print("\n1. row_acts_aligned 计算:")
print("   = 所有 DRAM level 因子的乘积")

# DRAM loops (Level 2 + Level 3)
# Level 0 (Buffer): R=7, K=16, P=8, Q=2
# Level 2 (RowBuffer): S=7
# Level 3 (DRAM): P=7, Q=28, C=3, K=4

# 对于 Input，irrelevant dim = K
# row_acts_aligned = P_l3 × Q_l3 × S_l2 × C_l3 × (R_related) × (N_related)
# 但 R 和 N 没有 DRAM 级别的 tile，所以他们的 bound^x = 1

# 实际上 row_acts_aligned 计算的是访问的 unique tile 数
# = P_l3 × Q_l3 × S_l2 × C_l3 = 7 × 28 × 7 × 3

row_acts_aligned = P_l3 * Q_l3 * S_l2 * C_l3
print(f"   P_l3 × Q_l3 × S_l2 × C_l3 = {P_l3} × {Q_l3} × {S_l2} × {C_l3}")
print(f"   = {row_acts_aligned}")

# 但等等，这是 DRAM tile 数，不是 unique row 数
# 让我重新理解...

print("\n2. block_crossing_acts 计算:")
print("   = 2 × crossing_count × reuse_penalty")

# crossing_count 是从 precompute_input_block_crossing_table 得到的
# 根据前面的分析，ILP 使用了 tile_h = 2 (Q_rb × stride + S_rb × dilation - ... = 2)
# 这导致 crossing_count 很小

# reuse_penalty = Π_{irrelevant dims} bound^x
# 对于 Input, irrelevant = K
# reuse_penalty = K_l3 = 4

reuse_penalty = K_l3
print(f"   reuse_penalty = K_l3 = {reuse_penalty}")

# crossing_count 根据前面计算 = 1 (用 tile_h=2)
# 但实际上 ILP 可能使用了不同的方式

# 让我检查 crossing_count 的计算
# 从 precompute_input_block_crossing_table 看:
# - block_h = 31 (选中的 H_rb)
# - q_rb = 2, s_rb = 1
# - tile_h = stride × q_rb + dilation × s_rb - stride - dilation + 1 = 2
# - step = q_rb × stride = 2
# - 这会导致 crossing_count 很小

# 但这不对，因为 2384 = 2352 + 32
# H crossing tiles = 2352
# 32 可能来自 W crossing 或其他

print("\n" + "=" * 80)
print("反推 ILP 的计算")
print("=" * 80)

print("\nILP 预测 = 2384")
print("我们的 H crossing tiles = 2352")
print("差值 = 2384 - 2352 = 32")

# 可能的解释：
# ILP 计算: row_acts_aligned + block_crossing_acts
# 其中 row_acts_aligned 是某个基础值

# 让我检查 unique rows
# Input 访问 12 个 unique rows (from trace)
# 如果 row_acts_aligned = unique_rows × K_l3 = 12 × 4 = 48? 不对

# 或者：
# row_acts_aligned = unique spatial tiles = P_l3 * Q_l3 = 7 * 28 = 196? 不对

# 让我用另一种方式
# ILP 的公式是:
# total = row_aligned × row_acts_aligned + block_crossing_acts
# 当 row_aligned = 1 时:
# total = row_acts_aligned + block_crossing_acts

# row_acts_aligned 可能是基于 "每个 spatial tile 一次 row activation"
# 但这需要更仔细分析 _build_log_product_expr 的实现

print("\n可能的 ILP 计算方式:")
print("  row_acts_aligned = ?")
print("  block_crossing_acts = 2 × crossing_count × reuse_penalty")
print("  total = row_acts_aligned + block_crossing_acts")

# 假设 block_crossing = 2 × 2352 / 2 = 2352 (H crossing)
# 那么 row_acts_aligned = 2384 - 2352 = 32

# 或者 ILP 只计算了 H crossing，没有乘以 2
# block_crossing_acts = crossing_count × reuse_penalty = 2352 × ? = 2352
# row_acts_aligned = 32

# 32 是什么？
# 可能是 W crossing 的某种简化: 336 / 10.5 ≈ 32
# 或者是 unique W blocks = 2 (block 0 和 block 1) × 某个因子

print("\n分解 2384:")
print("  如果 H crossing = 2352, 剩余 = 32")
print("  32 可能是:")
print("    - unique H blocks × unique W blocks = 2 × 2 × 8 = 32 ???")
print("    - 或者某种 W 方向的 crossing 简化")

# 实际上让我检查 W crossing 的计算
w_crossing_tiles = 336
# 但 ILP 的 block_crossing 只考虑 H 方向
# W 方向的 crossing 可能在别处处理

print("\n" + "=" * 80)
print("核心问题：ILP 只计算 H 方向的 block crossing")
print("=" * 80)

print("\n从 precompute_input_block_crossing_table 代码看:")
print("  只计算了 block_h 方向的 crossing")
print("  没有计算 block_w 方向的 crossing!")
print("\n这解释了为什么:")
print("  ILP crossing ≈ H crossing = 2352")
print("  剩余 32 可能是 row_acts_aligned 的基础值")

# 让我验证 row_acts_aligned
# 从代码看，all_dims 包含所有维度
# 对于 Input，log_aligned_expr 是所有 DRAM 相关维度的 log sum
# 但如果是 row_aligned 模式，很多 crossing 被消除了

# row_acts_aligned 可能代表 "unique L3 tiles 的数量"
# = P_l3 × Q_l3 / tiles_per_row ???

# 或者更简单：row_acts_aligned = unique rows accessed
# = 12 (from trace)
# 但 12 ≠ 32

print("\n让我检查 row_acts_aligned 的真实含义:")
# row_acts_aligned = Π_{all relevant dims at DRAM level} bound^x
# 对于 Input: relevant = R, S, P, Q, C, N
# 在 DRAM level (Level 2+3): P=7, Q=28, S=7, C=3, (R, N 没有 DRAM factor)

# 但这些是 factor，不是 tile 数
# 等等，bound^x 的意思是：如果 x=1，则乘以 bound；如果 x=0，则乘以 1
# 所以 row_acts_aligned = 选中因子的乘积

# 假设选中的因子是：P=7, Q=28, S=7, C=3
# row_acts_aligned = 7 × 28 × 7 × 3 = 4116??? 太大了

# 或者 row_acts_aligned 只计算外层循环
# P_l3 × Q_l3 × C_l3 = 7 × 28 × 3 = 588??? 还是太大

# 让我重新看代码...
print("\n实际上，从代码看:")
print("  对于 row_aligned 模式，DRAM row crossing = 0 (因为 block 对齐)")
print("  所以 total = 0 + block_crossing_acts = block_crossing_acts")
print("  block_crossing_acts = 2 × H_crossing_count × reuse_penalty")

# 如果是这样：
# block_crossing_acts = 2 × crossing_count × K_l3
# 2384 = 2 × crossing_count × 4
# crossing_count = 2384 / 8 = 298

print(f"\n如果 block_crossing_acts = 2 × crossing_count × K_l3:")
print(f"  2384 = 2 × crossing_count × {K_l3}")
print(f"  crossing_count = 2384 / {2 * K_l3} = {2384 / (2 * K_l3)}")

# 298 是什么？
# H crossing 的 P tile 数 = 1 (只有 P=3 会 cross)
# 每个 P 有 Q_l3 × S_l2 × C_l3 = 28 × 7 × 3 = 588 个 tile
# 但 588 ≠ 298

# 让我试另一种算法
# 如果 reuse_penalty 只包含 irrelevant 的外层因子
# reuse_penalty = K_l3 = 4
# block_crossing = H_crossing / K_l3 = 2352 / 4 = 588
# 588 × 4 = 2352
# 2 × 588 × 4 = 4704??? 不对

print("\n让我重新计算:")
h_crossing_tiles = 2352
print(f"  H crossing tiles = {h_crossing_tiles}")
print(f"  如果每个 H crossing tile 贡献 1 次 row act:")
print(f"    block_crossing_acts = {h_crossing_tiles}")
print(f"    total = 0 + {h_crossing_tiles} = {h_crossing_tiles}")
print(f"  但 ILP = 2384, 差值 = {2384 - h_crossing_tiles}")

# 差值 32 可能是 row_acts_aligned 的贡献
# 在 row_aligned 模式下，seq_part = 0, aligned_part = row_acts_aligned × row_aligned_var
# 当 row_aligned = 1: aligned_part = row_acts_aligned
# total = aligned_part + block_crossing_acts = row_acts_aligned + block_crossing_acts

# 所以 row_acts_aligned = 32?
# 这代表什么？可能是 unique L3 tiles / 某个因子

print(f"\n结论:")
print(f"  ILP Input row_acts = row_acts_aligned + block_crossing_acts")
print(f"                     = 32 + 2352")
print(f"                     = 2384")
print(f"  row_acts_aligned = 32 (可能是 unique H blocks × unique W blocks × C_l3)")
print(f"                   = 2 × 2 × 8 = 32? 或其他计算")
print(f"  block_crossing_acts = 2352 = H crossing tiles")
