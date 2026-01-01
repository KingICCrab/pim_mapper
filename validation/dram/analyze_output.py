#!/usr/bin/env python3
"""分析 Output tensor 的行激活次数差异"""

# 从 analysis.txt 提取的信息
# DRAM 循环结构 (outer to inner):
# [0] Level 3, P: 4 iterations
# [1] Level 3, C: 8 iterations  
# [2] Level 2, P: 2 iterations
# [3] Level 2, K: 2 iterations

# Output 相关维度: P, Q, K, N
# Output 不相关维度: R, S, C

# Output Buffer tile: N=1, K=8, P=2, Q=16
# Output tile size = 1*8*2*16 = 256 elements = 256 bytes

# Output layout: sequential
# Row size: 1024 bytes

print("Output 访问分析:")
print("=" * 60)

# 总迭代次数
total_iters = 4 * 8 * 2 * 2
print(f"Total DRAM loop iterations: {total_iters}")

# dram_loop_dims
DIM_P, DIM_C, DIM_K = 2, 4, 5
dram_loop_dims = [
    {"dim": DIM_P, "bound": 4, "level": 3},  # P_l3
    {"dim": DIM_C, "bound": 8, "level": 3},  # C_l3
    {"dim": DIM_P, "bound": 2, "level": 2},  # P_l2
    {"dim": DIM_K, "bound": 2, "level": 2},  # K_l2
]

output_relevant = {DIM_P, 3, DIM_K, 6}  # P=2, Q=3, K=5, N=6


def compute_flat_tile_index(tile_indices, relevant_dims, dram_loop_dims):
    """从 trace_generator.py 复制的函数"""
    flat_idx = 0
    stride = 1
    for loop in reversed(dram_loop_dims):
        dim, bound = loop["dim"], loop["bound"]
        if dim in relevant_dims:
            flat_idx += tile_indices.get(dim, 0) * stride
            stride *= bound
    return flat_idx


def iterate_loops(dram_loops, level_idx, indices, results):
    """模拟 DRAM 循环迭代"""
    if level_idx >= len(dram_loops):
        results.append(indices.copy())
        return

    loop = dram_loops[level_idx]
    dim = loop["dim"]
    bound = loop["bound"]
    base = indices.get(dim, 0)

    for tile_idx in range(bound):
        new_indices = indices.copy()
        new_indices[dim] = base + tile_idx  # 累加!
        iterate_loops(dram_loops, level_idx + 1, new_indices, results)


# 模拟实际循环迭代
print("\n模拟实际循环迭代 (代码逻辑):")
all_indices = []
iterate_loops(dram_loop_dims, 0, {d: 0 for d in range(7)}, all_indices)
print(f"Total iterations: {len(all_indices)}")

# 计算每个迭代的 output flat_idx
flat_indices = []
for indices in all_indices:
    flat_idx = compute_flat_tile_index(indices, output_relevant, dram_loop_dims)
    flat_indices.append(flat_idx)

print(f"Unique output flat indices: {len(set(flat_indices))}")
print(f"Max flat_idx: {max(flat_indices)}")

# 检查 output_changed 逻辑
print("\n检查 output_changed 判断:")
prev_indices = None
output_access_count = 0
for indices in all_indices:
    output_changed = True
    if prev_indices is not None:
        # Output relevant: P, Q, K, N (not R, S, C)
        output_changed = any(
            indices.get(d, 0) != prev_indices.get(d, 0)
            for d in [DIM_P, 3, DIM_K, 6]  # P, Q, K, N
        )
    if output_changed:
        output_access_count += 1
    prev_indices = indices.copy()

print(f"Output 访问次数 (output_changed=True): {output_access_count}")

# 计算地址和行号
output_base = 0x02000000
tile_size = 256
row_size = 1024

print("\n地址计算:")
addresses = []
rows = []
for flat_idx in flat_indices:
    addr = output_base + flat_idx * tile_size
    row = (addr - output_base) // row_size
    addresses.append(addr)
    rows.append(row)

print(f"Unique rows accessed: {sorted(set(rows))}")
print(f"Number of unique rows: {len(set(rows))}")

# 计算行切换次数
row_switches = 1  # 初始访问
prev_row = None
for row in rows:
    if prev_row is not None and row != prev_row:
        row_switches += 1
    prev_row = row

print(f"Row switches (activations) 基于 flat_idx: {row_switches}")

# 现在模拟正确的行为: 只在 output_changed 时访问
print("\n只统计 output_changed=True 时的行激活:")
prev_indices = None
prev_row = None
row_switches_filtered = 0
access_sequence = []

for i, indices in enumerate(all_indices):
    output_changed = True
    if prev_indices is not None:
        output_changed = any(
            indices.get(d, 0) != prev_indices.get(d, 0)
            for d in [DIM_P, 3, DIM_K, 6]
        )

    if output_changed:
        flat_idx = flat_indices[i]
        row = rows[i]
        access_sequence.append((flat_idx, row, indices.copy()))
        if prev_row is None:
            row_switches_filtered = 1
        elif row != prev_row:
            row_switches_filtered += 1
        prev_row = row

    prev_indices = indices.copy()

print(f"Output 访问次数: {len(access_sequence)}")
print(f"Row switches (只计 output_changed): {row_switches_filtered}")

# 详细打印访问序列
print("\n详细访问序列 (前20个):")
print(f"{'idx':<5} {'P':<5} {'C':<5} {'K':<5} {'flat':<8} {'row':<5} {'switch':<8}")
print("-" * 50)
prev_row = None
for i, (flat_idx, row, indices) in enumerate(access_sequence[:30]):
    switch = "NEW" if (prev_row is None or row != prev_row) else ""
    print(f"{i:<5} {indices[DIM_P]:<5} {indices[DIM_C]:<5} {indices[DIM_K]:<5} {flat_idx:<8} {row:<5} {switch:<8}")
    prev_row = row

# 分析问题
print("\n" + "=" * 60)
print("问题分析:")
print("=" * 60)

# 问题1: P 出现两次
print("\n问题1: P 维度在循环中出现两次")
print("  P_l3: 4 iterations")
print("  P_l2: 2 iterations")
print("  indices[P] = P_l3 + P_l2 (累加), 范围 0-5")
print("  但实际应该是 P_l3 * 2 + P_l2, 范围 0-7")

# 问题2: C 循环导致重复访问
print("\n问题2: C 循环导致 Output 被重复访问")
print("  C 是 Output 的不相关维度")
print("  但 C 循环 (8次) 在 P_l3 和 P_l2 之间")
print("  每次 C 变化时, output_changed 检测不到 (因为 C 不是 Output 相关维度)")
print("  但当 P_l2 或 K_l2 变化时, 会产生新访问")

# 计算预期行激活
print("\n预期行激活计算:")
print("  Output tile size: 256 bytes")
print("  Row size: 1024 bytes")
print("  tiles_per_row: 4")

# 实际访问模式分析
print("\n实际访问模式:")
# 按 (P, K) 分组统计
from collections import defaultdict
pk_counts = defaultdict(list)
for flat_idx, row, indices in access_sequence:
    pk = (indices[DIM_P], indices[DIM_K])
    pk_counts[pk].append((flat_idx, row, indices[DIM_C]))

print(f"Unique (P, K) combinations: {len(pk_counts)}")
for pk, accesses in sorted(pk_counts.items())[:10]:
    print(f"  P={pk[0]}, K={pk[1]}: {len(accesses)} accesses, rows={set(r for _, r, _ in accesses)}")
