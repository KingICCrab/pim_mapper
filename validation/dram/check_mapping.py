#!/usr/bin/env python3
import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from pim_optimizer.workload import ConvWorkload
from pim_optimizer.optimizer import PIMOptimizer
from validation.dram.trace_generator import TraceGenerator, DRAMConfig

workload = ConvWorkload(name="ResNet-L1", R=7, S=7, P=56, Q=56, C=3, K=64, N=1)
optimizer = PIMOptimizer()
result = optimizer.optimize([workload])
mapping = result.mappings[0]

# 看 L3 Q stride
print("Mapping Layout:")
print(f"  Input layout: {mapping.layout}")
print(f"  block_h: {mapping.metrics.get('block_h')}")
print(f"  block_w: {mapping.metrics.get('block_w')}")
print()

# 看 L3 factors
print("L3 factors:")
l3_bounds = mapping.loop_bounds.get(3, {}).get("temporal", {})
print(f"  {l3_bounds}")
print()

print("L2 factors:")
l2_bounds = mapping.loop_bounds.get(2, {}).get("temporal", {})
print(f"  {l2_bounds}")
print()

# 生成 trace 并分析第一个 row switch 的位置
dram_config = DRAMConfig()
gen = TraceGenerator(dram_config)
trace = gen.generate_trace(mapping, workload)

row_size = 1024
bank_size = row_size * 16384

# 收集所有 Input 访问
input_accesses = []
for i, line in enumerate(trace):
    parts = line.strip().split()
    if len(parts) < 2:
        continue
    addr = int(parts[1], 16)
    bank = addr // bank_size
    if bank == 0:  # Input bank
        row = (addr % bank_size) // row_size
        col = addr % row_size
        input_accesses.append((i, addr, row, col))

print(f"Total Input accesses: {len(input_accesses)}")
print(f"Unique rows: {sorted(set(a[2] for a in input_accesses))}")

# 找第一个 row switch
prev_row = input_accesses[0][2]
for i, (trace_idx, addr, row, col) in enumerate(input_accesses[1:], 1):
    if row != prev_row:
        print(f"\nFirst row switch at Input index {i} (trace index {trace_idx}):")
        print(f"  Previous: Row {prev_row}")
        print(f"  Current:  Row {row}, Addr 0x{addr:08X}, Col {col}")
        print()
        # 打印 switch 前后的访问
        print("Context (10 accesses before and after):")
        start = max(0, i - 10)
        end = min(len(input_accesses), i + 10)
        for j in range(start, end):
            trace_idx, addr, row, col = input_accesses[j]
            marker = " <-- SWITCH" if j == i else ""
            print(f"  [{j}] Addr=0x{addr:08X} Row={row} Col={col}{marker}")
        break
    prev_row = row

# 检查 block_h/block_w 来源
print()
print("=== Block Size Analysis ===")
print(f"mapping.tile_info = {mapping.tile_info}")
print()

# 重新计算 H_per_tile, W_per_tile
P_per_tile = 8
Q_per_tile = 2
R_per_tile = 7
S_per_tile = 7
stride_h, stride_w = 1, 1
dilation_h, dilation_w = 1, 1

H_per_tile = (P_per_tile - 1) * stride_h + (R_per_tile - 1) * dilation_h + 1
W_per_tile = (Q_per_tile - 1) * stride_w + (S_per_tile - 1) * dilation_w + 1

print(f"Calculated H_per_tile = {H_per_tile}")
print(f"Calculated W_per_tile = {W_per_tile}")
print()

# 如果 tile_info 为空, block_h/block_w 就用 H_per_tile/W_per_tile
block_h = mapping.tile_info.get('block_h', H_per_tile)
block_w = mapping.tile_info.get('block_w', W_per_tile)
print(f"Actual block_h = {block_h}")
print(f"Actual block_w = {block_w}")
print()

# 检查 L2 Q 迭代后的最大 W
Q_l2 = 4
max_w = (Q_l2 - 1) * Q_per_tile + W_per_tile - 1
print(f"Max W in L3 tile (with L2 Q={Q_l2}): {max_w}")
print(f"Will w_block change? max_w >= block_w: {max_w} >= {block_w} = {max_w >= block_w}")
