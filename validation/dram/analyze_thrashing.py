#!/usr/bin/env python3
"""深入分析 Input 的 row buffer thrashing 原因"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/validation/dram')

from trace_generator import DRAMConfig
import numpy as np

cfg = DRAMConfig()
row_buffer_bytes = cfg.row_buffer_bytes  # 1024
col_bits = int(np.log2(row_buffer_bytes))  # 10

print(f'=== DRAM Configuration ===')
print(f'row_buffer_bytes = {row_buffer_bytes}')
print(f'col_bits = {col_bits}')
print(f'Row 0: 0x000 - 0x3FF (0-1023)')
print(f'Row 1: 0x400 - 0x7FF (1024-2047)')
print()

# 读取 trace 并分析 Input (Bank 0) 的访问
with open('validation/dram/validation_output/small_trace_report.txt', 'r') as f:
    lines = f.readlines()

print(f'=== Input Tensor Access Analysis ===')
print()

# 只看 Bank 0 (Input) 的访问
input_accesses = []
for i, line in enumerate(lines):
    parts = line.strip().split()
    if len(parts) < 2:
        continue
    addr = int(parts[1], 16)
    bank = addr >> (col_bits + 14)
    if bank == 0:
        row = (addr >> col_bits) & 0x3FFF
        col = addr & ((1 << col_bits) - 1)
        input_accesses.append((i, addr, row, col))

print(f'Total Input accesses: {len(input_accesses)}')
print(f'Unique addresses: {len(set(a[1] for a in input_accesses))}')

# 找出 row 的范围
rows_accessed = set(a[2] for a in input_accesses)
print(f'Rows accessed: {sorted(rows_accessed)}')
print()

# 分析 thrashing 发生的位置
print(f'=== Row Thrashing Analysis ===')
print()

# 找到第一次 row change 的位置
prev_row = None
thrash_start = None
for i, (line_num, addr, row, col) in enumerate(input_accesses):
    if prev_row is not None and row != prev_row:
        if thrash_start is None:
            thrash_start = i
            print(f'First row change at access #{i} (line {line_num}):')
            print(f'  Previous: addr=0x{input_accesses[i-1][1]:08X}, row={prev_row}')
            print(f'  Current:  addr=0x{addr:08X}, row={row}')
            print()
            break
    prev_row = row

# 显示 thrashing 区域附近的访问
if thrash_start:
    print(f'=== Accesses around thrashing (#{thrash_start-5} to #{thrash_start+20}) ===')
    for i in range(max(0, thrash_start-5), min(len(input_accesses), thrash_start+20)):
        line_num, addr, row, col = input_accesses[i]
        mark = " <-- ROW CHANGE" if i > 0 and input_accesses[i][2] != input_accesses[i-1][2] else ""
        print(f'  #{i}: line={line_num}, addr=0x{addr:08X}, row={row}, col={col}{mark}')
    print()

# 分析地址分布
print(f'=== Address Distribution ===')
print()

# 按 row 分组
row_to_addrs = {}
for i, addr, row, col in input_accesses:
    if row not in row_to_addrs:
        row_to_addrs[row] = []
    row_to_addrs[row].append(addr)

for row in sorted(row_to_addrs.keys()):
    addrs = row_to_addrs[row]
    print(f'Row {row}: {len(addrs)} accesses, addr range 0x{min(addrs):08X} - 0x{max(addrs):08X}')

print()

# 分析访问顺序 - 找出导致 thrashing 的模式
print(f'=== Thrashing Pattern ===')
print()

# 统计连续访问不同 row 的次数
row_changes = 0
current_row = None
change_positions = []
for i, (line_num, addr, row, col) in enumerate(input_accesses):
    if current_row is not None and row != current_row:
        row_changes += 1
        if len(change_positions) < 50:
            change_positions.append((i, line_num, addr, current_row, row))
    current_row = row

print(f'Total row changes (Input): {row_changes}')
print()

if change_positions:
    print(f'First {min(20, len(change_positions))} row changes:')
    for i, (acc_idx, line_num, addr, from_row, to_row) in enumerate(change_positions[:20]):
        print(f'  #{i+1}: access={acc_idx}, line={line_num}, addr=0x{addr:08X}, row {from_row} -> {to_row}')
