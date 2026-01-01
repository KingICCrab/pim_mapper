#!/usr/bin/env python3
"""分析 trace 文件的 row activation 模式"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/validation/dram')

from trace_generator import DRAMConfig
import numpy as np
from collections import defaultdict

cfg = DRAMConfig()
row_buffer_bytes = cfg.row_buffer_bytes  # 1024
col_bits = int(np.log2(row_buffer_bytes))  # 10

print(f'row_buffer_bytes = {row_buffer_bytes}')
print(f'col_bits = {col_bits}')
print()

# 读取 trace 并分析
with open('validation/dram/validation_output/small_trace_report.txt', 'r') as f:
    lines = f.readlines()

print(f'Total trace lines: {len(lines)}')
print()

# 模拟 row buffer，每 bank 一个
open_rows = {}  # bank -> current_row
row_acts = defaultdict(int)
row_changes = []

for i, line in enumerate(lines):
    parts = line.strip().split()
    if len(parts) < 2:
        continue
    addr = int(parts[1], 16)
    
    # 地址解码
    col = addr & ((1 << col_bits) - 1)  # 低10位
    row = (addr >> col_bits) & 0x3FFF   # 接下来14位
    bank = addr >> (col_bits + 14)      # 高位
    
    # 检查 row activation
    if bank not in open_rows:
        open_rows[bank] = row
        row_acts[bank] += 1
        row_changes.append((i, addr, bank, row, 'NEW_BANK'))
    elif open_rows[bank] != row:
        row_acts[bank] += 1
        old_row = open_rows[bank]
        open_rows[bank] = row
        row_changes.append((i, addr, bank, row, f'ROW_CHANGE from {old_row}'))

print('Row activation stats:')
for bank, acts in sorted(row_acts.items()):
    print(f'  Bank {bank}: {acts} row activations')
print(f'  Total: {sum(row_acts.values())}')
print()

print(f'Total row changes: {len(row_changes)}')
print()

# 分析每个 bank 的访问模式
print('Row changes by bank:')
for bank in sorted(row_acts.keys()):
    bank_changes = [(i, addr, row, event) for i, addr, b, row, event in row_changes if b == bank]
    print(f'\nBank {bank}: {len(bank_changes)} activations')
    for item in bank_changes[:10]:
        i, addr, row, event = item
        print(f'  Line {i}: addr=0x{addr:08X}, row={row}, {event}')
    if len(bank_changes) > 10:
        print(f'  ... ({len(bank_changes) - 10} more)')
