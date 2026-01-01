#!/usr/bin/env python3
"""分析 Block Crossing 问题"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/validation/dram')

from trace_generator import DRAMConfig
import numpy as np

cfg = DRAMConfig()
row_buffer_bytes = cfg.row_buffer_bytes  # 1024

print(f'=== Block Crossing Analysis ===')
print()

# Small workload 参数
# block_h=10, block_w=10 (从 generate_report 输出)
# Input size: H=10, W=10 (P=8 + R-1 = 8+2 = 10)
# C=16, N=1

block_h = 10
block_w = 10
H_in = 10
W_in = 10
C = 16
N = 1

block_size = block_h * block_w  # 100 elements per block
num_blocks_h = (H_in + block_h - 1) // block_h  # 1 block in H
num_blocks_w = (W_in + block_w - 1) // block_w  # 1 block in W
nc_block_size = num_blocks_h * num_blocks_w * block_size  # 100 elements per (n,c)

print(f'Workload: N={N}, C={C}, H_in={H_in}, W_in={W_in}')
print(f'Block: block_h={block_h}, block_w={block_w}, block_size={block_size}')
print(f'Blocks: num_blocks_h={num_blocks_h}, num_blocks_w={num_blocks_w}')
print(f'nc_block_size = {nc_block_size} elements per (n,c) slice')
print()

# 计算每个 (n, c) 的地址范围
print(f'=== Input Address Layout ===')
print()
print(f'Row size = {row_buffer_bytes} bytes = {row_buffer_bytes} elements')
print()

total_elements = N * C * nc_block_size
print(f'Total Input elements: {total_elements}')
print(f'Total Input size: {total_elements} bytes')
print(f'Number of rows needed: {(total_elements + row_buffer_bytes - 1) // row_buffer_bytes}')
print()

print(f'Address ranges for each (n, c):')
for n in range(N):
    for c in range(C):
        start_addr = (n * C + c) * nc_block_size
        end_addr = start_addr + nc_block_size - 1
        start_row = start_addr // row_buffer_bytes
        end_row = end_addr // row_buffer_bytes
        crossing = "CROSSES ROW BOUNDARY" if start_row != end_row else ""
        print(f'  (n={n}, c={c:2d}): addr 0x{start_addr:04X}-0x{end_addr:04X}, row {start_row}-{end_row} {crossing}')

print()
print(f'=== Block Crossing Issue ===')
print()

# 计算第一个跨 row 的 (n, c)
for n in range(N):
    for c in range(C):
        start_addr = (n * C + c) * nc_block_size
        end_addr = start_addr + nc_block_size - 1
        start_row = start_addr // row_buffer_bytes
        end_row = end_addr // row_buffer_bytes
        if start_row != end_row:
            print(f'First block crossing at (n={n}, c={c}):')
            print(f'  Block address range: 0x{start_addr:04X} - 0x{end_addr:04X}')
            print(f'  Row {start_row}: 0x{start_row * row_buffer_bytes:04X} - 0x{(start_row+1) * row_buffer_bytes - 1:04X}')
            print(f'  Row {end_row}: 0x{end_row * row_buffer_bytes:04X} - 0x{(end_row+1) * row_buffer_bytes - 1:04X}')
            
            # 计算跨越点
            row_boundary = (start_row + 1) * row_buffer_bytes
            elements_before_boundary = row_boundary - start_addr
            elements_after_boundary = end_addr - row_boundary + 1
            print(f'  Elements before row boundary: {elements_before_boundary}')
            print(f'  Elements after row boundary: {elements_after_boundary}')
            print()
            
            # 在 block 内的位置
            h_boundary = elements_before_boundary // block_w
            w_boundary = elements_before_boundary % block_w
            print(f'  Row boundary crosses block at approximately h={h_boundary}, w={w_boundary}')
            break
    else:
        continue
    break

print()
print(f'=== Why Thrashing Happens ===')
print()
print(f'When accessing Input tile that spans row boundary:')
print(f'  - Tile has elements in both Row 0 (0x000-0x3FF) and Row 1 (0x400-0x7FF)')
print(f'  - Access order from permutation causes interleaved access')
print(f'  - Example: access 0x400 (Row 1), then 0x3ED (Row 0), then 0x401 (Row 1)...')
print(f'  - Each row switch = 1 row activation')
print()
print(f'=== Solution: Block Crossing ===')
print()
print(f'The ILP model should:')
print(f'  1. Detect when a tile crosses row boundary')
print(f'  2. Count additional row activations for boundary-crossing tiles')
print(f'  3. Or ensure block_h * block_w <= row_buffer_bytes to avoid crossing')
