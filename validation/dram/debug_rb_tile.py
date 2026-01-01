"""Analyze RowBuffer tile layout."""
import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/validation/dram')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
from trace_generator import TraceGenerator, DRAMConfig, DIM_C, DIM_N

workload = ConvWorkload(name='small', N=1, K=16, C=16, P=8, Q=8, R=3, S=3)
optimizer = PIMOptimizer()
result = optimizer.optimize([workload], objective='latency')
mapping = result.mappings[0]

gen = TraceGenerator(DRAMConfig())
buffer_tile = gen._compute_buffer_tile_size(mapping)
H_in = workload.input_size['H']
W_in = workload.input_size['W']
layout_info = gen._compute_data_layouts(mapping, workload, H_in, W_in, buffer_tile)

print('=== Layout Analysis (Updated) ===')
print()

print('buffer_tile (Level 0+1):')
print(f'  {buffer_tile}')
print(f'  C_tile = {buffer_tile[DIM_C]}')
print(f'  N_tile = {buffer_tile[DIM_N]}')
print()

print('RowBuffer tile info:')
print(f'  input_rb_tile_c = {layout_info["input_rb_tile_c"]}')
print(f'  input_rb_tile_n = {layout_info["input_rb_tile_n"]}')
print(f'  input_rb_tile_stride = {layout_info["input_rb_tile_stride"]}')
print()

print('Layout info:')
print(f'  nc_slice_stride = {layout_info["nc_slice_stride"]}')
print(f'  block_h = {layout_info["block_h"]}')
print(f'  block_w = {layout_info["block_w"]}')
print(f'  block_size = {layout_info["block_size"]}')
print(f'  input_layout = {layout_info["input_layout"]}')
print()

# 计算 row_aligned 地址
nc_slice_stride = layout_info['nc_slice_stride']
input_rb_tile_c = layout_info['input_rb_tile_c']
input_rb_tile_n = layout_info['input_rb_tile_n']
input_rb_tile_stride = layout_info['input_rb_tile_stride']
input_rb_tile_nc = input_rb_tile_c * input_rb_tile_n
row_size = 1024

print('=== Address calculation with row_aligned ===')
for c in range(min(8, workload.C)):
    n = 0
    global_nc_idx = n * workload.C + c
    rb_tile_idx = global_nc_idx // input_rb_tile_nc
    offset_in_rb_tile = global_nc_idx % input_rb_tile_nc
    addr = rb_tile_idx * input_rb_tile_stride + offset_in_rb_tile * nc_slice_stride
    row = addr // row_size
    print(f'  c={c}: rb_tile={rb_tile_idx}, offset={offset_in_rb_tile}, addr={addr}, row={row}')
