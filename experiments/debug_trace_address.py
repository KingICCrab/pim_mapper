#!/usr/bin/env python3
"""Debug trace address calculation."""

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'validation/dram')

from pim_optimizer.arch.pim_arch import PIMArchitecture
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer import PIMOptimizer
from pim_optimizer.mapping import Mapping
from pim_optimizer.model.variables import SpatialDim
from trace_generator import TraceGenerator, DRAMConfig, DIM_NAMES

# Test with tiny workload
arch = PIMArchitecture.from_yaml('examples/configs/arch.yaml')
workload = ConvWorkload(name='tiny', N=1, K=8, C=8, P=4, Q=4, R=3, S=3)

optimizer = PIMOptimizer(arch, verbose=False)
result = optimizer.optimize([workload])

# Extract mapping
model = optimizer.model
vars = optimizer.vars
w = 0
num_mems = arch.num_mems

mapping = Mapping()
mapping.workload_name = workload.name
mapping.workload_bounds = list(workload.bounds)

for m in range(num_mems):
    if m == 0:
        mapping.loop_bounds[m] = {'H': {}, 'W': {}, 'Internal': {}, 'temporal': {}}
        s_names = {SpatialDim.H: 'H', SpatialDim.W: 'W', SpatialDim.INTERNAL: 'Internal', SpatialDim.TEMPORAL: 'temporal'}
        s_range = SpatialDim.num_dims_pe()
    else:
        mapping.loop_bounds[m] = {'spatial': {}, 'temporal': {}}
        s_names = {0: 'spatial', 1: 'temporal'}
        s_range = SpatialDim.num_dims_other()
    
    for j, divs in enumerate(workload.divisors):
        for s in range(s_range):
            s_name = s_names[s]
            for i, div in enumerate(divs):
                if vars.xb[w, m, s, j, i].X > 0.5:
                    mapping.loop_bounds[m][s_name][j] = div

for m in range(num_mems):
    mapping.permutation[m] = {}
    for p in range(len(workload.bounds)):
        for j in range(len(workload.bounds)):
            if vars.xp[w, m, p, j].X > 0.5:
                mapping.permutation[m][p] = j

for t in range(3):
    if (w, t, 'row_aligned') in vars.layout_choice:
        mapping.layout[t] = 'row_aligned' if vars.layout_choice[w, t, 'row_aligned'].X > 0.5 else 'sequential'

h_divisors = getattr(workload, 'hw_divisors', {}).get('H', [1])
w_divisors = getattr(workload, 'hw_divisors', {}).get('W', [1])
block_h = 1
for i, h_div in enumerate(h_divisors):
    if (w, i) in vars.rowbuf_input_block_h:
        if vars.rowbuf_input_block_h[w, i].X > 0.5:
            block_h = h_div
            break
block_w = 1
for j, w_div in enumerate(w_divisors):
    if (w, j) in vars.rowbuf_input_block_w:
        if vars.rowbuf_input_block_w[w, j].X > 0.5:
            block_w = w_div
            break
mapping.tile_info = {'block_h': block_h, 'block_w': block_w}

print('Mapping extracted')
print(f'block_h={block_h}, block_w={block_w}')
print(f'Permutation: {mapping.permutation}')

# Now debug trace generation
trace_gen = TraceGenerator(DRAMConfig())

# Get buffer_tile
buffer_tile = trace_gen._compute_buffer_tile_size(mapping)
print(f'Buffer tile: {buffer_tile}')

# Get layout info
H_in = workload.input_size['H']
W_in = workload.input_size['W']
layout_info = trace_gen._compute_data_layouts(mapping, workload, H_in, W_in, buffer_tile)

print(f'\nInput order: {[DIM_NAMES[d] for d in layout_info["input_order"]]}')
print(f'Input loop order: {[(m, DIM_NAMES[d]) for m, d in layout_info["input_loop_order"]]}')
print(f'Input strides: {layout_info["input_strides"]}')
print(f'Input tile sizes: {layout_info["input_tile_sizes"]}')
print(f'Input base: 0x{layout_info["input_base"]:08X}')

# Test address calculation manually
print('\n--- Testing address calculation ---')
from trace_generator import DIM_N, DIM_C, DIM_P, DIM_Q

# Test with some sample indices
test_indices = {DIM_N: 0, DIM_C: 0, DIM_P: 0, DIM_Q: 0}
addr = trace_gen._compute_tile_wise_address(
    test_indices,
    layout_info['input_loop_order'],
    layout_info['input_strides'],
    layout_info['input_tile_sizes']
)
print(f'Address for (0,0,0,0): {addr}')

test_indices = {DIM_N: 0, DIM_C: 0, DIM_P: 1, DIM_Q: 0}
addr = trace_gen._compute_tile_wise_address(
    test_indices,
    layout_info['input_loop_order'],
    layout_info['input_strides'],
    layout_info['input_tile_sizes']
)
print(f'Address for (0,0,1,0): {addr}')

test_indices = {DIM_N: 0, DIM_C: 0, DIM_P: 0, DIM_Q: 1}
addr = trace_gen._compute_tile_wise_address(
    test_indices,
    layout_info['input_loop_order'],
    layout_info['input_strides'],
    layout_info['input_tile_sizes']
)
print(f'Address for (0,0,0,1): {addr}')

test_indices = {DIM_N: 0, DIM_C: 1, DIM_P: 0, DIM_Q: 0}
addr = trace_gen._compute_tile_wise_address(
    test_indices,
    layout_info['input_loop_order'],
    layout_info['input_strides'],
    layout_info['input_tile_sizes']
)
print(f'Address for (0,1,0,0): {addr}')
