#!/usr/bin/env python3
"""Debug MobileNet-L1 address calculation."""

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'validation/dram')

from pim_optimizer.arch.pim_arch import PIMArchitecture
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer import PIMOptimizer
from pim_optimizer.mapping import Mapping
from pim_optimizer.model.variables import SpatialDim
from trace_generator import TraceGenerator, DRAMConfig, DIM_NAMES, DIM_N, DIM_C, DIM_P, DIM_Q

# MobileNet-L1
arch = PIMArchitecture.from_yaml('examples/configs/arch.yaml')
workload = ConvWorkload(name='MobileNet-L1', N=1, K=32, C=16, P=28, Q=28, R=1, S=1)

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

print(f'block_h={block_h}, block_w={block_w}')
print(f'Input size: H={workload.input_size["H"]}, W={workload.input_size["W"]}')

# Get layout info
trace_gen = TraceGenerator(DRAMConfig())
buffer_tile = trace_gen._compute_buffer_tile_size(mapping)
H_in = workload.input_size['H']
W_in = workload.input_size['W']
layout_info = trace_gen._compute_data_layouts(mapping, workload, H_in, W_in, buffer_tile)

print(f'\nInput strides: {layout_info["input_strides"]}')
print(f'Input tile sizes: {layout_info["input_tile_sizes"]}')
print(f'Row size: {layout_info["row_size"]} elements')
print(f'Input layout: {layout_info["input_layout"]}')

# Calculate addresses for first few elements
print('\n--- Sample addresses (stride pattern) ---')
print('Walking through Q (H) direction:')
for q in range(5):  # H direction
    indices = {DIM_N: 0, DIM_C: 0, DIM_P: 0, DIM_Q: q}
    addr = trace_gen._compute_tile_wise_address(
        indices,
        layout_info['input_loop_order'],
        layout_info['input_strides'],
        layout_info['input_tile_sizes']
    )
    row = addr // 1024
    col = addr % 1024
    print(f'  Q={q} -> addr={addr:5d}, row={row:3d}, col={col:4d}')

print('\nWalking through P (W) direction:')
for p in range(5):  # W direction  
    indices = {DIM_N: 0, DIM_C: 0, DIM_P: p, DIM_Q: 0}
    addr = trace_gen._compute_tile_wise_address(
        indices,
        layout_info['input_loop_order'],
        layout_info['input_strides'],
        layout_info['input_tile_sizes']
    )
    row = addr // 1024
    col = addr % 1024
    print(f'  P={p} -> addr={addr:5d}, row={row:3d}, col={col:4d}')

print('\nWalking through C direction:')
for c in range(3):  # C direction  
    indices = {DIM_N: 0, DIM_C: c, DIM_P: 0, DIM_Q: 0}
    addr = trace_gen._compute_tile_wise_address(
        indices,
        layout_info['input_loop_order'],
        layout_info['input_strides'],
        layout_info['input_tile_sizes']
    )
    row = addr // 1024
    col = addr % 1024
    print(f'  C={c} -> addr={addr:5d}, row={row:3d}, col={col:4d}')

# Check what ILP predicts
print('\n--- ILP prediction ---')
mapping_result = result.mappings[0] if hasattr(result, 'mappings') else result
print(f'Input row acts: {mapping_result.metrics.get("row_activations_input", "N/A")}')
print(f'Weight row acts: {mapping_result.metrics.get("row_activations_weight", "N/A")}')  
print(f'Output row acts: {mapping_result.metrics.get("row_activations_output", "N/A")}')
print(f'Total row acts: {mapping_result.metrics.get("row_activations", "N/A")}')

# Understand the issue
print('\n--- Analysis ---')
# With stride=784 (28*28), consecutive H accesses jump by 784 bytes
# Row size is 1024, so every ~1.3 H steps cross a row boundary
# For H=28, that's ~21 row crossings per C slice
# For C=16, total Input row acts should be ~336

# But ILP predicts 98 - why?
# ILP uses block-based calculation, while trace counts every access

print('Input total elements: N*C*H*W =', 1*16*28*28)
print('If sequential layout: row_acts = elements / 1024 =', (1*16*28*28) // 1024)
print('Stride(Q):', layout_info['input_strides'].get((2, DIM_Q), 'N/A'))
print('Stride(P):', layout_info['input_strides'].get((2, DIM_P), 'N/A'))
print('Stride(C):', layout_info['input_strides'].get((2, DIM_C), 'N/A'))
