#!/usr/bin/env python3
"""Debug Input address calculation for medium workload."""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/validation/dram')

from pim_optimizer.workload import ConvWorkload
from pim_optimizer import PIMOptimizer
from pim_optimizer.arch import PIMArchitecture
from trace_generator_v2 import TraceGeneratorV2

workload = ConvWorkload(name='medium', R=3, S=3, P=7, Q=7, C=32, K=32, N=1, stride=(1,1), dilation=(1,1))
arch = PIMArchitecture.from_yaml('examples/configs/arch.yaml')
optimizer = PIMOptimizer(arch=arch)
result = optimizer.optimize([workload])
mapping = result.mappings[0]

gen = TraceGeneratorV2()
loops = gen.build_loop_nesting(mapping)
strides = gen.compute_strides(loops, mapping, workload)

print('Input strides:', strides['input'])
print('Weight strides:', strides['weight'])
print('Output strides:', strides['output'])
print('block_h:', strides['block_h'])
print('block_w:', strides['block_w'])
print('row_aligned:', strides['row_aligned'])

# Show Level 3 loops
print('\nLevel 3 loops:')
level3_loops = [(l, k, d, b) for l, k, d, b in loops if l == 3]
dim_names = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
for l, k, d, b in level3_loops:
    print(f'  Level {l} {k} {dim_names[d]}: bound={b}')

print('\nAll loops:')
for l, k, d, b in loops:
    print(f'  Level {l} {k:8s} {dim_names[d]}: bound={b}')

# Show first few Input addresses
print('\nFirst few Input addresses:')
loop_vars = {(l, k, d): 0 for l, k, d, b in loops}

DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6

def compute_input_addr():
    r = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == DIM_R)
    s = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == DIM_S)
    p = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == DIM_P)
    q = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == DIM_Q)
    c_h = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == DIM_C and l == 0 and k == 'H')
    c3 = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == DIM_C and l == 3)
    h = q + r
    w = p + s
    return (h * strides['input']['h'] + w * strides['input']['w'] + 
            c_h * strides['input']['c_h'] + c3 * strides['input']['c3'])

count = 0
for i0 in range(level3_loops[0][3]):  # R
    loop_vars[level3_loops[0][:3]] = i0
    for i1 in range(level3_loops[1][3]):  # Q
        loop_vars[level3_loops[1][:3]] = i1
        for i2 in range(level3_loops[2][3]):  # K
            loop_vars[level3_loops[2][:3]] = i2
            addr = compute_input_addr()
            if count < 30:
                print(f'R={i0}, Q={i1}, K={i2}: addr={addr}')
                count += 1
