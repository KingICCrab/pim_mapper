"""Debug relevancy detection logic."""
import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
from trace_generator import (
    TraceGenerator, DRAMConfig, 
    DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N
)

workload = ConvWorkload(name='small', N=1, K=16, C=16, P=8, Q=8, R=3, S=3)
optimizer = PIMOptimizer()
result = optimizer.optimize([workload], objective='latency')
mapping = result.mappings[0]

gen = TraceGenerator(DRAMConfig())
buffer_tile = gen._compute_buffer_tile_size(mapping)
dram_loops = gen._build_dram_loop_structure(mapping, workload, buffer_tile)

print('buffer_tile:', {k: v for k, v in buffer_tile.items() if v > 1})
print()

dim_names = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
print('DRAM loops (outer to inner):')
for loop in dram_loops:
    print(f'  {dim_names[loop["dim"]]}: bound={loop["bound"]}')
print()

# Simulate the iterate_dram_loops logic
prev_indices = [None]  # list to allow mutation
input_regen_count = [0]
weight_regen_count = [0]
output_regen_count = [0]

def iterate_dram_loops(level_idx, indices):
    if level_idx >= len(dram_loops):
        input_changed = True
        weight_changed = True
        output_changed = True
        
        if prev_indices[0] is not None:
            prev = prev_indices[0]
            input_changed = any(
                indices.get(d, 0) != prev.get(d, 0)
                for d in [DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_N]
            )
            weight_changed = any(
                indices.get(d, 0) != prev.get(d, 0)
                for d in [DIM_R, DIM_S, DIM_C, DIM_K]
            )
            output_changed = any(
                indices.get(d, 0) != prev.get(d, 0)
                for d in [DIM_P, DIM_Q, DIM_K, DIM_N]
            )
        
        prev_indices[0] = indices.copy()
        
        if input_changed:
            input_regen_count[0] += 1
        if weight_changed:
            weight_regen_count[0] += 1
        if output_changed:
            output_regen_count[0] += 1
        return
    
    loop_info = dram_loops[level_idx]
    dim = loop_info['dim']
    bound = loop_info['bound']
    stride = loop_info['stride']
    
    base = indices.get(dim, 0)
    for i in range(bound):
        new_indices = indices.copy()
        new_indices[dim] = base + i * stride
        iterate_dram_loops(level_idx + 1, new_indices)

initial_indices = {d: 0 for d in range(7)}
iterate_dram_loops(0, initial_indices)

print('Regeneration counts (based on relevancy):')
print(f'  Input: {input_regen_count[0]} (expected 8, only C changes matter)')
print(f'  Weight: {weight_regen_count[0]} (expected 32, both C and K matter)')
print(f'  Output: {output_regen_count[0]} (expected 32, K matters)')
