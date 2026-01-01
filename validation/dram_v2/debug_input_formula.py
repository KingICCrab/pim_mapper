#!/usr/bin/env python3
"""
Debug script for Input row activation formula.
Analyzes the trace generator's actual behavior to derive correct formula.
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

import math
from validation.dram.trace_generator import TraceGenerator, DRAMConfig
from validation.dram_v2.experiments.mapping_sweep import SimpleMapping
from validation.dram_v2.core.mapping_space import MappingConfig, DIM_P, DIM_Q, DIM_C, DIM_K
from src.pim_optimizer.workload.conv import ConvWorkload


def analyze_input_pattern(workload, config, dram_config):
    """Analyze input access pattern and return statistics."""
    mapping = SimpleMapping(config, workload)
    generator = TraceGenerator(dram_config)
    trace = generator.generate_trace(mapping, workload)
    
    input_rows = []
    row_size = dram_config.row_buffer_bytes
    bank_size = row_size * dram_config.num_rows
    
    for line in trace:
        parts = line.split()
        if len(parts) >= 2:
            addr = int(parts[1], 16)
            bank = addr // bank_size
            if bank == 0:  # Input bank
                bank_addr = addr % bank_size
                row = bank_addr // row_size
                input_rows.append(row)
    
    row_switches = sum(1 for i in range(1, len(input_rows)) if input_rows[i] != input_rows[i-1])
    unique_rows = len(set(input_rows))
    
    return {
        'rows': input_rows,
        'row_switches': row_switches,
        'unique_rows': unique_rows,
        'total_accesses': len(input_rows)
    }


def main():
    # Create workload
    workload = ConvWorkload(name='test', R=3, S=3, P=8, Q=8, C=4, K=4, N=1)
    H, W = workload.input_size['H'], workload.input_size['W']
    print(f'Workload: P={workload.P}, Q={workload.Q}, C={workload.C}, K={workload.K}')
    print(f'Input: H={H}, W={W}')
    print()
    
    dram_config = DRAMConfig(row_buffer_bytes=1024, num_banks=4, num_rows=16384, element_size=1)
    row_size = dram_config.row_buffer_bytes
    
    print('=' * 80)
    print('Analysis: Understanding row_aligned address calculation')
    print('=' * 80)
    print()
    
    # Case 1: block covers entire input
    print('Case 1: block_h >= H and block_w >= W (single block covers all)')
    config1 = MappingConfig(
        P_l3=4, Q_l3=4, C_l3=2, K_l3=2,
        permutation=(DIM_P, DIM_Q, DIM_C, DIM_K),
        input_layout='row_aligned',
        weight_layout='sequential',
        output_layout='sequential',
        block_h=10, block_w=10
    )
    result1 = analyze_input_pattern(workload, config1, dram_config)
    print(f'  Config: block_h=10, block_w=10 (H={H}, W={W})')
    print(f'  Total elements: H*W*C = {H}*{W}*{workload.C} = {H*W*workload.C}')
    print(f'  Rows needed: ceil({H*W*workload.C}/{row_size}) = {math.ceil(H*W*workload.C/row_size)}')
    print(f'  Trace: unique_rows={result1["unique_rows"]}, row_switches={result1["row_switches"]}')
    print()
    
    # Case 2: smaller blocks
    print('Case 2: block_h < H or block_w < W (multiple blocks)')
    config2 = MappingConfig(
        P_l3=4, Q_l3=4, C_l3=2, K_l3=2,
        permutation=(DIM_P, DIM_Q, DIM_C, DIM_K),
        input_layout='row_aligned',
        weight_layout='sequential',
        output_layout='sequential',
        block_h=5, block_w=5
    )
    result2 = analyze_input_pattern(workload, config2, dram_config)
    num_h_blocks = math.ceil(H / 5)
    num_w_blocks = math.ceil(W / 5)
    print(f'  Config: block_h=5, block_w=5')
    print(f'  num_h_blocks = ceil({H}/5) = {num_h_blocks}')
    print(f'  num_w_blocks = ceil({W}/5) = {num_w_blocks}')
    print(f'  Elements per block = 5*5*C = {5*5*workload.C}')
    print(f'  Rows per block = ceil({5*5*workload.C}/{row_size}) = {math.ceil(5*5*workload.C/row_size)}')
    print(f'  Trace: unique_rows={result2["unique_rows"]}, row_switches={result2["row_switches"]}')
    print()
    
    # Case 3: Even smaller blocks
    print('Case 3: Very small blocks (more row switches)')
    config3 = MappingConfig(
        P_l3=8, Q_l3=8, C_l3=4, K_l3=4,
        permutation=(DIM_P, DIM_Q, DIM_C, DIM_K),
        input_layout='row_aligned',
        weight_layout='sequential',
        output_layout='sequential',
        block_h=2, block_w=5
    )
    result3 = analyze_input_pattern(workload, config3, dram_config)
    num_h_blocks = math.ceil(H / 2)
    num_w_blocks = math.ceil(W / 5)
    print(f'  Config: block_h=2, block_w=5, P_l3=8, Q_l3=8, C_l3=4, K_l3=4')
    print(f'  num_h_blocks = ceil({H}/2) = {num_h_blocks}')
    print(f'  num_w_blocks = ceil({W}/5) = {num_w_blocks}')
    print(f'  Total spatial blocks = {num_h_blocks * num_w_blocks}')
    print(f'  Elements per spatial block = 2*5 = 10')
    print(f'  Trace: unique_rows={result3["unique_rows"]}, row_switches={result3["row_switches"]}')
    
    # Print row pattern
    rows = result3['rows']
    print()
    print('Row sequence pattern (first 100):')
    for i in range(0, min(100, len(rows)), 10):
        print(f'  [{i:3d}-{i+9:3d}]: {rows[i:i+10]}')
    
    print()
    print('=' * 80)
    print('Key observations:')
    print('=' * 80)
    print()
    print('1. For row_aligned layout, each (h_block, w_block, c) combination has a unique')
    print('   row-aligned address region.')
    print()
    print('2. Row switches happen when:')
    print('   - Moving to a different (h_block, w_block) combination')
    print('   - Moving to a different C index (if C_l3 > 1)')
    print('   - Within a large enough region that spans multiple rows')
    print()
    print('3. K dimension does NOT cause input row switches (input doesnt depend on K)')
    print()
    
    # Analyze switch pattern
    switch_analysis(workload, dram_config, result3['rows'])


def switch_analysis(workload, dram_config, rows):
    """Analyze when row switches occur."""
    print('=' * 80)
    print('Switch analysis:')
    print('=' * 80)
    print()
    
    switches = []
    for i in range(1, len(rows)):
        if rows[i] != rows[i-1]:
            switches.append({
                'position': i,
                'from_row': rows[i-1],
                'to_row': rows[i]
            })
    
    print(f'Total switches: {len(switches)}')
    print()
    print('First 30 switches:')
    for s in switches[:30]:
        print(f'  Position {s["position"]:5d}: row {s["from_row"]:3d} -> {s["to_row"]:3d}')


if __name__ == '__main__':
    main()
