#!/usr/bin/env python3
"""
Deep analysis: Why does trace have row_switches when block >= input size?

When block_h=6, block_w=6 and H=6, W=6:
- The entire input fits in ONE block
- All C channels are in a single block
- Total elements = H × W × C = 6 × 6 × 4 = 144 elements
- Row size = 1024 elements
- So everything should fit in ONE row!

Why then does trace show row_switches > 0?
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

import math
from validation.dram.trace_generator import TraceGenerator, DRAMConfig
from validation.dram_v2.experiments.mapping_sweep import SimpleMapping
from validation.dram_v2.core.mapping_space import MappingConfig, DIM_P, DIM_Q, DIM_C, DIM_K
from src.pim_optimizer.workload.conv import ConvWorkload


def analyze_detailed(workload, config, dram_config):
    """Detailed analysis of input access pattern."""
    mapping = SimpleMapping(config, workload)
    generator = TraceGenerator(dram_config)
    trace = generator.generate_trace(mapping, workload)
    
    row_size = dram_config.row_buffer_bytes
    bank_size = row_size * dram_config.num_rows
    
    input_accesses = []
    for line in trace:
        parts = line.split()
        if len(parts) >= 2:
            addr = int(parts[1], 16)
            bank = addr // bank_size
            if bank == 0:  # Input bank
                bank_addr = addr % bank_size
                row = bank_addr // row_size
                col = bank_addr % row_size
                input_accesses.append({
                    'addr': addr,
                    'bank_addr': bank_addr,
                    'row': row,
                    'col': col
                })
    
    # Print first 50 accesses
    print(f"First 50 input accesses:")
    print(f"{'Idx':>5} {'Addr':>12} {'BankAddr':>10} {'Row':>5} {'Col':>5}")
    print("-" * 45)
    
    prev_row = None
    for i, acc in enumerate(input_accesses[:50]):
        switch = " <-- SWITCH" if prev_row is not None and acc['row'] != prev_row else ""
        print(f"{i:>5} 0x{acc['addr']:08x} {acc['bank_addr']:>10} {acc['row']:>5} {acc['col']:>5}{switch}")
        prev_row = acc['row']
    
    # Count row switches
    row_switches = sum(1 for i in range(1, len(input_accesses)) 
                       if input_accesses[i]['row'] != input_accesses[i-1]['row'])
    
    # Row distribution
    row_counts = {}
    for acc in input_accesses:
        row_counts[acc['row']] = row_counts.get(acc['row'], 0) + 1
    
    print(f"\nRow distribution:")
    for row in sorted(row_counts.keys()):
        print(f"  Row {row}: {row_counts[row]} accesses")
    
    print(f"\nTotal: {len(input_accesses)} accesses, {row_switches} row switches")
    print(f"Unique rows: {len(row_counts)}")
    
    return row_switches, len(row_counts)


def main():
    print("=" * 80)
    print("Deep Analysis: Row Switches with Full Block Coverage")
    print("=" * 80)
    
    workload = ConvWorkload(name='test', R=3, S=3, P=4, Q=4, C=4, K=4, N=1)
    H, W = workload.input_size['H'], workload.input_size['W']
    dram_config = DRAMConfig(row_buffer_bytes=1024, num_banks=4, num_rows=16384, element_size=1)
    
    print(f"\nWorkload: P={workload.P}, Q={workload.Q}, C={workload.C}, K={workload.K}")
    print(f"Input size: H={H}, W={W}")
    print(f"Total input elements: {H * W * workload.C}")
    print(f"Row buffer: {dram_config.row_buffer_bytes} elements")
    print()
    
    # Test case: P2Q2C2K2 with full block
    print("=" * 80)
    print("Test Case: P2Q2C2K2 with block 6x6 (covers full input)")
    print("=" * 80)
    
    config = MappingConfig(
        P_l3=2, Q_l3=2, C_l3=2, K_l3=2,
        permutation=(DIM_P, DIM_Q, DIM_C, DIM_K),
        input_layout='row_aligned',
        weight_layout='sequential',
        output_layout='sequential',
        block_h=6, block_w=6
    )
    
    print(f"\nConfig: P_l3={config.P_l3}, Q_l3={config.Q_l3}, C_l3={config.C_l3}, K_l3={config.K_l3}")
    print(f"Block: {config.block_h} x {config.block_w}")
    print()
    
    analyze_detailed(workload, config, dram_config)
    
    # Analysis
    print()
    print("=" * 80)
    print("Expected vs Actual")
    print("=" * 80)
    print(f"""
Expected (ILP model):
- block_h=6 >= H=6, block_w=6 >= W=6 → NO block crossing
- All input data in one block: 6×6×4 = 144 elements
- 144 elements < 1024 row size → fits in ONE row
- Expected row_switches = 0 (ignoring reuse)
- With K reuse: K_l3=2, but K doesn't cause new rows for Input

Actual (Trace):
- row_switches > 0

WHY?
- The ILP formula counts "row activations" not "row switches"
- row_acts_aligned = P_l3×Q_l3×C_l3×K_l3 = tile visits
- Each "tile visit" might hit the same row (if in row buffer)
- But if accessing OTHER tensors evicts row buffer, need re-activation

The trace generator is correct:
- It shows actual memory access patterns
- Row switches happen when accessing different C channels
- Each C_tile accesses different address ranges

The ILP model assumption:
- "row_acts_aligned" = upper bound on row activations
- Assumes worst case: every tile access = new row activation
""")


if __name__ == "__main__":
    main()
