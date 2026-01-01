#!/usr/bin/env python3
"""
Detailed analysis of Input row switches to derive correct formula.

Key insight from trace_generator.py:
- For row_aligned layout, addr = input_base + block_base + l2_tile_base + offset_in_block
- block_base = h_block * stride_p_l3 + w_block * stride_q_l3 + c_l3_idx * stride_c_l3 + n_l3_idx * stride_n_l3
- L3 strides are row_aligned (multiples of row_buffer_bytes)
- The loop order is: for h_block: for w_block: for h_in_block: for w_in_block
- This means we iterate by spatial blocks, and within each block, iterate h then w

Key question: When do row switches happen?
1. Moving to different (h_block, w_block) -> always causes row switch (different L3 region)
2. Moving to different c_l3_idx -> always causes row switch
3. Within a block, if block elements span multiple rows, that causes row switches
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from validation.dram.trace_generator import TraceGenerator, DRAMConfig
from validation.dram_v2.experiments.mapping_sweep import SimpleMapping
from validation.dram_v2.core.mapping_space import MappingConfig, DIM_P, DIM_Q, DIM_C, DIM_K
from src.pim_optimizer.workload.conv import ConvWorkload


def analyze_input_row_switches(workload, config: MappingConfig, dram_config):
    """Analyze when Input row switches happen."""
    
    # Generate trace
    mapping = SimpleMapping(config, workload)
    generator = TraceGenerator(dram_config)
    trace = generator.generate_trace(mapping, workload)
    
    row_buffer_bytes = dram_config.row_buffer_bytes
    bank_size = row_buffer_bytes * dram_config.num_rows
    
    # Extract row sequence for Input (bank 0)
    rows = []
    for line in trace:
        parts = line.split()
        if len(parts) >= 2 and parts[0] == 'LD':
            addr = int(parts[1], 16)
            bank = addr // bank_size
            if bank == 0:  # Input bank
                bank_addr = addr % bank_size
                row = bank_addr // row_buffer_bytes
                rows.append(row)
    
    # Count row switches
    row_switches = sum(1 for i in range(1, len(rows)) if rows[i] != rows[i-1])
    unique_rows = len(set(rows))
    
    return rows, row_switches, unique_rows


def create_mapping_config(P_l3, Q_l3, C_l3, K_l3, block_h, block_w, 
                          input_layout='row_aligned'):
    """Helper to create MappingConfig"""
    return MappingConfig(
        P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3,
        permutation=(DIM_Q, DIM_P, DIM_C, DIM_K),  # Inner to outer
        input_layout=input_layout,
        weight_layout='row_aligned',
        output_layout='row_aligned',
        block_h=block_h,
        block_w=block_w
    )


def derive_formula_components():
    """Analyze what factors determine row switches."""
    
    print("="*80)
    print("Factor Analysis: What determines Input row switches?")
    print("="*80)
    
    dram_config = DRAMConfig(row_buffer_bytes=1024, num_banks=4, num_rows=16384)
    
    # Fixed workload
    P, Q, C, K, R, S = 8, 8, 4, 4, 3, 3
    N = 1
    workload = ConvWorkload(name='test', R=R, S=S, P=P, Q=Q, C=C, K=K, N=N)
    
    H = P + R - 1
    W = Q + S - 1
    
    print(f"\nWorkload: P={P}, Q={Q}, C={C}, K={K}, R={R}, S={S}")
    print(f"Input: H={H}, W={W}, C={C}")
    print(f"row_buffer_bytes = {dram_config.row_buffer_bytes}")
    
    # Test 1: Effect of block_h, block_w
    print("\n" + "-"*60)
    print("Test 1: Effect of block_h, block_w on row switches")
    print("-"*60)
    
    for block_h, block_w in [(2, 2), (5, 5), (10, 10), (H, W)]:
        config = create_mapping_config(8, 8, 4, 4, block_h, block_w)
        
        rows, switches, unique = analyze_input_row_switches(workload, config, dram_config)
        
        num_h_blocks = (H + block_h - 1) // block_h
        num_w_blocks = (W + block_w - 1) // block_w
        total_blocks = num_h_blocks * num_w_blocks
        elements_per_block = block_h * block_w
        
        print(f"\nblock_h={block_h:2}, block_w={block_w:2}")
        print(f"  num_h_blocks={num_h_blocks}, num_w_blocks={num_w_blocks}, total_blocks={total_blocks}")
        print(f"  elements_per_block={elements_per_block}")
        print(f"  unique_rows={unique}, row_switches={switches}")
    
    # Test 2: Effect of tiling
    print("\n" + "-"*60)
    print("Test 2: Effect of P_l3, Q_l3 tiling (with fixed block)")
    print("-"*60)
    
    block_h, block_w = 5, 5
    
    for P_l3, Q_l3 in [(8, 8), (4, 4), (2, 4), (4, 2), (2, 2)]:
        config = create_mapping_config(P_l3, Q_l3, 4, 4, block_h, block_w)
        
        rows, switches, unique = analyze_input_row_switches(workload, config, dram_config)
        
        print(f"\nP_l3={P_l3}, Q_l3={Q_l3}")
        print(f"  unique_rows={unique}, row_switches={switches}")
    
    # Test 3: Effect of C_l3 tiling  
    print("\n" + "-"*60)
    print("Test 3: Effect of C_l3 tiling (C iterates outer loop)")
    print("-"*60)
    
    for C_l3 in [4, 2, 1]:
        config = create_mapping_config(8, 8, C_l3, 4, 5, 5)
        
        rows, switches, unique = analyze_input_row_switches(workload, config, dram_config)
        
        print(f"\nC_l3={C_l3}")
        print(f"  unique_rows={unique}, row_switches={switches}")


def analyze_access_pattern():
    """Look at actual access pattern to understand row switch pattern."""
    
    print("\n" + "="*80)
    print("Access Pattern Analysis")
    print("="*80)
    
    dram_config = DRAMConfig(row_buffer_bytes=1024, num_banks=4, num_rows=16384)
    
    P, Q, C, K, R, S = 8, 8, 4, 4, 3, 3
    N = 1
    workload = ConvWorkload(name='test', R=R, S=S, P=P, Q=Q, C=C, K=K, N=N)
    H = P + R - 1  # 10
    W = Q + S - 1  # 10
    
    # Use small block to see block-level pattern
    block_h, block_w = 5, 5
    
    config = create_mapping_config(4, 4, 2, 4, block_h, block_w)
    
    print(f"\nConfig: P_l3=4, Q_l3=4, C_l3=2, block={block_h}x{block_w}")
    print(f"H={H}, W={W}")
    
    # Generate trace
    mapping = SimpleMapping(config, workload)
    generator = TraceGenerator(dram_config)
    trace = generator.generate_trace(mapping, workload)
    
    row_buffer_bytes = dram_config.row_buffer_bytes
    bank_size = row_buffer_bytes * dram_config.num_rows
    
    prev_row = None
    switch_count = 0
    
    print("\nFirst 50 Input accesses (with row info):")
    input_count = 0
    for line in trace:
        parts = line.split()
        if len(parts) >= 2 and parts[0] == 'LD':
            addr = int(parts[1], 16)
            bank = addr // bank_size
            if bank == 0:  # Input bank
                if input_count >= 50:
                    break
                    
                bank_addr = addr % bank_size
                row = bank_addr // row_buffer_bytes
                col = bank_addr % row_buffer_bytes
                
                is_switch = (prev_row is not None and row != prev_row)
                if is_switch:
                    switch_count += 1
                
                marker = " *SWITCH*" if is_switch else ""
                print(f"  [{input_count:3}] addr=0x{addr:08X} row={row:4} col={col:4}{marker}")
                
                prev_row = row
                input_count += 1
    
    print(f"\nSwitches in first 50: {switch_count}")
    
    # Get total stats
    rows, total_switches, unique = analyze_input_row_switches(workload, config, dram_config)
    print(f"\nTotal: unique_rows={unique}, row_switches={total_switches}")
    
    # Analyze switch positions
    switch_positions = [i for i in range(1, len(rows)) if rows[i] != rows[i-1]]
    print(f"\nFirst 30 switch positions: {switch_positions[:30]}")
    
    # Look for pattern
    if len(switch_positions) > 1:
        gaps = [switch_positions[i+1] - switch_positions[i] for i in range(min(30, len(switch_positions)-1))]
        print(f"Gaps between switches: {gaps}")


def derive_correct_formula():
    """Try to derive the correct formula based on analysis."""
    
    print("\n" + "="*80)
    print("Deriving Correct Formula")
    print("="*80)
    
    dram_config = DRAMConfig(row_buffer_bytes=1024, num_banks=4, num_rows=16384)
    row_buffer_bytes = dram_config.row_buffer_bytes
    
    P, Q, C, K, R, S = 8, 8, 4, 4, 3, 3
    N = 1
    workload = ConvWorkload(name='test', R=R, S=S, P=P, Q=Q, C=C, K=K, N=N)
    H = P + R - 1  # 10
    W = Q + S - 1  # 10
    
    print(f"\nWorkload: P={P}, Q={Q}, C={C}, K={K}")
    print(f"Input: H={H}, W={W}, C={C}")
    
    configs = [
        {'P_l3': 8, 'Q_l3': 8, 'C_l3': 4, 'K_l3': 4, 'block_h': 5, 'block_w': 5},
        {'P_l3': 4, 'Q_l3': 4, 'C_l3': 2, 'K_l3': 4, 'block_h': 5, 'block_w': 5},
        {'P_l3': 2, 'Q_l3': 2, 'C_l3': 2, 'K_l3': 2, 'block_h': 5, 'block_w': 5},
        {'P_l3': 8, 'Q_l3': 8, 'C_l3': 4, 'K_l3': 4, 'block_h': 10, 'block_w': 10},
        {'P_l3': 8, 'Q_l3': 8, 'C_l3': 4, 'K_l3': 4, 'block_h': 2, 'block_w': 2},
        {'P_l3': 4, 'Q_l3': 8, 'C_l3': 4, 'K_l3': 4, 'block_h': 5, 'block_w': 5},  # Asymmetric tiling
    ]
    
    print("\n" + "-"*80)
    print(f"{'Config':<30} {'Trace':>8} {'Formula':>8} {'Error':>8} {'Note'}")
    print("-"*80)
    
    for cfg in configs:
        P_l3, Q_l3, C_l3, K_l3 = cfg['P_l3'], cfg['Q_l3'], cfg['C_l3'], cfg['K_l3']
        block_h, block_w = cfg['block_h'], cfg['block_w']
        
        config = create_mapping_config(P_l3, Q_l3, C_l3, K_l3, block_h, block_w)
        
        rows, trace_switches, unique = analyze_input_row_switches(workload, config, dram_config)
        
        # Number of L3 DRAM tiles
        num_P_l3 = (P + P_l3 - 1) // P_l3
        num_Q_l3 = (Q + Q_l3 - 1) // Q_l3
        num_C_l3 = (C + C_l3 - 1) // C_l3
        num_K_l3 = (K + K_l3 - 1) // K_l3
        
        # Number of spatial blocks in data layout
        num_h_blocks = (H + block_h - 1) // block_h
        num_w_blocks = (W + block_w - 1) // block_w
        total_spatial_blocks = num_h_blocks * num_w_blocks
        
        # Elements per spatial block
        elements_per_block = block_h * block_w
        
        # Rows per block (if block doesn't fit in one row)
        rows_per_block = (elements_per_block + row_buffer_bytes - 1) // row_buffer_bytes
        
        # Key insight: K iterations cause us to revisit Input data
        # Each K_l3 tile needs to load Input again
        # 
        # Formula hypothesis:
        # - unique_rows = total_spatial_blocks * C * rows_per_block (approx)
        # - Each K_l3 iteration accesses all Input data
        # - row_switches = unique_rows * num_K_l3 - 1 (approximate, doesn't account for sequential access)
        
        # More refined formula:
        # Within each K iteration, we iterate over P, Q, C tiles
        # For each (p_tile, q_tile, c_tile), we access certain h_blocks and w_blocks
        # Due to row_aligned layout, each L3 tile access starts at a row boundary
        # So row_switches ≈ num_L3_tile_accesses - 1
        
        # Total L3 tile accesses for Input = sum over all tiles of 1
        # But this is complicated by which tiles need which Input regions
        
        # Simpler empirical approach:
        # row_switches ≈ unique_rows * num_K_l3 - 1
        formula_v1 = unique * num_K_l3 - 1
        
        # Another approach: Each L3 iteration over (K, C, P, Q) accesses Input
        # But Input only depends on (C, P, Q), not K
        # So we have num_K_l3 iterations, each accessing the same Input pattern
        # row_switches per K iter = unique - 1
        # Total = (unique - 1) * num_K_l3 + (num_K_l3 - 1)
        formula_v2 = (unique - 1) * num_K_l3 + (num_K_l3 - 1)
        
        # Yet another: row_switches = (unique - 1) * num_K_l3
        formula_v3 = (unique - 1) * num_K_l3
        
        # Try formula_v1 for now
        formula_val = formula_v1
        
        error = abs(trace_switches - formula_val) / max(1, trace_switches) * 100
        
        cfg_str = f"P{P_l3}Q{Q_l3}C{C_l3}K{K_l3} blk{block_h}x{block_w}"
        note = f"unique={unique}, K_l3={num_K_l3}"
        print(f"{cfg_str:<30} {trace_switches:>8} {formula_val:>8} {error:>7.1f}%  {note}")


if __name__ == '__main__':
    derive_formula_components()
    analyze_access_pattern()
    derive_correct_formula()
