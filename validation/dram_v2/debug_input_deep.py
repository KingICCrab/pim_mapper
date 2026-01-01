#!/usr/bin/env python3
"""
深入分析 Input row switches 的规律。

关键发现：
1. row_switches >> unique_rows，说明同一个 row 被多次访问
2. K_l3 影响不大（因为 K 对 Input 无影响，Input 被多次重复访问）
3. P_l3, Q_l3 影响显著：减小 tiling 会减少 row_switches

让我们分析实际的访问模式来推导公式。
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from validation.dram.trace_generator import TraceGenerator, DRAMConfig
from validation.dram_v2.experiments.mapping_sweep import SimpleMapping
from validation.dram_v2.core.mapping_space import MappingConfig, DIM_P, DIM_Q, DIM_C, DIM_K
from src.pim_optimizer.workload.conv import ConvWorkload


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


def analyze_access_count(workload, config: MappingConfig, dram_config):
    """Analyze access count per row."""
    
    mapping = SimpleMapping(config, workload)
    generator = TraceGenerator(dram_config)
    trace = generator.generate_trace(mapping, workload)
    
    row_buffer_bytes = dram_config.row_buffer_bytes
    bank_size = row_buffer_bytes * dram_config.num_rows
    
    # Count accesses per row
    row_access_count = {}
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
                row_access_count[row] = row_access_count.get(row, 0) + 1
    
    row_switches = sum(1 for i in range(1, len(rows)) if rows[i] != rows[i-1])
    unique_rows = len(set(rows))
    total_accesses = len(rows)
    
    return {
        'rows': rows,
        'row_switches': row_switches,
        'unique_rows': unique_rows,
        'total_accesses': total_accesses,
        'row_access_count': row_access_count
    }


def main():
    print("="*80)
    print("Input Row Switches: Deep Analysis")
    print("="*80)
    
    dram_config = DRAMConfig(row_buffer_bytes=1024, num_banks=4, num_rows=16384)
    row_buffer_bytes = dram_config.row_buffer_bytes
    
    P, Q, C, K, R, S = 8, 8, 4, 4, 3, 3
    N = 1
    workload = ConvWorkload(name='test', R=R, S=S, P=P, Q=Q, C=C, K=K, N=N)
    H = P + R - 1  # 10
    W = Q + S - 1  # 10
    
    print(f"\nWorkload: P={P}, Q={Q}, C={C}, K={K}, R={R}, S={S}")
    print(f"Input: H={H}, W={W}, C={C}")
    print(f"Total Input elements: {H * W * C} = {H}×{W}×{C}")
    
    # Test different configurations
    configs = [
        {'P_l3': 8, 'Q_l3': 8, 'C_l3': 4, 'K_l3': 4, 'block_h': 5, 'block_w': 5},
        {'P_l3': 4, 'Q_l3': 4, 'C_l3': 4, 'K_l3': 4, 'block_h': 5, 'block_w': 5},
        {'P_l3': 2, 'Q_l3': 2, 'C_l3': 4, 'K_l3': 4, 'block_h': 5, 'block_w': 5},
        {'P_l3': 8, 'Q_l3': 8, 'C_l3': 4, 'K_l3': 4, 'block_h': 10, 'block_w': 10},
    ]
    
    print("\n" + "-"*80)
    print("Configuration Analysis")
    print("-"*80)
    
    for cfg in configs:
        P_l3, Q_l3, C_l3, K_l3 = cfg['P_l3'], cfg['Q_l3'], cfg['C_l3'], cfg['K_l3']
        block_h, block_w = cfg['block_h'], cfg['block_w']
        
        config = create_mapping_config(P_l3, Q_l3, C_l3, K_l3, block_h, block_w)
        result = analyze_access_count(workload, config, dram_config)
        
        # Analyze
        total = result['total_accesses']
        unique = result['unique_rows']
        switches = result['row_switches']
        
        # Avg accesses per row
        avg_per_row = total / unique if unique > 0 else 0
        
        # Number of L3 tiles
        num_P_tiles = (P + P_l3 - 1) // P_l3
        num_Q_tiles = (Q + Q_l3 - 1) // Q_l3
        num_C_tiles = (C + C_l3 - 1) // C_l3
        num_K_tiles = (K + K_l3 - 1) // K_l3
        
        # Input tile size (H_tile, W_tile depends on P_l3, Q_l3, R, S)
        H_tile = P_l3 + R - 1  # With sliding window
        W_tile = Q_l3 + S - 1
        
        # Theoretical analysis:
        # Total input accesses = P * Q * C * R * S * K (with reuse)
        # No, that's if no buffering. With tiling:
        # Each L3 tile (p_tile, q_tile, c_tile, k_tile) accesses:
        #   H_tile * W_tile * C_l3 Input elements
        # Total tiles = num_P * num_Q * num_C * num_K
        # But K doesn't change Input access (same Input for all K tiles)
        # So: total accesses = num_K * num_C * num_P * num_Q * H_tile * W_tile * C_l3
        
        expected_accesses = num_K_tiles * num_C_tiles * num_P_tiles * num_Q_tiles * H_tile * W_tile * C_l3
        
        # Number of spatial blocks
        num_h_blocks = (H + block_h - 1) // block_h
        num_w_blocks = (W + block_w - 1) // block_w
        
        print(f"\nConfig: P_l3={P_l3}, Q_l3={Q_l3}, C_l3={C_l3}, K_l3={K_l3}, block={block_h}x{block_w}")
        print(f"  L3 tiles: P={num_P_tiles}, Q={num_Q_tiles}, C={num_C_tiles}, K={num_K_tiles}")
        print(f"  Input tile: {H_tile}×{W_tile}×{C_l3}")
        print(f"  Expected accesses: {expected_accesses}, Actual: {total}")
        print(f"  Unique rows: {unique}, Row switches: {switches}")
        print(f"  Avg accesses per row: {avg_per_row:.1f}")
        print(f"  Spatial blocks: {num_h_blocks}×{num_w_blocks} = {num_h_blocks * num_w_blocks}")
        
        # Row access distribution
        counts = list(result['row_access_count'].values())
        print(f"  Row access distribution: min={min(counts)}, max={max(counts)}, avg={sum(counts)/len(counts):.1f}")
    
    # Now let's understand the pattern
    print("\n" + "="*80)
    print("Deriving Formula")
    print("="*80)
    
    # Key insight:
    # With row_aligned layout, each (h_block, w_block, c) combination maps to a unique row region.
    # The loop structure is:
    #   for k_tile in K_l3:      # Outer
    #     for c_tile in C_l3:
    #       for p_tile in P_l3:
    #         for q_tile in Q_l3:  # Inner
    #           load input[c_tile][p_tile*stride..p_tile*stride+H_tile][q_tile*stride..q_tile*stride+W_tile]
    #
    # For each (k_tile, c_tile, p_tile, q_tile) iteration:
    # - Access a region of Input with shape (H_tile, W_tile) for c_tile channel(s)
    # - This region may span multiple (h_block, w_block) combinations
    #
    # Row switches happen when:
    # 1. Moving to a different (h_block, w_block) within a tile -> fixed for each tile
    # 2. Moving between tiles (p_tile or q_tile changes) -> may return to previously accessed blocks
    # 3. Moving between c_tile -> different rows (different C region)
    #
    # Key formula insight:
    # Within each K iteration (K doesn't affect Input), we iterate over C * P * Q tiles.
    # For each tile, we access some rows and cause some switches.
    # 
    # With row_aligned layout:
    # - Each (h_block, w_block, c_l3_idx) combination is padded to row boundary
    # - So moving between different combinations always causes a row switch
    #
    # Total row switches ≈ (number of unique (h_block, w_block, c) visits) * K iterations - 1
    #                    + (number of L3 tile boundaries crossed)
    
    print("\nFormula derivation:")
    print("-" * 40)
    
    for cfg in configs:
        P_l3, Q_l3, C_l3, K_l3 = cfg['P_l3'], cfg['Q_l3'], cfg['C_l3'], cfg['K_l3']
        block_h, block_w = cfg['block_h'], cfg['block_w']
        
        config = create_mapping_config(P_l3, Q_l3, C_l3, K_l3, block_h, block_w)
        result = analyze_access_count(workload, config, dram_config)
        
        num_P_tiles = (P + P_l3 - 1) // P_l3
        num_Q_tiles = (Q + Q_l3 - 1) // Q_l3
        num_C_tiles = (C + C_l3 - 1) // C_l3
        num_K_tiles = (K + K_l3 - 1) // K_l3
        
        num_h_blocks = (H + block_h - 1) // block_h
        num_w_blocks = (W + block_w - 1) // block_w
        
        # Input tile coverage
        H_tile = P_l3 + R - 1
        W_tile = Q_l3 + S - 1
        
        # Number of h_blocks and w_blocks covered by one Input tile
        h_blocks_per_tile = (H_tile + block_h - 1) // block_h
        w_blocks_per_tile = (W_tile + block_w - 1) // block_w
        blocks_per_tile = h_blocks_per_tile * w_blocks_per_tile
        
        # Total L3 iterations (ignoring K which doesn't affect Input data)
        # But K DOES affect traversal - we repeat Input loading for each K
        total_L3_iters = num_K_tiles * num_C_tiles * num_P_tiles * num_Q_tiles
        
        # Approximate: each L3 iteration visits blocks_per_tile * C_l3 block-channel combinations
        # Each such combination is in a separate row-aligned region
        # So row switches per L3 iter ≈ blocks_per_tile * C_l3 - 1 (for transitions within iter)
        # Plus 1 for transition between iterations (unless sequential in memory)
        
        # Simplified formula:
        # row_switches ≈ (blocks_per_tile * C_l3) * total_L3_iters - 1
        
        formula_v1 = (blocks_per_tile * C_l3) * total_L3_iters - 1
        
        # Alternative: unique_rows * num_times_accessed - 1
        unique = result['unique_rows']
        total_accesses = result['total_accesses']
        num_visits = total_accesses // (block_h * block_w) if block_h * block_w > 0 else 1
        formula_v2 = unique * (total_L3_iters // num_C_tiles) - 1  # Each unique row visited once per (P,Q,K) iteration
        
        actual = result['row_switches']
        
        print(f"\nConfig: P{P_l3}Q{Q_l3}C{C_l3}K{K_l3} blk{block_h}x{block_w}")
        print(f"  blocks_per_tile={blocks_per_tile}, C_l3={C_l3}")
        print(f"  total_L3_iters={total_L3_iters}")
        print(f"  Formula v1: {formula_v1}, Actual: {actual}, Error: {abs(formula_v1-actual)/max(1,actual)*100:.1f}%")
        print(f"  Formula v2: {formula_v2}, Actual: {actual}, Error: {abs(formula_v2-actual)/max(1,actual)*100:.1f}%")


if __name__ == '__main__':
    main()
