#!/usr/bin/env python3
"""
最终精确公式：基于 trace_generator 的实际迭代模式

实际迭代顺序 (从 trace_generator.py):
for n_local:
  for c_local:
    for h_block:
      for w_block:
        for h in range(h_lo, h_hi):
          for w in range(w_lo, w_hi):
            access[h, w, c]

这意味着:
1. 在一个 (h_block, w_block, c) 组合内的元素是连续访问的 -> 0 row switch
2. 切换 w_block (同一 h_block 和 c) -> 1 row switch
3. 切换 h_block -> 1 row switch  
4. 切换 c_local -> 1 row switch

每个 L3 tile 访问的 block 数量和 channel 数量决定了 row switches。
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from validation.dram.trace_generator import TraceGenerator, DRAMConfig
from validation.dram_v2.experiments.mapping_sweep import SimpleMapping
from validation.dram_v2.core.mapping_space import MappingConfig, DIM_P, DIM_Q, DIM_C, DIM_K
from src.pim_optimizer.workload.conv import ConvWorkload


def create_mapping_config(P_l3, Q_l3, C_l3, K_l3, block_h, block_w):
    return MappingConfig(
        P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3,
        permutation=(DIM_Q, DIM_P, DIM_C, DIM_K),
        input_layout='row_aligned',
        weight_layout='row_aligned',
        output_layout='row_aligned',
        block_h=block_h,
        block_w=block_w
    )


def get_trace_stats(workload, config: MappingConfig, dram_config):
    """Get row switch stats from trace."""
    mapping = SimpleMapping(config, workload)
    generator = TraceGenerator(dram_config)
    trace = generator.generate_trace(mapping, workload)
    
    row_buffer_bytes = dram_config.row_buffer_bytes
    bank_size = row_buffer_bytes * dram_config.num_rows
    
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
    
    row_switches = sum(1 for i in range(1, len(rows)) if rows[i] != rows[i-1])
    return row_switches, len(rows)


def compute_formula(P, Q, C, K, R, S, P_l3, Q_l3, C_l3, K_l3, block_h, block_w):
    """Compute row switches using derived formula."""
    
    H = P + R - 1
    W = Q + S - 1
    
    # Number of L3 tiles
    num_P_tiles = (P + P_l3 - 1) // P_l3
    num_Q_tiles = (Q + Q_l3 - 1) // Q_l3
    num_C_tiles = (C + C_l3 - 1) // C_l3
    num_K_tiles = (K + K_l3 - 1) // K_l3
    
    # For each L3 tile (p_tile, q_tile, c_tile, k_tile):
    # - We access Input region: [c_tile*C_l3 : (c_tile+1)*C_l3]
    #                           [h_start : h_end]
    #                           [w_start : w_end]
    # where h_start = p_tile * P_l3 (as output p, mapped to input h)
    #       h_end = h_start + H_tile, H_tile = P_l3 + R - 1
    
    H_tile = P_l3 + R - 1
    W_tile = Q_l3 + S - 1
    
    # Number of blocks covered by each L3 tile
    h_blocks_per_tile = (H_tile + block_h - 1) // block_h
    w_blocks_per_tile = (W_tile + block_w - 1) // block_w
    blocks_per_tile = h_blocks_per_tile * w_blocks_per_tile
    
    # Iteration order within each L3 tile:
    # for c_local in C_l3:
    #   for h_block in h_blocks_per_tile:
    #     for w_block in w_blocks_per_tile:
    #       <access elements>
    #
    # Row switches within one L3 tile:
    # - Every time we change (c_local, h_block, w_block), we switch row
    # - Except the very first one
    # - Total = C_l3 * blocks_per_tile - 1
    
    switches_within_tile = C_l3 * blocks_per_tile - 1
    
    # Total L3 tiles: num_K * num_C * num_P * num_Q
    # For Input, K doesn't affect data but affects iteration count
    # L3 loop order: K (outer) -> C -> P -> Q (inner)
    total_L3_tiles = num_K_tiles * num_C_tiles * num_P_tiles * num_Q_tiles
    
    # Between L3 tiles:
    # When we move from tile (k, c, p, q) to next tile, we switch row
    # (unless by chance they access the same row, which is rare with row_aligned)
    between_tiles = total_L3_tiles - 1
    
    # Total = switches_within_tile * total_L3_tiles + between_tiles
    total_switches = switches_within_tile * total_L3_tiles + between_tiles
    
    return total_switches, {
        'H_tile': H_tile,
        'W_tile': W_tile,
        'blocks_per_tile': blocks_per_tile,
        'h_blocks_per_tile': h_blocks_per_tile,
        'w_blocks_per_tile': w_blocks_per_tile,
        'switches_within_tile': switches_within_tile,
        'total_L3_tiles': total_L3_tiles,
        'between_tiles': between_tiles
    }


def main():
    print("="*80)
    print("Input Row Switches: Precise Formula Based on Trace Generator Logic")
    print("="*80)
    
    dram_config = DRAMConfig(row_buffer_bytes=1024, num_banks=4, num_rows=16384)
    
    P, Q, C, K, R, S = 8, 8, 4, 4, 3, 3
    N = 1
    workload = ConvWorkload(name='test', R=R, S=S, P=P, Q=Q, C=C, K=K, N=N)
    H = P + R - 1
    W = Q + S - 1
    
    print(f"\nWorkload: P={P}, Q={Q}, C={C}, K={K}, R={R}, S={S}")
    print(f"Input: H={H}, W={W}")
    
    configs = [
        # (P_l3, Q_l3, C_l3, K_l3, block_h, block_w)
        (8, 8, 4, 4, 5, 5),
        (4, 4, 4, 4, 5, 5),
        (2, 2, 4, 4, 5, 5),
        (8, 8, 4, 4, 10, 10),
        (8, 8, 4, 4, 2, 2),
        (4, 4, 2, 4, 5, 5),
        (2, 2, 2, 2, 5, 5),
        (1, 1, 1, 1, 5, 5),
        (8, 8, 1, 4, 5, 5),  # C_l3=1
    ]
    
    print("\n" + "-"*90)
    print(f"{'Config':<25} {'Trace':>8} {'Formula':>8} {'Error':>8}  {'Details'}")
    print("-"*90)
    
    for P_l3, Q_l3, C_l3, K_l3, block_h, block_w in configs:
        config = create_mapping_config(P_l3, Q_l3, C_l3, K_l3, block_h, block_w)
        
        trace_switches, total_accesses = get_trace_stats(workload, config, dram_config)
        formula_val, details = compute_formula(P, Q, C, K, R, S, P_l3, Q_l3, C_l3, K_l3, block_h, block_w)
        
        error = abs(trace_switches - formula_val) / max(1, trace_switches) * 100
        
        cfg_str = f"P{P_l3}Q{Q_l3}C{C_l3}K{K_l3} blk{block_h}x{block_w}"
        detail_str = f"L3={details['total_L3_tiles']}, blks={details['blocks_per_tile']}, sw/tile={details['switches_within_tile']}"
        
        print(f"{cfg_str:<25} {trace_switches:>8} {formula_val:>8} {error:>7.1f}%  {detail_str}")
    
    # Detailed analysis for failing case
    print("\n" + "="*80)
    print("Detailed Analysis for Failing Cases")
    print("="*80)
    
    # P8Q8C4K4 blk5x5 is far off
    P_l3, Q_l3, C_l3, K_l3, block_h, block_w = 8, 8, 4, 4, 5, 5
    
    config = create_mapping_config(P_l3, Q_l3, C_l3, K_l3, block_h, block_w)
    trace_switches, total_accesses = get_trace_stats(workload, config, dram_config)
    formula_val, details = compute_formula(P, Q, C, K, R, S, P_l3, Q_l3, C_l3, K_l3, block_h, block_w)
    
    print(f"\nConfig: P{P_l3}Q{Q_l3}C{C_l3}K{K_l3} blk{block_h}x{block_w}")
    print(f"Trace: {trace_switches}, Formula: {formula_val}")
    print(f"\nFormula breakdown:")
    print(f"  H_tile = P_l3 + R - 1 = {P_l3} + {R} - 1 = {details['H_tile']}")
    print(f"  W_tile = Q_l3 + S - 1 = {Q_l3} + {S} - 1 = {details['W_tile']}")
    print(f"  h_blocks_per_tile = ceil({details['H_tile']}/{block_h}) = {details['h_blocks_per_tile']}")
    print(f"  w_blocks_per_tile = ceil({details['W_tile']}/{block_w}) = {details['w_blocks_per_tile']}")
    print(f"  blocks_per_tile = {details['blocks_per_tile']}")
    print(f"  switches_within_tile = C_l3 * blocks_per_tile - 1 = {C_l3} * {details['blocks_per_tile']} - 1 = {details['switches_within_tile']}")
    print(f"  total_L3_tiles = {details['total_L3_tiles']}")
    print(f"  between_tiles = {details['between_tiles']}")
    print(f"  Total formula = {details['switches_within_tile']} * {details['total_L3_tiles']} + {details['between_tiles']} = {formula_val}")
    print(f"\nBut trace shows {trace_switches} switches with {total_accesses} total accesses")
    
    # The issue is that we're counting wrong
    # Let me trace through manually for a simple case


if __name__ == '__main__':
    main()
