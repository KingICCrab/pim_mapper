#!/usr/bin/env python3
"""
最终分析：Input row switches 精确公式推导

关键观察：
1. row_switches 与 total_accesses 强相关
2. 当 blocks_per_tile=1 时，Formula v2 准确 (255 vs 255)
3. 需要考虑 WITHIN tile 的 row switches

让我们深入分析实际的访问模式。
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from validation.dram.trace_generator import TraceGenerator, DRAMConfig
from validation.dram_v2.experiments.mapping_sweep import SimpleMapping
from validation.dram_v2.core.mapping_space import MappingConfig, DIM_P, DIM_Q, DIM_C, DIM_K
from src.pim_optimizer.workload.conv import ConvWorkload
from collections import defaultdict


def create_mapping_config(P_l3, Q_l3, C_l3, K_l3, block_h, block_w, 
                          input_layout='row_aligned'):
    return MappingConfig(
        P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3,
        permutation=(DIM_Q, DIM_P, DIM_C, DIM_K),
        input_layout=input_layout,
        weight_layout='row_aligned',
        output_layout='row_aligned',
        block_h=block_h,
        block_w=block_w
    )


def analyze_detailed(workload, config: MappingConfig, dram_config):
    """详细分析每次 L3 迭代的 row switches."""
    
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
    unique_rows = len(set(rows))
    total_accesses = len(rows)
    
    return rows, row_switches, unique_rows, total_accesses


def analyze_l3_iterations(workload, config: MappingConfig, dram_config):
    """分析每个 L3 迭代产生的 row switches."""
    
    P = workload.P
    Q = workload.Q
    C = workload.C
    K = workload.K
    R = workload.R
    S = workload.S
    H = P + R - 1
    W = Q + S - 1
    
    P_l3 = config.P_l3
    Q_l3 = config.Q_l3
    C_l3 = config.C_l3
    K_l3 = config.K_l3
    
    num_P = (P + P_l3 - 1) // P_l3
    num_Q = (Q + Q_l3 - 1) // Q_l3
    num_C = (C + C_l3 - 1) // C_l3
    num_K = (K + K_l3 - 1) // K_l3
    
    block_h = config.block_h
    block_w = config.block_w
    
    row_buffer_bytes = dram_config.row_buffer_bytes
    
    num_h_blocks = (H + block_h - 1) // block_h
    num_w_blocks = (W + block_w - 1) // block_w
    
    # Input tile dimensions
    H_tile = P_l3 + R - 1
    W_tile = Q_l3 + S - 1
    
    # For each L3 iteration, count how many row switches occur
    # Loop order: K (outer) -> C -> P -> Q (inner)
    
    print(f"\n  Analyzing L3 iterations:")
    print(f"  Loop order: K={num_K} -> C={num_C} -> P={num_P} -> Q={num_Q}")
    print(f"  Total L3 iterations: {num_K * num_C * num_P * num_Q}")
    print(f"  H_tile={H_tile}, W_tile={W_tile}, C_l3={C_l3}")
    
    # For each L3 tile (k, c, p, q), we access:
    # - Channels: c * C_l3 to (c+1) * C_l3
    # - H range: p * P_l3 (as output) -> p * P_l3 + R - 1 to p * P_l3 + P_l3 - 1 + R - 1 (as input H)
    #   Simplified: h_start = p * P_l3, h_end = p * P_l3 + H_tile
    # - W range: similar for Q
    
    # Each L3 tile accesses Input[c:c+C_l3, h_start:h_end, w_start:w_end]
    # This spans multiple (h_block, w_block) combinations
    
    # Row switches within one L3 tile:
    # We iterate: for h_block: for w_block: for h_in_block: for w_in_block: for c:
    # Each (h_block, w_block, c) combination is in a unique row-aligned region
    # So switches happen when changing (h_block, w_block, c)
    
    # Number of (h_block, w_block) combinations per tile
    h_blocks_per_tile = (H_tile + block_h - 1) // block_h
    w_blocks_per_tile = (W_tile + block_w - 1) // block_w
    blocks_per_tile = h_blocks_per_tile * w_blocks_per_tile
    
    # Switches within one tile = (blocks_per_tile * C_l3 - 1)
    # Because we iterate over blocks_per_tile blocks, each with C_l3 channels
    # And we switch when changing block or channel
    
    switches_per_tile = blocks_per_tile * C_l3 - 1
    
    print(f"  blocks_per_tile={blocks_per_tile}, switches_per_tile={switches_per_tile}")
    
    # Total L3 iterations
    total_iters = num_K * num_C * num_P * num_Q
    
    # Between tiles:
    # When we move from one tile to the next, we usually change row
    # So add 1 switch per tile transition
    tile_transitions = total_iters - 1
    
    # But wait - if consecutive tiles access the same row (sequential memory), no switch
    # With row_aligned layout, different tiles usually access different row regions
    # So tile_transitions ≈ total_iters - 1
    
    # Total formula v1: switches_per_tile * total_iters + tile_transitions
    # But this overcounts because last element of tile i and first element of tile i+1
    # both contribute to the switch count
    
    # Simpler model:
    # Each L3 tile accesses blocks_per_tile * C_l3 row-aligned regions
    # Each region access starts and ends in a contiguous way (within the region)
    # So switches = (regions accessed across all tiles) - 1
    # 
    # But regions are reused across tiles! The key is counting region VISITS, not unique regions.
    
    total_region_visits = blocks_per_tile * C_l3 * total_iters
    formula_visits = total_region_visits - 1
    
    print(f"  total_region_visits={total_region_visits}")
    print(f"  formula (visits - 1)={formula_visits}")
    
    return formula_visits


def main():
    print("="*80)
    print("Input Row Switches: Precise Formula")
    print("="*80)
    
    dram_config = DRAMConfig(row_buffer_bytes=1024, num_banks=4, num_rows=16384)
    
    P, Q, C, K, R, S = 8, 8, 4, 4, 3, 3
    N = 1
    workload = ConvWorkload(name='test', R=R, S=S, P=P, Q=Q, C=C, K=K, N=N)
    H = P + R - 1  # 10
    W = Q + S - 1  # 10
    
    print(f"\nWorkload: P={P}, Q={Q}, C={C}, K={K}, R={R}, S={S}")
    print(f"Input: H={H}, W={W}")
    
    configs = [
        # (P_l3, Q_l3, C_l3, K_l3, block_h, block_w)
        (8, 8, 4, 4, 5, 5),
        (4, 4, 4, 4, 5, 5),
        (2, 2, 4, 4, 5, 5),
        (8, 8, 4, 4, 10, 10),
        (8, 8, 4, 4, 2, 2),
        (4, 4, 2, 4, 5, 5),  # With C tiling
        (2, 2, 2, 2, 5, 5),
    ]
    
    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)
    print(f"{'Config':<25} {'Trace':>8} {'Formula':>8} {'Error':>8}")
    print("-"*55)
    
    for P_l3, Q_l3, C_l3, K_l3, block_h, block_w in configs:
        config = create_mapping_config(P_l3, Q_l3, C_l3, K_l3, block_h, block_w)
        
        rows, trace_switches, unique_rows, total_accesses = analyze_detailed(
            workload, config, dram_config)
        
        # Compute formula
        num_P = (P + P_l3 - 1) // P_l3
        num_Q = (Q + Q_l3 - 1) // Q_l3
        num_C = (C + C_l3 - 1) // C_l3
        num_K = (K + K_l3 - 1) // K_l3
        
        H_tile = P_l3 + R - 1
        W_tile = Q_l3 + S - 1
        
        h_blocks_per_tile = (H_tile + block_h - 1) // block_h
        w_blocks_per_tile = (W_tile + block_w - 1) // block_w
        blocks_per_tile = h_blocks_per_tile * w_blocks_per_tile
        
        total_iters = num_K * num_C * num_P * num_Q
        
        # Formula: (blocks_per_tile * C_l3) * total_iters - 1
        # This counts total row-aligned region visits minus 1
        formula_val = blocks_per_tile * C_l3 * total_iters - 1
        
        error = abs(trace_switches - formula_val) / max(1, trace_switches) * 100
        
        cfg_str = f"P{P_l3}Q{Q_l3}C{C_l3}K{K_l3} blk{block_h}x{block_w}"
        print(f"{cfg_str:<25} {trace_switches:>8} {formula_val:>8} {error:>7.1f}%")
        
        # Debug info
        print(f"  (L3 iters={total_iters}, blks/tile={blocks_per_tile}, C_l3={C_l3}, total_accesses={total_accesses})")
    
    # Let's try a different approach: analyze based on total accesses
    print("\n" + "="*80)
    print("Alternative Analysis: Based on Total Accesses")
    print("="*80)
    
    for P_l3, Q_l3, C_l3, K_l3, block_h, block_w in configs:
        config = create_mapping_config(P_l3, Q_l3, C_l3, K_l3, block_h, block_w)
        
        rows, trace_switches, unique_rows, total_accesses = analyze_detailed(
            workload, config, dram_config)
        
        # Number of elements per row-aligned region = block_h * block_w
        elements_per_region = block_h * block_w
        
        # If accesses are grouped by region, switches = (total_accesses / elements_per_region) - 1
        # But with row_aligned, each region might span multiple DRAM rows
        
        # Actually, with row_aligned layout and block_h * block_w < row_buffer_bytes,
        # each (h_block, w_block, c) region fits in ONE row (padded)
        # So the number of row-region visits = total_accesses / (block_h * block_w)
        
        if elements_per_region > 0:
            region_visits = total_accesses // elements_per_region
        else:
            region_visits = total_accesses
        
        # But wait, total_accesses includes ALL elements, not just one per region
        # Let's think differently:
        # - Total Input accesses in trace = total_accesses
        # - Unique addresses = H * W * C (assuming no padding waste)
        # - Each unique address is in one row
        # - Row switches depend on access ORDER, not just count
        
        # With block-wise iteration:
        # for h_block: for w_block: for c: for h_in_block: for w_in_block:
        #   access[h, w, c]
        # 
        # All elements in one (h_block, w_block, c) are accessed sequentially,
        # so they cause 0 row switches (within the same row-aligned region).
        # Switch happens only when moving to next (h_block, w_block, c) combination.
        
        # Total (h_block, w_block, c) visits per L3 tile = blocks_per_tile * C_l3
        # Total L3 tiles = total_iters
        # Total region visits = blocks_per_tile * C_l3 * total_iters
        # Row switches = region_visits - 1
        
        num_P = (P + P_l3 - 1) // P_l3
        num_Q = (Q + Q_l3 - 1) // Q_l3
        num_C = (C + C_l3 - 1) // C_l3
        num_K = (K + K_l3 - 1) // K_l3
        total_iters = num_K * num_C * num_P * num_Q
        
        H_tile = P_l3 + R - 1
        W_tile = Q_l3 + S - 1
        h_blocks_per_tile = (H_tile + block_h - 1) // block_h
        w_blocks_per_tile = (W_tile + block_w - 1) // block_w
        blocks_per_tile = h_blocks_per_tile * w_blocks_per_tile
        
        region_visits = blocks_per_tile * C_l3 * total_iters
        formula_v3 = region_visits - 1
        
        # Alternative: use actual total_accesses to estimate
        # total_accesses / elements_per_region should approximate region_visits
        estimated_region_visits = total_accesses // elements_per_region if elements_per_region > 0 else 1
        formula_v4 = estimated_region_visits - 1
        
        error_v3 = abs(trace_switches - formula_v3) / max(1, trace_switches) * 100
        error_v4 = abs(trace_switches - formula_v4) / max(1, trace_switches) * 100
        
        cfg_str = f"P{P_l3}Q{Q_l3}C{C_l3}K{K_l3} blk{block_h}x{block_w}"
        print(f"\n{cfg_str}")
        print(f"  Trace: {trace_switches}, Accesses: {total_accesses}")
        print(f"  region_visits (calculated): {region_visits}, formula_v3: {formula_v3}, error: {error_v3:.1f}%")
        print(f"  region_visits (estimated): {estimated_region_visits}, formula_v4: {formula_v4}, error: {error_v4:.1f}%")


if __name__ == '__main__':
    main()
