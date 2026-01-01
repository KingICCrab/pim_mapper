#!/usr/bin/env python3
"""
Compare FULL ILP formula (including block crossing) vs trace for Input row switches.

ILP Formula (row_aligned):
    total = row_acts_aligned + block_crossing_acts
    
    row_acts_aligned = Π_{j ∈ all_dims} bound_j^{xj}
    block_crossing_acts = 2 × (H_crossing + W_crossing) × reuse_penalty
    reuse_penalty = Π_{j ∈ irrelevant} bound_j = K_l3 × N_l3 (for Input)
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')

import math
from validation.dram.trace_generator import TraceGenerator, DRAMConfig
from validation.dram_v2.experiments.mapping_sweep import SimpleMapping
from validation.dram_v2.core.mapping_space import MappingConfig, DIM_P, DIM_Q, DIM_C, DIM_K
from src.pim_optimizer.workload.conv import ConvWorkload
from src.pim_optimizer.model.row_activation import compute_input_block_crossing_count


def analyze_input_pattern(workload, config, dram_config):
    """Analyze input access pattern."""
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
    
    return row_switches, unique_rows, len(input_rows)


def compute_full_ilp_formula(workload, config: MappingConfig):
    """
    Compute FULL ILP formula for Input row activations.
    
    total = row_acts_aligned + block_crossing_acts
    """
    H = workload.input_size['H']
    W = workload.input_size['W']
    R, S = workload.R, workload.S
    stride_h, stride_w = 1, 1
    dilation_h, dilation_w = 1, 1
    
    # L3 factors
    P_l3, Q_l3, C_l3, K_l3 = config.P_l3, config.Q_l3, config.C_l3, config.K_l3
    N_l3 = 1  # Usually N=1
    R_l3 = R  # Usually full kernel at buffer level
    S_l3 = S
    
    # Step 1: row_acts_aligned = product of all L3 factors
    # NOTE: For row_aligned mode with full block, this is the baseline
    row_acts_aligned = P_l3 * Q_l3 * C_l3 * K_l3 * 1 * 1 * N_l3
    
    # Step 2: reuse_penalty = K_l3 × N_l3 (irrelevant dims for Input)
    reuse_penalty = K_l3 * N_l3
    
    # Step 3: Compute block crossing
    # Buffer tile sizes
    P_tile = workload.P // P_l3
    Q_tile = workload.Q // Q_l3
    
    tile_h = stride_h * (P_tile - 1) + dilation_h * (R - 1) + 1
    tile_w = stride_w * (Q_tile - 1) + dilation_w * (S - 1) + 1
    step_h = P_tile * stride_h
    step_w = Q_tile * stride_w
    
    block_h = config.block_h
    block_w = config.block_w
    
    # H direction crossing
    h_crossing, h_total = compute_input_block_crossing_count(
        block_h=block_h,
        tile_h=tile_h,
        step=step_h,
        tile_s=R,  # full kernel at buffer level
        total_S=R,
        dilation=dilation_h,
        num_tiles=P_l3
    )
    
    # W direction crossing
    w_crossing, w_total = compute_input_block_crossing_count(
        block_h=block_w,  # Note: using block_w
        tile_h=tile_w,
        step=step_w,
        tile_s=S,  # full kernel at buffer level
        total_S=S,
        dilation=dilation_w,
        num_tiles=Q_l3
    )
    
    # Step 4: block_crossing_acts = 2 × (H_crossing + W_crossing) × reuse_penalty
    block_crossing_acts = 2 * (h_crossing + w_crossing) * reuse_penalty
    
    # Total
    total = row_acts_aligned + block_crossing_acts
    
    return {
        'row_acts_aligned': row_acts_aligned,
        'reuse_penalty': reuse_penalty,
        'h_crossing': h_crossing,
        'w_crossing': w_crossing,
        'block_crossing_acts': block_crossing_acts,
        'total': total,
        'tile_h': tile_h,
        'tile_w': tile_w,
        'block_h': block_h,
        'block_w': block_w,
    }


def main():
    print("=" * 120)
    print("FULL ILP Formula vs Trace: Input Row Switches Analysis")
    print("=" * 120)
    
    # Setup
    workload = ConvWorkload(name='test', R=3, S=3, P=4, Q=4, C=4, K=4, N=1)
    H, W = workload.input_size['H'], workload.input_size['W']
    dram_config = DRAMConfig(row_buffer_bytes=1024, num_banks=4, num_rows=16384, element_size=1)
    
    print(f"\nWorkload: P={workload.P}, Q={workload.Q}, C={workload.C}, K={workload.K}")
    print(f"         R={workload.R}, S={workload.S}, N={workload.N}")
    print(f"Input size: H={H}, W={W}, elements = {H * W * workload.C}")
    print(f"Row size: {dram_config.row_buffer_bytes} elements")
    print()
    
    # Test configurations
    configs = [
        # (P_l3, Q_l3, C_l3, K_l3, block_h, block_w)
        (2, 2, 2, 2, 6, 6),  # Full block
        (4, 4, 4, 4, 6, 6),  # Max L3
        (1, 1, 4, 4, 6, 6),  # No P/Q tiling
        (2, 2, 2, 2, 3, 3),  # Half block
        (2, 2, 2, 2, 2, 3),  # Smaller block
        (2, 2, 4, 4, 6, 6),  # Mixed
        (4, 4, 2, 2, 6, 6),  # Opposite
    ]
    
    print(f"{'Config':<25} {'Trace':>8} {'ILP':>8} {'Aligned':>8} {'BlkCross':>9} {'H_x':>5} {'W_x':>5} {'Reuse':>6} {'Err%':>8}")
    print("-" * 100)
    
    results = []
    
    for (P_l3, Q_l3, C_l3, K_l3, block_h, block_w) in configs:
        config = MappingConfig(
            P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3,
            permutation=(DIM_P, DIM_Q, DIM_C, DIM_K),
            input_layout='row_aligned',
            weight_layout='sequential',
            output_layout='sequential',
            block_h=block_h, block_w=block_w
        )
        
        # Run trace
        trace_switches, unique_rows, total_accesses = analyze_input_pattern(workload, config, dram_config)
        
        # Compute full ILP formula
        ilp = compute_full_ilp_formula(workload, config)
        
        # Error
        if trace_switches > 0:
            error = (ilp['total'] - trace_switches) / trace_switches * 100
        else:
            error = 0 if ilp['total'] == 0 else float('inf')
        
        config_str = f"P{P_l3}Q{Q_l3}C{C_l3}K{K_l3}_b{block_h}x{block_w}"
        print(f"{config_str:<25} {trace_switches:>8} {ilp['total']:>8} {ilp['row_acts_aligned']:>8} "
              f"{ilp['block_crossing_acts']:>9} {ilp['h_crossing']:>5} {ilp['w_crossing']:>5} "
              f"{ilp['reuse_penalty']:>6} {error:>8.1f}")
        
        results.append({
            'config': config_str,
            'trace': trace_switches,
            'ilp_total': ilp['total'],
            'aligned': ilp['row_acts_aligned'],
            'crossing': ilp['block_crossing_acts'],
            'error': error
        })
    
    print()
    print("=" * 120)
    print("Analysis:")
    print("=" * 120)
    
    print("""
ILP Formula:
    total = row_acts_aligned + block_crossing_acts
    
    row_acts_aligned = P_l3 × Q_l3 × C_l3 × K_l3 (R=S=N=1 at L3)
    block_crossing_acts = 2 × (H_crossing + W_crossing) × K_l3 (reuse penalty)
    
Key observations:
    1. When block >= input size (H×W), crossing = 0
    2. When block < tile size, every tile crosses → high crossing count
    3. K_l3 acts as reuse penalty (Input is re-accessed K_l3 times)
""")
    
    # Correlation
    trace_values = [r['trace'] for r in results]
    ilp_values = [r['ilp_total'] for r in results]
    
    from scipy.stats import spearmanr, pearsonr
    spearman_corr, sp_pval = spearmanr(trace_values, ilp_values)
    pearson_corr, pe_pval = pearsonr(trace_values, ilp_values)
    
    print(f"\nCorrelation (Trace vs ILP):")
    print(f"  Spearman: {spearman_corr:.3f} (p={sp_pval:.4f})")
    print(f"  Pearson:  {pearson_corr:.3f} (p={pe_pval:.4f})")
    
    # Mean absolute error
    mae = sum(abs(r['trace'] - r['ilp_total']) for r in results) / len(results)
    mape = sum(abs(r['error']) for r in results) / len(results)
    print(f"\n  Mean Absolute Error: {mae:.1f}")
    print(f"  Mean Absolute % Error: {mape:.1f}%")


if __name__ == "__main__":
    main()
