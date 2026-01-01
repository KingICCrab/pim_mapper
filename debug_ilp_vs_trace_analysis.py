#!/usr/bin/env python3
"""
Compare ILP formula vs trace for Input row switches.

ILP Formula (row_aligned):
    row_acts_aligned = Π_{j ∈ all_dims} bound_j^{xj}
    
This is the PRODUCT of all L3 iteration counts (including K, which is irrelevant to Input).

Hypothesis: ILP formula counts "tile visits" not "row switches"
    - tile_visits = P_l3 × Q_l3 × C_l3 × K_l3 × R_l3 × S_l3 × N_l3
    - But Input is reused across K iterations (K is irrelevant)
    - So unique_input_visits = P_l3 × Q_l3 × C_l3 × R_l3 × S_l3 × N_l3

The K factor in ILP represents the "reuse penalty" - how many times the same data
is accessed due to irrelevant dimension iterations.
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

import math
from validation.dram.trace_generator import TraceGenerator, DRAMConfig
from validation.dram_v2.experiments.mapping_sweep import SimpleMapping
from validation.dram_v2.core.mapping_space import MappingConfig, DIM_P, DIM_Q, DIM_C, DIM_K
from src.pim_optimizer.workload.conv import ConvWorkload


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


def compute_ilp_formula(config: MappingConfig, R, S, N):
    """
    Compute ILP formula for Input row_acts_aligned.
    
    ILP: row_acts_aligned = P_l3 × Q_l3 × C_l3 × K_l3 × R × S × N
    
    Note: In ILP, R, S, N are usually 1 at L3 level (full R/S at buffer level).
    """
    return config.P_l3 * config.Q_l3 * config.C_l3 * config.K_l3 * R * S * N


def compute_unique_input_visits(config: MappingConfig, R, S, N):
    """
    Compute unique input tile visits (excluding K reuse).
    
    Input doesn't depend on K, so unique visits = P_l3 × Q_l3 × C_l3 × R × S × N
    """
    return config.P_l3 * config.Q_l3 * config.C_l3 * R * S * N


def main():
    print("=" * 100)
    print("ILP Formula vs Trace: Input Row Switches Analysis")
    print("=" * 100)
    
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
        (2, 2, 2, 2, 6, 6),  # Small L3 factors, full block
        (4, 4, 4, 4, 6, 6),  # Max L3 factors, full block
        (1, 1, 4, 4, 6, 6),  # No P/Q tiling, full block
        (2, 2, 4, 4, 6, 6),  # Mixed
        (4, 4, 2, 2, 6, 6),  # Opposite
        (2, 2, 2, 2, 3, 3),  # Small block
        (2, 2, 2, 2, 2, 3),  # Smaller block
    ]
    
    # R=3, S=3, N=1 -> all at L3 (buffer handles full kernel)
    R_l3, S_l3, N_l3 = 1, 1, 1  # kernel loop at buffer level, not L3
    
    print(f"{'Config':<30} {'Trace':>8} {'ILP*':>8} {'Unique':>8} {'K_reuse':>8} {'Accesses':>10}")
    print("-" * 80)
    
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
        
        # Compute ILP formula
        # NOTE: ILP expects R_l3=S_l3=N_l3=1 since R/S/N are handled at buffer level
        ilp_value = P_l3 * Q_l3 * C_l3 * K_l3 * R_l3 * S_l3 * N_l3
        
        # Unique input visits (excluding K)
        unique_visits = P_l3 * Q_l3 * C_l3 * R_l3 * S_l3 * N_l3
        
        config_str = f"P{P_l3}Q{Q_l3}C{C_l3}K{K_l3}_blk{block_h}x{block_w}"
        print(f"{config_str:<30} {trace_switches:>8} {ilp_value:>8} {unique_rows:>8} {K_l3:>8} {total_accesses:>10}")
        
        results.append({
            'config': config_str,
            'trace': trace_switches,
            'ilp': ilp_value,
            'unique_rows': unique_rows,
            'unique_visits': unique_visits,
            'accesses': total_accesses
        })
    
    print()
    print("=" * 100)
    print("Key Observations:")
    print("=" * 100)
    
    print("""
1. ILP formula: row_acts_aligned = P_l3 × Q_l3 × C_l3 × K_l3 × R_l3 × S_l3 × N_l3
   - For kernel at buffer level: R_l3=S_l3=N_l3=1
   - So ILP = P_l3 × Q_l3 × C_l3 × K_l3

2. K is IRRELEVANT to Input (Input doesn't depend on K)
   - But ILP includes K_l3 factor as "reuse penalty"
   - This counts how many times the same Input data is accessed

3. IMPORTANT: ILP counts "tile visits" not "row switches"!
   - If data stays in row buffer, no new row activation needed
   - ILP model is an UPPER BOUND (worst case: every visit = new row activation)
   
4. Trace counts ACTUAL row switches based on:
   - Row buffer capacity
   - Access pattern (sequential vs random)
   - Data layout (row_aligned vs sequential)
""")
    
    # Analyze correlation
    print("\nCorrelation analysis:")
    trace_values = [r['trace'] for r in results]
    ilp_values = [r['ilp'] for r in results]
    
    # Simple Spearman rank correlation
    from scipy.stats import spearmanr
    corr, pval = spearmanr(trace_values, ilp_values)
    print(f"  Spearman correlation: {corr:.3f} (p={pval:.4f})")
    
    # Are they in same order?
    trace_ranks = sorted(range(len(trace_values)), key=lambda i: trace_values[i])
    ilp_ranks = sorted(range(len(ilp_values)), key=lambda i: ilp_values[i])
    
    print(f"\n  Trace ranking: {trace_ranks}")
    print(f"  ILP ranking:   {ilp_ranks}")


if __name__ == "__main__":
    main()
