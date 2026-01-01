#!/usr/bin/env python3
"""
Debug script: Compare ILP formula vs Trace generator for Input row switches.

ILP Formula (row_aligned mode):
    row_acts_aligned = Π_{j ∈ all_dims} bound_j^{xj}
    
Where:
    - all_dims = all 7 dimensions (P, Q, C, K, R, S, N)
    - bound_j = iteration count at DRAM level for dimension j
    - xj = 1 if dimension j iterates at DRAM level, else 0
"""

import sys
sys.path.insert(0, "/Users/haochenzhao/Projects/pim_optimizer/src")
sys.path.insert(0, "/Users/haochenzhao/Projects/pim_optimizer/validation/dram")

from trace_generator import TraceGenerator, DRAMConfig
from mapping_config import MappingConfig


def compute_ilp_input_formula(mapping_config: MappingConfig, verbose=False):
    """
    Compute Input row activations using ILP formula.
    
    ILP Formula (row_aligned mode):
        row_acts_aligned = Π_{j ∈ all_dims} bound_j
        
    Where bound_j is the L3 (DRAM) iteration count for each dimension.
    """
    tiling = mapping_config.tiling
    
    # Get L3 tiling factors (DRAM level iterations)
    P_l3 = tiling['P'][3] if len(tiling['P']) > 3 else 1
    Q_l3 = tiling['Q'][3] if len(tiling['Q']) > 3 else 1
    C_l3 = tiling['C'][3] if len(tiling['C']) > 3 else 1
    K_l3 = tiling['K'][3] if len(tiling['K']) > 3 else 1
    R_l3 = tiling['R'][3] if len(tiling['R']) > 3 else 1
    S_l3 = tiling['S'][3] if len(tiling['S']) > 3 else 1
    N_l3 = tiling['N'][3] if len(tiling['N']) > 3 else 1
    
    # Compute row_acts_aligned = Π_{j} bound_j
    # For ALL dimensions that iterate at DRAM level
    row_acts_aligned = P_l3 * Q_l3 * C_l3 * K_l3 * R_l3 * S_l3 * N_l3
    
    if verbose:
        print(f"  L3 tiling factors:")
        print(f"    P_l3={P_l3}, Q_l3={Q_l3}, C_l3={C_l3}, K_l3={K_l3}")
        print(f"    R_l3={R_l3}, S_l3={S_l3}, N_l3={N_l3}")
        print(f"  row_acts_aligned = {P_l3}*{Q_l3}*{C_l3}*{K_l3}*{R_l3}*{S_l3}*{N_l3} = {row_acts_aligned}")
    
    # TODO: Add block crossing calculation
    block_crossing_acts = 0
    
    total = row_acts_aligned + block_crossing_acts
    return total, row_acts_aligned, block_crossing_acts


def generate_trace_input_row_switches(mapping_config: MappingConfig, dram_config: DRAMConfig):
    """Generate trace and count input row switches."""
    generator = DRAMTraceGenerator(mapping_config, dram_config)
    traces = generator.generate_traces()
    input_trace = traces.get('input', [])
    
    if not input_trace:
        return 0, 0
    
    row_switches = 0
    prev_row = None
    unique_rows = set()
    
    for entry in input_trace:
        row_addr = entry['row_address']
        unique_rows.add(row_addr)
        if prev_row is not None and row_addr != prev_row:
            row_switches += 1
        prev_row = row_addr
    
    return row_switches, len(unique_rows)


def main():
    dram_config = DRAMConfig(
        row_buffer_bytes=1024,
        num_banks=4,
        num_rows=16384,
        element_size=1,
        layout='row_aligned'
    )
    
    # Test configurations (Conv layer with 3x3 kernel)
    configs = [
        ("P2Q2C2K2", [2, 1, 1, 2], [2, 1, 1, 2], [1, 1, 1, 4], [1, 1, 1, 4]),
        ("P4Q4C4K4", [4, 1, 1, 1], [4, 1, 1, 1], [1, 1, 1, 4], [1, 1, 1, 4]),
        ("P1Q1C4K4", [1, 1, 1, 4], [1, 1, 1, 4], [1, 1, 1, 4], [1, 1, 1, 4]),
        ("P4Q4C2K2", [4, 1, 1, 1], [4, 1, 1, 1], [1, 1, 1, 2], [1, 1, 1, 2]),
    ]
    
    # Loop orders to test
    loop_orders = [
        ("PQCK", ['P', 'Q', 'C', 'K', 'R', 'S', 'N']),
        ("KCQP", ['K', 'C', 'Q', 'P', 'R', 'S', 'N']),
        ("CPKQ", ['C', 'P', 'K', 'Q', 'R', 'S', 'N']),
        ("KPQC", ['K', 'P', 'Q', 'C', 'R', 'S', 'N']),
    ]
    
    # Use same workload as before
    P, Q, C, K = 4, 4, 4, 4
    R, S, N = 3, 3, 1
    
    print("=" * 90)
    print("ILP Formula vs Trace Generator: Input Row Switches")
    print("=" * 90)
    print(f"Workload: P={P}, Q={Q}, C={C}, K={K}, R={R}, S={S}, N={N}")
    print(f"DRAM Config: row_buffer={dram_config.row_buffer_bytes}, layout=row_aligned")
    print()
    
    results = []
    
    for config_name, P_tile, Q_tile, C_tile, K_tile in configs:
        for loop_name, loop_order in loop_orders:
            tiling = {
                'P': P_tile, 'Q': Q_tile, 'C': C_tile, 'K': K_tile,
                'R': [1, 1, 1, R], 'S': [1, 1, 1, S], 'N': [1, 1, 1, N]
            }
            
            mapping_config = MappingConfig(
                workload={'P': P, 'Q': Q, 'C': C, 'K': K, 'R': R, 'S': S, 'N': N},
                tiling=tiling,
                loop_orders={'L0': loop_order, 'L1': loop_order, 
                             'L2': loop_order, 'L3': loop_order}
            )
            
            # Get trace data
            trace_switches, unique_rows = generate_trace_input_row_switches(mapping_config, dram_config)
            
            # Get ILP formula result
            ilp_total, ilp_aligned, ilp_crossing = compute_ilp_input_formula(mapping_config)
            
            # Calculate error
            if trace_switches > 0:
                error = abs(ilp_total - trace_switches) / trace_switches * 100
            else:
                error = 0 if ilp_total == 0 else float('inf')
            
            results.append({
                'config': f"{config_name}_{loop_name}",
                'trace': trace_switches,
                'ilp_aligned': ilp_aligned,
                'ilp_crossing': ilp_crossing,
                'ilp_total': ilp_total,
                'unique_rows': unique_rows,
                'error': error
            })
    
    # Print results table
    print(f"{'Config':<20} {'Trace':>10} {'ILP_aligned':>12} {'ILP_cross':>10} {'ILP_total':>10} {'Unique':>8} {'Error%':>10}")
    print("-" * 90)
    
    for r in results:
        print(f"{r['config']:<20} {r['trace']:>10} {r['ilp_aligned']:>12} {r['ilp_crossing']:>10} {r['ilp_total']:>10} {r['unique_rows']:>8} {r['error']:>10.1f}")
    
    print()
    print("=" * 90)
    print("Analysis: Is ILP formula = product of L3 tiling factors?")
    print("=" * 90)
    
    # Calculate some derived metrics
    print("\nNote: For ALL configs, R_l3=3, S_l3=3, N_l3=1 (full kernel at L3)")
    print("So row_acts_aligned = P_l3 × Q_l3 × C_l3 × K_l3 × 3 × 3 × 1")
    print()
    
    # Show a few examples with breakdown
    for config_name, P_tile, Q_tile, C_tile, K_tile in configs[:2]:
        P_l3, Q_l3, C_l3, K_l3 = P_tile[3], Q_tile[3], C_tile[3], K_tile[3]
        product = P_l3 * Q_l3 * C_l3 * K_l3 * R * S * N
        print(f"{config_name}: P_l3={P_l3}, Q_l3={Q_l3}, C_l3={C_l3}, K_l3={K_l3}")
        print(f"  row_acts_aligned = {P_l3}×{Q_l3}×{C_l3}×{K_l3}×{R}×{S}×{N} = {product}")
        print()


if __name__ == "__main__":
    main()
