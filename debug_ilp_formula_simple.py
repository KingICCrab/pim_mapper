#!/usr/bin/env python3
"""
Compare ILP formula vs Trace generator for Input row switches.

ILP Formula (row_aligned mode):
    row_acts_aligned = Π_{j ∈ all_dims} bound_j^{xj}
    
This is the PRODUCT of all L3 tiling factors.
"""

import sys
sys.path.insert(0, "/Users/haochenzhao/Projects/pim_optimizer/validation/dram")

from trace_generator import TraceGenerator, DRAMConfig


def compute_ilp_input_formula(tiling):
    """
    Compute Input row activations using ILP formula.
    
    ILP Formula (row_aligned mode):
        row_acts_aligned = Π_{j ∈ all_dims} bound_j
        
    Where bound_j is the L3 (DRAM) iteration count for each dimension.
    """
    # Get L3 tiling factors (DRAM level iterations)
    P_l3 = tiling['P'][3] if len(tiling['P']) > 3 else 1
    Q_l3 = tiling['Q'][3] if len(tiling['Q']) > 3 else 1
    C_l3 = tiling['C'][3] if len(tiling['C']) > 3 else 1
    K_l3 = tiling['K'][3] if len(tiling['K']) > 3 else 1
    R_l3 = tiling['R'][3] if len(tiling['R']) > 3 else 1
    S_l3 = tiling['S'][3] if len(tiling['S']) > 3 else 1
    N_l3 = tiling['N'][3] if len(tiling['N']) > 3 else 1
    
    # Compute row_acts_aligned = Π_{j} bound_j
    row_acts_aligned = P_l3 * Q_l3 * C_l3 * K_l3 * R_l3 * S_l3 * N_l3
    
    return row_acts_aligned, (P_l3, Q_l3, C_l3, K_l3, R_l3, S_l3, N_l3)


def generate_trace_and_count(generator, mapping, workload):
    """Generate trace and count input row switches."""
    trace = generator.generate_trace(mapping, workload)
    
    # Parse input addresses (bank 0)
    input_addrs = []
    for line in trace:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        addr = int(parts[1], 16)
        bank = (addr >> 10) % 4
        if bank == 0:  # Input
            row = addr >> 12
            input_addrs.append((addr, row))
    
    # Count row switches
    row_switches = 0
    prev_row = None
    unique_rows = set()
    
    for addr, row in input_addrs:
        unique_rows.add(row)
        if prev_row is not None and row != prev_row:
            row_switches += 1
        prev_row = row
    
    return row_switches, len(unique_rows), len(input_addrs)


def create_workload_and_mapping(P, Q, C, K, R, S, N, tiling, loop_order):
    """Create workload and mapping objects."""
    from dataclasses import dataclass
    
    @dataclass
    class SimpleWorkload:
        N: int
        K: int
        C: int
        P: int
        Q: int
        R: int
        S: int
        
        @property
        def input_size(self):
            H = self.P + self.R - 1
            W = self.Q + self.S - 1
            return {'H': H, 'W': W}
        
        @property
        def stride(self):
            return (1, 1)
        
        @property
        def dilation(self):
            return (1, 1)
    
    @dataclass
    class SimpleMapping:
        """Minimal mapping for trace generator."""
        t: dict  # tiling
        u: dict  # loop order permutation
        
        def to_dict(self):
            return {'t': self.t, 'u': self.u}
    
    # Build t dict: t[mem_level][temporal_spatial][dim] = factor
    # mem_level: 0-3, temporal_spatial: 1 (temporal)
    t = {}
    for level in range(4):
        t[level] = {1: {}}  # temporal
        for dim_idx, dim_name in enumerate(['R', 'S', 'P', 'Q', 'C', 'K', 'N']):
            t[level][1][dim_idx] = tiling[dim_name][level]
    
    # Build u dict: u[mem_level][temporal_spatial][position] = dim
    u = {}
    for level in range(4):
        u[level] = {1: {}}
        dim_map = {'R': 0, 'S': 1, 'P': 2, 'Q': 3, 'C': 4, 'K': 5, 'N': 6}
        for pos, dim_name in enumerate(loop_order):
            u[level][1][pos] = dim_map[dim_name]
    
    workload = SimpleWorkload(N=N, K=K, C=C, P=P, Q=Q, R=R, S=S)
    mapping = SimpleMapping(t=t, u=u)
    
    return workload, mapping


def main():
    print("=" * 90)
    print("ILP Formula vs Trace Generator: Input Row Switches")
    print("=" * 90)
    
    dram_config = DRAMConfig(
        row_buffer_bytes=1024,
        num_banks=4,
        num_rows=16384,
        element_size=1
    )
    generator = TraceGenerator(dram_config)
    
    # Workload dimensions
    P, Q, C, K = 4, 4, 4, 4
    R, S, N = 3, 3, 1
    
    print(f"Workload: P={P}, Q={Q}, C={C}, K={K}, R={R}, S={S}, N={N}")
    print(f"Input size: H={P+R-1}, W={Q+S-1}")
    print(f"DRAM Config: row_buffer={dram_config.row_buffer_bytes} bytes")
    print()
    
    # Test configurations
    configs = [
        ("P2Q2C2K2", [2, 1, 1, 2], [2, 1, 1, 2], [1, 1, 1, 4], [1, 1, 1, 4]),
        ("P4Q4C4K4", [4, 1, 1, 1], [4, 1, 1, 1], [1, 1, 1, 4], [1, 1, 1, 4]),
        ("P1Q1C4K4", [1, 1, 1, 4], [1, 1, 1, 4], [1, 1, 1, 4], [1, 1, 1, 4]),
        ("P4Q4C2K2", [4, 1, 1, 1], [4, 1, 1, 1], [1, 1, 1, 2], [1, 1, 1, 2]),
    ]
    
    loop_orders = [
        ("PQCK", ['P', 'Q', 'C', 'K', 'R', 'S', 'N']),
        ("KCQP", ['K', 'C', 'Q', 'P', 'R', 'S', 'N']),
        ("CPKQ", ['C', 'P', 'K', 'Q', 'R', 'S', 'N']),
    ]
    
    results = []
    
    print(f"{'Config':<20} {'Loop':>6} {'Trace':>8} {'ILP':>8} {'Unique':>8} {'Accesses':>10} {'Error%':>10}")
    print("-" * 80)
    
    for config_name, P_tile, Q_tile, C_tile, K_tile in configs:
        tiling = {
            'P': P_tile, 'Q': Q_tile, 'C': C_tile, 'K': K_tile,
            'R': [1, 1, 1, R], 'S': [1, 1, 1, S], 'N': [1, 1, 1, N]
        }
        
        ilp_result, l3_factors = compute_ilp_input_formula(tiling)
        
        for loop_name, loop_order in loop_orders:
            workload, mapping = create_workload_and_mapping(
                P, Q, C, K, R, S, N, tiling, loop_order
            )
            
            try:
                trace_switches, unique_rows, total_accesses = generate_trace_and_count(
                    generator, mapping, workload
                )
                
                # Calculate error
                if trace_switches > 0:
                    error = abs(ilp_result - trace_switches) / trace_switches * 100
                else:
                    error = 0 if ilp_result == 0 else float('inf')
                
                print(f"{config_name:<20} {loop_name:>6} {trace_switches:>8} {ilp_result:>8} "
                      f"{unique_rows:>8} {total_accesses:>10} {error:>10.1f}")
                
                results.append({
                    'config': f"{config_name}_{loop_name}",
                    'trace': trace_switches,
                    'ilp': ilp_result,
                    'unique': unique_rows,
                    'error': error
                })
                
            except Exception as e:
                print(f"{config_name:<20} {loop_name:>6} ERROR: {e}")
    
    print()
    print("=" * 90)
    print("Summary:")
    print("=" * 90)
    print()
    print("ILP formula: row_acts_aligned = P_l3 × Q_l3 × C_l3 × K_l3 × R_l3 × S_l3 × N_l3")
    print()
    print("For these configs with R=3, S=3, N=1:")
    for config_name, P_tile, Q_tile, C_tile, K_tile in configs:
        P_l3, Q_l3, C_l3, K_l3 = P_tile[3], Q_tile[3], C_tile[3], K_tile[3]
        result = P_l3 * Q_l3 * C_l3 * K_l3 * R * S * N
        print(f"  {config_name}: {P_l3}×{Q_l3}×{C_l3}×{K_l3}×{R}×{S}×{N} = {result}")


if __name__ == "__main__":
    main()
