"""
Debug script for comparing ILP vs Trace row activation counts.
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from collections import defaultdict
from typing import List, Dict

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
from validation.dram.trace_generator import TraceGenerator, DRAMConfig


def count_row_activations(trace: List[str], dram_config: DRAMConfig) -> Dict[int, int]:
    """Count row activations per bank from trace."""
    row_size = dram_config.row_buffer_bytes
    bank_size = row_size * dram_config.num_rows
    
    current_row = {}  # bank -> current row
    activations = defaultdict(int)  # bank -> activation count
    
    for line in trace:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        addr = int(parts[1], 16)
        bank = addr // bank_size
        row = (addr % bank_size) // row_size
        
        if bank not in current_row or current_row[bank] != row:
            activations[bank] += 1
            current_row[bank] = row
    
    return dict(activations)


def debug_workload(name, N, K, C, P, Q, R, S, stride=(1,1)):
    """Debug a single workload."""
    print(f"\n{'='*80}")
    print(f"Workload: {name}")
    print(f"Config: N={N}, K={K}, C={C}, P={P}, Q={Q}, R={R}, S={S}, stride={stride}")
    print(f"{'='*80}")
    
    # Create workload
    workload = ConvWorkload(
        name=name,
        N=N, K=K, C=C, P=P, Q=Q, R=R, S=S,
        stride=stride,
    )
    
    print(f"\nDerived sizes:")
    print(f"  H_in = {workload.input_size['H']}, W_in = {workload.input_size['W']}")
    print(f"  Input:  {workload.N} x {workload.C} x {workload.input_size['H']} x {workload.input_size['W']}")
    print(f"  Weight: {workload.K} x {workload.C} x {workload.R} x {workload.S}")
    print(f"  Output: {workload.N} x {workload.K} x {workload.P} x {workload.Q}")
    
    # Run optimizer
    print("\nRunning optimizer...")
    optimizer = PIMOptimizer()
    result = optimizer.optimize([workload], objective="latency")
    
    if result is None or result.mappings is None or len(result.mappings) == 0:
        print("ERROR: No solution found")
        return
    
    mapping = result.mappings[0]
    
    print(f"\nMapping result:")
    print(f"  Block: {mapping.tile_info.get('block_h', 0)} x {mapping.tile_info.get('block_w', 0)}")
    print(f"  Layout: Input={mapping.layout.get(0)}, Weight={mapping.layout.get(1)}, Output={mapping.layout.get(2)}")
    
    print(f"\n  Loop bounds:")
    for m in sorted(mapping.loop_bounds.keys()):
        print(f"    Level {m}: {mapping.loop_bounds[m]}")
    
    print(f"\n  Metrics:")
    for key, val in mapping.metrics.items():
        print(f"    {key}: {val}")
    
    # Get ILP predictions
    ilp_input = mapping.metrics.get("row_activations_input", 0)
    ilp_weight = mapping.metrics.get("row_activations_weight", 0)
    ilp_output = mapping.metrics.get("row_activations_output", 0)
    
    # Generate trace
    print("\nGenerating trace...")
    dram_config = DRAMConfig(row_buffer_bytes=1024, num_banks=4, num_rows=16384, element_size=1)
    gen = TraceGenerator(dram_config)
    trace = gen.generate_trace(mapping, workload)
    
    print(f"  Total trace lines: {len(trace)}")
    
    # Count activations
    trace_acts = count_row_activations(trace, dram_config)
    
    print(f"\n  Trace row activations per bank:")
    for bank in sorted(trace_acts.keys()):
        tensor_name = {0: "Input", 1: "Weight", 2: "Output"}.get(bank, f"Bank{bank}")
        print(f"    Bank {bank} ({tensor_name}): {trace_acts[bank]}")
    
    print(f"\n  Comparison:")
    print(f"    Input:  ILP={ilp_input:.2f}, Trace={trace_acts.get(0, 0)}")
    print(f"    Weight: ILP={ilp_weight:.2f}, Trace={trace_acts.get(1, 0)}")
    print(f"    Output: ILP={ilp_output:.2f}, Trace={trace_acts.get(2, 0)}")
    
    # Analyze trace patterns
    print(f"\n  First 20 trace entries:")
    for i, line in enumerate(trace[:20]):
        parts = line.strip().split()
        addr = int(parts[1], 16)
        row_size = dram_config.row_buffer_bytes
        bank_size = row_size * dram_config.num_rows
        bank = addr // bank_size
        row = (addr % bank_size) // row_size
        col = addr % row_size
        print(f"    {line} -> Bank={bank}, Row={row}, Col={col}")


if __name__ == "__main__":
    # Test tiny workload
    debug_workload("tiny", N=1, K=4, C=4, P=4, Q=4, R=3, S=3)
    
    # Test kernel_1x1 (closest to matching)
    debug_workload("kernel_1x1", N=1, K=16, C=16, P=8, Q=8, R=1, S=1)
