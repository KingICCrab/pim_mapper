"""
Debug trace generation to understand row activation discrepancy.
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from collections import defaultdict
from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
from validation.dram.trace_generator import TraceGenerator, DRAMConfig


def analyze_trace(trace, dram_config, tensor_name, bank_id):
    """Analyze trace for a specific tensor/bank."""
    row_size = dram_config.row_buffer_bytes
    bank_size = row_size * dram_config.num_rows
    
    # Filter accesses for this bank
    accesses = []
    for line in trace:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        addr = int(parts[1], 16)
        bank = addr // bank_size
        if bank == bank_id:
            row = (addr % bank_size) // row_size
            col = addr % row_size
            accesses.append((addr, row, col))
    
    print(f"\n{tensor_name} (Bank {bank_id}):")
    print(f"  Total accesses: {len(accesses)}")
    
    # Count unique addresses
    unique_addrs = set(a[0] for a in accesses)
    print(f"  Unique addresses: {len(unique_addrs)}")
    
    # Count unique rows accessed
    unique_rows = set(a[1] for a in accesses)
    print(f"  Unique rows: {len(unique_rows)}")
    print(f"  Rows accessed: {sorted(unique_rows)[:20]}{'...' if len(unique_rows) > 20 else ''}")
    
    # Count row activations (row switches)
    current_row = None
    row_acts = 0
    for addr, row, col in accesses:
        if current_row != row:
            row_acts += 1
            current_row = row
    print(f"  Row activations (switches): {row_acts}")
    
    # Analyze access pattern - how many times each row is visited
    row_visit_counts = defaultdict(int)
    current_row = None
    for addr, row, col in accesses:
        if current_row != row:
            row_visit_counts[row] += 1
            current_row = row
    
    print(f"  Row visit pattern (row: times_activated):")
    for row in sorted(row_visit_counts.keys())[:10]:
        print(f"    Row {row}: activated {row_visit_counts[row]} times")
    if len(row_visit_counts) > 10:
        print(f"    ... ({len(row_visit_counts)} rows total)")
    
    # Show first 30 accesses with row info
    print(f"\n  First 30 accesses (showing row changes):")
    current_row = None
    for i, (addr, row, col) in enumerate(accesses[:30]):
        marker = " <-- NEW ROW" if row != current_row else ""
        print(f"    {i:3d}: 0x{addr:08X} Row={row:3d} Col={col:4d}{marker}")
        current_row = row


def main():
    dram_config = DRAMConfig(row_buffer_bytes=1024, num_banks=4, num_rows=16384, element_size=1)
    
    # Test with 'small' workload
    workload = ConvWorkload(name="small", N=1, K=16, C=16, P=8, Q=8, R=3, S=3)
    
    print(f"Workload: small")
    print(f"  Input:  {workload.N} x {workload.C} x {workload.input_size['H']} x {workload.input_size['W']}")
    print(f"  Weight: {workload.K} x {workload.C} x {workload.R} x {workload.S}")
    print(f"  Output: {workload.N} x {workload.K} x {workload.P} x {workload.Q}")
    
    # Get mapping
    optimizer = PIMOptimizer()
    result = optimizer.optimize([workload], objective="latency")
    mapping = result.mappings[0]
    
    print(f"\nMapping:")
    print(f"  Block: {mapping.tile_info.get('block_h')} x {mapping.tile_info.get('block_w')}")
    print(f"  Layout: I={mapping.layout.get(0)}, W={mapping.layout.get(1)}, O={mapping.layout.get(2)}")
    print(f"  ILP row_acts: I={mapping.metrics.get('row_activations_input', 0):.1f}, "
          f"W={mapping.metrics.get('row_activations_weight', 0):.1f}, "
          f"O={mapping.metrics.get('row_activations_output', 0):.1f}")
    
    print(f"\nLoop bounds (DRAM levels):")
    for m in [2, 3]:
        if m in mapping.loop_bounds:
            print(f"  Level {m}: {mapping.loop_bounds[m]}")
    
    # Generate trace
    print(f"\nGenerating trace...")
    gen = TraceGenerator(dram_config)
    trace = gen.generate_trace(mapping, workload)
    print(f"Total trace lines: {len(trace)}")
    
    # Analyze each tensor
    analyze_trace(trace, dram_config, "Input", 0)
    analyze_trace(trace, dram_config, "Weight", 1)
    analyze_trace(trace, dram_config, "Output", 2)


if __name__ == "__main__":
    main()
