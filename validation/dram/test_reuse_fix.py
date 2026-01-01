"""Test the reuse-aware trace generation fix."""

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
from trace_generator import TraceGenerator, DRAMConfig
from typing import Dict, List


def count_row_activations(trace: List[str], dram_config: DRAMConfig) -> Dict[str, int]:
    """Count row activations per tensor from trace."""
    row_size = dram_config.row_buffer_bytes
    num_rows = dram_config.num_rows
    bank_size = row_size * num_rows  # 1024 * 16384 = 16MB
    
    # Bank to tensor mapping (based on base addresses)
    # Input: bank 0 (0x00000000), Weight: bank 1 (0x01000000), Output: bank 2 (0x02000000)
    
    # Track current row per tensor
    current_row = {}
    row_acts = {'Input': 0, 'Weight': 0, 'Output': 0}
    
    for line in trace:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        addr = int(parts[1], 16)
        
        # Determine tensor from base address
        bank_idx = addr // bank_size
        if bank_idx == 0:
            tensor = 'Input'
        elif bank_idx == 1:
            tensor = 'Weight'
        elif bank_idx == 2:
            tensor = 'Output'
        else:
            continue
        
        # Row within the bank
        addr_in_bank = addr % bank_size
        row = addr_in_bank // row_size
        
        if tensor not in current_row or current_row[tensor] != row:
            current_row[tensor] = row
            row_acts[tensor] += 1
    
    return row_acts


def main():
    workload = ConvWorkload(name='small', N=1, K=16, C=16, P=8, Q=8, R=3, S=3)
    optimizer = PIMOptimizer()
    result = optimizer.optimize([workload], objective='latency')
    mapping = result.mappings[0]

    print('=== Testing reuse-aware trace generation ===')
    print()

    # Generate trace
    gen = TraceGenerator(DRAMConfig())
    trace = gen.generate_trace(mapping, workload)
    print(f'Total trace lines: {len(trace)}')
    print()

    # Count row activations
    row_acts = count_row_activations(trace, DRAMConfig())
    print('Trace-based row activations:')
    for tensor, count in row_acts.items():
        print(f'  {tensor}: {count}')
    print()

    # Compare with ILP (row activations stored in mapping.metrics)
    ilp_acts = {
        'Input': mapping.metrics.get('row_activations_input', 0),
        'Weight': mapping.metrics.get('row_activations_weight', 0),
        'Output': mapping.metrics.get('row_activations_output', 0),
    }
    print('ILP row activations:')
    for tensor, val in ilp_acts.items():
        print(f'  {tensor}: {val:.0f}')
    print()

    print('Comparison:')
    all_match = True
    for tensor in ['Input', 'Weight', 'Output']:
        ilp_val = ilp_acts[tensor]
        trace_val = row_acts[tensor]
        match = '✓' if abs(ilp_val - trace_val) < 0.5 else '✗'
        if match == '✗':
            all_match = False
        print(f'  {tensor}: ILP={ilp_val:.0f}, Trace={trace_val}, {match}')
    
    print()
    if all_match:
        print('✓ All values match!')
    else:
        print('✗ Some values do not match')


if __name__ == '__main__':
    main()
