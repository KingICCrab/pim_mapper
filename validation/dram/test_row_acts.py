#!/usr/bin/env python3
"""Test row activation counting after stride fix."""
import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/validation/dram')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
from trace_generator import TraceGenerator, DRAMConfig

workload = ConvWorkload(name='small', N=1, K=16, C=16, P=8, Q=8, R=3, S=3)
optimizer = PIMOptimizer()
result = optimizer.optimize([workload], objective='latency')
mapping = result.mappings[0]

gen = TraceGenerator(DRAMConfig())
trace = gen.generate_trace(mapping, workload)

# Count row activations
bank_size = 1024 * 16384  # 16MB per bank
row_size = 1024
input_base = 0
weight_base = bank_size
output_base = 2 * bank_size

def count_row_activations(trace, base_addr, bank_size, row_size):
    last_row = None
    activations = 0
    for line in trace:
        if 'LD' in line or 'ST' in line:
            addr = int(line.split()[1], 16)
            # Check if in this tensor's bank
            bank = addr // bank_size
            if bank == base_addr // bank_size:
                row = (addr % bank_size) // row_size
                if row != last_row:
                    activations += 1
                    last_row = row
    return activations

input_acts = count_row_activations(trace, input_base, bank_size, row_size)
weight_acts = count_row_activations(trace, weight_base, bank_size, row_size)
output_acts = count_row_activations(trace, output_base, bank_size, row_size)

# ILP predictions
ilp = mapping.metrics
print('=== Row Activation Comparison ===')
print('Tensor     Trace    ILP')
print(f'Input      {input_acts:5d}    {ilp.get("row_activations_input", 0):5.0f}')
print(f'Weight     {weight_acts:5d}    {ilp.get("row_activations_weight", 0):5.0f}')
print(f'Output     {output_acts:5d}    {ilp.get("row_activations_output", 0):5.0f}')
