#!/usr/bin/env python3
"""
Row Activation Validation Report Generator
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/validation/dram')

from pim_optimizer.arch.pim_arch import PIMArchitecture
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer import PIMOptimizer
from trace_generator import DRAMConfig
from full_validation import generate_trace_for_mapping, count_row_activations_from_trace
import numpy as np
from collections import defaultdict
import os


def main():
    arch = PIMArchitecture.from_yaml('/Users/haochenzhao/Projects/pim_optimizer/examples/configs/arch.yaml')
    cfg = DRAMConfig()
    row_bits = int(np.log2(cfg.row_buffer_bytes))
    bank_bits = int(np.log2(cfg.num_banks))

    workloads = [
        ('tiny', {'N': 1, 'K': 8, 'C': 8, 'P': 4, 'Q': 4, 'R': 3, 'S': 3}),
        ('small', {'N': 1, 'K': 16, 'C': 16, 'P': 8, 'Q': 8, 'R': 3, 'S': 3}),
    ]

    print('=' * 90)
    print('Row Activation Validation Report')
    print('=' * 90)
    print(f'DRAMConfig: row_buffer_bytes={cfg.row_buffer_bytes}, num_banks={cfg.num_banks}')
    print()

    for name, params in workloads:
        wl = ConvWorkload(name=name, **params)
        
        optimizer = PIMOptimizer(arch, verbose=False)
        result = optimizer.optimize([wl])
        model = optimizer.model
        
        # Generate trace
        trace_path = f'/Users/haochenzhao/Projects/pim_optimizer/validation/dram/validation_output/{name}_trace_report.txt'
        generate_trace_for_mapping(optimizer, wl, trace_path)
        
        # ILP predictions
        ilp_acts = {}
        for t_id, t_name in [(0, 'input'), (1, 'weight'), (2, 'output')]:
            var = model.getVarByName(f'total_row_acts_(0,{t_id})')
            ilp_acts[t_name] = var.X if var else 0
        
        # Trace statistics
        stats = count_row_activations_from_trace(trace_path, cfg)
        
        # Count unique rows
        with open(trace_path, 'r') as f:
            lines = f.readlines()
        unique_rows = defaultdict(set)
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            addr = int(parts[1], 16)
            addr_shifted = addr >> row_bits
            bank = addr_shifted & ((1 << bank_bits) - 1)
            row = addr_shifted >> bank_bits
            unique_rows[bank].add(row)
        total_unique_rows = sum(len(rows) for rows in unique_rows.values())
        
        print(f'Workload: {name}')
        print(f'  MACs: {wl.macs}')
        print(f'  ILP Predictions:')
        print(f'    Input:  {ilp_acts["input"]:.1f}')
        print(f'    Weight: {ilp_acts["weight"]:.1f}')
        print(f'    Output: {ilp_acts["output"]:.1f}')
        print(f'    Total:  {sum(ilp_acts.values()):.1f}')
        print(f'  Trace Statistics:')
        print(f'    Total row activations: {stats["total_row_acts"]}')
        print(f'    Unique (bank, row) pairs: {total_unique_rows}')
        print(f'    Per-bank activations: {dict(stats["per_bank_acts"])}')
        
        # Error calculation
        if stats["total_row_acts"] > 0:
            error = abs(sum(ilp_acts.values()) - stats["total_row_acts"]) / stats["total_row_acts"] * 100
        else:
            error = 0
        print(f'  Error: {error:.1f}%')
        print()

    print('=' * 90)
    print('Analysis:')
    print('=' * 90)
    print('''
The ILP model's row_aligned layout assumes that each tensor requires only 1 row activation.
However, actual trace counting shows many more row activations due to:
1. Tensor data spanning multiple DRAM rows
2. Interleaved access patterns causing row buffer conflicts

Key differences:
- ILP row_acts_row_aligned = product of bounds^(selection indicators)
- If all dimensions are processed at lower memory levels, row_acts_row_aligned = 1
- This doesn't account for the actual tensor size vs row_buffer_size

To fix the ILP model accuracy:
1. Consider tensor_bytes / row_buffer_bytes as a baseline
2. Account for reuse patterns that cause row re-activations
3. Consider bank interleaving effects
''')


if __name__ == '__main__':
    main()
