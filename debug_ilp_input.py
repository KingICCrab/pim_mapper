#!/usr/bin/env python3
"""Debug ILP Input row_acts_aligned calculation."""

import sys
sys.path.insert(0, 'src')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload

# small-v2 workload
workload = ConvWorkload(
    name='small-v2',
    N=1, C=16, P=16, Q=16,
    K=16, R=3, S=3,
    stride=(1, 1), dilation=(1, 1)
)

print(f'Workload: {workload.name}')
print(f'Dimensions: R={workload.R}, S={workload.S}, P={workload.P}, Q={workload.Q}, C={workload.C}, K={workload.K}, N={workload.N}')

# Relevancy matrix
dim_names = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
print(f'\nRelevancy O[dim][tensor]:')
print(f'  {"Dim":<4} {"Input":<6} {"Weight":<6} {"Output":<6}')
for i, name in enumerate(dim_names):
    print(f'  {name:<4} {workload.O[i][0]:<6} {workload.O[i][1]:<6} {workload.O[i][2]:<6}')

# Input relevant dims
input_relevant = [i for i in range(7) if workload.O[i][0] == 1]
input_irrelevant = [i for i in range(7) if workload.O[i][0] == 0]
print(f'\nInput relevant dims: {[dim_names[i] for i in input_relevant]}')
print(f'Input irrelevant dims: {[dim_names[i] for i in input_irrelevant]}')

# Run optimizer
print('\n' + '='*60)
print('Running optimizer...')
optimizer = PIMOptimizer()
result = optimizer.optimize([workload], objective="latency")
mapping = result.mappings[0]

print(f'\nSolver Status: {result.solver_status}')

# Get DRAM factors
print('\n' + '='*60)
print('DRAM Level Factors (Level 3):')
dram_factors = {}
if 3 in mapping.loop_bounds:
    for key in ['spatial', 'temporal']:
        if key in mapping.loop_bounds[3]:
            for dim, bound in mapping.loop_bounds[3][key].items():
                dram_factors[dim] = dram_factors.get(dim, 1) * bound

for i, name in enumerate(dim_names):
    factor = dram_factors.get(i, 1)
    print(f'  {name}: {factor}')

# Expected row_acts_aligned for Input
print('\n' + '='*60)
print('Expected row_acts_aligned for Input:')

# Method 1: Product of all DRAM factors (current implementation with all_dims)
all_product = 1
for i in range(7):
    all_product *= dram_factors.get(i, 1)
print(f'  Using all_dims: Π(all DRAM factors) = {all_product}')

# Method 2: Product of relevant DRAM factors only
relevant_product = 1
for i in input_relevant:
    relevant_product *= dram_factors.get(i, 1)
print(f'  Using relevant_dims only: Π(relevant DRAM factors) = {relevant_product}')

# Actual ILP result
ilp_input = mapping.metrics.get('row_activations_input', 0)
print(f'\nActual ILP result: {ilp_input}')

print('\n' + '='*60)
print('Analysis:')
print(f'  Expected (relevant only): {relevant_product}')
print(f'  ILP result: {ilp_input}')

if abs(ilp_input - relevant_product) < 1:
    print('  ✓ ILP matches expected (relevant dims)')
elif abs(ilp_input - all_product) < 1:
    print('  ✗ ILP matches all_dims (WRONG - includes irrelevant K)')
else:
    print(f'  ? ILP value ({ilp_input}) does not match either expected value')
    print(f'    Ratio to relevant: {ilp_input / relevant_product:.4f}')
    print(f'    Ratio to all: {ilp_input / all_product:.4f}')
