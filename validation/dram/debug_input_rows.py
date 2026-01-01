"""Debug Input row activations."""
import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
from trace_generator import TraceGenerator, DRAMConfig

workload = ConvWorkload(name='small', N=1, K=16, C=16, P=8, Q=8, R=3, S=3)
optimizer = PIMOptimizer()
result = optimizer.optimize([workload], objective='latency')
mapping = result.mappings[0]

gen = TraceGenerator(DRAMConfig())
buffer_tile = gen._compute_buffer_tile_size(mapping)

print('buffer_tile:', {k: v for k, v in buffer_tile.items() if v > 1})
print()

# Input per tile calculation
print('Input per tile calculation:')
print(f'  Q_tile={buffer_tile[3]}, S_tile={buffer_tile[1]}, P_tile={buffer_tile[2]}, R_tile={buffer_tile[0]}')
print(f'  C_tile={buffer_tile[4]}, N=1')
print()

# row_size = 1024 elements
H_in = workload.input_size['H']  # 10
W_in = workload.input_size['W']  # 10
print(f'  H_in={H_in}, W_in={W_in}')

row_size = 1024
nc_slice_size = H_in * W_in  # 100 elements
print(f'  (n,c) slice size = {nc_slice_size} elements')
print(f'  row_size = {row_size} elements')
print()

# slices per row
slices_per_row = row_size // nc_slice_size
print(f'  slices per row = {slices_per_row}')

# total (n,c) slices
total_slices = workload.N * workload.C  # 1 * 16 = 16
print(f'  total (n,c) slices = {total_slices}')

# rows needed
rows_needed = (total_slices + slices_per_row - 1) // slices_per_row
print(f'  rows needed = {rows_needed}')
print()

# ILP calculation
print('ILP expects:')
print(f'  C_dram = 16/2 = 8 iterations')
print(f'  Each iteration accesses C_tile=2 slices')
print(f'  If slices are contiguous within row, row_acts = ceil(16/10) = 2')
print(f'  But if thrashing between C slices... could be more')
print()

# Check actual trace
trace = gen.generate_trace(mapping, workload)
input_addrs = []
for line in trace:
    parts = line.strip().split()
    if len(parts) < 2:
        continue
    addr = int(parts[1], 16)
    bank = (addr >> 10) % 4
    if bank == 0:
        row = addr >> 12
        input_addrs.append((addr, row))

print(f'Actual trace:')
print(f'  Total Input accesses: {len(input_addrs)}')

# row switches
prev_row = None
row_switches = 0
for addr, row in input_addrs:
    if prev_row is not None and row != prev_row:
        row_switches += 1
    prev_row = row

print(f'  Row switches: {row_switches}')
print(f'  Row activations: {row_switches + 1}')
