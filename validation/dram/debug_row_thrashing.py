"""Debug row thrashing in Input accesses."""
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
trace = gen.generate_trace(mapping, workload)

# 只看 Input (bank 0) 的访问
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

print(f'Total Input accesses: {len(input_addrs)}')
print()

# 统计每个 row 被访问的次数
row_counts = {}
for addr, row in input_addrs:
    row_counts[row] = row_counts.get(row, 0) + 1

print(f'Rows accessed: {sorted(row_counts.keys())}')
print(f'Access counts per row:')
for row in sorted(row_counts.keys()):
    print(f'  Row {row}: {row_counts[row]} accesses')
print()

# 分析 row switch 模式
print('Row switch pattern (first 100 switches):')
prev_row = None
switches = []
for i, (addr, row) in enumerate(input_addrs):
    if prev_row is not None and row != prev_row:
        switches.append((i, prev_row, row))
    prev_row = row

for i, (idx, from_row, to_row) in enumerate(switches[:100]):
    print(f'  Switch {i+1} at access {idx}: row {from_row} -> {to_row}')

print()
print(f'Total row switches: {len(switches)}')
