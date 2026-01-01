"""Debug specific high row addresses."""
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

row_size = 1024
num_rows = 16384
bank_size = row_size * num_rows

# 统计每个 tensor 的 row 分布
input_rows = set()
weight_rows = set()
output_rows = set()

for line in trace:
    parts = line.strip().split()
    if len(parts) < 2:
        continue
    addr = int(parts[1], 16)
    bank_idx = addr // bank_size
    addr_in_bank = addr % bank_size
    row = addr_in_bank // row_size
    
    if bank_idx == 0:
        input_rows.add(row)
    elif bank_idx == 1:
        weight_rows.add(row)
    elif bank_idx == 2:
        output_rows.add(row)

print('Input rows accessed:', sorted(input_rows))
print(f'  Total unique rows: {len(input_rows)}')
print()
print('Weight rows accessed:', sorted(weight_rows)[:20], '...' if len(weight_rows) > 20 else '')
print(f'  Total unique rows: {len(weight_rows)}')
print()
print('Output rows accessed:', sorted(output_rows))
print(f'  Total unique rows: {len(output_rows)}')
