import sys
sys.path.insert(0, '.')
from trace_generator import TraceGenerator
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer import PIMOptimizer
from pim_optimizer.arch import PIMArch

workload = ConvWorkload(name='small', R=3, S=3, P=8, Q=8, C=16, K=16, N=1)
arch_config = {
    'pe_array': {'height': 4, 'width': 16},
    'global_buffer': {'size_kb': 128},
    'dram': {'row_buffer_bytes': 1024, 'num_banks': 4, 'num_rows': 16384}
}
arch = PIMArch(arch_config)
optimizer = PIMOptimizer(arch)
result = optimizer.optimize(workload, objective='latency')
mapping = result.mapping

trace_gen = TraceGenerator(
    row_buffer_bytes=1024, num_banks=4, num_rows=16384, element_size=1
)

layout_info = trace_gen.compute_layout_info(workload, mapping)

print('Input strides from trace_generator:')
for key, val in sorted(layout_info['input_strides'].items()):
    print(f'  {key}: {val}')

print()
print('DRAM factors:')
print(f'  P_l3={layout_info["P_l3"]}, Q_l3={layout_info["Q_l3"]}, C_l3={layout_info["C_l3"]}, N_l3={layout_info["N_l3"]}')

print()
print('Expected strides (manual calc):')
aligned_size = layout_info['input_aligned_tile_size']
P_l3, Q_l3, C_l3 = layout_info['P_l3'], layout_info['Q_l3'], layout_info['C_l3']
stride_p = aligned_size
stride_q = stride_p * P_l3
stride_c = stride_q * Q_l3
stride_n = stride_c * C_l3
print(f'  stride_p_l3 = {stride_p}')
print(f'  stride_q_l3 = {stride_p} * {P_l3} = {stride_q}')
print(f'  stride_c_l3 = {stride_q} * {Q_l3} = {stride_c}')
print(f'  stride_n_l3 = {stride_c} * {C_l3} = {stride_n}')
