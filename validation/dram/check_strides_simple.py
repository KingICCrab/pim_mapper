#!/usr/bin/env python3
"""Simple script to check input_strides values."""
import sys
sys.path.insert(0, '.')

from trace_generator import TraceGenerator
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer import PIMOptimizer
from pim_optimizer.arch.pim_arch import PIMArchitecture

# Constants
DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6
dim_names = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']

# Setup
workload = ConvWorkload(name='small', R=3, S=3, P=8, Q=8, C=16, K=16, N=1)
arch_config = {
    'pe_array': {'height': 4, 'width': 16},
    'global_buffer': {'size_kb': 128},
    'dram': {'row_buffer_bytes': 1024, 'num_banks': 4, 'num_rows': 16384}
}
arch = PIMArchitecture(arch_config)
optimizer = PIMOptimizer(arch)
result = optimizer.optimize([workload], objective='latency')
mapping = result.mapping

trace_gen = TraceGenerator(
    row_buffer_bytes=1024, num_banks=4, num_rows=16384, element_size=1
)

layout_info = trace_gen.compute_layout_info(workload, mapping)

print("=" * 60)
print("INPUT STRIDES ANALYSIS")
print("=" * 60)

print("\nRaw input_strides dictionary:")
for key, val in sorted(layout_info['input_strides'].items()):
    level, dim = key
    print(f"  {key} -> ({level}, {dim_names[dim]}): {val}")

print("\nDRAM factors:")
print(f"  P_l3 = {layout_info['P_l3']}")
print(f"  Q_l3 = {layout_info['Q_l3']}")
print(f"  C_l3 = {layout_info['C_l3']}")
print(f"  N_l3 = {layout_info['N_l3']}")

print("\nTile sizes:")
print(f"  input_dram_tile_size = {layout_info['input_dram_tile_size']}")
print(f"  input_aligned_tile_size = {layout_info['input_aligned_tile_size']}")
print(f"  H_per_tile = {layout_info['H_per_tile']}")
print(f"  W_per_tile = {layout_info['W_per_tile']}")
print(f"  C_per_tile = {layout_info['C_per_tile']}")

print("\nExpected L3 strides (manual calculation):")
aligned_size = layout_info['input_aligned_tile_size']
P_l3 = layout_info['P_l3']
Q_l3 = layout_info['Q_l3']
C_l3 = layout_info['C_l3']
N_l3 = layout_info['N_l3']

expected_stride_p = aligned_size
expected_stride_q = expected_stride_p * P_l3
expected_stride_c = expected_stride_q * Q_l3
expected_stride_n = expected_stride_c * C_l3

print(f"  stride_p_l3 = aligned_size = {expected_stride_p}")
print(f"  stride_q_l3 = {expected_stride_p} * P_l3({P_l3}) = {expected_stride_q}")
print(f"  stride_c_l3 = {expected_stride_q} * Q_l3({Q_l3}) = {expected_stride_c}")
print(f"  stride_n_l3 = {expected_stride_c} * C_l3({C_l3}) = {expected_stride_n}")

print("\nExpected L2 strides (within tile):")
H_per_tile = layout_info['H_per_tile']
W_per_tile = layout_info['W_per_tile']
C_per_tile = layout_info['C_per_tile']

print(f"  stride_p_l2 = 1 (innermost)")
print(f"  stride_q_l2 = W_per_tile = {W_per_tile}")
print(f"  stride_c_l2 = H_per_tile * W_per_tile = {H_per_tile} * {W_per_tile} = {H_per_tile * W_per_tile}")
print(f"  stride_n_l2 = H * W * C_per_tile = {H_per_tile * W_per_tile * C_per_tile}")

print("\n" + "=" * 60)
print("COMPARISON: Actual vs Expected")
print("=" * 60)
actual_strides = layout_info['input_strides']

comparisons = [
    ((2, DIM_P), 1, "L2, P"),
    ((2, DIM_Q), W_per_tile, "L2, Q"),
    ((2, DIM_C), H_per_tile * W_per_tile, "L2, C"),
    ((2, DIM_N), H_per_tile * W_per_tile * C_per_tile, "L2, N"),
    ((3, DIM_P), expected_stride_p, "L3, P"),
    ((3, DIM_Q), expected_stride_q, "L3, Q"),
    ((3, DIM_C), expected_stride_c, "L3, C"),
    ((3, DIM_N), expected_stride_n, "L3, N"),
]

for key, expected, name in comparisons:
    actual = actual_strides.get(key, "MISSING")
    match = "✓" if actual == expected else "✗ MISMATCH"
    print(f"  {name}: actual={actual}, expected={expected} {match}")
