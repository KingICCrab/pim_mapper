#!/usr/bin/env python3
"""Debug Input row count discrepancy."""

import sys
sys.path.insert(0, 'src')

from validation.dram.trace_generator import TraceGenerator, DRAMConfig
from validation.dram.trace_generator import DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N
import pim_optimizer.workload.conv as conv_module
import pim_optimizer.mapping as mapping_module
ConvWorkload = conv_module.ConvWorkload
Mapping = mapping_module.Mapping

# small-v2 workload
workload = ConvWorkload(
    N=1, C=16, P=16, Q=16,
    K=16, R=3, S=3,
    stride=(1, 1), dilation=(1, 1)
)

print(f'Workload: N=1, C=16, P=16, Q=16, K=16, R=3, S=3')
print(f'Input: N=1, C=16, H=18, W=18 = {1*16*18*18} elements')

# Mapping from analysis.txt
mapping_cfg = {
    'loop_bounds': {
        0: {'H': {3: 8, 5: 2}, 'W': {2: 4, 4: 4}, 'Internal': {}, 'temporal': {0: 3, 1: 3}},
        1: {'spatial': {}, 'temporal': {}},
        2: {'spatial': {}, 'temporal': {}},
        3: {'spatial': {}, 'temporal': {2: 4, 3: 2, 4: 4, 5: 8}},  # P=4, Q=2, C=4, K=8
    },
    'permutation': {
        0: {0: 0, 1: 1},
        1: {},
        2: {},
        3: {0: 5, 1: 4, 2: 2, 3: 3},  # K -> C -> P -> Q (inner to outer)
    },
    'layout': {0: 'row_aligned', 1: 'sequential', 2: 'sequential'},
    'tile_info': {'block_h': 18, 'block_w': 18},
}

mapping = Mapping.from_dict(mapping_cfg)

print('\n=== DRAM Tiling Analysis ===')
# DRAM level (Level 3) factors
P_l3 = mapping.loop_bounds[3]['temporal'].get(2, 1)
Q_l3 = mapping.loop_bounds[3]['temporal'].get(3, 1)
C_l3 = mapping.loop_bounds[3]['temporal'].get(4, 1)
K_l3 = mapping.loop_bounds[3]['temporal'].get(5, 1)
N_l3 = mapping.loop_bounds[3]['temporal'].get(6, 1)
print(f'DRAM factors: P_l3={P_l3}, Q_l3={Q_l3}, C_l3={C_l3}, K_l3={K_l3}, N_l3={N_l3}')

# Per-tile dimensions
P_per_tile = workload.P // P_l3  # 16 / 4 = 4
Q_per_tile = workload.Q // Q_l3  # 16 / 2 = 8
C_per_tile = workload.C // C_l3  # 16 / 4 = 4
R_per_tile = workload.R  # 3 (no DRAM tiling on R)
S_per_tile = workload.S  # 3 (no DRAM tiling on S)
print(f'Per-tile: P={P_per_tile}, Q={Q_per_tile}, C={C_per_tile}, R={R_per_tile}, S={S_per_tile}')

# Input DRAM tile H, W (sliding window)
H_per_tile = P_per_tile + R_per_tile - 1  # 4 + 3 - 1 = 6
W_per_tile = Q_per_tile + S_per_tile - 1  # 8 + 3 - 1 = 10
print(f'Input tile: H_per_tile={H_per_tile}, W_per_tile={W_per_tile}')

# Input DRAM tile size
input_tile_size = H_per_tile * W_per_tile * C_per_tile
print(f'Input DRAM tile size = {H_per_tile} × {W_per_tile} × {C_per_tile} = {input_tile_size} bytes')

# Row alignment
row_size = 1024
rows_per_tile = (input_tile_size + row_size - 1) // row_size
aligned_tile_size = rows_per_tile * row_size
print(f'Row size = {row_size}, rows_per_tile = {rows_per_tile}, aligned_size = {aligned_tile_size}')

# Expected row activations
# For Input: changes when P, Q, C, or N tile changes (not K)
expected_row_acts = P_l3 * Q_l3 * C_l3 * N_l3
print(f'\nExpected Input row activations = P_l3 × Q_l3 × C_l3 × N_l3 = {P_l3} × {Q_l3} × {C_l3} × {N_l3} = {expected_row_acts}')

print('\n=== Current Implementation Analysis ===')
dram_cfg = DRAMConfig(element_size=1, row_buffer_bytes=1024)
gen = TraceGenerator(dram_config=dram_cfg)

buffer_tile = gen._compute_buffer_tile_size(mapping)
print(f'buffer_tile: {buffer_tile}')

# Generate trace
trace = gen.generate_trace(mapping, workload)
input_acc = [l for l in trace if l.startswith('LD 0x00')]
print(f'Input accesses: {len(input_acc)}')

# Count unique rows
rows = set()
for acc in input_acc:
    addr = int(acc.split('0x')[1], 16)
    row = addr // 1024
    rows.add(row)

print(f'Unique rows (current): {len(rows)}')
print(f'Expected: {expected_row_acts}')

if len(rows) == expected_row_acts:
    print('\n✅ SUCCESS! Input row activations match expected value!')
else:
    print(f'\n❌ MISMATCH! Got {len(rows)}, expected {expected_row_acts}')

# Show layout info
H_in = workload.input_size['H']
W_in = workload.input_size['W']
layout_info = gen._compute_data_layouts(mapping, workload, H_in, W_in, buffer_tile)
print(f'\n=== Layout Info ===')
print(f"H_per_tile = {layout_info['H_per_tile']} (access tile)")
print(f"W_per_tile = {layout_info['W_per_tile']} (access tile)")
print(f"C_per_tile = {layout_info['C_per_tile']}")
print(f"block_h = {layout_info['block_h']} (data layout)")
print(f"block_w = {layout_info['block_w']} (data layout)")
print(f"input_dram_tile_size = {layout_info['input_dram_tile_size']} bytes")
print(f"input_aligned_tile_size = {layout_info['input_aligned_tile_size']} bytes")

print(f'\n=== Input Strides (row_aligned) ===')
for key, val in sorted(layout_info['input_strides'].items()):
    level, dim = key
    dim_name = {0: 'R', 1: 'S', 2: 'P', 3: 'Q', 4: 'C', 5: 'K', 6: 'N'}.get(dim, str(dim))
    print(f'  Level {level}, {dim_name}: {val}')

dram_cfg = DRAMConfig(element_size=1, row_buffer_bytes=1024)
gen = TraceGenerator(dram_config=dram_cfg)

# 看 layout_info
buffer_tile = gen._compute_buffer_tile_size(mapping)
print(f'Buffer tile: {buffer_tile}')

H_in = workload.input_size['H']
W_in = workload.input_size['W']
layout_info = gen._compute_data_layouts(mapping, workload, H_in, W_in, buffer_tile)

print(f'\nInput strides:')
for key, val in sorted(layout_info['input_strides'].items()):
    print(f'  {key}: {val}')

print(f'\nblock_h={layout_info["block_h"]}, block_w={layout_info["block_w"]}')
print(f'row_size={layout_info["row_size"]}')

# Check what stride_c_l3 actually is
input_strides = layout_info['input_strides']
stride_c_l3 = input_strides.get((3, DIM_C), None)
print(f'\nstride_c_l3 = input_strides.get((3, {DIM_C})) = {stride_c_l3}')

# Check if (3, 4) is in the strides
print(f'(3, 4) in input_strides: {(3, 4) in input_strides}')
print(f'DIM_C == 4: {DIM_C == 4}')

# Input DRAM tile size and row alignment
C = workload.C  # 16
C_l3 = mapping.loop_bounds[3]['temporal'].get(4, 1)  # DRAM factor for C
print(f'\nC={C}, C_l3={C_l3}')

# Input DRAM tile size (per C tile)
# tile contains: R×S×(C/C_l3)×H×W = 3×3×4×18×18 = 11664 elements?
# No wait, tile is for subset of channels: C/C_l3 = 16/4 = 4 channels
C_per_tile = C // C_l3
print(f'C_per_tile = {C_per_tile}')

# Input data size per DRAM tile: block_h × block_w × C_per_tile
block_h = layout_info['block_h']  # 18
block_w = layout_info['block_w']  # 18
tile_size = block_h * block_w * C_per_tile
print(f'Input DRAM tile size: {block_h}×{block_w}×{C_per_tile} = {tile_size} bytes')

# With row_aligned, each tile should occupy ceil(tile_size/1024) rows
import math
row_size = 1024
rows_per_tile = math.ceil(tile_size / row_size)
aligned_tile_size = rows_per_tile * row_size
print(f'Rows per tile: {rows_per_tile}')
print(f'Aligned tile size: {aligned_tile_size}')

# DRAM factors for Input: P=4, Q=2, C=4, N=1 (K=8 is irrelevant)
# Number of DRAM tiles = P × Q × C × N = 4 × 2 × 4 × 1 = 32
num_tiles = 4 * 2 * 4 * 1
print(f'\nNumber of DRAM tiles: {num_tiles}')
print(f'Expected total rows: {num_tiles * rows_per_tile}')

# 生成 trace
trace = gen.generate_trace(mapping, workload)
input_acc = [l for l in trace if l.startswith('LD 0x00')]
print(f'\nInput accesses: {len(input_acc)}')

# Analyze unique rows
rows = set()
row_list = []
for acc in input_acc:
    addr = int(acc.split('0x')[1], 16)
    row = addr // 1024
    rows.add(row)
    row_list.append(row)

print(f'Unique rows: {len(rows)}')
print(f'Expected: 32 (4×2×4×1)')

# Show first few addresses
print('\nFirst 20 accesses:')
for i, acc in enumerate(input_acc[:20]):
    addr = int(acc.split('0x')[1], 16)
    row = addr // 1024
    offset = addr % 1024
    print(f'  {i}: addr={addr}, row={row}, offset={offset}')

# Show row distribution
print(f'\nRows accessed: {sorted(rows)}')

# Analyze tile distribution - which c_tile values are being used?
print('\n=== Analyzing address pattern ===')
tile_addresses = {}
for acc in input_acc:
    addr = int(acc.split('0x')[1], 16)
    # Calculate which c_tile this should belong to based on stride_c_l3 = 1024
    c_tile_from_addr = addr // 1024  # Since stride_c_l3 = 1024
    if c_tile_from_addr not in tile_addresses:
        tile_addresses[c_tile_from_addr] = set()
    tile_addresses[c_tile_from_addr].add(addr)

print(f'Addresses grouped by c_tile (addr // 1024):')
for c_tile in sorted(tile_addresses.keys()):
    addrs = sorted(tile_addresses[c_tile])
    print(f'  c_tile {c_tile}: {len(addrs)} addresses, range [{addrs[0]}, {addrs[-1]}]')

# Expected: 4 C tiles × 4 P tiles × 2 Q tiles = 32 tiles
# But we only see 5 rows! Something is very wrong with the address calculation

# Let's trace through the first few iterations manually
print('\n=== Manual trace of c_tile calculation ===')
print('buffer_tile[DIM_C] =', buffer_tile[DIM_C])
print('c_size should be buffer_tile[DIM_C] =', buffer_tile[DIM_C])
print('For c in range(c_start, c_start + c_size):')
print('  c_tile = c // c_size')
print('  c_local = c % c_size')

# The actual C values being iterated
# Since DRAM loop has C_l3 = 4 iterations, and buffer_tile[C] = 4
# c_start = c_tile_idx * tile_stride (where tile_stride = buffer_tile[C] = 4)
# So c_start goes: 0, 4, 8, 12
# And c ranges: [0,3], [4,7], [8,11], [12,15]

# For each DRAM tile (c_tile_idx):
# c_tile = c // c_size = c // 4
# When c_start=0: c=0,1,2,3 → c_tile = 0,0,0,0 (all same!)
# When c_start=4: c=4,5,6,7 → c_tile = 1,1,1,1 (all same!)
# ...
# This looks correct! So why are addresses all small?

print('\nLet me check: stride_c_l3 * c_tile for each c_tile:')
for c_tile in range(4):
    base = c_tile * 1024  # stride_c_l3 = 1024
    print(f'  c_tile={c_tile}: tile_base = {base} (row {base // 1024})')

print('\nExpected address range for C=16, P=16, Q=16 (4 C tiles × block layout):')
print('  Total DRAM tiles (for Input): P_l3 × Q_l3 × C_l3 × N_l3 = 4 × 2 × 4 × 1 = 32')
print('  Expected address range: [0, 32 × 1024) = [0, 32768)')
print(f'  Actual max address: {max(int(acc.split("0x")[1], 16) for acc in input_acc)}')

# Check DRAM loop structure
print('\n=== DRAM loop structure ===')
dram_loops = gen._build_dram_loop_structure(mapping, workload, buffer_tile)
for i, loop in enumerate(dram_loops):
    print(f'  Loop {i}: dim={loop["dim"]}, bound={loop["bound"]}, level={loop.get("level", "?")}')

# Compute expected iterations
total_iterations = 1
for loop in dram_loops:
    total_iterations *= loop['bound']
print(f'\nTotal DRAM loop iterations: {total_iterations}')
print(f'Input accesses per iteration: ~{len(input_acc) / total_iterations:.1f}')

# Let me add debug output to trace_generator temporarily
print('\n=== Debug address calculation with smaller example ===')

# Create a simplified test case to trace through
# For this we need to see what c_start and c_tile values are
# Let's hook into the calculation

# Check: for each C DRAM loop iteration, what is c_start?
# DRAM loop has C_l3 = 4 iterations
# tile_strides[DIM_C] should give the stride between tiles

# From _build_dram_loop_structure, tile_stride is computed as tile_size[m][d]
# tile_size at DRAM level (level 3) = buffer_tile accumulated

print('\nExpected C iteration:')
print('  c_tile_idx in range(4):  # C_l3 = 4')
print('  c_start = c_tile_idx * tile_stride[C]')
print('  tile_stride[C] = buffer_tile[C] = 4')
print('  So: c_start = 0, 4, 8, 12 for each DRAM iteration')
print('')
print('  Then in _generate_tile_accesses:')
print('  for c in range(c_start, c_start + c_size):')
print('    where c_size = buffer_tile[C] = 4')
print('  So: c = 0,1,2,3 | 4,5,6,7 | 8,9,10,11 | 12,13,14,15')
print('')
print('  But then c_tile = c // c_size computes:')
print('    c=0..3 → c_tile=0')
print('    c=4..7 → c_tile=1')
print('    c=8..11 → c_tile=2')
print('    c=12..15 → c_tile=3')
print('')
print('  This is correct! So why do we see c_tile=4?')
print('')

# The issue might be in block_idx_in_tile calculation
# block_idx_in_tile = c_local * num_blocks_h * num_blocks_w + h_block * num_blocks_w + w_block
# For H_in=18, W_in=18, block_h=18, block_w=18:
# num_blocks_h = ceil(18/18) = 1
# num_blocks_w = ceil(18/18) = 1
# So block_idx_in_tile = c_local * 1 * 1 + h_block * 1 + w_block
#                      = c_local + h_block + w_block
# With c_local=0..3, h_block=0, w_block=0:
# block_idx_in_tile = 0, 1, 2, 3

# Wait! The issue is num_blocks is computed from H_in/block_h
# But block_h = 18 (full input size), not the actual DRAM block size

print('num_blocks_h = H_in / block_h = 18 / 18 = 1')
print('num_blocks_w = W_in / block_w = 18 / 18 = 1')
print('block_size = block_h * block_w = 18 * 18 = 324')
print('')
print('Address calculation:')
print('  tile_base = c_tile * stride_c_l3 = c_tile * 1024')
print('  block_idx_in_tile = c_local * 1 * 1 + h_block * 1 + w_block = c_local')
print('  addr = tile_base + block_idx_in_tile * block_size + offset')
print('       = c_tile * 1024 + c_local * 324 + offset')
print('')
print('For c_tile=0, c_local=0..3:')
for c_local in range(4):
    base = 0 * 1024 + c_local * 324
    print(f'  c_local={c_local}: addr_base = {base} (row {base // 1024})')

print('')
print('For c_tile=1, c_local=0..3:')
for c_local in range(4):
    base = 1 * 1024 + c_local * 324
    print(f'  c_local={c_local}: addr_base = {base} (row {base // 1024})')

print('\n=== ROOT CAUSE ANALYSIS ===')
print('DRAM tile size = H × W × C_per_tile = 18 × 18 × 4 = 1296 bytes')
print('Row size = 1024 bytes')
print('Rows needed per tile = ceil(1296 / 1024) = 2 rows')
print('')
print('For row_aligned layout, stride_c_l3 should be:')
print('  aligned_tile_size = 2 * 1024 = 2048 bytes')
print('')
print('But current stride_c_l3 = 1024 (only 1 row!)')
print('This is WRONG!')
print('')
print('The issue is in _compute_tile_wise_strides():')
print('  It only pads the innermost spatial dims (P, Q), not the full tile')
print('')
print('Expected row activations with correct stride:')
print('  C tiles = 4, each takes 2 rows → 8 rows for C')
print('  P × Q tiles = 4 × 2 = 8')
print('  But P, Q dont need additional rows (they are in same DRAM tile as C)')
print('  Wait no - for Input, P and Q are part of H,W which is INSIDE the DRAM tile')
print('')
print('Let me reconsider Input DRAM layout:')
print('  Input shape: [N, C, H, W] = [1, 16, 18, 18]')
print('  DRAM tiling: N_l3=1, C_l3=4, (P,Q,R,S maps to H,W through sliding window)')
print('  For Input, DRAM tile = all data needed for one (N_tile, C_tile)')
print('  tile size = H × W × C_per_tile = 18 × 18 × 4 = 1296 bytes')
print('')
print('  With row_aligned:')
print('    rows_per_tile = ceil(1296 / 1024) = 2')
print('    stride_c_l3 should = 2 * 1024 = 2048')
print('')
print('  Number of DRAM tiles = N_l3 × C_l3 = 1 × 4 = 4')
print('  Expected row activations = 4 tiles × 2 rows/tile = 8 rows? No wait...')
print('')
print('Actually for row_aligned, each unique DRAM tile access = 1 row activation')
print('(even if tile spans multiple rows, it counts as 1 row activation per tile)')
print('')
print('So expected = 4 C tiles = 4 row activations? No thats unique tiles.')
print('')
print('Wait, the formula: row_acts = P_l3 × Q_l3 × C_l3 × N_l3 = 4 × 2 × 4 × 1 = 32')
print('This counts DRAM LOOP ITERATIONS that change Input!')
print('')
print('Input changes when P, Q, C, or N tile changes')
print('But K changes dont affect Input!')
print('')
print('DRAM loops: Q=2, P=4, C=4, K=8 (outer to inner)')
print('For Input: K is irrelevant, so Input changes when Q,P,C change')
print('Input row activations = Q_l3 × P_l3 × C_l3 = 2 × 4 × 4 = 32')
print('')
print('So we expect 32 unique row activations for Input!')
print('Each is a row-aligned DRAM tile access')
print('')
print('Current implementation only gets 5 unique rows because:')
print('  1. stride_c_l3 = 1024 (should be 2048)')
print('  2. P_l3, Q_l3 dont have separate strides computed')
