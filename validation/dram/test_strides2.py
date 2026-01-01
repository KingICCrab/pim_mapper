import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')
from pim_optimizer import PIMOptimizer
from pim_optimizer.workload import ConvWorkload

workload = ConvWorkload('small', P=8, Q=8, C=16, K=16, R=3, S=3, N=1)
optimizer = PIMOptimizer()
result = optimizer.optimize([workload], objective='latency')
mapping = result.mappings[0]

DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6
DIM_NAMES = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']

# Simulating parameters from debug_small_detail.py analysis
buffer_tile = {0:3, 1:3, 2:4, 3:8, 4:2, 5:16, 6:1}  # R,S,P,Q,C,K,N from Level 0+1
level2_factors = {d:1 for d in range(7)}
level2_factors[5] = 2  # K=2 at Level 2

H_in, W_in = 10, 10
block_h, block_w = 10, 10
num_blocks_h = (H_in + block_h - 1) // block_h  # 1
num_blocks_w = (W_in + block_w - 1) // block_w  # 1

input_rb_n = buffer_tile[DIM_N] * level2_factors[DIM_N]  # 1
input_rb_c = buffer_tile[DIM_C] * level2_factors[DIM_C]  # 2

input_tile_sizes_for_stride = {
    2: {
        DIM_N: input_rb_n,  # 1
        DIM_C: input_rb_c,  # 2
        DIM_P: block_w,     # 10
        DIM_Q: block_h,     # 10
    },
    3: {
        DIM_N: (workload.N + input_rb_n - 1) // input_rb_n,  # 1
        DIM_C: (workload.C + input_rb_c - 1) // input_rb_c,  # 8
        DIM_P: num_blocks_w,  # 1
        DIM_Q: num_blocks_h,  # 1
    },
}

print('tile_sizes[2]:', {DIM_NAMES[k]: v for k, v in input_tile_sizes_for_stride[2].items()})
print('tile_sizes[3]:', {DIM_NAMES[k]: v for k, v in input_tile_sizes_for_stride[3].items()})

# Build input loop order
input_layout_dims = [DIM_N, DIM_C, DIM_P, DIM_Q]
loop_order = []
for m in [3, 2]:
    if m not in mapping.permutation:
        continue
    perm = mapping.permutation[m]
    sorted_perms = sorted(perm.items(), key=lambda x: x[0], reverse=True)
    for p_level, dim in sorted_perms:
        if dim in input_layout_dims:
            loop_order.append((m, dim))

print('loop_order:', [(lv, DIM_NAMES[d]) for lv, d in loop_order])

# Simulate _compute_tile_wise_strides
row_size = 1024
layout = 'row_aligned'
tile_sizes = input_tile_sizes_for_stride
spatial_dims_2d = {DIM_P, DIM_Q}

# Build complete order
loop_order_set = set(loop_order)
complete_order = list(loop_order)
for level in [3, 2]:
    if level in tile_sizes:
        for dim in tile_sizes[level]:
            if (level, dim) not in loop_order_set:
                complete_order.append((level, dim))
                loop_order_set.add((level, dim))

print('complete_order:', [(lv, DIM_NAMES[d]) for lv, d in complete_order])

reversed_order = list(reversed(complete_order))
print('reversed_order:', [(lv, DIM_NAMES[d]) for lv, d in reversed_order])

# Calculate strides
strides = {}
stride = 1
padded_rb_tile_size = None

print('\n--- Stride Calculation ---')
for i, (level, dim) in enumerate(reversed_order):
    strides[(level, dim)] = stride
    
    if level in tile_sizes and dim in tile_sizes[level]:
        dim_tile_size = tile_sizes[level][dim]
    else:
        dim_tile_size = 1
    
    stride *= dim_tile_size
    
    # row_aligned logic
    if layout == 'row_aligned' and level == 2 and dim in spatial_dims_2d:
        remaining = any(lv == 2 and d in spatial_dims_2d for lv, d in reversed_order[i+1:])
        if not remaining:
            padded_rb_tile_size = ((stride + row_size - 1) // row_size) * row_size
            print(f'  -> After L2 spatial: stride={stride}, padded={padded_rb_tile_size}')
    
    if layout == 'row_aligned' and level == 3 and padded_rb_tile_size is not None:
        prev_level = reversed_order[i-1][0] if i > 0 else None
        if prev_level == 2:
            stride = padded_rb_tile_size
            strides[(level, dim)] = stride
            stride *= dim_tile_size
            print(f'  -> First L3 dim ({DIM_NAMES[dim]}): reset stride to {padded_rb_tile_size}')
    
    print(f'({level}, {DIM_NAMES[dim]}): stride={strides[(level, dim)]}, tile_size={dim_tile_size}, next_stride={stride}')

print('\n--- Final strides ---')
for (lv, dim), s in sorted(strides.items()):
    print(f'  ({lv}, {DIM_NAMES[dim]}): {s}')
