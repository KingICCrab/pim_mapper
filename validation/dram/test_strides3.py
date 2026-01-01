"""Test stride calculation with actual small workload parameters"""

DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = range(7)
DIM_NAMES = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']

def compute_tile_wise_strides(loop_order, tile_sizes, layout, row_size):
    """Simplified version of _compute_tile_wise_strides"""
    strides = {}
    stride = 1
    
    loop_order_set = set(loop_order)
    complete_order = list(loop_order)
    
    # Add missing dims from tile_sizes (Level 3 first, then Level 2)
    for level in [3, 2]:
        if level in tile_sizes:
            for dim in tile_sizes[level]:
                if (level, dim) not in loop_order_set:
                    complete_order.append((level, dim))
                    loop_order_set.add((level, dim))
    
    print(f"complete_order = {[(l, DIM_NAMES[d]) for l, d in complete_order]}")
    
    reversed_order = list(reversed(complete_order))
    print(f"reversed_order = {[(l, DIM_NAMES[d]) for l, d in reversed_order]}")
    
    spatial_dims_2d = {DIM_P, DIM_Q}
    stride_before_padding = None
    padded_rb_tile_size = None
    
    for i, (level, dim) in enumerate(reversed_order):
        dim_name = DIM_NAMES[dim]
        
        strides[(level, dim)] = stride
        
        if level in tile_sizes and dim in tile_sizes[level]:
            dim_tile_size = tile_sizes[level][dim]
        else:
            dim_tile_size = 1
        
        print(f"  [{i}] Level {level}, {dim_name}: stride={stride}, dim_tile_size={dim_tile_size}")
        
        stride *= dim_tile_size
        
        # row_aligned padding logic
        if layout == "row_aligned" and level == 2 and dim in spatial_dims_2d:
            remaining_spatial_at_level2 = False
            for j in range(i + 1, len(reversed_order)):
                lv, d = reversed_order[j]
                if lv == 2 and d in spatial_dims_2d:
                    remaining_spatial_at_level2 = True
                    break
            
            if not remaining_spatial_at_level2:
                stride_before_padding = stride
                if stride % row_size != 0:
                    padded_rb_tile_size = ((stride + row_size - 1) // row_size) * row_size
                else:
                    padded_rb_tile_size = stride
                print(f"      -> After spatial dims: stride_before_padding={stride_before_padding}, padded={padded_rb_tile_size}")
        
        # Level 3 dims after padding
        if layout == "row_aligned" and level == 3 and padded_rb_tile_size is not None:
            prev_level = reversed_order[i-1][0] if i > 0 else None
            if prev_level == 2:
                stride = padded_rb_tile_size
                strides[(level, dim)] = stride
                stride *= dim_tile_size
                print(f"      -> Reset stride to padded: {padded_rb_tile_size}, now stride={stride}")
    
    return strides

# Small workload Input parameters
input_layout_loop_order = []  # Simulate empty/default
input_tile_sizes = {
    2: {DIM_N: 1, DIM_C: 2, DIM_P: 10, DIM_Q: 10},  # block_w=10, block_h=10, input_rb_c=2
    3: {DIM_N: 1, DIM_C: 8, DIM_P: 2, DIM_Q: 1},   # level3_factors: P=2, Q=1, C=8, N=1
}
layout = "row_aligned"
row_size = 1024

print("=" * 60)
print("Input tile sizes:")
print(f"  Level 2: N={input_tile_sizes[2][DIM_N]}, C={input_tile_sizes[2][DIM_C]}, P={input_tile_sizes[2][DIM_P]}, Q={input_tile_sizes[2][DIM_Q]}")
print(f"  Level 3: N={input_tile_sizes[3][DIM_N]}, C={input_tile_sizes[3][DIM_C]}, P={input_tile_sizes[3][DIM_P]}, Q={input_tile_sizes[3][DIM_Q]}")
print()

strides = compute_tile_wise_strides(input_layout_loop_order, input_tile_sizes, layout, row_size)

print()
print("Final strides:")
for level in [2, 3]:
    for dim in [DIM_N, DIM_C, DIM_P, DIM_Q]:
        if (level, dim) in strides:
            print(f"  (L{level}, {DIM_NAMES[dim]}): {strides[(level, dim)]}")

print()
print("=" * 60)
print("EXPECTED for row_aligned:")
print("  Each DRAM tile = 200 elements (10x10x2), padded to 1024")
print("  P_l3=2, C_l3=8 -> 16 DRAM tiles")
print("  L3 strides should all be multiples of 1024 (one tile per row)")
print("  (L3, P): 1024 (first P tile at row 0, second at row 1)")
print("  (L3, C): 2048 (after 2 P tiles)")
print("  Total Input rows = P_l3 * C_l3 = 2 * 8 = 16")
