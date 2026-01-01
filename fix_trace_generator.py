#!/usr/bin/env python3
"""Fix trace_generator.py to add H_per_tile fields."""

import re

with open('validation/dram/trace_generator.py', 'r') as f:
    content = f.read()

# Find and replace the block_h/block_w section
old_text = """        # Get input block_h and block_w from mapping
        block_h = mapping.tile_info.get('block_h', 1)
        block_w = mapping.tile_info.get('block_w', 1)
        
        # Number of blocks in each dimension (ceiling division)
        num_blocks_h = (H_in + block_h - 1) // block_h
        num_blocks_w = (W_in + block_w - 1) // block_w"""

new_text = """        # =======================================================================
        # Compute Input access tile size (H_per_tile × W_per_tile)
        # This is the size of data accessed in one DRAM loop iteration
        # =======================================================================
        # For Input, the access tile is determined by the sliding window over P, Q, R, S
        # H_per_tile = (P_per_tile - 1) * stride_h + (R_per_tile - 1) * dilation_h + 1
        # W_per_tile = (Q_per_tile - 1) * stride_w + (S_per_tile - 1) * dilation_w + 1
        #
        # buffer_tile contains Level 0+1 cumulative factors
        P_per_tile = buffer_tile[DIM_P]
        Q_per_tile = buffer_tile[DIM_Q]
        R_per_tile = buffer_tile[DIM_R]
        S_per_tile = buffer_tile[DIM_S]
        
        # Get convolution parameters
        stride_h = workload.stride[0] if hasattr(workload, 'stride') else 1
        stride_w = workload.stride[1] if hasattr(workload, 'stride') else 1
        dilation_h = workload.dilation[0] if hasattr(workload, 'dilation') else 1
        dilation_w = workload.dilation[1] if hasattr(workload, 'dilation') else 1
        
        # Input access tile dimensions (sliding window)
        H_per_tile = (P_per_tile - 1) * stride_h + (R_per_tile - 1) * dilation_h + 1
        W_per_tile = (Q_per_tile - 1) * stride_w + (S_per_tile - 1) * dilation_w + 1
        C_per_tile = buffer_tile[DIM_C]
        
        # Data layout block size (from tile_info, or use H_per_tile/W_per_tile as default)
        block_h = mapping.tile_info.get('block_h', H_per_tile)
        block_w = mapping.tile_info.get('block_w', W_per_tile)
        
        # Number of blocks in each dimension (ceiling division)
        num_blocks_h = (H_in + block_h - 1) // block_h
        num_blocks_w = (W_in + block_w - 1) // block_w"""

if old_text in content:
    content = content.replace(old_text, new_text)
    print("Replaced block_h/block_w section")
else:
    print("ERROR: Could not find block_h/block_w section")
    exit(1)

# Now update the return dictionary to include H_per_tile, W_per_tile, C_per_tile
old_return = """        return {
            'block_h': block_h,
            'block_w': block_w,
            'block_size': block_size,
            'num_blocks_h': num_blocks_h,
            'num_blocks_w': num_blocks_w,
            'input_layout': input_layout,
            'weight_layout': weight_layout,
            'output_layout': output_layout,
            'row_size': row_size,
            'input_order': input_order,  # outer to inner for iteration
            'weight_order': weight_order,
            'output_order': output_order,
            'input_strides': input_strides,
            'weight_strides': weight_strides,
            'output_strides': output_strides,
            'input_base': input_base,
            'weight_base': weight_base,
            'output_base': output_base,
            # Tile-wise loop order for stride calculation
            'input_loop_order': input_layout_loop_order,
            'weight_loop_order': weight_loop_order,
            'output_loop_order': output_loop_order,
            'input_tile_sizes': input_tile_sizes_for_stride,
            'weight_tile_sizes': {
                2: {d: buffer_tile[d] * level2_factors[d] for d in weight_dims},
            },
            'output_tile_sizes': {
                2: {d: buffer_tile[d] * level2_factors[d] for d in output_dims},
            },
        }"""

new_return = """        # =======================================================================
        # Compute DRAM factors and aligned tile sizes
        # =======================================================================
        # DRAM factors come from Level 3 loop bounds
        level3_factors = {d: 1 for d in range(7)}
        if 3 in mapping.loop_bounds:
            for key in ['spatial', 'temporal']:
                if key in mapping.loop_bounds[3]:
                    for d, bound in mapping.loop_bounds[3][key].items():
                        level3_factors[d] *= bound
        
        P_l3 = level3_factors[DIM_P]
        Q_l3 = level3_factors[DIM_Q]
        C_l3 = level3_factors[DIM_C]
        N_l3 = level3_factors[DIM_N]
        
        # Input DRAM tile size = H_per_tile × W_per_tile × C_per_tile
        input_dram_tile_size = H_per_tile * W_per_tile * C_per_tile
        
        # For row_aligned layout: pad to row boundary
        if input_layout == "row_aligned":
            input_aligned_tile_size = ((input_dram_tile_size + row_size - 1) // row_size) * row_size
        else:
            input_aligned_tile_size = input_dram_tile_size
        
        return {
            'block_h': block_h,
            'block_w': block_w,
            'block_size': block_size,
            'num_blocks_h': num_blocks_h,
            'num_blocks_w': num_blocks_w,
            # Input access tile size (different from data layout block)
            'H_per_tile': H_per_tile,
            'W_per_tile': W_per_tile,
            'C_per_tile': C_per_tile,
            'input_dram_tile_size': input_dram_tile_size,
            'input_aligned_tile_size': input_aligned_tile_size,
            # DRAM factors
            'P_l3': P_l3,
            'Q_l3': Q_l3,
            'C_l3': C_l3,
            'N_l3': N_l3,
            # Layout modes
            'input_layout': input_layout,
            'weight_layout': weight_layout,
            'output_layout': output_layout,
            'row_size': row_size,
            'input_order': input_order,  # outer to inner for iteration
            'weight_order': weight_order,
            'output_order': output_order,
            'input_strides': input_strides,
            'weight_strides': weight_strides,
            'output_strides': output_strides,
            'input_base': input_base,
            'weight_base': weight_base,
            'output_base': output_base,
            # Tile-wise loop order for stride calculation
            'input_loop_order': input_layout_loop_order,
            'weight_loop_order': weight_loop_order,
            'output_loop_order': output_loop_order,
            'input_tile_sizes': input_tile_sizes_for_stride,
            'weight_tile_sizes': {
                2: {d: buffer_tile[d] * level2_factors[d] for d in weight_dims},
            },
            'output_tile_sizes': {
                2: {d: buffer_tile[d] * level2_factors[d] for d in output_dims},
            },
        }"""

if old_return in content:
    content = content.replace(old_return, new_return)
    print("Replaced return dictionary")
else:
    print("ERROR: Could not find return dictionary")
    # Try to find what's there
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "return {" in line and "'block_h'" in lines[i+1]:
            print(f"Found return at line {i+1}, showing context:")
            for j in range(i, min(i+30, len(lines))):
                print(f"{j+1:4d}: {lines[j]}")
            break
    exit(1)

with open('validation/dram/trace_generator.py', 'w') as f:
    f.write(content)

print("Done!")
