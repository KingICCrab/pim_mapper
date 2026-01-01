"""
Debug script to understand Input address calculation and row activation counting.
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer.optimizer import PIMOptimizer
from validation.dram.trace_generator import TraceGenerator, DRAMConfig
from collections import defaultdict

def main():
    # small-v2 workload
    workload = ConvWorkload(name='small-v2', R=3, S=3, P=16, Q=16, C=16, K=16, N=1)
    
    # Run optimizer
    optimizer = PIMOptimizer()
    result = optimizer.optimize([workload], objective='latency')
    mapping = result.mappings[0]
    
    dim_names = ['N','K','C','R','S','P','Q']
    DIM_N, DIM_K, DIM_C, DIM_R, DIM_S, DIM_P, DIM_Q = 0, 1, 2, 3, 4, 5, 6
    
    print("DRAM loop bounds (Level 3 total):")
    if 3 in mapping.loop_bounds and 'total' in mapping.loop_bounds[3]:
        dram_factors = mapping.loop_bounds[3]['total']
        for i, name in enumerate(dim_names):
            if dram_factors.get(i, 1) > 1:
                print(f"  {name}: {dram_factors.get(i, 1)}")
    
    # Setup trace generator
    dram_config = DRAMConfig(row_buffer_bytes=1024, num_banks=4, num_rows=16384, element_size=1)
    gen = TraceGenerator(dram_config)
    
    buffer_tile = gen._compute_buffer_tile_size(mapping)
    dram_loops = gen._build_dram_loop_structure(mapping, workload, buffer_tile)
    layout_info = gen._compute_data_layouts(workload, mapping, buffer_tile, dram_loops, dram_config)
    
    print()
    print("DRAM Loop structure (outer to inner):")
    dim_to_level = {}
    for i, loop_info in enumerate(dram_loops):
        dim = loop_info['dim']
        level = loop_info['level']
        bound = loop_info['bound']
        dim_to_level[dim] = level
        print(f"  [{i}] {dim_names[dim]}: bound={bound}, level={level}")
    
    print()
    print("Input strides (from layout_info):")
    input_strides = layout_info['input_strides']
    for key, val in sorted(input_strides.items()):
        level, dim = key
        print(f"  (Level {level}, {dim_names[dim]}): {val}")
    
    print()
    print("=" * 60)
    print("ANALYSIS: How tile_base is calculated for Input")
    print("=" * 60)
    
    # Get the strides used in _generate_tile_accesses
    stride_p_l3 = input_strides.get((3, DIM_P), 1024)
    stride_q_l3 = input_strides.get((3, DIM_Q), 1024)
    stride_c_l3 = input_strides.get((3, DIM_C), 1024)
    stride_n_l3 = input_strides.get((3, DIM_N), 1024)
    
    print()
    print("Strides used in current code (hardcoded Level 3):")
    print(f"  stride_p_l3 = {stride_p_l3}")
    print(f"  stride_q_l3 = {stride_q_l3}")
    print(f"  stride_c_l3 = {stride_c_l3}")
    print(f"  stride_n_l3 = {stride_n_l3}")
    
    # What the strides SHOULD be based on dim_to_level
    print()
    print("What strides SHOULD be based on actual levels:")
    for dim in [DIM_P, DIM_Q, DIM_C, DIM_N]:
        actual_level = dim_to_level.get(dim)
        if actual_level:
            actual_stride = input_strides.get((actual_level, dim), 'NOT FOUND')
            print(f"  {dim_names[dim]}: level={actual_level}, stride={actual_stride}")
        else:
            print(f"  {dim_names[dim]}: not in DRAM loops")
    
    # Simulate iteration and see tile_base values
    print()
    print("=" * 60)
    print("SIMULATION: First 12 iterations of DRAM loops")
    print("=" * 60)
    
    # For small-v2, DRAM loops are: Q(16), S(3), C(16)
    # with C at Level 2, Q and S at Level 3
    row_size = 1024
    
    iteration = 0
    addresses = []
    print(f"{'Iter':<6} {'Q':<4} {'S':<4} {'C':<4} {'tile_base (current)':<20} {'row#':<10}")
    print("-" * 60)
    
    for q_tile in range(min(3, 16)):  # Q first 3
        for s_tile in range(3):        # S all 3
            for c_tile in range(min(2, 16)):  # C first 2
                # Current code uses Level 3 strides for all
                tile_base = (0 * stride_p_l3 + 
                            q_tile * stride_q_l3 + 
                            c_tile * stride_c_l3 +  # BUG: C is Level 2!
                            0 * stride_n_l3)
                row_num = tile_base // row_size
                addresses.append(tile_base)
                print(f"{iteration:<6} {q_tile:<4} {s_tile:<4} {c_tile:<4} {tile_base:<20} {row_num:<10}")
                iteration += 1
                if iteration >= 12:
                    break
            if iteration >= 12:
                break
        if iteration >= 12:
            break
    
    print()
    print("KEY INSIGHT:")
    print("  S changes do NOT affect tile_base (S not used in tile_base calculation)")
    print("  But input_changed=True when S changes, so trace regenerates addresses")
    print("  Q changes DO affect tile_base by stride_q_l3")
    print("  C changes: stride_c_l3 from Level 3 = ???")
    print()
    
    # Check what C stride we're getting
    print("C stride analysis:")
    print(f"  input_strides.get((3, DIM_C), 1024) = {input_strides.get((3, DIM_C), 1024)}")
    print(f"  input_strides.get((2, DIM_C), 1024) = {input_strides.get((2, DIM_C), 1024)}")
    print()
    
    # Now simulate with CORRECT strides
    print("=" * 60)
    print("SIMULATION with CORRECT stride selection")
    print("=" * 60)
    
    # C is Level 2, so use Level 2 stride for C
    stride_c_correct = input_strides.get((dim_to_level.get(DIM_C), DIM_C), 1024)
    stride_q_correct = input_strides.get((dim_to_level.get(DIM_Q), DIM_Q), 1024)
    
    print(f"Correct strides:")
    print(f"  stride_q (Level {dim_to_level.get(DIM_Q)}): {stride_q_correct}")
    print(f"  stride_c (Level {dim_to_level.get(DIM_C)}): {stride_c_correct}")
    print()
    
    iteration = 0
    correct_addresses = []
    print(f"{'Iter':<6} {'Q':<4} {'S':<4} {'C':<4} {'tile_base (correct)':<20} {'row#':<10}")
    print("-" * 60)
    
    for q_tile in range(min(3, 16)):
        for s_tile in range(3):
            for c_tile in range(min(2, 16)):
                tile_base = (0 * stride_p_l3 + 
                            q_tile * stride_q_correct + 
                            c_tile * stride_c_correct +
                            0 * stride_n_l3)
                row_num = tile_base // row_size
                correct_addresses.append(tile_base)
                print(f"{iteration:<6} {q_tile:<4} {s_tile:<4} {c_tile:<4} {tile_base:<20} {row_num:<10}")
                iteration += 1
                if iteration >= 12:
                    break
            if iteration >= 12:
                break
        if iteration >= 12:
            break

if __name__ == "__main__":
    main()
