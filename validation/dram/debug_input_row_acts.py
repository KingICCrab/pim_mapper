"""
Debug script to understand why Input has 5880 row activations.

Goal: Trace each DRAM tile iteration and understand:
1. How many row switches occur per tile
2. What causes the row switches (block boundary? C/N change?)
3. Total row activations breakdown

Key information from analysis.txt:
- Input layout: row_aligned
- 12 unique rows accessed: [0, 1, 7, 8, 196, 197, 203, 204, 392, 393, 399, 400]
- Each row activated 448-532 times
- Total: 5880 row activations
- ILP prediction: 2392

DRAM Loop Structure (outer to inner):
- Level 3: K=4 → C=3 → P=28 → Q=7
- Level 2: R=7
- Total iterations: 4 × 3 × 28 × 7 × 7 = 16464

Access tile: H_per_tile=2, W_per_tile=14, C_per_tile=1
Block size: block_h=31, block_w=31
Input size: H=62, W=62, C=3

Hypothesis: Row switches come from:
1. Different (p_tile, q_tile) accessing different h_block, w_block
2. Within a tile, crossing block boundaries (h_block or w_block change)
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from collections import defaultdict
from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload

# Constants
DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6
DIM_NAMES = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']


def analyze_input_row_activations():
    """Analyze Input row activations in detail."""
    
    # Create workload and get mapping
    workload = ConvWorkload(name="ResNet-L1", R=7, S=7, P=56, Q=56, C=3, K=64, N=1)
    optimizer = PIMOptimizer()
    result = optimizer.optimize([workload], objective="latency")
    mapping = result.mappings[0]
    
    # Extract key parameters
    H_in = workload.input_size['H']  # 62
    W_in = workload.input_size['W']  # 62
    stride_h, stride_w = workload.stride[0], workload.stride[1]  # (1, 1)
    
    # Buffer tile (Level 0+1)
    buffer_tile = {d: 1 for d in range(7)}
    for level in [0, 1]:
        if level not in mapping.loop_bounds:
            continue
        level_bounds = mapping.loop_bounds[level]
        if level == 0:
            for key in ['H', 'W', 'Internal', 'temporal']:
                if key in level_bounds:
                    for d, bound in level_bounds[key].items():
                        buffer_tile[d] *= bound
        else:
            for key in ['spatial', 'temporal']:
                if key in level_bounds:
                    for d, bound in level_bounds[key].items():
                        buffer_tile[d] *= bound
    
    # Access tile dimensions
    P_per_tile = buffer_tile[DIM_P]  # 2
    Q_per_tile = buffer_tile[DIM_Q]  # 8
    R_per_tile = buffer_tile[DIM_R]  # 1
    S_per_tile = buffer_tile[DIM_S]  # 7
    
    H_per_tile = (P_per_tile - 1) * stride_h + R_per_tile  # 2
    W_per_tile = (Q_per_tile - 1) * stride_w + S_per_tile  # 14
    
    # Block size
    block_h = mapping.tile_info.get('block_h', 31)
    block_w = mapping.tile_info.get('block_w', 31)
    
    # DRAM factors
    level3_factors = {d: 1 for d in range(7)}
    if 3 in mapping.loop_bounds:
        for key in ['spatial', 'temporal']:
            if key in mapping.loop_bounds[3]:
                for d, bound in mapping.loop_bounds[3][key].items():
                    level3_factors[d] *= bound
    
    level2_factors = {d: 1 for d in range(7)}
    if 2 in mapping.loop_bounds:
        for key in ['spatial', 'temporal']:
            if key in mapping.loop_bounds[2]:
                for d, bound in mapping.loop_bounds[2][key].items():
                    level2_factors[d] *= bound
    
    P_l3 = level3_factors[DIM_P]  # 28
    Q_l3 = level3_factors[DIM_Q]  # 7
    C_l3 = level3_factors[DIM_C]  # 3
    K_l3 = level3_factors[DIM_K]  # 4
    R_l2 = level2_factors[DIM_R]  # 7
    
    # Row aligned strides (from analysis.txt)
    row_size = 1024  # row buffer bytes
    input_dram_tile_size = H_per_tile * W_per_tile  # 2 × 14 = 28
    input_aligned_tile_size = ((input_dram_tile_size + row_size - 1) // row_size) * row_size  # 1024
    
    # Strides
    stride_q_l3 = input_aligned_tile_size  # 1024
    stride_p_l3 = stride_q_l3 * Q_l3  # 7168
    stride_c_l3 = stride_p_l3 * P_l3  # 200704
    stride_n_l3 = stride_c_l3 * C_l3  # 602112
    
    stride_q_l2 = 1
    stride_p_l2 = W_per_tile  # 14
    
    print("=" * 80)
    print("INPUT ROW ACTIVATION ANALYSIS")
    print("=" * 80)
    
    print(f"\nWorkload: H={H_in}, W={W_in}, C={workload.C}, N={workload.N}")
    print(f"Access tile: H_per_tile={H_per_tile}, W_per_tile={W_per_tile}")
    print(f"Block size: block_h={block_h}, block_w={block_w}")
    print(f"Num blocks: num_h={(H_in + block_h - 1) // block_h}, num_w={(W_in + block_w - 1) // block_w}")
    
    print(f"\nDRAM Loop Structure (outer to inner):")
    print(f"  K_l3={K_l3} × C_l3={C_l3} × P_l3={P_l3} × Q_l3={Q_l3} × R_l2={R_l2}")
    print(f"  Total iterations: {K_l3 * C_l3 * P_l3 * Q_l3 * R_l2}")
    
    print(f"\nStrides:")
    print(f"  stride_q_l3 = {stride_q_l3}")
    print(f"  stride_p_l3 = {stride_p_l3}")
    print(f"  stride_c_l3 = {stride_c_l3}")
    print(f"  stride_n_l3 = {stride_n_l3}")
    
    print(f"\n" + "=" * 80)
    print("SIMULATING INPUT ACCESS PATTERN")
    print("=" * 80)
    
    # Simulate the access pattern
    total_row_acts = 0
    current_row = None
    row_visit_counts = defaultdict(int)
    tile_row_acts = []  # row activations per tile
    
    # Track per-loop statistics
    per_k_row_acts = defaultdict(int)
    per_c_row_acts = defaultdict(int)
    per_p_row_acts = defaultdict(int)
    per_q_row_acts = defaultdict(int)
    
    # Track row switches per tile
    tiles_with_row_switch = 0
    tiles_without_row_switch = 0
    
    # Outer loops: K (irrelevant for Input) → C → P → Q → R
    for k_tile in range(K_l3):  # K is irrelevant, but we still iterate for each K tile
        for c_tile in range(C_l3):
            for p_tile in range(P_l3):
                for q_tile in range(Q_l3):
                    for r_tile in range(R_l2):
                        # This is one DRAM tile iteration for Input
                        # Since K is irrelevant, Input access only changes when C, P, Q change
                        
                        # For this tile, compute all addresses accessed
                        tile_current_row = current_row
                        tile_row_switches = 0
                        
                        # Starting coordinates
                        p_start = p_tile * P_per_tile
                        q_start = q_tile * Q_per_tile
                        
                        # H, W range for sliding window
                        h_start = p_start * stride_h
                        h_end = min((p_start + P_per_tile - 1) * stride_h + R_per_tile, H_in)
                        w_start = q_start * stride_w
                        w_end = min((q_start + Q_per_tile - 1) * stride_w + S_per_tile, W_in)
                        
                        # Block ranges
                        h_block_start = h_start // block_h
                        h_block_end = (h_end - 1) // block_h
                        w_block_start = w_start // block_w
                        w_block_end = (w_end - 1) // block_w
                        
                        # Iterate by block (as in trace_generator)
                        for h_block in range(h_block_start, h_block_end + 1):
                            for w_block in range(w_block_start, w_block_end + 1):
                                # Valid h, w range within this block
                                h_lo = max(h_start, h_block * block_h)
                                h_hi = min(h_end, (h_block + 1) * block_h)
                                w_lo = max(w_start, w_block * block_w)
                                w_hi = min(w_end, (w_block + 1) * block_w)
                                
                                for h in range(h_lo, h_hi):
                                    for w in range(w_lo, w_hi):
                                        h_in_block = h % block_h
                                        w_in_block = w % block_w
                                        
                                        # row_aligned address calculation
                                        block_base = (h_block * stride_p_l3 + 
                                                     w_block * stride_q_l3 + 
                                                     c_tile * stride_c_l3)
                                        
                                        offset_in_block = h_in_block * stride_p_l2 + w_in_block * stride_q_l2
                                        
                                        addr = block_base + offset_in_block
                                        row = addr // row_size
                                        
                                        if tile_current_row != row:
                                            tile_row_switches += 1
                                            tile_current_row = row
                        
                        # Update statistics
                        if tile_row_switches > 0:
                            tiles_with_row_switch += 1
                        else:
                            tiles_without_row_switch += 1
                        
                        # Track per-loop row activations
                        per_k_row_acts[k_tile] += tile_row_switches
                        per_c_row_acts[c_tile] += tile_row_switches
                        per_p_row_acts[p_tile] += tile_row_switches
                        per_q_row_acts[q_tile] += tile_row_switches
                        
                        total_row_acts += tile_row_switches
                        tile_row_acts.append(tile_row_switches)
                        
                        # Update global current_row for next tile
                        current_row = tile_current_row
    
    print(f"\nTotal row activations: {total_row_acts}")
    print(f"Total tiles: {len(tile_row_acts)}")
    print(f"Tiles with row switch: {tiles_with_row_switch}")
    print(f"Tiles without row switch: {tiles_without_row_switch}")
    
    print(f"\nRow activations by tile count:")
    from collections import Counter
    tile_acts_dist = Counter(tile_row_acts)
    for acts, count in sorted(tile_acts_dist.items()):
        print(f"  {acts} row switches: {count} tiles")
    
    print(f"\nRow activations per K_tile (should be same since K is irrelevant):")
    for k, acts in sorted(per_k_row_acts.items()):
        print(f"  K={k}: {acts}")
    
    print(f"\nRow activations per C_tile:")
    for c, acts in sorted(per_c_row_acts.items()):
        print(f"  C={c}: {acts}")
    
    # Analyze why row switches happen
    print(f"\n" + "=" * 80)
    print("ANALYZING ROW SWITCH CAUSES")
    print("=" * 80)
    
    # Sample some tiles to understand the pattern
    print(f"\nSampling first few tiles to understand row switch pattern:")
    
    current_row = None
    for k_tile in range(1):  # Just K=0
        for c_tile in range(1):  # Just C=0
            for p_tile in range(3):  # First 3 P tiles
                for q_tile in range(2):  # First 2 Q tiles
                    for r_tile in range(1):  # Just R=0
                        p_start = p_tile * P_per_tile
                        q_start = q_tile * Q_per_tile
                        h_start = p_start * stride_h
                        h_end = min((p_start + P_per_tile - 1) * stride_h + R_per_tile, H_in)
                        w_start = q_start * stride_w
                        w_end = min((q_start + Q_per_tile - 1) * stride_w + S_per_tile, W_in)
                        
                        h_block_start = h_start // block_h
                        h_block_end = (h_end - 1) // block_h
                        w_block_start = w_start // block_w
                        w_block_end = (w_end - 1) // block_w
                        
                        print(f"\n  Tile (K={k_tile}, C={c_tile}, P={p_tile}, Q={q_tile}, R={r_tile}):")
                        print(f"    h_range=[{h_start}, {h_end}), w_range=[{w_start}, {w_end})")
                        print(f"    h_block_range=[{h_block_start}, {h_block_end}], w_block_range=[{w_block_start}, {w_block_end}]")
                        
                        # Show addresses for first few elements
                        elements_shown = 0
                        prev_row = current_row
                        for h_block in range(h_block_start, h_block_end + 1):
                            for w_block in range(w_block_start, w_block_end + 1):
                                h_lo = max(h_start, h_block * block_h)
                                h_hi = min(h_end, (h_block + 1) * block_h)
                                w_lo = max(w_start, w_block * block_w)
                                w_hi = min(w_end, (w_block + 1) * block_w)
                                
                                for h in range(h_lo, h_hi):
                                    for w in range(w_lo, w_hi):
                                        h_in_block = h % block_h
                                        w_in_block = w % block_w
                                        
                                        block_base = (h_block * stride_p_l3 + 
                                                     w_block * stride_q_l3 + 
                                                     c_tile * stride_c_l3)
                                        offset_in_block = h_in_block * stride_p_l2 + w_in_block * stride_q_l2
                                        addr = block_base + offset_in_block
                                        row = addr // row_size
                                        
                                        if elements_shown < 10 or prev_row != row:
                                            marker = " <-- ROW SWITCH" if prev_row is not None and prev_row != row else ""
                                            print(f"      h={h}, w={w}, h_blk={h_block}, w_blk={w_block}, "
                                                  f"block_base={block_base}, offset={offset_in_block}, "
                                                  f"addr={addr}, row={row}{marker}")
                                            elements_shown += 1
                                        prev_row = row
                                        current_row = row
    
    # Compute expected row activations
    print(f"\n" + "=" * 80)
    print("COMPUTING EXPECTED ROW ACTIVATIONS")
    print("=" * 80)
    
    # For row_aligned with block_h=31, block_w=31:
    # - stride_p_l3 = 7168 = 7 rows
    # - stride_q_l3 = 1024 = 1 row
    # - Each (h_block, w_block) maps to a distinct row pattern
    
    # Unique rows accessed:
    # Row 0: h_block=0, w_block=0, offset ∈ [0, 28)
    # Row 1: h_block=0, w_block=1, offset ∈ [0, 28)
    # Row 7: h_block=1, w_block=0, offset ∈ [0, 28)
    # Row 8: h_block=1, w_block=1, offset ∈ [0, 28)
    # etc. for different C values
    
    print(f"\nBlock → Row mapping:")
    for c_tile in range(C_l3):
        print(f"  C={c_tile}:")
        for h_block in range(2):
            for w_block in range(2):
                block_base = h_block * stride_p_l3 + w_block * stride_q_l3 + c_tile * stride_c_l3
                row = block_base // row_size
                print(f"    (h_blk={h_block}, w_blk={w_block}): block_base={block_base}, row={row}")
    
    # Expected row activations for row_aligned:
    # Each unique (c_tile, h_block, w_block) combination is a distinct row
    # But we visit each combination multiple times due to K and (P, Q, R) iterations
    #
    # For Input with row_aligned:
    # - Unique rows = C_l3 × num_h_blocks × num_w_blocks = 3 × 2 × 2 = 12 ✓
    # - But row activations count SWITCHES, not unique rows
    
    # Let's compute how often we switch between these rows
    # Loop order: K → C → P → Q → R
    # Within one (K, C) iteration, we iterate P × Q × R = 28 × 7 × 7 = 1372 tiles
    # Each tile accesses some subset of the 4 blocks (num_h=2, num_w=2)
    
    # Key insight: Row switch happens when:
    # 1. Moving to next P_tile might change h_block (when p crosses block_h boundary)
    # 2. Moving to next Q_tile might change w_block (when w crosses block_w boundary)
    # 3. Within a tile, if tile spans multiple blocks → internal row switches
    
    print(f"\nP → h mapping (P_per_tile={P_per_tile}, stride_h={stride_h}, block_h={block_h}):")
    for p_tile in range(P_l3):
        p_start = p_tile * P_per_tile
        h_start = p_start * stride_h
        h_end = h_start + H_per_tile
        h_block_start = h_start // block_h
        h_block_end = (h_end - 1) // block_h
        if p_tile < 10 or h_block_start != h_block_end:
            crosses = " (CROSSES BLOCK)" if h_block_start != h_block_end else ""
            print(f"  P={p_tile}: h=[{h_start}, {h_end}), h_blocks=[{h_block_start}, {h_block_end}]{crosses}")
    
    print(f"\nQ → w mapping (Q_per_tile={Q_per_tile}, stride_w={stride_w}, block_w={block_w}):")
    for q_tile in range(Q_l3):
        q_start = q_tile * Q_per_tile
        w_start = q_start * stride_w
        w_end = w_start + W_per_tile
        w_block_start = w_start // block_w
        w_block_end = (w_end - 1) // block_w
        crosses = " (CROSSES BLOCK)" if w_block_start != w_block_end else ""
        print(f"  Q={q_tile}: w=[{w_start}, {w_end}), w_blocks=[{w_block_start}, {w_block_end}]{crosses}")


if __name__ == "__main__":
    analyze_input_row_activations()
