"""
Test script to verify address calculation logic based on user's explanation.

For small workload:
- Loop nesting (outer to inner): C3 -> K3 -> S0 -> P0 -> R0 -> parallel(Q_H, C_H, P_W, K_W)
- Weight: K_W * 1 + C_H*4 + R0*4*2 + S0*4*2*3 + K3*4*2*3*3 + C3*4*2*3*3*4
- Input (row_aligned): h*1 + w*BLOCK_H + c_h*BLOCK_H*BLOCK_W + c3*Rowsize
- Input (sequential): h*1 + w*BLOCK_H + c_h*BLOCK_H*BLOCK_W + c3*BLOCK_H*BLOCK_W*C_H
- Output: K_W * 1 + P_W*4 + Q_H*4*2 + P0*4*2*8 + K3*4*2*8*2
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from pim_optimizer.workload import ConvWorkload
from pim_optimizer import PIMOptimizer
from pim_optimizer.arch import PIMArchitecture


def analyze_mapping(mapping, workload):
    """Analyze mapping to extract loop structure and compute strides."""
    
    print("=" * 60)
    print("MAPPING ANALYSIS")
    print("=" * 60)
    
    # Extract loop bounds from mapping
    print("\nLoop Bounds:")
    for level in [0, 1, 2, 3]:
        if level in mapping.loop_bounds:
            print(f"  Level {level}: {mapping.loop_bounds[level]}")
    
    # Extract permutation
    print("\nPermutation (inner to outer):")
    for level in [0, 1, 2, 3]:
        if level in mapping.permutation:
            perm = mapping.permutation[level]
            sorted_perm = sorted(perm.items(), key=lambda x: x[0])
            dim_names = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
            order = [dim_names[d] for _, d in sorted_perm]
            print(f"  Level {level}: {' -> '.join(order)}")
    
    # Extract tile info
    print(f"\nTile Info:")
    print(f"  block_h = {mapping.tile_info.get('block_h', 1)}")
    print(f"  block_w = {mapping.tile_info.get('block_w', 1)}")
    
    # Extract layout
    print(f"\nLayout:")
    print(f"  Input:  {mapping.layout.get(0, 'sequential')}")
    print(f"  Weight: {mapping.layout.get(1, 'sequential')}")
    print(f"  Output: {mapping.layout.get(2, 'sequential')}")
    
    return mapping


def build_loop_nesting(mapping):
    """
    Build complete loop nesting from outer to inner.
    
    Order: Level 3 -> Level 2 -> Level 1 -> Level 0 temporal -> Level 0 spatial
    
    Returns list of (level, key, dim, bound) tuples
    """
    loops = []
    dim_names = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
    
    # Level 3 temporal (outer)
    if 3 in mapping.permutation:
        perm = mapping.permutation[3]
        # Sort by permutation position (outer to inner = high to low)
        sorted_perm = sorted(perm.items(), key=lambda x: x[0], reverse=True)
        for pos, dim in sorted_perm:
            if 3 in mapping.loop_bounds and 'temporal' in mapping.loop_bounds[3]:
                bound = mapping.loop_bounds[3]['temporal'].get(dim, 1)
                if bound > 1:
                    loops.append((3, 'temporal', dim, bound))
    
    # Level 2 temporal
    if 2 in mapping.permutation:
        perm = mapping.permutation[2]
        sorted_perm = sorted(perm.items(), key=lambda x: x[0], reverse=True)
        for pos, dim in sorted_perm:
            if 2 in mapping.loop_bounds and 'temporal' in mapping.loop_bounds[2]:
                bound = mapping.loop_bounds[2]['temporal'].get(dim, 1)
                if bound > 1:
                    loops.append((2, 'temporal', dim, bound))
    
    # Level 1 temporal
    if 1 in mapping.permutation:
        perm = mapping.permutation[1]
        sorted_perm = sorted(perm.items(), key=lambda x: x[0], reverse=True)
        for pos, dim in sorted_perm:
            if 1 in mapping.loop_bounds and 'temporal' in mapping.loop_bounds[1]:
                bound = mapping.loop_bounds[1]['temporal'].get(dim, 1)
                if bound > 1:
                    loops.append((1, 'temporal', dim, bound))
    
    # Level 0 temporal
    if 0 in mapping.permutation:
        perm = mapping.permutation[0]
        sorted_perm = sorted(perm.items(), key=lambda x: x[0], reverse=True)
        for pos, dim in sorted_perm:
            if 0 in mapping.loop_bounds and 'temporal' in mapping.loop_bounds[0]:
                bound = mapping.loop_bounds[0]['temporal'].get(dim, 1)
                if bound > 1:
                    loops.append((0, 'temporal', dim, bound))
    
    # Level 0 H spatial (Q, C in H direction)
    if 0 in mapping.loop_bounds and 'H' in mapping.loop_bounds[0]:
        for dim, bound in mapping.loop_bounds[0]['H'].items():
            if bound > 1:
                loops.append((0, 'H', dim, bound))
    
    # Level 0 W spatial (P, K in W direction)
    if 0 in mapping.loop_bounds and 'W' in mapping.loop_bounds[0]:
        for dim, bound in mapping.loop_bounds[0]['W'].items():
            if bound > 1:
                loops.append((0, 'W', dim, bound))
    
    return loops


def compute_strides_for_weight(loops, row_size=1024):
    """
    Compute strides for Weight tensor.
    Weight relevant dims: K, C, R, S
    
    Strides are computed from innermost to outermost.
    """
    # Filter to only Weight-relevant loops
    weight_dims = {4, 5, 0, 1}  # C, K, R, S
    weight_loops = [(l, k, d, b) for l, k, d, b in loops if d in weight_dims]
    
    # Reverse to process from inner to outer
    weight_loops_inner_to_outer = list(reversed(weight_loops))
    
    print("\n=== Weight Stride Calculation ===")
    print("Loops (inner to outer):")
    dim_names = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
    
    strides = {}
    stride = 1
    for level, key, dim, bound in weight_loops_inner_to_outer:
        strides[(level, key, dim)] = stride
        print(f"  Level {level} {key} {dim_names[dim]}: bound={bound}, stride={stride}")
        stride *= bound
    
    return strides


def compute_strides_for_input(loops, block_h, block_w, row_size=1024, row_aligned=True):
    """
    Compute strides for Input tensor.
    Input relevant dims: N, C, Q (via h), P (via w), R (via h), S (via w)
    
    For Input, we use tile-wise layout with block_h x block_w.
    - h coordinate: stride = 1
    - w coordinate: stride = block_h
    - C in Level 0 H: stride = block_h * block_w
    - C in Level 3: stride = row_size (if row_aligned) or block_h * block_w * C_H_bound
    """
    # Find C_H bound (C in Level 0 H)
    c_h_bound = 1
    for level, key, dim, bound in loops:
        if level == 0 and key == 'H' and dim == 4:  # C
            c_h_bound = bound
            break
    
    print("\n=== Input Stride Calculation ===")
    print(f"block_h={block_h}, block_w={block_w}, row_size={row_size}")
    print(f"C_H bound = {c_h_bound}")
    print(f"row_aligned = {row_aligned}")
    
    # Input layout strides (simplified)
    # h: stride = 1
    # w: stride = block_h
    # c_h (Level 0 H): stride = block_h * block_w
    # c3 (Level 3): stride = row_size (if row_aligned) else block_h * block_w * c_h_bound
    
    h_stride = 1
    w_stride = block_h
    c_h_stride = block_h * block_w
    if row_aligned:
        c3_stride = row_size
    else:
        c3_stride = block_h * block_w * c_h_bound
    
    print(f"h stride = {h_stride}")
    print(f"w stride = {w_stride}")
    print(f"c_h stride = {c_h_stride}")
    print(f"c3 stride = {c3_stride}")
    
    return {
        'h': h_stride,
        'w': w_stride,
        'c_h': c_h_stride,
        'c3': c3_stride,
        'c_h_bound': c_h_bound,
    }


def compute_strides_for_output(loops, row_size=1024):
    """
    Compute strides for Output tensor.
    Output relevant dims: N, K, P, Q
    """
    # Filter to only Output-relevant loops
    output_dims = {2, 3, 5, 6}  # P, Q, K, N
    output_loops = [(l, k, d, b) for l, k, d, b in loops if d in output_dims]
    
    # Reverse to process from inner to outer
    output_loops_inner_to_outer = list(reversed(output_loops))
    
    print("\n=== Output Stride Calculation ===")
    print("Loops (inner to outer):")
    dim_names = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
    
    strides = {}
    stride = 1
    for level, key, dim, bound in output_loops_inner_to_outer:
        strides[(level, key, dim)] = stride
        print(f"  Level {level} {key} {dim_names[dim]}: bound={bound}, stride={stride}")
        stride *= bound
    
    return strides


def test_weight_address(loops, workload):
    """
    Test Weight address calculation.
    
    Weight[k, c, r, s] address = sum of (loop_var * stride) for each relevant loop
    """
    weight_strides = compute_strides_for_weight(loops)
    
    print("\n=== Weight Address Test ===")
    print("First few Weight addresses (only when relevant vars change):")
    
    dim_names = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
    weight_dims = {4, 5, 0, 1}  # C, K, R, S
    
    # Initialize loop variables
    loop_vars = {}
    for l, k, d, b in loops:
        loop_vars[(l, k, d)] = 0
    
    # Helper to compute Weight address from loop variables
    def compute_weight_addr():
        addr = 0
        for (l, k, d), stride in weight_strides.items():
            val = loop_vars.get((l, k, d), 0)
            addr += val * stride
        return addr
    
    # Show first few addresses
    count = 0
    max_count = 30
    prev_addr = None
    
    def iterate_loops(idx):
        nonlocal count, prev_addr
        if count >= max_count:
            return
        if idx >= len(loops):
            addr = compute_weight_addr()
            # Only print when address changes
            if addr != prev_addr:
                vars_str = ", ".join(
                    f"{dim_names[d]}={loop_vars[(l,k,d)]}"
                    for l, k, d, b in loops if d in weight_dims
                )
                print(f"  addr={addr:4d}: {vars_str}")
                count += 1
                prev_addr = addr
            return
        
        level, key, dim, bound = loops[idx]
        for i in range(bound):
            loop_vars[(level, key, dim)] = i
            iterate_loops(idx + 1)
            if count >= max_count:
                return
    
    iterate_loops(0)


def test_input_address(loops, workload, mapping):
    """Test Input address calculation with h = q + r, w = p + s."""
    
    block_h = mapping.tile_info.get('block_h', 1)
    block_w = mapping.tile_info.get('block_w', 1)
    row_size = 1024
    input_layout = mapping.layout.get(0, "sequential")
    row_aligned = (input_layout == "row_aligned")
    
    print("\n=== Input Address Test ===")
    print(f"block_h={block_h}, block_w={block_w}, row_aligned={row_aligned}")
    
    # Find C bounds at different levels
    c_h_bound = 1
    c3_bound = 1
    for level, key, dim, bound in loops:
        if dim == 4:  # C
            if level == 0 and key == 'H':
                c_h_bound = bound
            elif level == 3 and key == 'temporal':
                c3_bound = bound
    
    print(f"C in Level 0 H: {c_h_bound}, C in Level 3: {c3_bound}")
    
    # Input strides
    h_stride = 1
    w_stride = block_h
    c_h_stride = block_h * block_w
    c3_stride = row_size if row_aligned else block_h * block_w * c_h_bound
    
    print(f"Strides: h={h_stride}, w={w_stride}, c_h={c_h_stride}, c3={c3_stride}")
    
    dim_names = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
    
    # Initialize loop variables
    loop_vars = {}
    for l, k, d, b in loops:
        loop_vars[(l, k, d)] = 0
    
    # Helper to compute Input address from loop variables
    def compute_input_addr():
        # Get R, S, P, Q values from all levels
        r = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == 0)  # R
        s = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == 1)  # S
        p = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == 2)  # P
        q = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == 3)  # Q
        
        # Get C values at different levels
        c_h = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == 4 and l == 0 and k == 'H')
        c3 = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == 4 and l == 3)
        
        # h = q + r, w = p + s (assuming stride=1, dilation=1)
        h = q + r
        w = p + s
        
        addr = h * h_stride + w * w_stride + c_h * c_h_stride + c3 * c3_stride
        return addr, (h, w, c_h, c3)
    
    # Show first few addresses when they change
    count = 0
    max_count = 30
    prev_addr = None
    
    def iterate_loops(idx):
        nonlocal count, prev_addr
        if count >= max_count:
            return
        if idx >= len(loops):
            addr, coords = compute_input_addr()
            if addr != prev_addr:
                h, w, c_h, c3 = coords
                print(f"  addr={addr:6d}: h={h}, w={w}, c_h={c_h}, c3={c3}")
                count += 1
                prev_addr = addr
            return
        
        level, key, dim, bound = loops[idx]
        for i in range(bound):
            loop_vars[(level, key, dim)] = i
            iterate_loops(idx + 1)
            if count >= max_count:
                return
    
    iterate_loops(0)


def test_output_address(loops, workload, mapping):
    """Test Output address calculation."""
    
    output_strides = compute_strides_for_output(loops)
    
    print("\n=== Output Address Test ===")
    print("First few Output addresses:")
    
    dim_names = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
    output_dims = {2, 3, 5, 6}  # P, Q, K, N
    
    # Initialize loop variables
    loop_vars = {}
    for l, k, d, b in loops:
        loop_vars[(l, k, d)] = 0
    
    # Helper to compute Output address from loop variables
    def compute_output_addr():
        addr = 0
        for (l, k, d), stride in output_strides.items():
            val = loop_vars.get((l, k, d), 0)
            addr += val * stride
        return addr
    
    # Show first few addresses
    count = 0
    max_count = 30
    prev_addr = None
    
    def iterate_loops(idx):
        nonlocal count, prev_addr
        if count >= max_count:
            return
        if idx >= len(loops):
            addr = compute_output_addr()
            if addr != prev_addr:
                vars_str = ", ".join(
                    f"{dim_names[d]}={loop_vars[(l,k,d)]}"
                    for l, k, d, b in loops if d in output_dims
                )
                print(f"  addr={addr:4d}: {vars_str}")
                count += 1
                prev_addr = addr
            return
        
        level, key, dim, bound = loops[idx]
        for i in range(bound):
            loop_vars[(level, key, dim)] = i
            iterate_loops(idx + 1)
            if count >= max_count:
                return
    
    iterate_loops(0)


def simulate_row_activations(loops, mapping, workload, row_size=1024, num_banks=4, num_rows=16384):
    """
    Simulate all DRAM accesses and count row activations.
    
    IMPORTANT:
    1. Each tensor has its own row buffer state (independent banks or address space)
    2. Only Level 3 loops generate DRAM accesses (lower levels use cached data)
    """
    block_h = mapping.tile_info.get('block_h', 1)
    block_w = mapping.tile_info.get('block_w', 1)
    
    input_layout = mapping.layout.get(0, "sequential")
    row_aligned = (input_layout == "row_aligned")
    
    # Find C bounds at different levels
    c_h_bound = 1
    for level, key, dim, bound in loops:
        if dim == 4 and level == 0 and key == 'H':
            c_h_bound = bound
    
    # Input strides
    h_stride = 1
    w_stride = block_h
    c_h_stride = block_h * block_w
    c3_stride = row_size if row_aligned else block_h * block_w * c_h_bound
    
    # Weight strides (cumulative from inner to outer)
    weight_dims = {4, 5, 0, 1}  # C, K, R, S
    weight_loops = [(l, k, d, b) for l, k, d, b in loops if d in weight_dims]
    weight_loops_inner_to_outer = list(reversed(weight_loops))
    weight_strides = {}
    stride = 1
    for level, key, dim, bound in weight_loops_inner_to_outer:
        weight_strides[(level, key, dim)] = stride
        stride *= bound
    
    # Output strides (cumulative from inner to outer)
    output_dims = {2, 3, 5, 6}  # P, Q, K, N
    output_loops = [(l, k, d, b) for l, k, d, b in loops if d in output_dims]
    output_loops_inner_to_outer = list(reversed(output_loops))
    output_strides = {}
    stride = 1
    for level, key, dim, bound in output_loops_inner_to_outer:
        output_strides[(level, key, dim)] = stride
        stride *= bound
    
    # Separate loops into Level 3 and inner levels
    level3_loops = [(l, k, d, b) for l, k, d, b in loops if l == 3]
    inner_loops = [(l, k, d, b) for l, k, d, b in loops if l < 3]
    
    print(f"\nLevel 3 loops (DRAM access): {len(level3_loops)}")
    for l, k, d, b in level3_loops:
        print(f"  Level {l} {k} dim={d} bound={b}")
    print(f"Inner loops (cached): {len(inner_loops)}")
    for l, k, d, b in inner_loops:
        print(f"  Level {l} {k} dim={d} bound={b}")
    
    # Initialize loop variables
    loop_vars = {}
    for l, k, d, b in loops:
        loop_vars[(l, k, d)] = 0
    
    # Track row buffer state per bank PER TENSOR (independent)
    row_buffer_state = {
        'input': {},   # bank_id -> current_row
        'weight': {},
        'output': {}
    }
    
    row_acts = {'input': 0, 'weight': 0, 'output': 0}
    dram_accesses = 0
    total_inner_iterations = 0
    
    def compute_input_addr():
        r = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == 0)
        s = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == 1)
        p = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == 2)
        q = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == 3)
        c_h = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == 4 and l == 0 and k == 'H')
        c3 = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == 4 and l == 3)
        h = q + r
        w = p + s
        return h * h_stride + w * w_stride + c_h * c_h_stride + c3 * c3_stride
    
    def compute_weight_addr():
        addr = 0
        for (l, k, d), stride in weight_strides.items():
            addr += loop_vars.get((l, k, d), 0) * stride
        return addr
    
    def compute_output_addr():
        addr = 0
        for (l, k, d), stride in output_strides.items():
            addr += loop_vars.get((l, k, d), 0) * stride
        return addr
    
    def check_row_activation(tensor_name, addr):
        # Calculate bank and row
        row = addr // row_size
        bank = row % num_banks
        physical_row = row // num_banks
        
        # Check if row buffer hit (independent per tensor)
        state = row_buffer_state[tensor_name]
        if bank not in state or state[bank] != physical_row:
            state[bank] = physical_row
            return True
        return False
    
    def iterate_level3_loops(idx):
        nonlocal dram_accesses, total_inner_iterations
        if idx >= len(level3_loops):
            # Level 3 loop iteration complete - this generates DRAM access
            dram_accesses += 1
            
            # Reset row buffer for each DRAM access (inner loops are cached)
            # NO! Don't reset - the row buffer persists across DRAM accesses
            
            # Check each tensor
            input_addr = compute_input_addr()
            weight_addr = compute_weight_addr()
            output_addr = compute_output_addr()
            
            if check_row_activation('input', input_addr):
                row_acts['input'] += 1
            if check_row_activation('weight', weight_addr):
                row_acts['weight'] += 1
            if check_row_activation('output', output_addr):
                row_acts['output'] += 1
            
            # Count inner iterations
            inner_iters = 1
            for l, k, d, b in inner_loops:
                inner_iters *= b
            total_inner_iterations += inner_iters
            return
        
        level, key, dim, bound = level3_loops[idx]
        for i in range(bound):
            loop_vars[(level, key, dim)] = i
            iterate_level3_loops(idx + 1)
    
    iterate_level3_loops(0)
    
    return row_acts, dram_accesses, total_inner_iterations


def main():
    # Create small workload
    workload = ConvWorkload(
        name='small', R=3, S=3, P=8, Q=8, C=16, K=16, N=1,
        stride=(1,1), dilation=(1,1)
    )
    
    print(f"Workload: {workload.name}")
    print(f"  R={workload.R}, S={workload.S}, P={workload.P}, Q={workload.Q}")
    print(f"  C={workload.C}, K={workload.K}, N={workload.N}")
    
    # Run optimizer
    arch = PIMArchitecture.from_yaml('examples/configs/arch.yaml')
    optimizer = PIMOptimizer(arch=arch)
    result = optimizer.optimize([workload])
    mapping = result.mappings[0]
    model = optimizer.model
    
    # Get ILP row activation values
    print("\n" + "=" * 60)
    print("ILP ROW ACTIVATION PREDICTIONS")
    print("=" * 60)
    
    ilp_row_acts = {}
    for t_id, t_name in [(0, 'input'), (1, 'weight'), (2, 'output')]:
        var = model.getVarByName(f'total_row_acts_(0,{t_id})')
        if var:
            ilp_row_acts[t_name] = var.X
            print(f"  {t_name}: {var.X}")
        else:
            ilp_row_acts[t_name] = 0
            print(f"  {t_name}: 0 (variable not found)")
    
    print(f"  Total: {sum(ilp_row_acts.values())}")
    
    # Analyze mapping
    analyze_mapping(mapping, workload)
    
    # Build loop nesting
    loops = build_loop_nesting(mapping)
    
    print("\n" + "=" * 60)
    print("COMPLETE LOOP NESTING (outer to inner)")
    print("=" * 60)
    dim_names = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
    for level, key, dim, bound in loops:
        print(f"  Level {level} {key:8s} {dim_names[dim]}: bound={bound}")
    
    # Compute strides
    block_h = mapping.tile_info.get('block_h', 1)
    block_w = mapping.tile_info.get('block_w', 1)
    row_size = 1024
    
    input_layout = mapping.layout.get(0, "sequential")
    row_aligned = (input_layout == "row_aligned")
    
    weight_strides = compute_strides_for_weight(loops, row_size)
    input_strides = compute_strides_for_input(loops, block_h, block_w, row_size, row_aligned)
    output_strides = compute_strides_for_output(loops, row_size)
    
    # Test Weight addresses
    test_weight_address(loops, workload)
    
    # Test Input addresses
    test_input_address(loops, workload, mapping)
    
    # Test Output addresses  
    test_output_address(loops, workload, mapping)
    
    # Simulate row activations
    print("\n" + "=" * 60)
    print("SIMULATING ROW ACTIVATIONS")
    print("=" * 60)
    
    row_acts, dram_accesses, inner_iters = simulate_row_activations(loops, mapping, workload)
    
    print(f"\nDRAM accesses (Level 3 iterations): {dram_accesses}")
    print(f"Inner iterations per DRAM access: {inner_iters // max(1, dram_accesses)}")
    print(f"Row activations:")
    print(f"  Input:  {row_acts['input']}")
    print(f"  Weight: {row_acts['weight']}")
    print(f"  Output: {row_acts['output']}")
    print(f"  Total:  {sum(row_acts.values())}")


if __name__ == "__main__":
    main()
