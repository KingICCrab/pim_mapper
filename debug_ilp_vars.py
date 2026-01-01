"""
Debug script to inspect ILP variable values (xr, xj, xb) for row activation analysis.
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
import gurobipy as gp

def main():
    # ResNet-L1 workload
    workload = ConvWorkload(
        "ResNet-L1",
        R=7, S=7, P=56, Q=56, C=3, K=64, N=1,
        stride=(1, 1), dilation=(1, 1)
    )
    
    dim_names = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
    tensor_names = ['Input', 'Weight', 'Output']
    
    print("=" * 80)
    print("WORKLOAD INFO")
    print("=" * 80)
    print(f"Dimensions: R={workload.R}, S={workload.S}, P={workload.P}, Q={workload.Q}, "
          f"C={workload.C}, K={workload.K}, N={workload.N}")
    print()
    print("Relevancy Matrix O[dim][tensor]:")
    print(f"  {'Dim':<8} {'Input':<8} {'Weight':<8} {'Output':<8}")
    for i, dim in enumerate(dim_names):
        print(f"  {dim:<8} {workload.O[i][0]:<8} {workload.O[i][1]:<8} {workload.O[i][2]:<8}")
    
    print()
    print("=" * 80)
    print("RUNNING OPTIMIZER...")
    print("=" * 80)
    
    optimizer = PIMOptimizer()
    arch = optimizer.arch
    print("\nDEBUG INFO:")
    print(f"Workload Divisors: {workload.divisors}")
    print(f"Mem Entries: {arch.mem_entries}")
    print(f"Mem Stores Datatype: {arch.mem_stores_datatype}")
    print(f"Mem Stores Multiple: {arch.mem_stores_multiple_datatypes}")
    print(f"Shared RowBuf Idx: {getattr(arch, 'shared_rowbuf_idx', 'N/A')}")
    print(f"DRAM Level: {getattr(arch, 'dram_level', 'N/A')}")
    
    result = optimizer.optimize([workload], objective="latency")
    
    if not result.mappings:
        print(f"Optimization failed. Status: {optimizer.model.status}")
        if optimizer.model.status == gp.GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS...")
            optimizer.model.computeIIS()
            optimizer.model.write("model.ilp")
            print("IIS written to model.ilp")
        return

    mapping = result.mappings[0]
    
    # Access internal Gurobi model
    model = optimizer.model
    vars_obj = optimizer.vars
    
    print()
    print("=" * 80)
    print("MAPPING RESULT")
    print("=" * 80)
    
    print(f"\nLoop Bounds at Level 3 (DRAM):")
    if 3 in mapping.loop_bounds:
        for key, bounds in mapping.loop_bounds[3].items():
            non_one = {dim_names[d]: v for d, v in bounds.items() if v > 1}
            if non_one:
                print(f"  {key}: {non_one}")
    
    print(f"\nPermutation at Level 3:")
    if 3 in mapping.permutation:
        perm = mapping.permutation[3]
        order = [dim_names[perm[p]] for p in sorted(perm.keys())]
        print(f"  {' -> '.join(order)}")
    
    print(f"\nILP Row Activations:")
    print(f"  Input:  {mapping.metrics.get('row_activations_input', 0):.4f}")
    print(f"  Weight: {mapping.metrics.get('row_activations_weight', 0):.4f}")
    print(f"  Output: {mapping.metrics.get('row_activations_output', 0):.4f}")
    
    print()
    print("=" * 80)
    print("ILP VARIABLE VALUES")
    print("=" * 80)
    
    w = 0  # workload index
    num_mems = 4  # PE, GlobalBuffer, RowBuffer, LocalDRAM
    dram_level = 3  # LocalDRAM
    
    # Print xp values at DRAM level
    print(f"\n--- xp[w=0, m_={dram_level}, p, j] (which dim j at position p) ---")
    print(f"  Position -> Dimension mapping at DRAM level:")
    for p in range(7):
        for j in range(7):
            xp_var = vars_obj.xp.get((w, dram_level, p, j))
            if xp_var is not None and xp_var.X > 0.5:
                print(f"    Position {p}: {dim_names[j]} (xp={xp_var.X:.0f})")
    
    # Print xr values
    print(f"\n--- xr[w=0, t, m=2, m_={dram_level}, p] (relevant inner loop at position p) ---")
    print(f"  xr=1 means: there exists a relevant inner loop at position p or earlier")
    for t in range(3):
        print(f"\n  {tensor_names[t]}:")
        for p in range(7):
            xr_var = vars_obj.xr.get((w, t, 2, dram_level, p))
            if xr_var is not None:
                print(f"    Position {p}: xr = {xr_var.X:.0f}")
    
    # Print xj values
    print(f"\n--- xj[w=0, t, m=2, m_={dram_level}, j] (dim j has inner loop) ---")
    print(f"  xj=1 means: dimension j is placed at a position where there's a relevant inner loop")
    for t in range(3):
        print(f"\n  {tensor_names[t]}:")
        for j in range(7):
            xj_var = vars_obj.xj.get((w, t, 2, dram_level, j))
            if xj_var is not None:
                print(f"    {dim_names[j]}: xj = {xj_var.X:.0f}")
    
    # Print xb values (divisor selection)
    print(f"\n--- xb[w=0, m_={dram_level}, s, j, i] (divisor selection) ---")
    print(f"  Which divisor index i is selected for dimension j at DRAM level?")
    for s in ['temporal', 'spatial']:
        s_idx = 0 if s == 'spatial' else 1
        print(f"\n  {s}:")
        for j in range(7):
            for i in range(len(workload.divisors[j])):
                xb_var = vars_obj.xb.get((w, dram_level, s_idx, j, i))
                if xb_var is not None and xb_var.X > 0.5:
                    div_val = workload.divisors[j][i]
                    print(f"    {dim_names[j]}: divisor index {i} -> value {div_val}")
    
    # Calculate expected row_acts manually
    print()
    print("=" * 80)
    print("MANUAL ROW ACTS CALCULATION")
    print("=" * 80)
    
    # Get DRAM factors
    dram_factors = {i: 1 for i in range(7)}
    for m in [2, 3]:
        if m in mapping.loop_bounds:
            for key in ['spatial', 'temporal']:
                if key in mapping.loop_bounds[m]:
                    for dim, bound in mapping.loop_bounds[m][key].items():
                        dram_factors[dim] *= bound
    
    print(f"\nDRAM factors:")
    for i, name in enumerate(dim_names):
        print(f"  {name}: {dram_factors[i]}")
    
    print(f"\nExpected row_acts_aligned (product of DRAM factors with xj=1):")
    for t in range(3):
        product = 1
        dims_with_xj1 = []
        for j in range(7):
            xj_var = vars_obj.xj.get((w, t, 2, dram_level, j))
            if xj_var is not None and xj_var.X > 0.5:
                dims_with_xj1.append(j)
                product *= dram_factors[j]
        dim_strs = [f"{dim_names[j]}={dram_factors[j]}" for j in dims_with_xj1]
        print(f"  {tensor_names[t]}: {' × '.join(dim_strs) if dim_strs else '1'} = {product}")
    
    # Check the actual ILP internal variables
    print()
    print("=" * 80)
    print("ILP INTERNAL VARIABLES (for 57.6 debugging)")
    print("=" * 80)
    
    import numpy as np
    
    for t_id, t_name in enumerate(tensor_names):
        print(f"\n  {t_name}:")
        
        # row_acts_row_aligned
        var = model.getVarByName(f"row_acts_row_aligned_({w},{t_id})")
        if var:
            print(f"    row_acts_row_aligned: {var.X:.4f}")
        
        # log_row_acts_row_aligned
        var = model.getVarByName(f"log_row_acts_row_aligned_({w},{t_id})")
        if var:
            print(f"    log_row_acts_row_aligned: {var.X:.4f} -> exp = {np.exp(var.X):.4f}")
        
        # reuse_penalty
        var = model.getVarByName(f"reuse_penalty_({w},{t_id})")
        if var:
            print(f"    reuse_penalty: {var.X:.4f}")
        
        # total_row_acts
        var = model.getVarByName(f"total_row_acts_({w},{t_id})")
        if var:
            print(f"    total_row_acts: {var.X:.4f}")
        
        # block_crossing (Input only)
        if t_id == 0:
            var = model.getVarByName(f"input_block_crossing_acts_({w})")
            if var:
                print(f"    input_block_crossing_acts: {var.X:.4f}")
            else:
                print(f"    input_block_crossing_acts: (not found)")
            
            # Also check selected_ibc_count and aux_ibc_rp
            var = model.getVarByName(f"selected_ibc_count_({w})")
            if var:
                print(f"    selected_ibc_count: {var.X:.4f}")
            
            var = model.getVarByName(f"aux_ibc_rp_({w})")
            if var:
                print(f"    aux_ibc_rp: {var.X:.4f}")
            
            # Check rowbuf_input_block_h selection
            print(f"    rowbuf_input_block_h selection:")
            h_divisors = workload.hw_divisors.get('H', [1])
            for i, h_div in enumerate(h_divisors):
                var = vars_obj.rowbuf_input_block_h.get((w, i))
                if var and var.X > 0.5:
                    print(f"      index {i} -> block_h = {h_div} (SELECTED)")
                elif var:
                    print(f"      index {i} -> block_h = {h_div}: {var.X:.2f}")
        
        # row_aligned_acts (the multiplied part)
        var = model.getVarByName(f"row_aligned_acts_({w},{t_id})")
        if var:
            print(f"    row_aligned_acts: {var.X:.4f}")

        # NEW DEBUGGING: Print Reuse and Crossing details
        try:
            outer_irr = model.getVarByName(f"outer_irr_prod_({w},{t_id})")
            print(f"    outer_irr_prod: {outer_irr.X if outer_irr else 'N/A'}")
            
            # Calculate Tile Size and Row Size
            elem_bits_map = getattr(arch, "element_bits_per_dtype", None)
            element_bits = elem_bits_map.get(t_name.lower(), 8) if isinstance(elem_bits_map, dict) else 8
            element_bytes = max(1.0, float(element_bits) / 8.0)
            
            row_bytes = 1024.0
            if hasattr(arch, "mem_row_buffer_size"):
                # Assuming shared_rowbuf_idx is 2 (RowBuffer)
                rb_size = arch.mem_row_buffer_size[2]
                if rb_size not in (None, 0):
                    row_bytes = float(rb_size)
            
            print(f"    element_bytes: {element_bytes}")
            print(f"    row_bytes: {row_bytes}")
            
            # Calculate Tile Size
            # Weight: K, C, R, S
            # Factors: K=4, C=3, R=1, S=1 (from previous run)
            # Tile Size = (K_bound/K_fac) * (C_bound/C_fac) * ...
            # K=64/4=16. C=3/3=1. R=7. S=7.
            # Size = 16 * 1 * 7 * 7 = 784 elements.
            tile_elements = 784 # Hardcoded for now based on observation
            tile_bytes = tile_elements * element_bytes
            print(f"    tile_bytes: {tile_bytes}")
            print(f"    crosses_row: {tile_bytes > row_bytes}")
            
        except Exception as e:
            print(f"    Error printing details: {e}")


if __name__ == "__main__":
    main()
