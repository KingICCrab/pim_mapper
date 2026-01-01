"""
Detailed debug information for multiple workloads.
Outputs all relevant info for debugging ILP vs Trace row activation discrepancy.
Each workload outputs to a separate file in debug_output/<workload_name>/analysis.txt
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

import os
from pathlib import Path
from collections import defaultdict
from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
from validation.dram.trace_generator import TraceGenerator, DRAMConfig


def process_workload(workload, output_file):
    """Process a single workload and generate detailed debug output."""
    
    output = []
    dim_names = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
    
    def log(s=""):
        output.append(s)
    
    log("=" * 100)
    log(f"DETAILED DEBUG INFO FOR '{workload.name}' WORKLOAD")
    log("=" * 100)
    
    # =========================================================================
    # 1. Workload Configuration
    # =========================================================================
    log("\n" + "=" * 100)
    log("1. WORKLOAD CONFIGURATION")
    log("=" * 100)
    log(f"  Name: {workload.name}")
    log(f"  Dimensions: R={workload.R}, S={workload.S}, P={workload.P}, Q={workload.Q}, "
        f"C={workload.C}, K={workload.K}, N={workload.N}")
    log(f"  Bounds: {workload.bounds}")
    log(f"  MACs: {workload.macs}")
    log(f"  Stride: {workload.stride}")
    log(f"  Dilation: {workload.dilation}")
    log(f"")
    log(f"  Tensor Sizes:")
    log(f"    Input:  N={workload.N} × C={workload.C} × H={workload.input_size['H']} × W={workload.input_size['W']} = {workload.N * workload.C * workload.input_size['H'] * workload.input_size['W']} elements")
    log(f"    Weight: K={workload.K} × C={workload.C} × R={workload.R} × S={workload.S} = {workload.K * workload.C * workload.R * workload.S} elements")
    log(f"    Output: N={workload.N} × K={workload.K} × P={workload.P} × Q={workload.Q} = {workload.N * workload.K * workload.P * workload.Q} elements")
    
    log(f"\n  Relevancy Matrix O[dim][tensor] (1=relevant, 0=irrelevant):")
    log(f"    {'Dim':<8} {'Input':<8} {'Weight':<8} {'Output':<8}")
    for i, dim in enumerate(dim_names):
        log(f"    {dim:<8} {workload.O[i][0]:<8} {workload.O[i][1]:<8} {workload.O[i][2]:<8}")
    
    log(f"\n  Divisors per dimension:")
    for i, (name, divs) in enumerate(zip(dim_names, workload.divisors)):
        log(f"    {name}: {divs}")
    
    # =========================================================================
    # 2. Run Optimizer
    # =========================================================================
    log("\n" + "=" * 100)
    log("2. OPTIMIZER RESULT")
    log("=" * 100)
    
    optimizer = PIMOptimizer()
    result = optimizer.optimize([workload], objective="latency")
    mapping = result.mappings[0]
    
    log(f"\n  Solver Status: {result.solver_status}")
    log(f"  Solve Time: {result.solve_time:.2f}s")
    
    # =========================================================================
    # 3. Mapping Details
    # =========================================================================
    log("\n" + "=" * 100)
    log("3. MAPPING DETAILS")
    log("=" * 100)
    
    log(f"\n  Layout:")
    log(f"    Input:  {mapping.layout.get(0)}")
    log(f"    Weight: {mapping.layout.get(1)}")
    log(f"    Output: {mapping.layout.get(2)}")
    
    log(f"\n  Tile Info:")
    log(f"    block_h: {mapping.tile_info.get('block_h')}")
    log(f"    block_w: {mapping.tile_info.get('block_w')}")
    for k, v in mapping.tile_info.items():
        if k not in ['block_h', 'block_w']:
            log(f"    {k}: {v}")
    
    log(f"\n  Loop Bounds (per memory level):")
    log(f"  Memory Hierarchy: Level 0=PE, Level 1=GlobalBuffer, Level 2=RowBuffer, Level 3=LocalDRAM")
    for m in sorted(mapping.loop_bounds.keys()):
        log(f"\n    Level {m}:")
        for key, bounds in mapping.loop_bounds[m].items():
            non_one = {dim_names[d]: v for d, v in bounds.items() if v > 1}
            if non_one:
                log(f"      {key}: {non_one}")
            else:
                log(f"      {key}: (all 1s)")
    
    log(f"\n  Permutation (per memory level, inner to outer):")
    for m in sorted(mapping.permutation.keys()):
        perm = mapping.permutation[m]
        order = [dim_names[perm[p]] for p in sorted(perm.keys())]
        log(f"    Level {m}: {' -> '.join(order)}")
    
    log(f"\n  Tile Sizes (cumulative up to each level):")
    for m in sorted(mapping.loop_bounds.keys()):
        tile_at_level = {}
        for d in range(7):
            tile_at_level[dim_names[d]] = mapping.get_tile_size(m, d)
        non_one = {k: v for k, v in tile_at_level.items() if v > 1}
        log(f"    Level {m}: {non_one if non_one else '(all 1s)'}")
    
    # =========================================================================
    # 4. DRAM Level Analysis
    # =========================================================================
    log("\n" + "=" * 100)
    log("4. DRAM LEVEL ANALYSIS (Level 2 + Level 3)")
    log("=" * 100)
    
    # Calculate DRAM level factors (Level 2 + 3)
    dram_factors = {i: 1 for i in range(7)}
    for m in [2, 3]:
        if m in mapping.loop_bounds:
            for key in ['spatial', 'temporal']:
                if key in mapping.loop_bounds[m]:
                    for dim, bound in mapping.loop_bounds[m][key].items():
                        dram_factors[dim] *= bound
    
    log(f"\n  DRAM Level Factors (Level 2 × Level 3):")
    for i, name in enumerate(dim_names):
        if dram_factors[i] > 1:
            log(f"    {name}: {dram_factors[i]}")
        else:
            log(f"    {name}: 1")
    
    log(f"\n  Reuse Penalty per Tensor (irrelevant DRAM factors):")
    for t_id, t_name in [(0, "Input"), (1, "Weight"), (2, "Output")]:
        irrelevant = [i for i in range(7) if workload.O[i][t_id] == 0]
        penalty = 1
        for dim in irrelevant:
            penalty *= dram_factors[dim]
        irrelevant_names = [dim_names[i] for i in irrelevant]
        log(f"    {t_name}: irrelevant dims = {irrelevant_names}, penalty = {penalty}")
    
    log(f"\n  Expected Unique Row Activations (simplified):")
    log(f"    For row_aligned: total_tiles = Π(DRAM factors)")
    total_tiles = 1
    for i in range(7):
        total_tiles *= dram_factors[i]
    log(f"    Total DRAM tiles: {total_tiles}")
    
    # =========================================================================
    # 5. ILP Row Activation Predictions
    # =========================================================================
    log("\n" + "=" * 100)
    log("5. ILP ROW ACTIVATION PREDICTIONS")
    log("=" * 100)
    
    log(f"\n  ILP Predicted Row Activations:")
    log(f"    Input:  {mapping.metrics.get('row_activations_input', 0):.4f}")
    log(f"    Weight: {mapping.metrics.get('row_activations_weight', 0):.4f}")
    log(f"    Output: {mapping.metrics.get('row_activations_output', 0):.4f}")
    log(f"    Total:  {mapping.metrics.get('row_activations', 0):.4f}")
    
    # =========================================================================
    # 5.1 ILP Variable Values (xj, xr, xb for debugging)
    # =========================================================================
    log("\n" + "=" * 100)
    log("5.1 ILP VARIABLE VALUES")
    log("=" * 100)
    
    vars_obj = optimizer.vars
    w = 0
    dram_level = 3
    rowbuf_level = 2
    s_temporal = 1  # temporal index for non-PE levels
    
    # Print xp (permutation at DRAM level)
    log(f"\n  xp[m={dram_level}] - Permutation at DRAM level (which dim at which position):")
    for p in range(7):
        for j in range(7):
            xp_var = vars_obj.xp.get((w, dram_level, p, j))
            if xp_var is not None and xp_var.X > 0.5:
                log(f"    Position {p}: {dim_names[j]}")
    
    # Print xb (loop bounds at DRAM level, temporal)
    log(f"\n  xb[m={dram_level}, s={s_temporal}] - DRAM temporal bounds:")
    for j in range(7):
        for i, div in enumerate(workload.divisors[j]):
            xb_var = vars_obj.xb.get((w, dram_level, s_temporal, j, i))
            if xb_var is not None and xb_var.X > 0.5:
                log(f"    {dim_names[j]}: divisor[{i}] = {div}")
    
    # Print xr for each tensor
    log(f"\n  xr[m={rowbuf_level}, m_={dram_level}] - Relevant inner loop exists at position p:")
    for t_id, t_name in [(0, "Input"), (1, "Weight"), (2, "Output")]:
        log(f"\n    {t_name}:")
        for p in range(7):
            xr_var = vars_obj.xr.get((w, t_id, rowbuf_level, dram_level, p))
            if xr_var is not None:
                log(f"      Position {p}: xr = {int(xr_var.X)}")
    
    # Print xj for each tensor
    log(f"\n  xj[m={rowbuf_level}, m_={dram_level}] - Dimension j has inner loop for tensor:")
    for t_id, t_name in [(0, "Input"), (1, "Weight"), (2, "Output")]:
        log(f"\n    {t_name} (relevant dims: {[dim_names[i] for i in range(7) if workload.O[i][t_id] == 1]}):")
        for j in range(7):
            xj_var = vars_obj.xj.get((w, t_id, rowbuf_level, dram_level, j))
            if xj_var is not None:
                relevant = "relevant" if workload.O[j][t_id] == 1 else "irrelevant"
                log(f"      {dim_names[j]}: xj = {int(xj_var.X)} ({relevant})")
    
    # Manual row_acts calculation
    log(f"\n  Manual row_acts_aligned calculation (for verification):")
    for t_id, t_name in [(0, "Input"), (1, "Weight"), (2, "Output")]:
        product = 1
        factors = []
        for j in range(7):
            xj_var = vars_obj.xj.get((w, t_id, rowbuf_level, dram_level, j))
            if xj_var is not None and xj_var.X > 0.5:
                # Find which divisor is selected
                for i, div in enumerate(workload.divisors[j]):
                    xb_var = vars_obj.xb.get((w, dram_level, s_temporal, j, i))
                    if xb_var is not None and xb_var.X > 0.5:
                        product *= div
                        factors.append(f"{dim_names[j]}={div}")
                        break
        log(f"    {t_name}: {' × '.join(factors) if factors else '1'} = {product}")
    
    # =========================================================================
    # 5.2 Detailed Row Activation ILP Variables
    # =========================================================================
    log("\n" + "=" * 100)
    log("5.2 DETAILED ROW ACTIVATION ILP VARIABLES")
    log("=" * 100)
    
    model = optimizer.model
    tensor_names = {0: "Input", 1: "Weight", 2: "Output"}
    
    for t_id in range(3):
        t_name = tensor_names[t_id]
        log(f"\n  {'='*60}")
        log(f"  Tensor: {t_name}")
        log(f"  {'='*60}")
        
        # 1. reuse_penalty
        var = model.getVarByName(f"reuse_penalty_({w},{t_id})")
        if var is not None:
            log(f"\n    1. Reuse Penalty: {var.X:.4f}")
            log(f"       (Π {{irrelevant dims with xj=1}} bound_j)")
        
        # 2. row_acts_row_aligned
        var = model.getVarByName(f"row_acts_row_aligned_({w},{t_id})")
        if var is not None:
            log(f"\n    2. Row Acts (Row-Aligned mode): {var.X:.4f}")
            log(f"       (Π {{all dims with xj=1}} bound_j)")
        
        # 3. base_row_acts (for sequential)
        var = model.getVarByName(f"base_row_acts_({w},{t_id})")
        if var is not None:
            log(f"\n    3. Base Row Acts (Sequential): {var.X:.4f}")
            log(f"       (non_crossing_acts + 2 × crossing_count × reuse_penalty)")
        
        # 4. outer_irr_product
        var = model.getVarByName(f"outer_irr_prod_({w},{t_id})")
        if var is not None:
            log(f"\n    4. Outer Irrelevant Product: {var.X:.4f}")
            log(f"       (Π {{outer irrelevant dims with xj=1}} bound_j)")
        
        # 5. row_acts (sequential final)
        var = model.getVarByName(f"row_acts_({w},{t_id})")
        if var is not None:
            log(f"\n    5. Row Acts (Sequential final): {var.X:.4f}")
            log(f"       (base_row_acts × outer_irr_product)")
        
        # 6. seq_part (conditional)
        var = model.getVarByName(f"dram_crossing_seq_({w},{t_id})")
        if var is not None:
            log(f"\n    6. Sequential Part (conditional): {var.X:.4f}")
            log(f"       ((1 - row_aligned) × row_acts_seq)")
        
        # 7. aligned_part (conditional)
        var = model.getVarByName(f"row_aligned_acts_({w},{t_id})")
        if var is not None:
            log(f"\n    7. Row-Aligned Part (conditional): {var.X:.4f}")
            log(f"       (row_aligned × row_acts_row_aligned)")
        
        # 8. Input Block Crossing (only for Input)
        if t_id == 0:
            var = model.getVarByName(f"input_block_crossing_acts_({w})")
            if var is not None:
                log(f"\n    8. Input Block Crossing Acts: {var.X:.4f}")
                log(f"       (2 × selected_count × reuse_penalty)")
            
            ibc_count = model.getVarByName(f"selected_ibc_count_({w})")
            if ibc_count is not None:
                log(f"       - Selected IBC Count: {ibc_count.X:.4f}")
            
            aux_var = model.getVarByName(f"aux_ibc_rp_({w})")
            if aux_var is not None:
                log(f"       - aux_ibc_rp (selected_count × reuse_penalty): {aux_var.X:.4f}")
            
            rp_var = model.getVarByName(f"reuse_penalty_({w},0)")
            if ibc_count is not None and rp_var is not None:
                expected = 2 * ibc_count.X * rp_var.X
                log(f"       - Expected: 2 × {ibc_count.X:.1f} × {rp_var.X:.1f} = {expected:.1f}")
        
        # 9. total_row_acts
        var = model.getVarByName(f"total_row_acts_({w},{t_id})")
        if var is not None:
            log(f"\n    9. Total Row Acts: {var.X:.4f}")
            log(f"       (seq_part + aligned_part + block_crossing)")
        
        # 10. scaled acts
        var = model.getVarByName(f"row_acts_scaled_({w},{t_id})")
        if var is not None:
            log(f"\n    10. Scaled Row Acts: {var.X:.4f}")
            log(f"        (total_row_acts × macs_scale_factor)")
        
        # 11. cycles
        var = model.getVarByName(f"row_acts_cycles_({w},{t_id})")
        if var is not None:
            log(f"\n    11. Row Acts Cycles: {var.X:.4f}")
            log(f"        (scaled_acts × activation_latency)")
    
    # Layout choices
    log(f"\n  {'='*60}")
    log(f"  Layout Choices")
    log(f"  {'='*60}")
    for t_id in range(3):
        t_name = tensor_names[t_id]
        row_aligned_var = vars_obj.layout_choice.get((w, t_id, "row_aligned"))
        if row_aligned_var is not None:
            layout = "row_aligned" if row_aligned_var.X > 0.5 else "sequential"
            log(f"    {t_name}: {layout}")
        else:
            log(f"    {t_name}: sequential (no choice variable)")

    # =========================================================================
    # 5.5 Address Calculation Details (Strides and Loop Order)
    # =========================================================================
    log("\n" + "=" * 100)
    log("5.5 ADDRESS CALCULATION DETAILS")
    log("=" * 100)
    
    # Use TraceGenerator's internal methods to get stride/loop info
    dram_config = DRAMConfig(row_buffer_bytes=1024, num_banks=4, num_rows=16384, element_size=1)
    gen = TraceGenerator(dram_config)
    
    # Get buffer tile (Level 0+1)
    buffer_tile = gen._compute_buffer_tile_size(mapping)
    log(f"\n  Buffer Tile (Level 0+1):")
    for i, name in enumerate(dim_names):
        if buffer_tile[i] > 1:
            log(f"    {name}: {buffer_tile[i]}")
    
    # Get DRAM loop structure
    dram_loops = gen._build_dram_loop_structure(mapping, workload, buffer_tile)
    
    log(f"\n  DRAM Loop Structure (Level 2+3, outer to inner):")
    log(f"    Loops iterate using TILE INDEX (0, 1, 2, ...), NOT element coordinates")
    log(f"    Element coord = tile_index × tile_stride (computed at access time)")
    for i, loop_info in enumerate(dram_loops):
        dim = loop_info['dim']
        bound = loop_info['bound']
        tile_stride = loop_info['tile_stride']
        level = loop_info['level']
        log(f"    [{i}] Level {level}, {dim_names[dim]}: for tile_idx in range({bound})")
        log(f"         tile_stride={tile_stride} → elem_coord = tile_idx × {tile_stride}")
    
    # Get layout info including strides
    H_in = workload.input_size['H']
    W_in = workload.input_size['W']
    layout_info = gen._compute_data_layouts(mapping, workload, H_in, W_in, buffer_tile)
    
    # =========================================================================
    # DEBUG: Input stride calculation details
    # =========================================================================
    log(f"\n  " + "=" * 70)
    log(f"  DEBUG: INPUT STRIDE CALCULATION DETAILS")
    log(f"  " + "=" * 70)
    
    # Get input loop order
    from validation.dram.trace_generator import DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N
    input_dims = [DIM_N, DIM_C, DIM_Q, DIM_S, DIM_P, DIM_R]
    input_loop_order = gen._build_loop_order(mapping, input_dims)
    input_l3_dims_in_perm = [(lv, d) for (lv, d) in input_loop_order if lv == 3]
    
    log(f"    input_loop_order = {input_loop_order}")
    log(f"    input_l3_dims_in_perm = {input_l3_dims_in_perm}")
    
    # Get Level 3 factors
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
    
    log(f"    Level 3 factors: P_l3={P_l3}, Q_l3={Q_l3}, C_l3={C_l3}, N_l3={N_l3}")
    log(f"    input_aligned_tile_size = {layout_info['input_aligned_tile_size']}")
    
    input_l3_tile_counts = {DIM_P: P_l3, DIM_Q: Q_l3, DIM_C: C_l3, DIM_N: N_l3}
    log(f"    input_l3_tile_counts = {input_l3_tile_counts}")
    
    # Simulate stride calculation
    log(f"\n    Stride calculation process:")
    stride = layout_info['input_aligned_tile_size']
    log(f"      Initial stride = input_aligned_tile_size = {stride}")
    
    processed_dims = set()
    log(f"\n      Step 1: Process dims in permutation (reversed):")
    for (lv, dim) in reversed(input_l3_dims_in_perm):
        if dim in input_l3_tile_counts:
            log(f"        dim={dim_names[dim]}: stride={stride}, count={input_l3_tile_counts[dim]}")
            stride *= input_l3_tile_counts[dim]
            processed_dims.add(dim)
    
    log(f"\n      Step 2: Process remaining dims:")
    for dim in [DIM_Q, DIM_P, DIM_C, DIM_N]:
        if dim not in processed_dims and dim in input_l3_tile_counts:
            log(f"        dim={dim_names[dim]}: stride={stride}, count={input_l3_tile_counts[dim]}")
            stride *= input_l3_tile_counts[dim]
            processed_dims.add(dim)
    
    log(f"\n    Final input_strides from layout_info:")
    for (level, dim), stride_val in sorted(layout_info['input_strides'].items()):
        log(f"      (L{level}, {dim_names[dim]}): {stride_val}")
    
    log(f"  " + "=" * 70)
    
    log(f"\n  Block Configuration:")
    log(f"    block_h: {layout_info['block_h']} (data layout)")
    log(f"    block_w: {layout_info['block_w']} (data layout)")
    log(f"    block_size: {layout_info['block_size']}")
    log(f"    num_blocks_h: {layout_info['num_blocks_h']}")
    log(f"    num_blocks_w: {layout_info['num_blocks_w']}")
    log(f"    row_size (elements): {layout_info['row_size']}")
    
    log(f"\n  Input Access Tile Configuration:")
    log(f"    H_per_tile: {layout_info['H_per_tile']} (access tile)")
    log(f"    W_per_tile: {layout_info['W_per_tile']} (access tile)")
    log(f"    C_per_tile: {layout_info['C_per_tile']}")
    log(f"    input_dram_tile_size: {layout_info['input_dram_tile_size']} bytes")
    log(f"    input_aligned_tile_size: {layout_info['input_aligned_tile_size']} bytes (row_aligned)")
    log(f"    DRAM factors: P_l3={layout_info['P_l3']}, Q_l3={layout_info['Q_l3']}, C_l3={layout_info['C_l3']}, N_l3={layout_info['N_l3']}")

    log(f"\n  Input Address Calculation:")
    log(f"    Layout: {layout_info['input_layout']}")
    log(f"    Strides (Level 3 = between DRAM tiles, Level 2 = within tile):")
    for (level, dim), stride in sorted(layout_info['input_strides'].items()):
        log(f"      (L{level}, {dim_names[dim]}): {stride}")
    
    log(f"\n  Weight Address Calculation:")
    log(f"    Layout: {layout_info['weight_layout']}")
    weight_tile_size = buffer_tile[5] * buffer_tile[4] * buffer_tile[0] * buffer_tile[1]  # K, C, R, S
    log(f"    Buffer Tile Size: {weight_tile_size} (K={buffer_tile[5]} × C={buffer_tile[4]} × R={buffer_tile[0]} × S={buffer_tile[1]})")
    log(f"    Strides:")
    for (level, dim), stride in sorted(layout_info['weight_strides'].items()):
        log(f"      (L{level}, {dim_names[dim]}): {stride}")
    
    log(f"\n  Output Address Calculation:")
    log(f"    Layout: {layout_info['output_layout']}")
    output_tile_size = buffer_tile[6] * buffer_tile[5] * buffer_tile[2] * buffer_tile[3]  # N, K, P, Q
    log(f"    Buffer Tile Size: {output_tile_size} (N={buffer_tile[6]} × K={buffer_tile[5]} × P={buffer_tile[2]} × Q={buffer_tile[3]})")
    log(f"    Strides:")
    for (level, dim), stride in sorted(layout_info['output_strides'].items()):
        log(f"      (L{level}, {dim_names[dim]}): {stride}")
    
    log(f"\n  Base Addresses:")
    log(f"    Input:  0x{layout_info['input_base']:08X} (Bank 0)")
    log(f"    Weight: 0x{layout_info['weight_base']:08X} (Bank 1)")
    log(f"    Output: 0x{layout_info['output_base']:08X} (Bank 2)")
    
    # =========================================================================
    # 6. Trace Generation Details
    # =========================================================================
    log("\n" + "=" * 100)
    log("6. TRACE GENERATION DETAILS")
    log("=" * 100)
    
    dram_config = DRAMConfig(row_buffer_bytes=1024, num_banks=4, num_rows=16384, element_size=1)
    
    log(f"\n  DRAM Config:")
    log(f"    row_buffer_bytes: {dram_config.row_buffer_bytes}")
    log(f"    num_banks: {dram_config.num_banks}")
    log(f"    num_rows: {dram_config.num_rows}")
    log(f"    element_size: {dram_config.element_size}")
    log(f"    bank_size: {dram_config.row_buffer_bytes * dram_config.num_rows} bytes (16 MB)")
    
    gen = TraceGenerator(dram_config)
    trace = gen.generate_trace(mapping, workload)
    
    # =========================================================================
    # 6.1 INPUT ADDRESS DEBUG INFO
    # =========================================================================
    log("\n" + "=" * 100)
    log("6.1 INPUT ADDRESS CALCULATION DEBUG")
    log("=" * 100)
    
    if hasattr(gen, '_input_debug_info') and gen._input_debug_info:
        entries = gen._input_debug_info[:100]
        
        log(f"\n  First {len(entries)} Input accesses (detailed):")
        log(f"  {'idx':>5} | {'h':>3} {'w':>3} | {'h_blk':>5} {'w_blk':>5} | {'h_in':>4} {'w_in':>4} | "
            f"{'block_base':>12} {'l2_base':>8} {'offset':>8} | {'addr':>12} | {'row':>5} {'col':>5}")
        log("  " + "-" * 95)
        
        prev_row = None
        for entry in entries:
            l2_base = entry.get('l2_tile_base', 0)
            row_switch = ""
            if prev_row is not None and entry['row'] != prev_row:
                row_switch = " <-- ROW SWITCH"
            
            log(f"  {entry['idx']:>5} | {entry['h']:>3} {entry['w']:>3} | "
                f"{entry['h_block']:>5} {entry['w_block']:>5} | "
                f"{entry['h_in_block']:>4} {entry['w_in_block']:>4} | "
                f"{entry['block_base']:>12} {l2_base:>8} {entry['offset_in_block']:>8} | "
                f"0x{entry['addr']:08X} | {entry['row']:>5} {entry['col']:>5}{row_switch}")
            prev_row = entry['row']
        
        # Analyze row switch pattern in ALL debug entries
        log("\n  ROW SWITCH ANALYSIS:")
        row_switches = []
        prev_row = None
        for entry in gen._input_debug_info:
            if prev_row is not None and entry['row'] != prev_row:
                row_switches.append((entry['idx'], prev_row, entry['row'], entry['h'], entry['w']))
            prev_row = entry['row']
        
        log(f"    Row switches in first {len(gen._input_debug_info)} Input accesses: {len(row_switches)}")
        
        if row_switches:
            log(f"\n    First 30 row switches:")
            for idx, from_row, to_row, h, w in row_switches[:30]:
                log(f"      idx={idx}: row {from_row} -> {to_row} (h={h}, w={w})")
    
    # Show boundary crossing debug info
    if hasattr(gen, '_input_debug_boundary_info') and gen._input_debug_boundary_info:
        log(f"\n  W BOUNDARY CROSSING DEBUG (w=30 or w=31):")
        log(f"  {'idx':>6} | {'h':>3} {'w':>3} | {'w_range':<10} | {'h_blk':>5} {'w_blk':>5} | {'h_in':>4} {'w_in':>4} | "
            f"{'block_base':>12} {'offset':>8} | {'addr':>12} | {'row':>5}")
        log("  " + "-" * 100)
        
        prev_row = None
        for entry in gen._input_debug_boundary_info[:50]:
            row_switch = ""
            if prev_row is not None and entry['row'] != prev_row:
                row_switch = " <-- ROW SWITCH"
            
            w_range = f"[{entry['w_start']},{entry['w_end']})"
            log(f"  {entry['idx']:>6} | {entry['h']:>3} {entry['w']:>3} | {w_range:<10} | "
                f"{entry['h_block']:>5} {entry['w_block']:>5} | "
                f"{entry['h_in_block']:>4} {entry['w_in_block']:>4} | "
                f"{entry['block_base']:>12} {entry['offset_in_block']:>8} | "
                f"0x{entry['addr']:08X} | {entry['row']:>5}{row_switch}")
            prev_row = entry['row']
    
    log(f"\n  Trace Statistics:")
    log(f"    Total trace lines: {len(trace)}")
    
    # Analyze per bank
    row_size = dram_config.row_buffer_bytes
    bank_size = row_size * dram_config.num_rows
    
    trace_acts = {}
    for bank_id, tensor_name in [(0, "Input"), (1, "Weight"), (2, "Output")]:
        accesses = []
        for line in trace:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            addr = int(parts[1], 16)
            bank = addr // bank_size
            if bank == bank_id:
                row = (addr % bank_size) // row_size
                col = addr % row_size
                accesses.append((addr, row, col))
        
        log(f"\n  {tensor_name} (Bank {bank_id}):")
        log(f"    Total accesses: {len(accesses)}")
        log(f"    Unique addresses: {len(set(a[0] for a in accesses))}")
        
        unique_rows = sorted(set(a[1] for a in accesses))
        log(f"    Unique rows: {len(unique_rows)}")
        log(f"    Rows accessed: {unique_rows}")
        
        # Count row activations
        current_row = None
        row_acts = 0
        row_visit_counts = defaultdict(int)
        for addr, row, col in accesses:
            if current_row != row:
                row_acts += 1
                row_visit_counts[row] += 1
                current_row = row
        
        trace_acts[bank_id] = row_acts
        
        log(f"    Row activations (switches): {row_acts}")
        log(f"    Row visit pattern:")
        for row in sorted(row_visit_counts.keys()):
            log(f"      Row {row}: activated {row_visit_counts[row]} times")
    
    # =========================================================================
    # 7. Sample Trace Entries
    # =========================================================================
    log("\n" + "=" * 100)
    log("7. SAMPLE TRACE ENTRIES (First 100)")
    log("=" * 100)
    
    current_rows = {}
    for i, line in enumerate(trace[:100]):
        parts = line.strip().split()
        addr = int(parts[1], 16)
        bank = addr // bank_size
        row = (addr % bank_size) // row_size
        col = addr % row_size
        
        tensor = {0: "Input", 1: "Weight", 2: "Output"}.get(bank, f"Bank{bank}")
        marker = ""
        if bank not in current_rows or current_rows[bank] != row:
            marker = " <-- NEW ROW"
            current_rows[bank] = row
        
        log(f"  {i:4d}: {line} -> {tensor:<6} Row={row:3d} Col={col:4d}{marker}")

    # =========================================================================
    # 8. DISCREPANCY ANALYSIS
    # =========================================================================
    log("\n" + "=" * 100)
    log("8. DISCREPANCY ANALYSIS")
    log("=" * 100)
    
    log(f"\n  Comparison:")
    log(f"  {'Tensor':<10} {'ILP':<15} {'Trace':<15} {'Ratio (Trace/ILP)':<20}")
    log(f"  {'-'*60}")
    
    ilp_input = mapping.metrics.get('row_activations_input', 0)
    ilp_weight = mapping.metrics.get('row_activations_weight', 0)
    ilp_output = mapping.metrics.get('row_activations_output', 0)
    
    for t_id, name, ilp in [(0, "Input", ilp_input), (1, "Weight", ilp_weight), (2, "Output", ilp_output)]:
        trace_val = trace_acts[t_id]
        ratio = trace_val / ilp if ilp > 0 else float('inf')
        log(f"  {name:<10} {ilp:<15.2f} {trace_val:<15} {ratio:<20.2f}")
    
    log(f"\n  Key Observations:")
    log(f"    - ILP computes row_acts based on DRAM level tiling factors")
    log(f"    - Trace counts actual row switches during execution")
    log(f"    - The large discrepancy suggests:")
    log(f"      1. ILP model may only consider Level 3 factors, not Level 2+3")
    log(f"      2. Or trace generator iterates both Level 2 and Level 3 loops")
    log(f"      3. Or the formula for row_aligned mode is incorrect")
    
    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(output))
    
    # Return stats for summary
    return {
        'ilp_input': ilp_input,
        'ilp_weight': ilp_weight,
        'ilp_output': ilp_output,
        'trace_input': trace_acts[0],
        'trace_weight': trace_acts[1],
        'trace_output': trace_acts[2],
    }


def main():
    # Define 10 workloads
    workloads = [
        ConvWorkload(name="tiny", R=3, S=3, P=8, Q=8, C=8, K=4, N=1),
        ConvWorkload(name="small", R=3, S=3, P=8, Q=8, C=16, K=16, N=1),
        ConvWorkload(name="small-v2", R=3, S=3, P=16, Q=16, C=16, K=16, N=1),
        ConvWorkload(name="medium", R=3, S=3, P=28, Q=28, C=32, K=32, N=1),
        ConvWorkload(name="ResNet-L1", R=7, S=7, P=56, Q=56, C=3, K=64, N=1),
        ConvWorkload(name="ResNet-L2", R=3, S=3, P=28, Q=28, C=64, K=64, N=1),
        ConvWorkload(name="ResNet-L3", R=3, S=3, P=14, Q=14, C=128, K=128, N=1),
        ConvWorkload(name="VGG-L1", R=3, S=3, P=56, Q=56, C=3, K=64, N=1),
        ConvWorkload(name="MobileNet-L1", R=3, S=3, P=56, Q=56, C=32, K=32, N=1),
        ConvWorkload(name="AlexNet-L1", R=11, S=11, P=55, Q=55, C=3, K=96, N=1),
    ]
    
    # Output directory
    output_dir = Path('/Users/haochenzhao/Projects/pim_optimizer/validation/dram/debug_output/')
    
    print(f"Processing {len(workloads)} workloads...")
    print(f"Output directory: {output_dir}")
    print()
    
    results = []
    for workload in workloads:
        print(f"Processing {workload.name}...")
        output_file = output_dir / workload.name / "analysis.txt"
        try:
            stats = process_workload(workload, output_file)
            results.append((workload.name, stats))
            print(f"  -> Generated: {output_file}")
        except Exception as e:
            print(f"  -> Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print()
    print("=" * 100)
    print("SUMMARY (Weight & Output Comparison)")
    print("=" * 100)
    print(f"{'Workload':<15} | {'ILP Weight':<12} {'Trace Weight':<12} | {'ILP Output':<12} {'Trace Output':<12}")
    print("-" * 100)
    for name, stats in results:
        print(f"{name:<15} | {stats['ilp_weight']:<12.2f} {stats['trace_weight']:<12d} | {stats['ilp_output']:<12.2f} {stats['trace_output']:<12d}")
    
    print()
    print(f"Done! Check {output_dir} for detailed results.")


if __name__ == "__main__":
    main()
