#!/usr/bin/env python3
"""
Debug script for Row Activation validation.

Prints detailed mapping information for workloads with high error.
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'validation', 'dram'))

import traceback
import numpy as np

from pim_optimizer.arch.pim_arch import PIMArchitecture
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer import PIMOptimizer
from pim_optimizer.mapping import Mapping
from pim_optimizer.model.variables import SpatialDim

from trace_generator import TraceGenerator, DRAMConfig, DIM_NAMES


# Test workloads
WORKLOADS = {
    'tiny': {'N': 1, 'K': 8, 'C': 8, 'P': 4, 'Q': 4, 'R': 3, 'S': 3},
    'small': {'N': 1, 'K': 16, 'C': 16, 'P': 8, 'Q': 8, 'R': 3, 'S': 3},
    'medium': {'N': 1, 'K': 32, 'C': 32, 'P': 14, 'Q': 14, 'R': 3, 'S': 3},
    'ResNet-L1': {'N': 1, 'K': 32, 'C': 3, 'P': 28, 'Q': 28, 'R': 7, 'S': 7},
    'ResNet-L2': {'N': 1, 'K': 64, 'C': 32, 'P': 14, 'Q': 14, 'R': 3, 'S': 3},
    'ResNet-L3': {'N': 1, 'K': 128, 'C': 64, 'P': 7, 'Q': 7, 'R': 3, 'S': 3},
    'VGG-L1': {'N': 1, 'K': 32, 'C': 3, 'P': 56, 'Q': 56, 'R': 3, 'S': 3},
    'MobileNet-L1': {'N': 1, 'K': 32, 'C': 16, 'P': 28, 'Q': 28, 'R': 1, 'S': 1},
}


def extract_mapping(optimizer, workload) -> Mapping:
    """Extract mapping from optimizer result with full details."""
    model = optimizer.model
    arch = optimizer.arch
    num_mems = arch.num_mems
    vars = optimizer.vars
    w = 0
    
    mapping = Mapping()
    mapping.workload_name = workload.name
    mapping.workload_bounds = list(workload.bounds)
    
    # Extract loop bounds
    for m in range(num_mems):
        if m == 0:
            mapping.loop_bounds[m] = {"H": {}, "W": {}, "Internal": {}, "temporal": {}}
            s_names = {SpatialDim.H: "H", SpatialDim.W: "W", SpatialDim.INTERNAL: "Internal", SpatialDim.TEMPORAL: "temporal"}
            s_range = SpatialDim.num_dims_pe()
        else:
            mapping.loop_bounds[m] = {"spatial": {}, "temporal": {}}
            s_names = {0: "spatial", 1: "temporal"}
            s_range = SpatialDim.num_dims_other()
        
        for j, divs in enumerate(workload.divisors):
            for s in range(s_range):
                s_name = s_names[s]
                for i, div in enumerate(divs):
                    if vars.xb[w, m, s, j, i].X > 0.5:
                        mapping.loop_bounds[m][s_name][j] = div
    
    # Extract permutation
    for m in range(num_mems):
        mapping.permutation[m] = {}
        for p in range(len(workload.bounds)):
            for j in range(len(workload.bounds)):
                if vars.xp[w, m, p, j].X > 0.5:
                    mapping.permutation[m][p] = j
    
    # Extract layout
    for t in range(3):
        if (w, t, "row_aligned") in vars.layout_choice:
            if vars.layout_choice[w, t, "row_aligned"].X > 0.5:
                mapping.layout[t] = "row_aligned"
            else:
                mapping.layout[t] = "sequential"
    
    # Extract input block_h and block_w
    h_divisors = getattr(workload, 'hw_divisors', {}).get('H', [1])
    w_divisors = getattr(workload, 'hw_divisors', {}).get('W', [1])
    
    block_h = 1
    for i, h_div in enumerate(h_divisors):
        if (w, i) in vars.rowbuf_input_block_h:
            if vars.rowbuf_input_block_h[w, i].X > 0.5:
                block_h = h_div
                break
    
    block_w = 1
    for j, w_div in enumerate(w_divisors):
        if (w, j) in vars.rowbuf_input_block_w:
            if vars.rowbuf_input_block_w[w, j].X > 0.5:
                block_w = w_div
                break
    
    mapping.tile_info = {'block_h': block_h, 'block_w': block_w}
    
    return mapping


def print_mapping_details(mapping, workload):
    """Print detailed mapping information."""
    print("\n" + "=" * 80)
    print(f"MAPPING DETAILS for {workload.name}")
    print("=" * 80)
    
    # Workload info
    print(f"\nWorkload: N={workload.N}, K={workload.K}, C={workload.C}, "
          f"P={workload.P}, Q={workload.Q}, R={workload.R}, S={workload.S}")
    print(f"Input size: H={workload.input_size['H']}, W={workload.input_size['W']}")
    print(f"Stride: {workload.stride}, Dilation: {workload.dilation}")
    
    # Tile info
    print(f"\nTile Info:")
    print(f"  block_h = {mapping.tile_info.get('block_h', 1)}")
    print(f"  block_w = {mapping.tile_info.get('block_w', 1)}")
    
    # Layout
    print(f"\nLayout:")
    print(f"  Input:  {mapping.layout.get(0, 'sequential')}")
    print(f"  Weight: {mapping.layout.get(1, 'sequential')}")
    print(f"  Output: {mapping.layout.get(2, 'sequential')}")
    
    # Loop bounds per level
    print(f"\nLoop Bounds:")
    for m in sorted(mapping.loop_bounds.keys()):
        print(f"  Level {m}:")
        for key, bounds in mapping.loop_bounds[m].items():
            if bounds:
                bounds_str = ", ".join([f"{DIM_NAMES[d]}={v}" for d, v in bounds.items()])
                print(f"    {key}: {bounds_str}")
    
    # Permutation per level
    print(f"\nPermutation (position -> dim):")
    for m in sorted(mapping.permutation.keys()):
        perm = mapping.permutation[m]
        if perm:
            # Sort by position
            sorted_perm = sorted(perm.items(), key=lambda x: x[0])
            perm_str = " -> ".join([f"[{p}]{DIM_NAMES[d]}" for p, d in sorted_perm])
            print(f"  Level {m}: {perm_str}")
    
    # Compute buffer tile size (Level 0+1)
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
    
    print(f"\nBuffer Tile (Level 0+1):")
    print(f"  " + ", ".join([f"{DIM_NAMES[d]}={buffer_tile[d]}" for d in range(7)]))
    
    # Level 2 factors
    level2_factors = {d: 1 for d in range(7)}
    if 2 in mapping.loop_bounds:
        for key in ['spatial', 'temporal']:
            if key in mapping.loop_bounds[2]:
                for d, bound in mapping.loop_bounds[2][key].items():
                    level2_factors[d] *= bound
    
    print(f"\nLevel 2 Factors:")
    print(f"  " + ", ".join([f"{DIM_NAMES[d]}={level2_factors[d]}" for d in range(7)]))
    
    # RowBuffer tile (Level 0+1+2)
    rb_tile = {d: buffer_tile[d] * level2_factors[d] for d in range(7)}
    print(f"\nRowBuffer Tile (Level 0+1+2):")
    print(f"  " + ", ".join([f"{DIM_NAMES[d]}={rb_tile[d]}" for d in range(7)]))
    
    print()


def count_row_activations(trace_path: str, dram_config: DRAMConfig) -> dict:
    """Count row activations from trace file."""
    row_buffer_bytes = dram_config.row_buffer_bytes
    num_rows = dram_config.num_rows
    
    col_bits = int(np.log2(row_buffer_bytes))
    row_bits = int(np.log2(num_rows))
    
    open_rows = {}
    total_row_acts = 0
    per_bank_acts = {}
    
    with open(trace_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                addr = int(parts[1], 16)
            except ValueError:
                continue
            
            row = (addr >> col_bits) & ((1 << row_bits) - 1)
            bank = addr >> (col_bits + row_bits)
            
            if bank not in open_rows or open_rows[bank] != row:
                open_rows[bank] = row
                total_row_acts += 1
                per_bank_acts[bank] = per_bank_acts.get(bank, 0) + 1
    
    return {
        'total': total_row_acts,
        'per_bank': per_bank_acts,
    }


def debug_workload(name: str, params: dict, arch, dram_config: DRAMConfig):
    """Debug a single workload."""
    print("\n" + "#" * 80)
    print(f"# DEBUGGING: {name}")
    print("#" * 80)
    
    # Create workload
    stride = params.pop('stride', (1, 1))
    workload = ConvWorkload(name=name, stride=stride, **params)
    params['stride'] = stride
    
    # Run optimizer
    print(f"\nRunning ILP optimizer...")
    optimizer = PIMOptimizer(arch, verbose=False)
    result = optimizer.optimize([workload])
    model = optimizer.model
    
    # Get ILP row activations
    ilp_row_acts = {}
    for t_id, t_name in [(0, 'input'), (1, 'weight'), (2, 'output')]:
        var = model.getVarByName(f'total_row_acts_(0,{t_id})')
        if var:
            ilp_row_acts[t_name] = var.X
        else:
            ilp_row_acts[t_name] = 0
    ilp_row_acts['total'] = sum(ilp_row_acts.values())
    
    print(f"\nILP Row Activations:")
    print(f"  Input:  {ilp_row_acts['input']:.0f}")
    print(f"  Weight: {ilp_row_acts['weight']:.0f}")
    print(f"  Output: {ilp_row_acts['output']:.0f}")
    print(f"  Total:  {ilp_row_acts['total']:.0f}")
    
    # Extract and print mapping
    mapping = extract_mapping(optimizer, workload)
    print_mapping_details(mapping, workload)
    
    # Try to generate trace
    print("Generating trace...")
    trace_gen = TraceGenerator(dram_config)
    
    try:
        traces = trace_gen.generate_trace(mapping, workload)
        
        # Write trace
        trace_path = f"/tmp/{name}_trace.txt"
        with open(trace_path, 'w') as f:
            f.write('\n'.join(traces))
        
        print(f"Generated {len(traces)} trace lines")
        
        # Count row activations
        trace_stats = count_row_activations(trace_path, dram_config)
        
        print(f"\nTrace Row Activations:")
        print(f"  Total: {trace_stats['total']}")
        print(f"  Per Bank: {dict(trace_stats['per_bank'])}")
        
        # Calculate error
        if trace_stats['total'] > 0:
            error = abs(ilp_row_acts['total'] - trace_stats['total']) / trace_stats['total'] * 100
        else:
            error = 100
        
        print(f"\nError: {error:.2f}%")
        
        # Print first few traces
        print(f"\nFirst 20 trace lines:")
        for i, line in enumerate(traces[:20]):
            print(f"  {line}")
        
        # Print unique addresses
        unique_addrs = set()
        for line in traces[:1000]:
            parts = line.split()
            if len(parts) >= 2:
                unique_addrs.add(parts[1])
        print(f"\nUnique addresses in first 1000 traces: {len(unique_addrs)}")
        print(f"Sample: {list(unique_addrs)[:10]}")
        
        return error
        
    except Exception as e:
        print(f"\n!!! TraceGenerator ERROR !!!")
        print(f"Exception: {e}")
        traceback.print_exc()
        
        # Print more debug info
        print(f"\nDebug Info:")
        print(f"  block_h = {mapping.tile_info.get('block_h', 1)}")
        print(f"  block_w = {mapping.tile_info.get('block_w', 1)}")
        print(f"  H_in = {workload.input_size['H']}")
        print(f"  W_in = {workload.input_size['W']}")
        
        return None


def main():
    arch = PIMArchitecture.from_yaml('examples/configs/arch.yaml')
    dram_config = DRAMConfig()
    
    print("=" * 80)
    print("Row Activation Debug Script")
    print("=" * 80)
    
    results = []
    
    for name, params in WORKLOADS.items():
        error = debug_workload(name, params.copy(), arch, dram_config)
        results.append((name, error))
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Workload':<20} {'Error':>10}")
    print("-" * 30)
    for name, error in results:
        if error is not None:
            print(f"{name:<20} {error:>9.2f}%")
        else:
            print(f"{name:<20} {'FAILED':>10}")


if __name__ == "__main__":
    main()
