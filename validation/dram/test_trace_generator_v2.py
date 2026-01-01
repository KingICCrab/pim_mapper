#!/usr/bin/env python3
"""
Test the new simplified TraceGeneratorV2.
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from pim_optimizer.workload import ConvWorkload
from pim_optimizer import PIMOptimizer
from pim_optimizer.arch import PIMArchitecture

from trace_generator_v2 import TraceGeneratorV2, DRAMConfig


def test_small_workload():
    """Test with small workload."""
    
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
    
    ilp_total = sum(ilp_row_acts.values())
    print(f"  Total: {ilp_total}")
    
    # Use new TraceGeneratorV2
    print("\n" + "=" * 60)
    print("TRACE GENERATOR V2 ROW ACTIVATIONS")
    print("=" * 60)
    
    gen = TraceGeneratorV2()
    trace_row_acts = gen.count_row_activations(mapping, workload)
    
    print(f"  input:  {trace_row_acts['input']}")
    print(f"  weight: {trace_row_acts['weight']}")
    print(f"  output: {trace_row_acts['output']}")
    trace_total = sum(trace_row_acts.values())
    print(f"  Total:  {trace_total}")
    
    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    match = (ilp_total == trace_total)
    print(f"ILP Total: {ilp_total}")
    print(f"Trace Total: {trace_total}")
    print(f"Match: {'✓ YES' if match else '✗ NO'}")
    
    if not match:
        print(f"\nPer-tensor comparison:")
        for t_name in ['input', 'weight', 'output']:
            ilp_val = ilp_row_acts.get(t_name, 0)
            trace_val = trace_row_acts.get(t_name, 0)
            match_t = (ilp_val == trace_val)
            print(f"  {t_name}: ILP={ilp_val}, Trace={trace_val}, {'✓' if match_t else '✗'}")
    
    return match


def test_tiny_workload():
    """Test with tiny workload."""
    
    workload = ConvWorkload(
        name='tiny', R=3, S=3, P=4, Q=4, C=8, K=8, N=1,
        stride=(1,1), dilation=(1,1)
    )
    
    print(f"\n\nWorkload: {workload.name}")
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
    
    ilp_total = sum(ilp_row_acts.values())
    print(f"  Total: {ilp_total}")
    
    # Use new TraceGeneratorV2
    print("\n" + "=" * 60)
    print("TRACE GENERATOR V2 ROW ACTIVATIONS")
    print("=" * 60)
    
    gen = TraceGeneratorV2()
    trace_row_acts = gen.count_row_activations(mapping, workload)
    
    print(f"  input:  {trace_row_acts['input']}")
    print(f"  weight: {trace_row_acts['weight']}")
    print(f"  output: {trace_row_acts['output']}")
    trace_total = sum(trace_row_acts.values())
    print(f"  Total:  {trace_total}")
    
    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    match = (ilp_total == trace_total)
    print(f"ILP Total: {ilp_total}")
    print(f"Trace Total: {trace_total}")
    print(f"Match: {'✓ YES' if match else '✗ NO'}")
    
    return match


if __name__ == "__main__":
    print("Testing TraceGeneratorV2...")
    
    small_ok = test_small_workload()
    tiny_ok = test_tiny_workload()
    
    # Test medium workload
    workload = ConvWorkload(
        name='medium', R=3, S=3, P=7, Q=7, C=32, K=32, N=1,
        stride=(1,1), dilation=(1,1)
    )
    
    print(f"\n\nWorkload: {workload.name}")
    print(f"  R={workload.R}, S={workload.S}, P={workload.P}, Q={workload.Q}")
    print(f"  C={workload.C}, K={workload.K}, N={workload.N}")
    
    arch = PIMArchitecture.from_yaml('examples/configs/arch.yaml')
    optimizer = PIMOptimizer(arch=arch)
    result = optimizer.optimize([workload])
    mapping = result.mappings[0]
    model = optimizer.model
    
    # Print mapping details
    print("\nMapping details:")
    print(f"  Loop bounds Level 0: {mapping.loop_bounds.get(0, {})}")
    print(f"  Loop bounds Level 3: {mapping.loop_bounds.get(3, {})}")
    print(f"  Permutation Level 0: {mapping.permutation.get(0, {})}")
    print(f"  Permutation Level 3: {mapping.permutation.get(3, {})}")
    print(f"  Layout: {mapping.layout}")
    print(f"  Tile info: {mapping.tile_info}")
    
    # Build loops
    gen = TraceGeneratorV2()
    loops = gen.build_loop_nesting(mapping)
    print("\nLoop nesting (outer to inner):")
    for level, key, dim, bound in loops:
        dim_names = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
        print(f"  Level {level} {key:8s} {dim_names[dim]}: bound={bound}")
    
    level3_loops = [(l, k, d, b) for l, k, d, b in loops if l == 3]
    print(f"\nLevel 3 loops: {len(level3_loops)}")
    for l, k, d, b in level3_loops:
        print(f"  Level {l} {k} dim={d} bound={b}")
    
    ilp_row_acts = {}
    for t_id, t_name in [(0, 'input'), (1, 'weight'), (2, 'output')]:
        var = model.getVarByName(f'total_row_acts_(0,{t_id})')
        if var:
            ilp_row_acts[t_name] = var.X
    
    ilp_total = sum(ilp_row_acts.values())
    
    trace_row_acts = gen.count_row_activations(mapping, workload)
    trace_total = sum(trace_row_acts.values())
    
    medium_ok = (ilp_total == trace_total)
    
    print(f"\nILP: input={ilp_row_acts['input']}, weight={ilp_row_acts['weight']}, output={ilp_row_acts['output']}, total={ilp_total}")
    print(f"Trace: input={trace_row_acts['input']}, weight={trace_row_acts['weight']}, output={trace_row_acts['output']}, total={trace_total}")
    print(f"Match: {'✓ YES' if medium_ok else '✗ NO'}")
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"small workload:  {'✓ PASS' if small_ok else '✗ FAIL'}")
    print(f"tiny workload:   {'✓ PASS' if tiny_ok else '✗ FAIL'}")
    print(f"medium workload: {'✓ PASS' if medium_ok else '✗ FAIL'}")
