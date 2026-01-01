"""
Debug ILP row activation model vs trace.
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload


def main():
    # Test with 'small' workload
    workload = ConvWorkload(name="small", N=1, K=16, C=16, P=8, Q=8, R=3, S=3)
    
    print(f"Workload: small")
    print(f"  Dimensions: R={workload.R}, S={workload.S}, P={workload.P}, Q={workload.Q}, "
          f"C={workload.C}, K={workload.K}, N={workload.N}")
    print(f"  Input:  {workload.N} x {workload.C} x {workload.input_size['H']} x {workload.input_size['W']}")
    print(f"  Weight: {workload.K} x {workload.C} x {workload.R} x {workload.S}")
    print(f"  Output: {workload.N} x {workload.K} x {workload.P} x {workload.Q}")
    
    # Relevancy matrix O[dim][tensor]
    # Input:  relevant to R, S, P, Q, C, N (not K)
    # Weight: relevant to R, S, C, K (not P, Q, N)
    # Output: relevant to P, Q, K, N (not R, S, C)
    print(f"\n  Relevancy matrix O[dim][tensor]:")
    print(f"    {'Dim':<8} {'Input':<8} {'Weight':<8} {'Output':<8}")
    for i, dim in enumerate(['R', 'S', 'P', 'Q', 'C', 'K', 'N']):
        print(f"    {dim:<8} {workload.O[i][0]:<8} {workload.O[i][1]:<8} {workload.O[i][2]:<8}")
    
    # Get mapping
    optimizer = PIMOptimizer()
    result = optimizer.optimize([workload], objective="latency")
    mapping = result.mappings[0]
    
    print(f"\n  Mapping loop bounds:")
    for m in sorted(mapping.loop_bounds.keys()):
        print(f"    Level {m}: {mapping.loop_bounds[m]}")
    
    # Calculate DRAM level factors
    print(f"\n  DRAM level (2+3) factors:")
    dram_factors = {i: 1 for i in range(7)}
    for m in [2, 3]:
        if m in mapping.loop_bounds:
            for key in ['spatial', 'temporal']:
                if key in mapping.loop_bounds[m]:
                    for dim, bound in mapping.loop_bounds[m][key].items():
                        dram_factors[dim] *= bound
    
    dim_names = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
    for i, name in enumerate(dim_names):
        if dram_factors[i] > 1:
            print(f"    {name}: {dram_factors[i]}")
    
    # Calculate reuse penalty for each tensor
    print(f"\n  Reuse penalty (irrelevant DRAM factors):")
    for t_id, t_name in [(0, "Input"), (1, "Weight"), (2, "Output")]:
        irrelevant = [i for i in range(7) if workload.O[i][t_id] == 0]
        penalty = 1
        for dim in irrelevant:
            penalty *= dram_factors[dim]
        irrelevant_names = [dim_names[i] for i in irrelevant]
        print(f"    {t_name}: irrelevant dims = {irrelevant_names}, penalty = {penalty}")
    
    # Calculate expected row activations
    print(f"\n  Expected row activations (manual calculation):")
    
    # For row_aligned Input:
    # row_acts = Π_{j} DRAM_bound_j = all dimensions
    input_all_factors = 1
    for i in range(7):
        input_all_factors *= dram_factors[i]
    print(f"    Input (row_aligned): Π(all DRAM factors) = {input_all_factors}")
    
    # But ILP says 8, let's see what the ILP model actually computes
    print(f"\n  ILP predicted row activations:")
    print(f"    Input:  {mapping.metrics.get('row_activations_input', 0):.1f}")
    print(f"    Weight: {mapping.metrics.get('row_activations_weight', 0):.1f}")
    print(f"    Output: {mapping.metrics.get('row_activations_output', 0):.1f}")
    
    # What trace observes
    print(f"\n  Trace observed (from previous run):")
    print(f"    Input:  505 (unique rows: 9, each re-activated many times)")
    print(f"    Weight: 16")
    print(f"    Output: 1")
    
    # Explanation
    print(f"\n  Analysis:")
    print(f"    ILP model computes row_acts based on tile counts at DRAM level")
    print(f"    Trace counts actual row switches due to access pattern")
    print(f"    ")
    print(f"    The discrepancy arises because:")
    print(f"    1. ILP assumes each row is activated once per 'tile load'")
    print(f"    2. Trace sees interleaved accesses across tiles, causing row thrashing")


if __name__ == "__main__":
    main()
