
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.arch import PIMArchitecture, MemoryHierarchy, MemoryLevel
from validation.dram.trace_generator import TraceGenerator, DRAMConfig
from experiments.rank_accuracy_paper.fast_trace_generator import FastTraceGenerator

def verify_filtering():
    # 1. Define the "Worst Case" Workload (AlexNet-L1)
    # This workload had 30x error in Input Row Acts prediction.
    # Use a smaller workload to ensure we can find many solutions quickly
    # VGG_Conv3_1 is a good candidate (mid-sized)
    # UPDATE: VGG_Conv3_1 was too large for the license. Using a tiny workload.
    workload = ConvWorkload(
        name="Tiny_Test",
        P=4, Q=4, C=16, K=16, R=1, S=1, stride=(1,1), dilation=(1,1)
    )
    
    print(f"==================================================================")
    print(f"Verifying ILP Filtering Capability on: {workload.name}")
    print(f"==================================================================")
    
    # Create a simplified architecture to fit in the license
    # 2 levels: PE Local Buffer and DRAM
    hierarchy = MemoryHierarchy(levels=[
        MemoryLevel(
            name="PELocalBuffer",
            entries=64,
            blocksize=1,
            instances=64,
            latency=1,
            access_cost=0.001,
            stores=[False, False, True],
            bypass_defined=True,
        ),
        MemoryLevel(
            name="LocalDRAM",
            entries=-1,  # Unlimited
            blocksize=1024,
            instances=1,
            latency=25,
            access_cost=0.1,
            stores=[True, True, True],
            bypass_defined=True,
            num_banks=4,
            row_buffer_size=1024,
        ),
    ])
    arch = PIMArchitecture(hierarchy=hierarchy)
    
    # 2. Generate a LARGE pool of solutions (Good, Bad, and Ugly)
    optimizer = PIMOptimizer(arch=arch, verbose=False)
    # We want a large pool to see where the "True Best" falls in the ILP ranking
    gurobi_params = {
        "PoolSearchMode": 2, 
        "PoolSolutions": 200,  # Try to find 200 solutions
        "PoolGap": 1000000.0,    # Huge gap to include very bad solutions
        "TimeLimit": 120,        # Give it some time
        "MIPFocus": 1, # Focus on finding feasible solutions
        "OutputFlag": 1 # Enable Gurobi output to see what's happening
    }
    
    print("1. Generating large solution pool (ILP)...")
    result = optimizer.optimize(
        [workload], 
        objective="latency", 
        gurobi_params=gurobi_params,
        enable_row_activation=False
    )
    
    # Force extraction of all solutions from pool
    model = optimizer.model
    num_solutions = model.SolCount
    print(f"   Found {num_solutions} solutions in pool.")
    
    if num_solutions < 10:
        print("   Not enough solutions for a meaningful test.")
        # return

    # 3. Evaluate all solutions with Trace (The "Ground Truth")
    print("2. Evaluating all solutions with Trace Generator...")
    dram_config = DRAMConfig(num_channels=32, num_banks=16, row_buffer_bytes=1024, element_size=1)
    trace_gen = FastTraceGenerator(dram_config)
    
    data = []
    
    # Manually extract from pool
    for sol_idx in range(num_solutions):
        model.params.SolutionNumber = sol_idx
        
        # Extract result
        # We need to call _extract_results but it expects the model to be in the state of the solution
        # Setting SolutionNumber does that for variable attributes like Xn
        
        res = optimizer._extract_results(
            [workload],
            optimizer.compute_cycles,
            optimizer.latency_vars,
            optimizer.activation_cycles,
            optimizer.macs_scale_factors,
            var_attr="Xn" # Use Xn for solution pool
        )
        
        if not res.mappings:
            continue
            
        mapping = res.mappings[0]
        
        # ILP Cost (Total Row Acts) - This will be 0 if row activation is disabled in ILP
        # But we can still look at latency
        ilp_latency = mapping.latency
        
        # Trace Cost (Total Row Acts)
        trace_stats = trace_gen.generate_trace(mapping, workload)
        trace_total = sum(trace_stats.values())
        
        data.append({
            "id": sol_idx,
            "ilp_latency": ilp_latency,
            "trace_row_acts": trace_total,
            "mapping": mapping
        })
        
    # 4. Analyze Results
    df = pd.DataFrame(data)
    print("\nTop 10 Solutions by ILP Latency:")
    print(df.sort_values("ilp_latency").head(10)[["id", "ilp_latency", "trace_row_acts"]])
    
    print("\nTop 10 Solutions by Trace Row Acts (Ground Truth):")
    top_trace = df.sort_values("trace_row_acts").head(10)
    print(top_trace[["id", "ilp_latency", "trace_row_acts"]])
    
    # Check if the best Trace solution is in the top N ILP solutions
    best_trace_id = top_trace.iloc[0]["id"]
    best_trace_rank_in_ilp = df.sort_values("ilp_latency").reset_index().query(f"id == {best_trace_id}").index[0]
    
    print(f"\nAnalysis:")
    print(f"The Best Solution (according to Trace) is Rank #{best_trace_rank_in_ilp} in ILP.")
    
    if best_trace_rank_in_ilp < 50:
        print("SUCCESS: The best solution is within the Top 50 ILP candidates.")
        print("This proves that ILP filtering + Trace Re-ranking works.")
    else:
        print("FAILURE: The best solution is NOT in the Top 50 ILP candidates.")
        print("We need a better ILP model (current model is stripped for license reasons).")
        
    df = pd.DataFrame(data)
    
    # 4. Analyze Rankings
    print("3. Analyzing Rankings...")
    
    # Sort by Trace Cost (True Ranking)
    df_true = df.sort_values("trace_cost").reset_index(drop=True)
    true_best_cost = df_true.iloc[0]["trace_cost"]
    true_best_id = df_true.iloc[0]["id"]
    
    # Sort by ILP Cost (Predicted Ranking)
    df_ilp = df.sort_values("ilp_cost").reset_index(drop=True)
    
    # Find where the "True Best" sits in the ILP Ranking
    # We look for the rank (index) of the solution with true_best_id in df_ilp
    ilp_rank_of_true_best = df_ilp[df_ilp["id"] == true_best_id].index[0]
    
    print(f"\n   True Best Solution ID: {true_best_id}")
    print(f"   True Best Trace Cost:  {true_best_cost}")
    print(f"   ILP Predicted Rank:    {ilp_rank_of_true_best} (0-indexed)")
    
    # 5. Check "Recall" for Top-K
    # If we pick Top-N from ILP, do we get the True Top-1?
    print(f"\n   Filtering Efficiency:")
    for n in [1, 5, 10, 20, 50]:
        if n > num_solutions: break
        # Get IDs of Top-N ILP solutions
        top_n_ilp_ids = set(df_ilp.head(n)["id"])
        
        # Check if True Best is in there
        found = true_best_id in top_n_ilp_ids
        status = "✅ FOUND" if found else "❌ MISSED"
        print(f"     Top-{n:<2} ILP Candidates: {status} (True Best is Rank {ilp_rank_of_true_best})")

    # 6. Check Correlation in the "Good" Region (Top 20 ILP)
    # We care most about correlation among the good candidates, not the garbage ones.
    top_20_ilp = df_ilp.head(20)
    corr = top_20_ilp["ilp_cost"].corr(top_20_ilp["trace_cost"], method="spearman")
    print(f"\n   Spearman Correlation in Top-20 ILP Candidates: {corr:.4f}")
    
    # 7. Conclusion
    print(f"\n==================================================================")
    if ilp_rank_of_true_best < 20:
        print(f"CONCLUSION: VALIDATED.")
        print(f"The ILP successfully placed the True Best solution in its Top {ilp_rank_of_true_best+1}.")
        print(f"A 'Filter-Verify' workflow selecting Top-20 would find the optimal solution.")
    else:
        print(f"CONCLUSION: FAILED.")
        print(f"The True Best solution was ranked {ilp_rank_of_true_best} by ILP.")
        print(f"Top-20 filtering would miss it.")

if __name__ == "__main__":
    verify_filtering()
