
import sys
import os
import gurobipy as gp

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.baselines import HeuristicOptimizer
from pim_optimizer.arch import PIMArchitecture
from pim_optimizer.workload import ConvWorkload
from validation.dram.trace_generator import DRAMConfig
from experiments.rank_accuracy_paper.fast_trace_generator import FastTraceGenerator

def debug_resnet_1x1_red():
    print("========================================================")
    print("      Debugging ResNet_1x1_Red Anomaly")
    print("========================================================")
    
    # 1. Define Workload
    # ResNet-50 Bottleneck 1x1 (Reduce)
    # P=7, Q=7, C=64, K=16, R=1, S=1
    workload = ConvWorkload(
        name="ResNet_1x1_Red",
        P=7, Q=7, C=64, K=16, R=1, S=1, stride=(1,1), dilation=(1,1)
    )
    print(f"Workload: {workload}")

    # 2. Setup Environment
    arch = PIMArchitecture()
    dram_config = DRAMConfig(
        num_channels=32,
        num_banks=16,
        row_buffer_bytes=1024,
        element_size=1
    )
    trace_gen = FastTraceGenerator(dram_config)

    # 3. Run Heuristic (Weight Stationary) - The "Good" Result
    print("\n--- Running Heuristic (Weight Stationary) ---")
    ws_opt = HeuristicOptimizer(arch, strategy="weight_stationary")
    ws_res = ws_opt.optimize([workload])
    ws_mapping = ws_res.mappings[0]
    
    print("WS Mapping:")
    print(ws_mapping)
    
    ws_stats = trace_gen.generate_trace(ws_mapping, workload)
    print(f"WS Stats: Max={max(ws_stats.values())}, Total={sum(ws_stats.values())}")
    print(f"WS Trace Stats: {ws_stats}")

    # 4. Run ILP Optimizer - The "Bad" Result
    print("\n--- Running ILP Optimizer ---")
    # Enable verbose to see solver output
    ilp_opt = PIMOptimizer(verbose=True, time_limit=30) 
    ilp_res = ilp_opt.optimize([workload])
    
    if not ilp_res.mappings:
        print("ILP failed to find a solution.")
        return

    ilp_mapping = ilp_res.mappings[0]
    print("ILP Mapping:")
    print(ilp_mapping)

    ilp_stats = trace_gen.generate_trace(ilp_mapping, workload)
    print(f"ILP Stats: Max={max(ilp_stats.values())}, Total={sum(ilp_stats.values())}")
    print(f"ILP Trace Stats: {ilp_stats}")

    # 5. Comparison Analysis
    print("\n--- Comparison ---")
    # print(f"WS  Tile Sizes: {ws_mapping.tile_sizes}") # Mapping object might not have this populated by Heuristic
    print(f"WS  Loop Order: {ws_mapping.loop_order}")
    print(f"ILP Loop Order: {ilp_mapping.loop_order}")
    
    print("\n--- Latency Analysis ---")
    print(f"Arch Activation Latency: {getattr(arch, 'dram_activation_latency', 'Unknown')}")
    print(f"ILP Metrics: {ilp_mapping.metrics}")
    
    # Estimate Heuristic Compute Cycles
    # Total MACs = P*Q*C*K*R*S
    total_macs = workload.P * workload.Q * workload.C * workload.K * workload.R * workload.S
    print(f"Total MACs: {total_macs}")
    
    # Check how many banks WS uses
    # WS Mapping: spatial loop bounds
    # We need to interpret the mapping to see parallelism
    # But we can just look at the trace stats? No, trace stats is memory.
    
    # Let's try to infer parallelism from loop bounds
    # Level 2 is DRAM (Banks). If there is a spatial loop here, it's using banks.
    # ws_mapping.loop_bounds[2]['spatial']
    print(f"WS Spatial Bounds (DRAM): {ws_mapping.loop_bounds[2].get('spatial', {})}")
    
    # Level 0 is Compute (PEs).
    # ws_mapping.loop_bounds[0]['W'] or 'H'
    print(f"WS Spatial Bounds (PE): {ws_mapping.loop_bounds[0]}")

if __name__ == "__main__":
    debug_resnet_1x1_red()
