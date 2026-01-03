
import sys
import os
import csv
import copy
from pathlib import Path
import gurobipy as gp

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.baselines import HeuristicOptimizer
from pim_optimizer.arch import PIMArchitecture
from pim_optimizer.arch.memory import MemoryHierarchy, MemoryLevel
from pim_optimizer.workload import ConvWorkload
from validation.dram.trace_generator import DRAMConfig
from experiments.rank_accuracy_paper.fast_trace_generator import FastTraceGenerator

def create_custom_arch(row_buffer_bytes):
    """
    Create a PIMArchitecture with a specific Row Buffer size.
    """
    # Start with default hierarchy
    hierarchy = MemoryHierarchy()
    
    # Find RowBuffer level and modify it
    # Note: We need to reconstruct the hierarchy or modify the existing object carefully
    # The default hierarchy is created in __init__ if None is passed.
    # Let's manually construct the levels based on the default but change RowBuffer.
    
    levels = [
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
            name="GlobalBuffer",
            entries=256,
            blocksize=32,
            instances=1,
            latency=1,
            access_cost=0.01,
            stores=[True, True, True],
            bypass_defined=True,
        ),
        MemoryLevel(
            name="RowBuffer",
            entries=row_buffer_bytes, # Change entries to match size (assuming 1 byte per entry)
            blocksize=16,
            instances=1,
            latency=1,
            access_cost=0.01,
            stores=[True, True, True],
            bypass_defined=True,
            num_banks=16, # Use 16 banks for this experiment
            row_buffer_size=row_buffer_bytes, # Explicitly set row buffer size
        ),
        MemoryLevel(
            name="LocalDRAM",
            entries=10000000,
            blocksize=64,
            instances=1,
            latency=25,
            access_cost=0.1,
            stores=[True, True, True],
            bypass_defined=True,
            num_banks=16,
            row_buffer_size=row_buffer_bytes,
        )
    ]
    
    hierarchy = MemoryHierarchy(levels)
    arch = PIMArchitecture(hierarchy=hierarchy)
    return arch

def run_sensitivity_analysis():
    print("========================================================")
    print("      Row Buffer Size Sensitivity Analysis")
    print("========================================================")
    
    # Parameters
    row_buffer_sizes = [256, 512, 1024, 2048, 4096]
    
    # Workloads to test
    # 1. ResNet_L1 (Standard)
    # 2. VGG_Conv2_1 (High sensitivity observed previously)
    workloads = [
        ConvWorkload(
            name="ResNet_L1",
            P=56, Q=56, C=64, K=64, R=3, S=3, stride=(1,1), dilation=(1,1)
        ),
        ConvWorkload(
            name="VGG_Conv2_1",
            P=112, Q=112, C=64, K=128, R=3, S=3, stride=(1,1), dilation=(1,1)
        )
    ]
    
    # CSV Writer
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "row_buffer_sensitivity.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            "Workload", "RowBuffer_Bytes", 
            "WS_Max_Acts", "OS_Max_Acts", "ILP_Max_Acts",
            "WS_Latency", "OS_Latency", "ILP_Latency" # Note: Heuristic latency is estimated or 0 if not available
        ]
        writer.writerow(header)
        
        for workload in workloads:
            print(f"\nProcessing Workload: {workload.name}")
            
            for rb_size in row_buffer_sizes:
                print(f"  Testing Row Buffer Size: {rb_size} Bytes...", end="", flush=True)
                
                # Create Arch
                arch = create_custom_arch(rb_size)
                
                # Create Trace Generator with matching config
                dram_config = DRAMConfig(
                    num_channels=32, # Keep consistent
                    num_banks=16,
                    row_buffer_bytes=rb_size,
                    element_size=1
                )
                trace_gen = FastTraceGenerator(dram_config)
                
                # 1. Weight Stationary
                ws_opt = HeuristicOptimizer(arch, strategy="weight_stationary")
                ws_res = ws_opt.optimize([workload])
                ws_mapping = ws_res.mappings[0]
                ws_stats = trace_gen.generate_trace(ws_mapping, workload)
                ws_max = max(ws_stats.values())
                
                # 2. Output Stationary
                os_opt = HeuristicOptimizer(arch, strategy="output_stationary")
                os_res = os_opt.optimize([workload])
                os_mapping = os_res.mappings[0]
                os_stats = trace_gen.generate_trace(os_mapping, workload)
                os_max = max(os_stats.values())
                
                # 3. ILP
                ilp_opt = PIMOptimizer(arch=arch, verbose=False, time_limit=45)
                try:
                    ilp_res = ilp_opt.optimize([workload], objective="latency")
                    if ilp_res.mappings:
                        ilp_mapping = ilp_res.mappings[0]
                        ilp_stats = trace_gen.generate_trace(ilp_mapping, workload)
                        ilp_max = max(ilp_stats.values())
                        ilp_lat = ilp_res.total_latency
                    else:
                        ilp_max = "Failed"
                        ilp_lat = "Failed"
                except Exception as e:
                    print(f" (ILP Error: {e})", end="")
                    ilp_max = "Error"
                    ilp_lat = "Error"
                
                print(" Done.")
                
                # Write row
                writer.writerow([
                    workload.name, rb_size,
                    ws_max, os_max, ilp_max,
                    0, 0, ilp_lat # Heuristic latency not calculated yet
                ])
                f.flush() # Ensure data is written

    print(f"\nSensitivity analysis complete. Results saved to {csv_path}")

if __name__ == "__main__":
    run_sensitivity_analysis()
