
import sys
import os
import csv
from pathlib import Path
import gurobipy as gp

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.arch import PIMArchitecture
from pim_optimizer.arch.memory import MemoryHierarchy, MemoryLevel
from pim_optimizer.workload import ConvWorkload
from validation.dram.trace_generator import DRAMConfig
from experiments.rank_accuracy_paper.fast_trace_generator import FastTraceGenerator

def create_small_buffer_arch(row_buffer_bytes=512):
    """
    Create a PIMArchitecture with a small Row Buffer to stress test the optimizer.
    """
    levels = [
        MemoryLevel(name="PELocalBuffer", entries=64, blocksize=1, instances=64, latency=1, access_cost=0.001, stores=[False, False, True], bypass_defined=True),
        MemoryLevel(name="GlobalBuffer", entries=256, blocksize=32, instances=1, latency=1, access_cost=0.01, stores=[True, True, True], bypass_defined=True),
        MemoryLevel(
            name="RowBuffer",
            entries=row_buffer_bytes, 
            blocksize=16,
            instances=1,
            latency=1,
            access_cost=0.01,
            stores=[True, True, True],
            bypass_defined=True,
            num_banks=16, 
            row_buffer_size=row_buffer_bytes,
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
    return PIMArchitecture(hierarchy=hierarchy)

def run_ablation_study():
    print("========================================================")
    print("      Ablation Study: Row Activation Cost Model")
    print("========================================================")
    
    # 1. Setup
    # Use 512B Row Buffer - small enough to cause thrashing if not careful
    ROW_BUFFER_SIZE = 512 
    arch = create_small_buffer_arch(ROW_BUFFER_SIZE)
    
    dram_config = DRAMConfig(
        num_channels=32,
        num_banks=16,
        row_buffer_bytes=ROW_BUFFER_SIZE,
        element_size=1
    )
    trace_gen = FastTraceGenerator(dram_config)
    
    # Workload: ResNet_L1
    workload = ConvWorkload(
        name="ResNet_L1",
        P=56, Q=56, C=64, K=64, R=3, S=3, stride=(1,1), dilation=(1,1)
    )
    
    print(f"Workload: {workload.name}")
    print(f"Row Buffer Size: {ROW_BUFFER_SIZE} Bytes")
    
    results = []
    
    # 2. Run Full ILP (With Row Activation Model)
    print("\n--- Running Full ILP (With Cost Model) ---")
    opt_full = PIMOptimizer(arch=arch, verbose=False, time_limit=60)
    res_full = opt_full.optimize([workload], enable_row_activation=True)
    
    if res_full.mappings:
        map_full = res_full.mappings[0]
        stats_full = trace_gen.generate_trace(map_full, workload)
        max_acts_full = max(stats_full.values())
        print(f"Full ILP Max Acts: {max_acts_full}")
        print(f"Full ILP Latency (Obj): {res_full.total_latency:.2f}")
    else:
        print("Full ILP Failed")
        max_acts_full = "Failed"

    # 3. Run Ablated ILP (Without Row Activation Model)
    print("\n--- Running Ablated ILP (Without Cost Model) ---")
    # We use the same optimizer instance or new one, doesn't matter.
    # Key is enable_row_activation=False
    opt_ablated = PIMOptimizer(arch=arch, verbose=False, time_limit=60)
    res_ablated = opt_ablated.optimize([workload], enable_row_activation=False)
    
    if res_ablated.mappings:
        map_ablated = res_ablated.mappings[0]
        stats_ablated = trace_gen.generate_trace(map_ablated, workload)
        max_acts_ablated = max(stats_ablated.values())
        print(f"Ablated ILP Max Acts: {max_acts_ablated}")
        # Note: The objective value returned here won't include row activation costs, 
        # so comparing 'total_latency' directly is misleading. 
        # We rely on the Trace Generator (max_acts) for the fair comparison.
    else:
        print("Ablated ILP Failed")
        max_acts_ablated = "Failed"

    # 4. Save Results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "ablation_results.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Row_Buffer_Size", "Max_Row_Activations", "Improvement_Factor"])
        writer.writerow(["Ablated_ILP", ROW_BUFFER_SIZE, max_acts_ablated, "1.0x"])
        
        if isinstance(max_acts_full, (int, float)) and isinstance(max_acts_ablated, (int, float)):
            improvement = max_acts_ablated / max_acts_full
            writer.writerow(["Full_ILP", ROW_BUFFER_SIZE, max_acts_full, f"{improvement:.2f}x"])
        else:
            writer.writerow(["Full_ILP", ROW_BUFFER_SIZE, max_acts_full, "N/A"])
            
    print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":
    run_ablation_study()
