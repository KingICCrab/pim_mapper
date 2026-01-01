
import sys
import os
from pathlib import Path
import gurobipy as gp

# Add src to path
sys.path.append(os.path.abspath('src'))

from pim_optimizer.workload import ConvWorkload
from pim_optimizer.mapping import Mapping
# from pim_optimizer.optimizer import Optimizer
from pim_optimizer.arch import PIMArchitecture as Architecture
from validation.dram.trace_generator import TraceGenerator, DRAMConfig

def main():
    # Define workload
    workload = ConvWorkload(name="medium", R=3, S=3, P=28, Q=28, C=32, K=32, N=1)
    
    # Define Architecture (dummy)
    # arch = Architecture() # Use default
    
    # Define DRAM Config
    dram_config = DRAMConfig()
    
    # Create Manual Mapping
    # Loop 0: Level 3 K (Bound 2)
    # Loop 1: Level 3 C (Bound 2)
    # Loop 2: Level 3 Q (Bound 4)
    # Loop 3: Level 3 P (Bound 7)
    # Loop 4: Level 2 K (Bound 4)
    
    mapping = Mapping()
    mapping.workload_name = workload.name
    mapping.workload_bounds = workload.bounds
    
    # Loop Bounds
    # Level 3 (LocalDRAM)
    mapping.loop_bounds[3] = {
        'spatial': {
            5: 2, # K
            4: 2, # C
            3: 4, # Q
            2: 7, # P
        },
        'temporal': {}
    }
    # Level 2 (RowBuffer)
    mapping.loop_bounds[2] = {
        'spatial': {
            5: 4, # K
        },
        'temporal': {}
    }
    # Level 1 (GlobalBuffer) - implied tile sizes
    # K_tile = 32 / (2*4) = 4
    # C_tile = 32 / 2 = 16
    # Q_tile = 28 / 4 = 7
    # P_tile = 28 / 7 = 4
    mapping.loop_bounds[1] = {
        'spatial': {
            5: 4, # K
            4: 16, # C
            3: 7, # Q
            2: 4, # P
        },
        'temporal': {}
    }
    # Level 0 (PE) - assume 1
    mapping.loop_bounds[0] = {
        'H': {}, 'W': {}, 'Internal': {}, 'temporal': {}
    }
    
    # Permutation
    # Level 3: K (outer) -> C -> Q -> P (inner)
    # get_loop_order returns [inner, ..., outer]
    # So [P, Q, C, K]
    # Permutation dict: index -> dim
    # 0: P, 1: Q, 2: C, 3: K
    mapping.permutation[3] = {0: 2, 1: 3, 2: 4, 3: 5}
    
    # Level 2: K
    mapping.permutation[2] = {0: 5}
    
    # Tile Info (for block size)
    # Assume block size = tile size
    # H_per_tile = (P_tile-1)*1 + (R-1)*1 + 1 = 3 + 2 = 5
    # W_per_tile = (Q_tile-1)*1 + (S-1)*1 + 1 = 6 + 2 = 8
    mapping.tile_info = {
        'block_h': 5,
        'block_w': 8,
    }
    
    # Run Trace Generator
    output_dir = Path('validation/dram/debug_output_manual')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = TraceGenerator(
        dram_config=dram_config
    )
    
    print("Generating trace...")
    trace = generator.generate_trace(mapping, workload)
    print(f"Trace generated with {len(trace)} lines")
    
    # Count activations
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
                accesses.append(row)
        
        # DEBUG: Print first 2000 accesses for Input with RLE
        if tensor_name == "Input":
            print(f"DEBUG: First 2000 Input Accesses (Bank {bank_id}):")
            
            rle = []
            if accesses:
                curr = accesses[0]
                count = 1
                for r in accesses[1:2000]:
                    if r == curr:
                        count += 1
                    else:
                        rle.append((curr, count))
                        curr = r
                        count = 1
                rle.append((curr, count))
            
            for r, c in rle:
                print(f"  Row {r}: {c} accesses")
            
            print(f"  Total RLE entries: {len(rle)}")

        # Count row activations
        current_row = None
        row_acts = 0
        for row in accesses:
            if row != current_row:
                row_acts += 1
                current_row = row
        
        trace_acts[tensor_name] = row_acts
        print(f"{tensor_name}: {row_acts} activations")

if __name__ == "__main__":
    main()
