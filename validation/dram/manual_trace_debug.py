
import sys
import os
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from pim_optimizer.workload.conv import ConvWorkload
from validation.dram.trace_generator import TraceGenerator, DRAMConfig
from pim_optimizer.mapping import Mapping

def main():
    # 1. Define Workload (Medium)
    workload = ConvWorkload(name="medium", R=3, S=3, P=28, Q=28, C=32, K=32, N=1)
    
    # Dim IDs
    DIM_R = 0
    DIM_S = 1
    DIM_P = 2
    DIM_Q = 3
    DIM_C = 4
    DIM_K = 5
    DIM_N = 6

    # 2. Define Mapping (from debug_output/medium/analysis.txt)
    # Loop Bounds (Use Integer Keys!)
    loop_bounds = {
        0: { # Level 0 (PE)
            'H': {DIM_P: 4, DIM_Q: 4},
            'W': {DIM_K: 16},
            'temporal': {DIM_R: 3, DIM_S: 3}
        },
        1: {}, # Level 1 (GlobalBuffer)
        2: { # Level 2 (RowBuffer)
            'temporal': {DIM_C: 2, DIM_K: 2}
        },
        3: { # Level 3 (LocalDRAM)
            'temporal': {DIM_P: 7, DIM_Q: 7, DIM_C: 16}
        }
    }
    
    permutation = {
        3: {6: DIM_C, 3: DIM_Q, 1: DIM_P},
        2: {4: DIM_C, 0: DIM_K}, # Level 2: C -> K (Inner to Outer). So K is Outer?
        # debug_output: "Level 2: C -> K".
        # If "Inner to Outer", then C is Inner, K is Outer.
        # So K has higher position.
        # Let's assume K=4, C=0?
        # Wait, debug_output says "Level 2: C -> K".
        # Usually means C is Inner.
        # Let's try K at Pos 4, C at Pos 0.
    }
    
    # Layouts
    layout = {
        0: "row_aligned", # Input
        1: "sequential",  # Weight
        2: "sequential"   # Output
    }
    
    mapping = Mapping(loop_bounds=loop_bounds, permutation=permutation, layout=layout)
    
    # 3. Run Trace Generator
    config = DRAMConfig() # Default config
    gen = TraceGenerator(config)
    
    print("Generating trace for medium workload...")
    trace = gen.generate_trace(mapping, workload)
    
    print(f"Generated {len(trace)} trace lines.")
    
    # 4. Analyze Weight Accesses (Bank 1)
    print("\nAnalyzing Weight Accesses (Bank 1)...")
    row_size = config.row_buffer_bytes
    bank_size = row_size * config.num_rows
    
    weight_accesses = []
    for line in trace:
        parts = line.strip().split()
        if len(parts) < 2: continue
        addr = int(parts[1], 16)
        bank = addr // bank_size
        if bank == 1:
            row = (addr % bank_size) // row_size
            col = addr % row_size
            weight_accesses.append((addr, row, col))
            
    print(f"Total Weight Accesses: {len(weight_accesses)}")
    
    # Count activations
    current_row = None
    acts = 0
    row_counts = {}
    
    for _, row, _ in weight_accesses:
        if row != current_row:
            acts += 1
            current_row = row
            row_counts[row] = row_counts.get(row, 0) + 1
            
    print(f"Total Activations: {acts}")
    print(f"Row Activation Counts: {dict(sorted(row_counts.items())[:20])}")

if __name__ == "__main__":
    main()
