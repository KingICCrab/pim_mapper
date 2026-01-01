
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.path.join(os.getcwd(), 'validation'))

from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer.mapping import Mapping
from validation.dram.trace_generator import TraceGenerator, DRAMConfig

def debug_weight_trace():
    # 1. Define Medium Workload
    # R=3, S=3, P=28, Q=28, C=32, K=32, N=1
    workload = ConvWorkload(
        name="medium",
        P=28, Q=28,
        K=32, C=32, R=3, S=3, N=1,
        stride=(1, 1),
        dilation=(1, 1)
    )
    
    # Define Dimension Constants
    DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6

    # 2. Define Mapping (from analysis.txt)
    # Level 0: H: {P:2, C:4, K:2}, W: {Q:14}, T: {R:3, S:3}
    # Level 1: (all 1s)
    # Level 2: T: {Q:2, K:8}
    # Level 3: T: {P:14, C:8, K:2}
    
    mapping_dict = {
        0: {'spatial': {DIM_P: 2, DIM_C: 4, DIM_K: 2, DIM_Q: 14}, 'temporal': {DIM_R: 3, DIM_S: 3}},
        1: {'spatial': {}, 'temporal': {}},
        2: {'spatial': {}, 'temporal': {DIM_Q: 2, DIM_K: 8}},
        3: {'spatial': {}, 'temporal': {DIM_P: 14, DIM_C: 8, DIM_K: 2}}
    }
    
    # Note: Level 0 structure in TraceGenerator expects 'H', 'W', 'Internal', 'temporal' keys?
    # Let's check TraceGenerator._compute_buffer_tile_size again.
    # Yes: if level == 0: for key in ['H', 'W', 'Internal', 'temporal']:
    # So I need to structure Level 0 correctly.
    
    mapping_dict = {
        0: {
            'H': {DIM_P: 2, DIM_C: 4, DIM_K: 2}, 
            'W': {DIM_Q: 14}, 
            'Internal': {},
            'temporal': {DIM_R: 3, DIM_S: 3}
        },
        1: {'spatial': {}, 'temporal': {}},
        2: {'spatial': {}, 'temporal': {DIM_Q: 2, DIM_K: 8}},
        3: {'spatial': {}, 'temporal': {DIM_P: 14, DIM_C: 8, DIM_K: 2}}
    }
    
    # Permutations (Inner to Outer)
    # Level 0: R -> S (implied?)
    # Level 2: K -> Q
    # Level 3: K -> C -> P
    
    # Note: Mapping object expects permutations in specific format if needed
    # But TraceGenerator infers loop order from mapping dict if not explicit?
    # Let's check TraceGenerator._build_dram_loop_structure
    # It sorts by some criteria if not provided.
    # We should explicitly set the loop orders if possible, or rely on the dict order?
    # The dicts in Python 3.7+ preserve insertion order.
    
    mapping = Mapping(mapping_dict)
    # Manually set loop orders to match analysis.txt
    # Level 2: K (inner) -> Q (outer)
    # Level 3: K (inner) -> C -> P (outer)
    # Wait, analysis.txt said:
    # Level 2: K -> Q (Inner to Outer) => K is Inner.
    # Level 3: K -> C -> P (Inner to Outer) => K is Inner.
    
    # We need to ensure TraceGenerator respects this.
    # TraceGenerator uses mapping.loop_orders if available?
    # No, it builds it. Let's trust the generator's default or insertion order.
    # In the dict above:
    # L2: Q, K. Insertion order Q, then K.
    # If generator iterates dict, K is last (inner?).
    # Let's verify the output.
    
    # 3. Generate Trace
    dram_config = DRAMConfig(row_buffer_bytes=1024)
    generator = TraceGenerator(dram_config)
    
    print("Generating trace...")
    # We only care about Weight (Bank 1)
    # We can hook into the generator or just parse the output.
    
    trace = generator.generate_trace(mapping, workload)
    
    print(f"Total trace lines: {len(trace)}")
    
    weight_accesses = []
    for line in trace:
        if "LD" in line:
            # Parse address
            parts = line.split()
            addr_str = parts[1]
            addr = int(addr_str, 16)
            
            # Bank 1 check (0x01000000 - 0x01FFFFFF)
            if 0x01000000 <= addr < 0x02000000:
                weight_accesses.append(addr)
                
    print(f"Total Weight Accesses: {len(weight_accesses)}")
    
    # Analyze accesses around tile boundaries
    print("\nAccesses around Tile 0 end (576):")
    prev_addr = None
    prev_row = None
    
    # Print range 570 to 600
    for i in range(570, 600):
        if i >= len(weight_accesses): break
        addr = weight_accesses[i]
        row = (addr >> 10) & 0x3FFF
        col = addr & 0x3FF
        
        diff = ""
        if i > 0:
            prev = weight_accesses[i-1]
            d = addr - prev
            diff = f"(+{d})"
            
        row_switch = ""
        if i > 0:
            prev_r = (weight_accesses[i-1] >> 10) & 0x3FFF
            if row != prev_r:
                row_switch = "<-- ROW SWITCH"
            
        print(f"{i:3d}: 0x{addr:08X} {diff:>6} | Row {row:3d} Col {col:4d} {row_switch}")

    print("\nAccesses around Row Boundary inside Tile 1 (approx 1600):")
    # Tile 1 starts at 1152 (trace index). Address 576.
    # Row boundary at Address 1024.
    # Offset = 1024 - 576 = 448.
    # Trace index = 1152 + 448 = 1600.
    for i in range(1595, 1605):
        if i >= len(weight_accesses): break
        addr = weight_accesses[i]
        row = (addr >> 10) & 0x3FFF
        col = addr & 0x3FF
        
        diff = ""
        if i > 0:
            prev = weight_accesses[i-1]
            d = addr - prev
            diff = f"(+{d})"
            
        row_switch = ""
        if i > 0:
            prev_r = (weight_accesses[i-1] >> 10) & 0x3FFF
            if row != prev_r:
                row_switch = "<-- ROW SWITCH"
            
        print(f"{i:3d}: 0x{addr:08X} {diff:>6} | Row {row:3d} Col {col:4d} {row_switch}")

    print("\nAccesses around Tile 1 end / Repeat (approx 1728):")
    # Tile 1 ends at 1152 + 576 = 1728.
    for i in range(1720, 1735):
        if i >= len(weight_accesses): break
        addr = weight_accesses[i]
        row = (addr >> 10) & 0x3FFF
        col = addr & 0x3FF
        
        diff = ""
        if i > 0:
            prev = weight_accesses[i-1]
            d = addr - prev
            diff = f"(+{d})"
            
        row_switch = ""
        if i > 0:
            prev_r = (weight_accesses[i-1] >> 10) & 0x3FFF
            if row != prev_r:
                row_switch = "<-- ROW SWITCH"
            
        print(f"{i:3d}: 0x{addr:08X} {diff:>6} | Row {row:3d} Col {col:4d} {row_switch}")


if __name__ == "__main__":
    debug_weight_trace()
