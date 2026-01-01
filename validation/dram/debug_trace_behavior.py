
import logging
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer.mapping import Mapping
from validation.dram.trace_generator import TraceGenerator, DIM_K, DIM_C, DIM_R, DIM_S, DIM_P, DIM_Q, DIM_N

# Configure logging
logging.basicConfig(level=logging.INFO)

def debug_trace_continuity():
    print("DEBUG: Verifying Trace Generator Address Continuity for Small Tiles")
    
    # 1. Define a small workload
    # K=2, C=1, R=1, S=1. Total Elements = 2.
    # We will split K into 2 tiles of size 1.
    wl = ConvWorkload(C=1, K=2, R=1, S=1, P=1, Q=1, stride=(1,1), dilation=(1,1), N=1)
    
    # 2. Define Mapping with Small Tiles
    # Tile Size = 1 element (4 bytes? or 1 byte depending on element size)
    # We want to see if Tile 0 and Tile 1 are contiguous.
    mapping = Mapping()
    
    # Level 2 (DRAM): Iterate K (2 steps)
    mapping.loop_bounds = {
        2: {'temporal': {DIM_K: 2, DIM_C: 1, DIM_R: 1, DIM_S: 1}},
        1: {'temporal': {DIM_K: 1, DIM_C: 1, DIM_R: 1, DIM_S: 1, DIM_P: 1, DIM_Q: 1, DIM_N: 1}}
    }
    
    # Permutation: K is the only loop
    mapping.permutation = {
        2: {0: DIM_K, 1: DIM_C, 2: DIM_R, 3: DIM_S},
        1: {0: DIM_K, 1: DIM_C, 2: DIM_R, 3: DIM_S, 4: DIM_P, 5: DIM_Q, 6: DIM_N}
    }
    
    mapping.layout = {'input': 'sequential', 'weight': 'sequential', 'output': 'sequential'}
    
    # 3. Generate Trace
    # We need to mock the config or pass minimal config
    # TraceGenerator expects a config object usually, but let's see if we can instantiate it simply.
    # It seems it takes (config) in __init__.
    
    class MockConfig:
        def __init__(self):
            self.element_size = 1 # 1 Byte for easy math
            self.align_to_row = False
            self.row_size_elements = 1024
            self.burst_length = 1
            self.banks = 16
            self.rows = 65536
            self.columns = 1024
            self.row_buffer_bytes = 1024
            self.num_rows = 65536
            
    gen = TraceGenerator(MockConfig())
    
    print("Generating trace...")
    trace = gen.generate_trace(mapping, wl)
    
    print(f"Trace Length: {len(trace)}")
    print("First 10 lines:")
    for line in trace[:10]:
        print(line)
        
    # Analyze Addresses per Tensor (Bank)
    # Based on TraceGenerator defaults:
    # Input Base: 0x00000000
    # Weight Base: 0x04000000
    # Output Base: 0x08000000
    
    input_addrs = []
    weight_addrs = []
    output_addrs = []
    
    for line in trace:
        parts = line.split()
        if len(parts) >= 2:
            addr = int(parts[1], 16)
            if addr >= 0x08000000:
                output_addrs.append(addr)
            elif addr >= 0x04000000:
                weight_addrs.append(addr)
            else:
                input_addrs.append(addr)
            
    print(f"\nWeight Addresses (Bank 4): {weight_addrs}")
    
    # Check continuity for Weight
    is_contiguous = True
    if not weight_addrs:
        print("No Weight accesses found.")
    else:
        for i in range(len(weight_addrs) - 1):
            if weight_addrs[i+1] != weight_addrs[i] + 1:
                is_contiguous = False
                print(f"Weight Discontinuity: {weight_addrs[i]} -> {weight_addrs[i+1]}")
                
        if is_contiguous:
            print("\nRESULT: Weight Addresses are PERFECTLY CONTIGUOUS.")
            print("Even though tiles are small (1 element), they are accessed sequentially.")
            print("DRAM Controller will keep the Row Buffer OPEN.")
            print("Original ILP (Cost=2 per tile) assumed Row Close after every tile -> WRONG.")
            print("New ILP (Streaming) assumes Row Open across tiles -> CORRECT.")
        else:
            print("\nRESULT: Weight Addresses are NOT contiguous.")

if __name__ == "__main__":
    debug_trace_continuity()
