
import math
import logging
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer.mapping import Mapping
from validation.dram.trace_generator import TraceGenerator, DRAMConfig, DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def count_input_row_acts(trace):
    """Count Row Activations for Bank 0 (Input)."""
    row_size = 1024 # Default
    bank_size = row_size * 1024 # Default num_rows
    
    current_row = None
    activations = 0
    
    for line in trace:
        parts = line.strip().split()
        if len(parts) < 2: continue
        if parts[0] != 'LD': continue 
        
        addr = int(parts[1], 16)
        bank = addr // bank_size
        row = (addr % bank_size) // row_size
        
        if bank == 0: # Input
            if current_row != row:
                activations += 1
                current_row = row
    return activations

def run_test(k_l2_val):
    # Define a standard workload
    workload = ConvWorkload(
        name="test",
        R=3, S=3, P=32, Q=32, C=64, K=64, N=1,
        stride=(1,1), dilation=(1,1)
    )
    
    # Fixed L2 Tile Geometry for Input
    p_l2 = 4
    q_l2 = 32
    c_l2 = 16
    
    # Loop Bounds
    loop_bounds = {
        2: {'temporal': {}, 'spatial': {}}, 
        3: {'temporal': {}, 'spatial': {}}
    }
    
    # L2 Bounds (Tile Size)
    loop_bounds[2]['temporal'][DIM_P] = p_l2
    loop_bounds[2]['temporal'][DIM_Q] = q_l2
    loop_bounds[2]['temporal'][DIM_C] = c_l2
    loop_bounds[2]['temporal'][DIM_K] = k_l2_val # Varying K
    
    # L3 Bounds (1 Tile)
    loop_bounds[3]['temporal'][DIM_P] = 1
    loop_bounds[3]['temporal'][DIM_Q] = 1
    loop_bounds[3]['temporal'][DIM_C] = 1
    loop_bounds[3]['temporal'][DIM_K] = 1
    
    # Loop Order (Standard)
    # Mapping expects permutation dict: [level][perm_level] = dim
    permutation = {
        2: {0: DIM_R, 1: DIM_S, 2: DIM_Q, 3: DIM_P, 4: DIM_C, 5: DIM_K}, # K is outermost in L2
        3: {0: DIM_R, 1: DIM_S, 2: DIM_Q, 3: DIM_P, 4: DIM_C, 5: DIM_K}
    }
    
    mapping = Mapping(loop_bounds=loop_bounds, permutation=permutation)
    
    dram_config = DRAMConfig(row_buffer_bytes=1024)
    trace_gen = TraceGenerator(dram_config)
    
    trace = trace_gen.generate_trace(mapping, workload)
    acts = count_input_row_acts(trace)
    
    return acts

def main():
    print("Verifying K_l2 Independence for Input Tensor Row Acts...")
    
    k1 = 1
    acts_1 = run_test(k1)
    print(f"K_l2 = {k1}: Input Row Acts = {acts_1}")
    
    k2 = 16
    acts_2 = run_test(k2)
    print(f"K_l2 = {k2}: Input Row Acts = {acts_2}")
    
    if acts_1 == acts_2:
        print("\nSUCCESS: Input Row Acts are independent of K_l2.")
    else:
        print("\nFAILURE: Input Row Acts changed!")

if __name__ == "__main__":
    main()
