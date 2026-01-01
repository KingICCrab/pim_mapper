
import sys
import os
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.getcwd())

from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer.mapping import Mapping
from validation.dram.trace_generator import TraceGenerator, DRAMConfig, DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N

def count_input_row_activations(trace, dram_config):
    """Count row activations for Input tensor (Bank 0)."""
    row_size = dram_config.row_buffer_bytes
    bank_size = row_size * dram_config.num_rows
    
    current_row = None
    activations = 0
    
    for line in trace:
        parts = line.strip().split()
        if len(parts) < 2: continue
        
        # Only count Loads (LD)
        if parts[0] != 'LD': continue
        
        addr = int(parts[1], 16)
        bank = addr // bank_size
        row = (addr % bank_size) // row_size
        
        # Input is usually mapped to Bank 0 (or low addresses)
        # In TraceGenerator, Input is Bank 0, Weight Bank 1, Output Bank 2
        # We can verify this by checking the address ranges or just assuming Bank 0 for Input
        # Based on debug_row_activation.py, Bank 0 is Input.
        if bank == 0:
            if current_row != row:
                activations += 1
                current_row = row
                
    return activations

def run_experiment(name, workload_params, mapping_params):
    print(f"\n--- Experiment: {name} ---")
    
    # 1. Create Workload
    workload = ConvWorkload(
        name="Test",
        **workload_params
    )
    
    # 2. Create Mapping
    # Default structure
    loop_bounds = {
        2: {'temporal': {}, 'spatial': {}}, # Row Buffer Level
        3: {'temporal': {}, 'spatial': {}}  # DRAM Level
    }
    
    # Fill bounds from params
    # params keys: 'P_l2', 'P_l3', etc.
    for key, val in mapping_params.items():
        if '_l2' in key:
            dim_char = key.split('_')[0]
            dim_idx = globals()[f'DIM_{dim_char}']
            loop_bounds[2]['temporal'][dim_idx] = val
        elif '_l3' in key:
            dim_char = key.split('_')[0]
            dim_idx = globals()[f'DIM_{dim_char}']
            loop_bounds[3]['temporal'][dim_idx] = val
            
    # Default Permutation (Outer to Inner)
    # Level 3: K, C, P, Q
    # Level 2: K, C, P, Q
    permutation = {
        3: {0: DIM_K, 1: DIM_C, 2: DIM_P, 3: DIM_Q},
        2: {0: DIM_K, 1: DIM_C, 2: DIM_P, 3: DIM_Q}
    }
    
    # Override permutation if provided
    if 'permutation' in mapping_params:
        permutation = mapping_params['permutation']

    # Layout
    layout = {0: 'row_aligned', 1: 'row_aligned', 2: 'row_aligned'}
    if 'layout' in mapping_params:
        layout = mapping_params['layout']
        
    # Tile Info (needed for TraceGenerator Scheme B)
    tile_info = {
        'block_h': mapping_params.get('block_h', 4),
        'block_w': mapping_params.get('block_w', 32)
    }

    mapping = Mapping(
        loop_bounds=loop_bounds,
        permutation=permutation,
        layout=layout,
        tile_info=tile_info
    )
    
    # 3. Run Trace
    dram_config = DRAMConfig()
    gen = TraceGenerator(dram_config)
    trace = gen.generate_trace(mapping, workload)
    
    acts = count_input_row_activations(trace, dram_config)
    print(f"Input Row Acts: {acts}")
    return acts

def verify_assumptions():
    # Base Params
    base_workload = {'N':1, 'C':64, 'K':64, 'P':32, 'Q':32, 'R':3, 'S':3, 'stride':(1,1)}
    # Base Mapping: 1 Tile (Everything in L2)
    # P_l2=32, Q_l2=32 -> 1 Tile
    base_mapping = {
        'P_l2': 32, 'Q_l2': 32, 'C_l2': 64, 'K_l2': 64,
        'P_l3': 1,  'Q_l3': 1,  'C_l3': 1,  'K_l3': 1,
        'block_h': 4, 'block_w': 32
    }
    
    print("=== 1. Verify K-Independence ===")
    # Case A: K=64
    acts_k64 = run_experiment("K=64", base_workload, base_mapping)
    
    # Case B: K=128 (K_l3=2)
    # We increase K in workload, and increase K_l3 in mapping
    wl_k128 = base_workload.copy(); wl_k128['K'] = 128
    map_k128 = base_mapping.copy(); map_k128['K_l3'] = 2
    acts_k128 = run_experiment("K=128 (K_l3=2)", wl_k128, map_k128)
    
    ratio_k = acts_k128 / acts_k64
    print(f"Ratio (K=128/K=64): {ratio_k:.2f} (Expected: ~2.0)")
    
    print("\n=== 2. Verify L3-Linearity (Tile Count) ===")
    # Case A: 1 Tile (Already run as K=64)
    
    # Case B: 4 Tiles (2x2)
    # Workload: P=64, Q=64
    # Mapping: P_l3=2, Q_l3=2 (P_l2, Q_l2 fixed at 32)
    wl_4tiles = base_workload.copy(); wl_4tiles['P'] = 64; wl_4tiles['Q'] = 64
    map_4tiles = base_mapping.copy(); map_4tiles['P_l3'] = 2; map_4tiles['Q_l3'] = 2
    acts_4tiles = run_experiment("4 Tiles (2x2)", wl_4tiles, map_4tiles)
    
    ratio_tiles = acts_4tiles / acts_k64
    print(f"Ratio (4 Tiles / 1 Tile): {ratio_tiles:.2f} (Expected: ~4.0)")
    
    print("\n=== 3. Verify Geometry Sensitivity ===")
    # Case A: P_l2=32 (Aligned with block_w=32? No, P maps to H usually, Q to W)
    # Let's look at Q_l2 vs block_w
    # block_w = 32.
    # Case A: Q_l2 = 32 (Aligned). Acts should be low.
    # Case B: Q_l2 = 33 (Misaligned). Acts should jump.
    
    # We need to adjust workload Q to allow Q_l2=33
    wl_mis = base_workload.copy(); wl_mis['Q'] = 33
    map_mis = base_mapping.copy(); map_mis['Q_l2'] = 33; map_mis['Q_l3'] = 1
    acts_mis = run_experiment("Q_l2=33 (Misaligned)", wl_mis, map_mis)
    
    # Compare with Q_l2=32 (we need to normalize per pixel or just look at raw jump)
    # Q=32 -> 32 pixels. Q=33 -> 33 pixels.
    # If linear, Acts(33) ~= Acts(32) * 33/32
    # If non-linear crossing, Acts(33) >> Acts(32)
    
    expected_linear = acts_k64 * (33/32)
    print(f"Acts(32): {acts_k64}")
    print(f"Acts(33): {acts_mis}")
    print(f"Expected if linear: {expected_linear:.2f}")
    print(f"Jump Factor: {acts_mis / expected_linear:.2f}")

if __name__ == "__main__":
    verify_assumptions()
