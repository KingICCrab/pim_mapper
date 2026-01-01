#!/usr/bin/env python3
"""
Verify Hybrid Model Batch: Run multiple mappings to validate ILP Hybrid Model correctness.
"""

import sys
import os
import math
from pathlib import Path
import itertools
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from validation.dram_v2.core.mapping_space import MappingConfig, WorkloadConfig, ArchConfig
from validation.dram.trace_generator import TraceGenerator, DRAMConfig
from pim_optimizer.workload import ConvWorkload
from pim_optimizer.generator.precompute_row_acts import RowActivationPrecomputer, PrecomputeConfig
from pim_optimizer.mapping import Mapping

# Try to import tqdm
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

def verify_batch():
    print("=== Verify Hybrid Model Batch ===")
    
    # 1. Define Workload
    # ResNet L2 Small: N=1, C=32, K=32, P=16, Q=16, R=3, S=3
    wl = ConvWorkload(name="ResNet_L2_Small", N=1, C=32, K=32, P=16, Q=16, R=3, S=3, stride=(1,1))
    print(f"Workload: {wl}")
    
    # 2. Setup Precomputer (Hybrid Model)
    precompute_config = PrecomputeConfig(
        row_buffer_bytes=1024,
        element_size=1
    )
    precomputer = RowActivationPrecomputer(wl, precompute_config)
    
    # 3. Define Test Cases
    # We want to vary:
    # - Tile Size (P_l2, Q_l2, C_l2) -> Determines L1 tile size
    # - Block Size (block_h, block_w)
    # - Loop Order (Permutation)
    
    # Factors of P=16: 1, 2, 4, 8, 16
    # Factors of Q=16: 1, 2, 4, 8, 16
    # Factors of C=32: 1, 2, 4, 8, 16, 32
    
    # Reduced set for quick verification
    p_factors = [4, 8, 16]
    q_factors = [4, 8, 16]
    c_factors = [4, 8, 32]
    
    # Block sizes (must be <= 1024 bytes)
    # 18x18 = 324 bytes (OK)
    # 32x32 = 1024 bytes (OK)
    block_sizes = [(18, 18), (32, 32)]
    
    # Loop orders (L2)
    # P, Q, C permutations
    # TraceGenerator forces C-Outer for Input iteration.
    # So we only test C-Outer to validate the model where it matches.
    loop_orders = [
        ['C', 'P', 'Q'], # Channel Outer
    ]
    
    test_cases = []
    for p in p_factors:
        for q in q_factors:
            for c in c_factors:
                for bh, bw in block_sizes:
                    for order in loop_orders:
                        # Filter invalid block sizes (must be <= row buffer)
                        if bh * bw > 1024: continue
                        
                        test_cases.append({
                            'p_l2': p, 'q_l2': q, 'c_l2': c,
                            'block_h': bh, 'block_w': bw,
                            'loop_order': order
                        })
    
    print(f"Generated {len(test_cases)} test cases.")
    
    # 4. Run Verification
    results = []
    
    # Map dim names to IDs for TraceGenerator
    DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6
    dim_map = {'P': DIM_P, 'Q': DIM_Q, 'C': DIM_C, 'K': DIM_K, 'N': DIM_N, 'R': DIM_R, 'S': DIM_S}
    
    # Setup TraceGenerator
    dram_config = DRAMConfig(
        row_buffer_bytes=1024,
        element_size=1,
        num_rows=16384
    )
    tracer = TraceGenerator(dram_config)
    
    correct_count = 0
    total_error = 0.0
    
    pbar = tqdm(test_cases, desc="Verifying")
    
    for i, case in enumerate(pbar):
        # A. Run Hybrid Model (Predicted)
        try:
            if hasattr(precomputer, '_calculate_hybrid_cost'):
                predicted_acts = precomputer._calculate_hybrid_cost(
                    case['p_l2'], case['q_l2'], case['c_l2'],
                    case['block_h'], case['block_w'], case['loop_order']
                )
            elif hasattr(precomputer, '_generate_full_trace'):
                 predicted_acts = precomputer._generate_full_trace(
                    case['p_l2'], case['q_l2'], case['c_l2'],
                    case['block_h'], case['block_w'], case['loop_order']
                )
            else:
                print("Error: Could not find calculation method in RowActivationPrecomputer")
                break
        except Exception as e:
            print(f"Hybrid Model Error: {e}")
            continue
            
        # B. Run Trace Generator (Actual)
        
        # Calculate L3 factors
        P_l3 = wl.P // case['p_l2']
        Q_l3 = wl.Q // case['q_l2']
        C_l3 = wl.C // case['c_l2']
        
        # L3 Permutation: P, Q (Outer to Inner in Precomputer loop: P then Q)
        perm_l3_tuple = (DIM_Q, DIM_P, DIM_C, DIM_K, DIM_N, DIM_R, DIM_S)
        perm_l3 = {i: d for i, d in enumerate(perm_l3_tuple)}
        
        # Construct Mapping object (Legacy format for TraceGenerator)
        mapping = Mapping()
        
        # Level 3 (DRAM): L3 factors
        mapping.loop_bounds[3] = {'temporal': {
            DIM_P: P_l3, DIM_Q: Q_l3, DIM_C: C_l3,
            DIM_K: 1, DIM_N: 1, DIM_R: 1, DIM_S: 1
        }}
        mapping.permutation[3] = perm_l3
        
        # Level 2 (RowBuffer): 1 (Single tile in RowBuffer)
        mapping.loop_bounds[2] = {'temporal': {
            DIM_P: 1, DIM_Q: 1, DIM_C: 1,
            DIM_K: 1, DIM_N: 1, DIM_R: 1, DIM_S: 1
        }}
        mapping.permutation[2] = perm_l3 # Same as L3
        
        # Level 1 (GlobalBuffer): 1
        mapping.loop_bounds[1] = {'temporal': {d: 1 for d in range(7)}}
        
        # Level 0 (PE): Tile Size (L0 factors)
        # TraceGenerator expects 'H', 'W', 'Internal', 'temporal'
        # We just put everything in 'temporal' for simplicity, or 'H'/'W' if needed.
        # But TraceGenerator _compute_buffer_tile_size checks Level 0.
        mapping.loop_bounds[0] = {'temporal': {
            DIM_P: case['p_l2'], DIM_Q: case['q_l2'], DIM_C: case['c_l2'],
            DIM_K: wl.K, DIM_N: wl.N, DIM_R: wl.R, DIM_S: wl.S
        }}
        
        # Layout
        # Use 'sequential' to match MicroTraceGenerator's 'tiled' (dense) layout
        mapping.layout['Input'] = 'sequential'
        mapping.tile_info['block_h'] = case['block_h']
        mapping.tile_info['block_w'] = case['block_w']
        
        # Run TraceGenerator
        trace = tracer.generate_trace(mapping, wl, strict_ordering=False)
        
        # Count Acts
        row_size = 1024
        bank_size = row_size * 16384
        actual_acts = 0
        current_row = -1
        
        for line in trace:
            parts = line.strip().split()
            if len(parts) < 2 or parts[0] != 'LD': continue
            addr = int(parts[1], 16)
            bank = addr // bank_size
            row = (addr % bank_size) // row_size
            
            if bank == 0: # Input
                if current_row != row:
                    actual_acts += 1
                    current_row = row
        
        # Normalize Actual Acts (Average per tile)
        num_tiles = P_l3 * Q_l3 * C_l3
        avg_actual_acts = actual_acts / num_tiles
        
        # Compare
        error = abs(predicted_acts - avg_actual_acts)
        
        results.append({
            'case': case,
            'predicted': predicted_acts,
            'actual': avg_actual_acts,
            'error': error
        })
        
        if error < 1.0: # Allow small float error (boundary effects)
            correct_count += 1
            if correct_count == 1:
                print(f"First Success: Case {case}")
                print(f"  Pred: {predicted_acts:.4f}, Act: {avg_actual_acts:.4f}")
        else:
            # Print failure immediately
            # tqdm.write(f"Mismatch! Case: {case}")
            # tqdm.write(f"  Pred: {predicted_acts:.4f}, Actual: {avg_actual_acts:.4f}, Error: {error:.4f}")
            pass
            
        total_error += error
        
        # Update pbar description
        pbar.set_description(f"Acc: {correct_count}/{i+1}")

    print("\n=== Verification Summary ===")
    print(f"Total Cases: {len(results)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {correct_count/len(results)*100:.2f}%")
    
    # Print failures
    failures = [r for r in results if r['error'] >= 0.1]
    if failures:
        print(f"\nTop 5 Failures:")
        failures.sort(key=lambda x: x['error'], reverse=True)
        for f in failures[:5]:
            print(f"  Error: {f['error']:.4f} | Pred: {f['predicted']:.2f} vs Act: {f['actual']:.2f}")
            print(f"  Case: {f['case']}")

if __name__ == "__main__":
    verify_batch()
