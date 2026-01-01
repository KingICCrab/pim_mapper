#!/usr/bin/env python3
"""
Debug Small Detail V2: Verify Row Activation Formula vs Trace Generator
Using the dram_v2 infrastructure.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src')) # For pim_optimizer

# Imports
from validation.dram_v2.core.mapping_space import MappingConfig, WorkloadConfig, ArchConfig
from validation.dram_v2.formula.row_activation_formula import compute_input_row_switches_formula, FormulaConfig
from validation.dram_v2.core.mapping.trace_adapter import TraceGeneratorMapping
from validation.dram.trace_generator import TraceGenerator, DRAMConfig
from pim_optimizer.workload import ConvWorkload

def debug_small_detail_v2():
    print("=== Debug Small Detail V2 ===")
    
    # 1. Define Workload
    # N=1, C=32, K=32, P=16, Q=16, R=3, S=3, stride=(1,1)
    # Use ConvWorkload for TraceGenerator
    wl_legacy = ConvWorkload(name="ResNet_L2_Small", N=1, C=32, K=32, P=16, Q=16, R=3, S=3, stride=(1,1))
    print(f"Workload (Legacy): {wl_legacy}")
    
    # Use WorkloadConfig for Formula (or just extract values)
    H_in = (wl_legacy.P - 1) * wl_legacy.stride[0] + (wl_legacy.R - 1) * wl_legacy.dilation[0] + 1
    W_in = (wl_legacy.Q - 1) * wl_legacy.stride[1] + (wl_legacy.S - 1) * wl_legacy.dilation[1] + 1
    
    wl_config = WorkloadConfig(
        N=wl_legacy.N, C=wl_legacy.C, K=wl_legacy.K, 
        P=wl_legacy.P, Q=wl_legacy.Q, R=wl_legacy.R, S=wl_legacy.S, 
        H=H_in, W=W_in,
        stride_h=wl_legacy.stride[0], stride_w=wl_legacy.stride[1], 
        dilation_h=wl_legacy.dilation[0], dilation_w=wl_legacy.dilation[1]
    )
    
    # 2. Define Architecture
    arch = ArchConfig(
        row_buffer_bytes=1024,
        num_banks=16,
        num_rows=16384,
        element_size=1
    )
    
    # 3. Define Mapping Config
    # Dimensions: R=0, S=1, P=2, Q=3, C=4, K=5, N=6
    DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6
    
    # Permutation: Inner -> Outer
    # Q, P, C are the main ones.
    perm = (DIM_Q, DIM_P, DIM_C, DIM_R, DIM_S, DIM_K, DIM_N)
    
    config = MappingConfig(
        # L3 factors (All 1)
        R_l3=1, S_l3=1, P_l3=1, Q_l3=1, C_l3=1, K_l3=1, N_l3=1,
        # L2 factors (All 1 - Single Tile in RowBuffer)
        R_l2=1, S_l2=1, P_l2=1, Q_l2=1, C_l2=1, K_l2=1, N_l2=1,
        # L0 factors (Full - The Tile Content)
        R_l0=wl_config.R, S_l0=wl_config.S, P_l0=wl_config.P, Q_l0=wl_config.Q, 
        C_l0=wl_config.C, K_l0=wl_config.K, N_l0=wl_config.N,
        # Permutation
        permutation_l3=perm,
        # Layout
        input_layout='sequential',
        weight_layout='sequential',
        output_layout='sequential',
        # Block Size
        block_h=18,
        block_w=18
    )
    
    print(f"Mapping Config: Block={config.block_h}x{config.block_w}, Layout={config.input_layout}")
    
    # 4. Run Formula
    print("\n--- Formula Calculation ---")
    formula_config = FormulaConfig(
        R=wl_config.R, S=wl_config.S, P=wl_config.P, Q=wl_config.Q, C=wl_config.C, K=wl_config.K, N=wl_config.N,
        H=wl_config.H, W=wl_config.W,
        stride=(wl_config.stride_h, wl_config.stride_w),
        dilation=(wl_config.dilation_h, wl_config.dilation_w),
        row_buffer_bytes=arch.row_buffer_bytes,
        element_size=arch.element_size,
        # Mapping params
        P_l3=config.P_l3, Q_l3=config.Q_l3, C_l3=config.C_l3, K_l3=config.K_l3,
        P_l2=config.P_l2, Q_l2=config.Q_l2, C_l2=config.C_l2, K_l2=config.K_l2,
        block_h=config.block_h, block_w=config.block_w,
        input_layout=config.input_layout
    )
    
    predicted_acts = compute_input_row_switches_formula(formula_config)
    print(f"Formula Predicted Acts: {predicted_acts}")
    
    # 5. Run Trace Generator
    print("\n--- Trace Generator ---")
    # Convert MappingConfig to TraceGeneratorMapping
    legacy_mapping = TraceGeneratorMapping(config)
    print(f"Loop Bounds: {legacy_mapping.loop_bounds}")
    
    # Create DRAM Config for TraceGenerator
    dram_config = DRAMConfig(
        row_buffer_bytes=arch.row_buffer_bytes,
        element_size=arch.element_size,
        num_rows=arch.num_rows
    )
    
    tracer = TraceGenerator(dram_config)
    # Use legacy workload
    trace = tracer.generate_trace(legacy_mapping, wl_legacy, strict_ordering=True)
    
    tracer.print_input_debug_info(num_entries=20)
    
    # Count acts
    row_size = arch.row_buffer_bytes
    bank_size = row_size * arch.num_rows
    
    actual_acts = 0
    current_row = -1
    
    print("DEBUG: Trace Analysis")
    for i, line in enumerate(trace):
        parts = line.strip().split()
        if len(parts) < 2: continue
        if parts[0] != 'LD': continue
        
        addr = int(parts[1], 16)
        bank = addr // bank_size
        row = (addr % bank_size) // row_size
        
        if bank == 0:
            if current_row != row:
                print(f"Line {i}: Row Switch {current_row} -> {row} (Addr {addr})")
                actual_acts += 1
                current_row = row
                
    print(f"Trace Actual Acts: {actual_acts}")
    
    error = abs(predicted_acts - actual_acts)
    print(f"Error: {error}")

if __name__ == "__main__":
    debug_small_detail_v2()
