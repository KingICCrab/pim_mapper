#!/usr/bin/env python3
"""
Detailed debug information for workloads using DRAM v2 infrastructure.
Replaces debug_small_detail.py with updated logic.
"""

import sys
import os
from pathlib import Path
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pim_optimizer.workload.conv import ConvWorkload
from validation.dram.trace_generator import TraceGenerator, DRAMConfig

# Import DRAM v2 modules
from validation.dram_v2.core.mapping import (
    MappingConfig, 
    to_trace_generator_mapping,
    DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N, DIM_R, DIM_S
)
from validation.dram_v2.formula.row_activation_formula import (
    FormulaConfig, compute_total_row_switches_formula
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def log(msg):
    print(msg) # Force print
    # logger.info(msg)

def debug_workload(workload: ConvWorkload, mapping_config: MappingConfig, dram_config: DRAMConfig):
    """
    Run detailed debug analysis for a workload and mapping.
    """
    log("=" * 80)
    log(f"DETAILED DEBUG INFO FOR '{workload.name}' WORKLOAD")
    log("=" * 80)
    
    # 1. Workload Info
    log("\n" + "-" * 40)
    log("1. WORKLOAD CONFIGURATION")
    log("-" * 40)
    log(f"  Name: {workload.name}")
    log(f"  Dimensions: R={workload.R}, S={workload.S}, P={workload.P}, Q={workload.Q}, "
        f"C={workload.C}, K={workload.K}, N={workload.N}")
    log(f"  Input Size: H={workload.input_size['H']}, W={workload.input_size['W']}")
    log(f"  Stride: {workload.stride}, Dilation: {workload.dilation}")
    
    # 2. Mapping Info
    log("\n" + "-" * 40)
    log("2. MAPPING CONFIGURATION")
    log("-" * 40)
    log(f"  Block Size: {mapping_config.block_h}x{mapping_config.block_w}")
    log(f"  L3 Tiling: P={mapping_config.P_l3}, Q={mapping_config.Q_l3}, C={mapping_config.C_l3}, K={mapping_config.K_l3}")
    log(f"  Layouts: Input={mapping_config.input_layout}, Weight={mapping_config.weight_layout}, Output={mapping_config.output_layout}")
    
    # 3. Formula Calculation
    log("\n" + "-" * 40)
    log("3. FORMULA PREDICTION")
    log("-" * 40)
    
    formula_config = FormulaConfig(
        R=workload.R, S=workload.S,
        P=workload.P, Q=workload.Q,
        C=workload.C, K=workload.K, N=workload.N,
        H=workload.input_size['H'], W=workload.input_size['W'],
        stride=workload.stride,
        dilation=workload.dilation,
        row_buffer_bytes=dram_config.row_buffer_bytes,
        element_size=dram_config.element_size,
        P_l3=mapping_config.P_l3, Q_l3=mapping_config.Q_l3,
        C_l3=mapping_config.C_l3, K_l3=mapping_config.K_l3,
        block_h=mapping_config.block_h, block_w=mapping_config.block_w,
        input_layout=mapping_config.input_layout
    )
    
    formula_result = compute_total_row_switches_formula(formula_config)
    predicted_acts = formula_result['input']
    log(f"  Formula Predicted Input Row Acts: {predicted_acts}")
    log(f"  Details: {formula_result}")

    # 4. Trace Simulation
    log("\n" + "-" * 40)
    log("4. TRACE SIMULATION")
    log("-" * 40)
    
    # Convert mapping
    tg_mapping = to_trace_generator_mapping(mapping_config)
    
    # Run TraceGenerator
    tracer = TraceGenerator(dram_config)
    # Use strict_ordering=False to match block-major assumption if needed, 
    # but let's see what the user prefers. Usually False for optimized.
    trace = tracer.generate_trace(tg_mapping, workload, strict_ordering=False)
    
    log(f"  Trace Length: {len(trace)}")
    
    # Count Row Acts
    row_size = dram_config.row_buffer_bytes
    bank_size = row_size * dram_config.num_rows
    
    actual_acts = 0
    current_row = -1
    
    # Debug: Track first few accesses
    debug_accesses = []
    
    for i, line in enumerate(trace):
        parts = line.strip().split()
        if len(parts) < 2 or parts[0] != 'LD': continue
        
        addr = int(parts[1], 16)
        bank = addr // bank_size
        row = (addr % bank_size) // row_size
        
        if bank == 0: # Input
            if current_row != row:
                actual_acts += 1
                current_row = row
            
            if len(debug_accesses) < 20:
                debug_accesses.append(f"Line {i}: Addr={hex(addr)}, Row={row}")

    log(f"  Trace Actual Input Row Acts: {actual_acts}")
    
    # 5. Comparison
    log("\n" + "-" * 40)
    log("5. COMPARISON RESULTS")
    log("-" * 40)
    
    error = abs(predicted_acts - actual_acts)
    error_pct = (error / actual_acts * 100) if actual_acts > 0 else 0.0
    
    log(f"  Predicted: {predicted_acts}")
    log(f"  Actual:    {actual_acts}")
    log(f"  Error:     {error} ({error_pct:.2f}%)")
    
    if error_pct < 1.0:
        log("  SUCCESS: Match!")
    else:
        log("  FAILURE: Mismatch!")
        log("\n  First 20 Input Accesses:")
        for line in debug_accesses:
            log("    " + line)

def main():
    # Define Workload (Same as verify_single_workload_hybrid.py)
    wl = ConvWorkload(name="ResNet_L2_Small", N=1, C=32, K=32, P=16, Q=16, R=3, S=3, stride=(1,1))
    
    # Define DRAM Config
    dram_config = DRAMConfig(
        row_buffer_bytes=1024,
        element_size=1,
        num_rows=16384
    )
    
    # Define Mapping Config
    # Case 1: The "Fixed" case (Block Size = Tile Size)
    # H_in = (16-1)*1 + 3 = 18. W_in = 18.
    # We put all dimensions in Level 1 (GlobalBuffer) so they form a single tile fetched from DRAM.
    config_fixed = MappingConfig(
        P_l3=1, Q_l3=1, C_l3=1, K_l3=1, # No L3 tiling (all in one tile)
        P_l1=16, Q_l1=16, C_l1=32, K_l1=32, N_l1=1, R_l1=3, S_l1=3, # SRAM tile
        block_h=18, block_w=18,         # Match Input Size
        input_layout='sequential'
    )
    
    log("\nRunning Case 1: Matched Block Size (18x18)")
    debug_workload(wl, config_fixed, dram_config)
    
    # Case 2: The "Problematic" case (Block Size = 16x16)
    # This represents a scenario where the data layout is tiled (16x16),
    # but the access pattern (18x18) is misaligned.
    # To make this a valid comparison, we should acknowledge that 16x16 is a valid layout choice,
    # but accessing it with strict row-major order causes thrashing.
    config_mismatch = MappingConfig(
        P_l3=1, Q_l3=1, C_l3=1, K_l3=1,
        P_l1=16, Q_l1=16, C_l1=32, K_l1=32, N_l1=1, R_l1=3, S_l1=3,
        block_h=16, block_w=16,
        input_layout='sequential'
    )
    
    log("\n\nRunning Case 2: Mismatched Block Size (16x16)")
    debug_workload(wl, config_mismatch, dram_config)

if __name__ == "__main__":
    main()
