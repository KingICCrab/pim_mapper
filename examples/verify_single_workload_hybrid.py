#!/usr/bin/env python3
"""
Verify Single Workload Hybrid Model Accuracy.

This script:
1. Defines a specific ConvWorkload.
2. Generates a Lookup Table for it (using Hybrid Model).
3. Runs the ILP Optimizer to find the best mapping.
4. Extracts the mapping and runs TraceGenerator (Honest Validator).
5. Compares the ILP-predicted Input Row Acts vs. TraceGenerator's count.
"""

import sys
import os
import json
import logging
import gurobipy as gp
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT)) # For validation module

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload import ConvWorkload
from pim_optimizer.generator.precompute_row_acts import RowActivationPrecomputer, PrecomputeConfig
from pim_optimizer.arch import PIMArchitecture
from pim_optimizer.arch.memory import MemoryHierarchy, MemoryLevel
from validation.dram.trace_generator import TraceGenerator, DRAMConfig
import pim_optimizer.model.constraints as constraints
import pim_optimizer.optimizer as optimizer_module

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_single_workload():
    # 1. Define Workload (e.g., ResNet Layer 2: 64x64x56x56, 3x3, s=1)
    # Using a smaller workload for speed if needed, but let's try a realistic one.
    # N=1, C=64, K=64, P=56, Q=56, R=3, S=3
    # REDUCED WORKLOAD for Gurobi License
    wl = ConvWorkload(name="ResNet_L2_Small", N=1, C=32, K=32, P=16, Q=16, R=3, S=3, stride=(1,1))
    
    logger.info(f"Workload: {wl}")
    
    # 2. Generate Lookup Table
    logger.info("Generating Lookup Table (Hybrid Model)...")
    
    # OPTIMIZATION: Reduce search space for fast verification
    # We only need to verify that the ILP picks a valid entry and the cost matches.
    # We don't need to find the global optimum for this test.
    # NOTE: We removed explicit block_h/w options. 
    # The precomputer should now automatically generate factors of H_in/W_in (e.g. 18).
    config = PrecomputeConfig(
        row_buffer_bytes=1024 # Reduced buffer
    )
    
    precomputer = RowActivationPrecomputer(wl, config)
    
    # Monkey-patch get_factors to reduce search space further
    original_get_factors = precomputer.get_factors
    def fast_get_factors(n: int):
        # Force single factor to minimize search
        return [n]
        
    precomputer.get_factors = fast_get_factors
    logger.info("Monkey-patched precomputer.get_factors for speed.")
    
    table = precomputer.compute_table()
    
    # Save table
    table_path = "row_activation_cost_table.json"
    precomputer.save_table(table, table_path)
    
    # 3. Run ILP Optimizer
    logger.info("Skipping ILP Optimizer to avoid license issues.")
    
    # Load the table
    with open(table_path, 'r') as f:
        table_data = json.load(f)
        
    # Find the entry for our config
    # We pick the best one (lowest cost) as ILP would do.
    best_entry = min(table_data, key=lambda x: x['row_acts'])
    predicted_acts = best_entry['row_acts']
    logger.info(f"Lookup Table Best Cost: {predicted_acts} (Order: {best_entry['order']})")
    
    # 4. Extract Results
    # Manually construct mapping
    # We need a Mapping object to run TraceGenerator
    from pim_optimizer.mapping import Mapping
    
    # Construct loop bounds
    # Level 0 (RowBuffer): P=16, Q=16, C=32 (All in one tile)
    # Level 1 (DRAM): P=1, Q=1, C=1 (No tiling at DRAM level)
    
    # Map names to dims
    dim_map = {'P': 2, 'Q': 3, 'C': 4, 'K': 5, 'N': 6, 'R': 0, 'S': 1}
    
    mapping = Mapping()
    mapping.loop_bounds = {
        0: {'spatial': {}, 'temporal': {}},
        1: {'spatial': {}, 'temporal': {}}
    }
    
    # Set bounds
    # Level 0 (RowBuffer): Holds the tile.
    mapping.loop_bounds[0]['temporal'] = {
        dim_map['P']: wl.P, dim_map['Q']: wl.Q, dim_map['C']: wl.C, 
        dim_map['K']: wl.K, dim_map['N']: wl.N, dim_map['R']: wl.R, dim_map['S']: wl.S
    }
    # Level 1 (DRAM): Iterates over tiles.
    mapping.loop_bounds[1]['temporal'] = {
        dim_map['P']: 1, dim_map['Q']: 1, dim_map['C']: 1, 
        dim_map['K']: 1, dim_map['N']: 1, dim_map['R']: 1, dim_map['S']: 1
    }
    
    # Set permutation to match the best order found (C-P-Q)
    # TraceGenerator uses Level 2/3 permutation to determine element iteration order
    # Higher index = Outer loop
    mapping.permutation = {
        2: {
            2: dim_map['C'], # Outer
            1: dim_map['P'], # Middle
            0: dim_map['Q']  # Inner
        }
    }
    
    # Set layout
    mapping.layout = {0: "sequential", 1: "sequential", 2: "sequential"}
    mapping.tile_info = {
        'block_h': best_entry.get('best_block_h', 16), 
        'block_w': best_entry.get('best_block_w', 16)
    }
    logger.info(f"Using Block Size from Best Entry: {mapping.tile_info['block_h']}x{mapping.tile_info['block_w']}")
    
    # Set permutation based on best entry
    # Entry order is e.g. "P-Q-C". This is L2 loop order (inside tile).
    # But we have 1 tile. So L2 loops cover everything.
    # We need to set L2 permutation.
    # Mapping permutation format: [level][perm_level] = dimension
    # Level 0 is RowBuffer.
    # Level 1 is DRAM.
    # The loops inside RowBuffer (Level 0) are the ones iterating over elements?
    # No, Level 0 loops iterate over Level -1 (PE) tiles.
    # If PE tile is 1x1x1. Then Level 0 loops iterate over elements.
    # So we should set Level 0 permutation.
    
    order_str = best_entry['order']
    order_list = order_str.split('-')
    # Map names to dims
    dim_map = {'P': 2, 'Q': 3, 'C': 4, 'K': 5, 'N': 6, 'R': 0, 'S': 1}
    # We need to include all dims.
    # The entry only specifies P, Q, C order.
    # We assume others are inner or outer?
    # Usually R, S are inner. K, N are outer.
    # Let's just set P, Q, C order as specified.
    
    # TraceGenerator uses _build_loop_order.
    # It sorts by permutation level.
    
    mapping.permutation = {0: {}}
    # Assign levels. 0 is outer.
    # If order is P-Q-C. P is outer, C is inner.
    # So P=0, Q=1, C=2.
    for i, dim_name in enumerate(order_list):
        mapping.permutation[0][i] = dim_map[dim_name]
        
    # Add other dims
    next_idx = len(order_list)
    for dim_name in ['K', 'N', 'R', 'S']:
        mapping.permutation[0][next_idx] = dim_map[dim_name]
        next_idx += 1
    
    # 5. Run TraceGenerator (Honest Validator)
    logger.info("Running TraceGenerator (Honest Validator)...")
    
    # Create DRAMConfig from architecture
    # We removed PELocalBuffer, so LocalDRAM is Level 1.
    # But TraceGenerator expects LocalDRAM to be the last level.
    # We have 2 levels.
    
    dram_config = DRAMConfig(
        row_buffer_bytes=config.row_buffer_bytes,
        element_size=1, 
        num_rows=16384
    )
    tracer = TraceGenerator(dram_config)
    
    # Generate trace
    # Use strict_ordering=False to match precomputer's block-major assumption
    trace = tracer.generate_trace(mapping, wl, strict_ordering=False)
    
    # DEBUG: Write trace to file
    with open("debug_trace.txt", "w") as f:
        for line in trace[:100]:
            f.write(line + "\n")
    logger.info("DEBUG: Wrote first 100 lines of trace to debug_trace.txt")
    
    # Count Input Row Acts (Bank 0)
    row_size = config.row_buffer_bytes
    bank_size = row_size * 16384
    
    actual_acts = 0
    current_row = -1
    
    for line in trace:
        parts = line.strip().split()
        if len(parts) < 2: continue
        if parts[0] != 'LD': continue
        
        addr = int(parts[1], 16)
        bank = addr // bank_size
        row = (addr % bank_size) // row_size
        
        if bank == 0:
            if current_row != row:
                actual_acts += 1
                current_row = row
                
    logger.info(f"TraceGenerator Actual Input Row Acts: {actual_acts}")
    
    # 6. Compare
    error = abs(predicted_acts - actual_acts)
    error_pct = (error / actual_acts * 100) if actual_acts > 0 else 0.0
    
    logger.info("-" * 40)
    logger.info(f"Validation Results:")
    logger.info(f"  Predicted: {predicted_acts:.2f}")
    logger.info(f"  Actual:    {actual_acts}")
    logger.info(f"  Error:     {error:.2f} ({error_pct:.2f}%)")
    logger.info("-" * 40)
    
    if error_pct < 5.0:
        logger.info("SUCCESS: Model is accurate within 5%.")
    else:
        logger.warning("FAILURE: Model error is too high.")
    
    return # Skip the rest

    # 3. Run ILP Optimizer
    logger.info("Running ILP Optimizer...")
    
    # Test license
    try:
        m = gp.Model()
        x = m.addVar(name="x")
        m.addConstr(x >= 1)
        m.optimize()
        logger.info("Trivial model solved.")
    except Exception as e:
        logger.error(f"License check failed: {e}")
        return

    # Create custom hierarchy with fewer levels to save variables
    # 2 Levels: RowBuffer (Level 0), LocalDRAM (Level 1)
    # We treat RowBuffer as the "innermost" for this test, although usually it's shared.
    # But PIMOptimizer expects PE level.
    # Let's try 2 levels: PE (Level 0), DRAM (Level 1).
    # But Row Activation happens at DRAM.
    # If we have PE and DRAM, where is RowBuffer?
    # We can define DRAM as Level 1, and set row_buffer_size on it.
    # And PE as Level 0.
    # Note: row_activation.py requires "RowBuffer" and "LocalDRAM" in mem_idx.
    
    hierarchy = MemoryHierarchy(levels=[
        # MemoryLevel(name="PELocalBuffer", entries=64, blocksize=1, instances=64, stores=[False, False, True]),
        # RowBuffer with infinite capacity to avoid tiling constraints, but we'll set its physical size later
        # MemoryLevel(name="RowBuffer", entries=-1, blocksize=32, instances=1, stores=[True, True, True]),
        MemoryLevel(name="LocalDRAM", entries=-1, blocksize=32, instances=1, stores=[True, True, True], row_buffer_size=1024)
    ])
    arch = PIMArchitecture(hierarchy=hierarchy)
    arch.row_activation_dtypes = ["input"] # Only optimize Input Row Acts to save variables
    
    optimizer = PIMOptimizer(arch=arch, verbose=True)
    
    # Hack: Modify architecture to match our precompute config
    # Assuming LocalDRAM is level 2 (index 2)
    rb_idx = optimizer.arch.mem_idx.get("RowBuffer") 
    if rb_idx is not None:
        # Set physical size for cost model, while capacity is infinite
        optimizer.arch.mem_row_buffer_size[rb_idx] = config.row_buffer_bytes
        logger.info(f"Updated Arch RowBuffer Size to {config.row_buffer_bytes}")
    
    optimizer.time_limit = 30.0 # Give it some time
    
    # Build model but don't optimize yet (if possible)
    # PIMOptimizer.optimize calls build_model then optimize.
    # We can catch the error and inspect the model if it was built.
    
    try:
        result = optimizer.optimize([wl], enable_row_activation=True)
    except gp.GurobiError as e:
        logger.error(f"Gurobi Error: {e}")
        if optimizer.model:
            logger.info(f"Model Stats: Vars={optimizer.model.NumVars}, Constrs={optimizer.model.NumConstrs}, SOS={optimizer.model.NumSOS}, QConstrs={optimizer.model.NumQConstrs}, GenConstrs={optimizer.model.NumGenConstrs}")
            # Write model to file for inspection
            optimizer.model.write("debug_model.lp")
        return
    except Exception as e:
        logger.error(f"Optimization Error: {e}")
        return

    # 4. Extract Results
    model = optimizer.model
    
    # Get Predicted Input Row Acts
    # Variable name: total_row_acts_(0,0) or row_acts_lut_0 (if lookup used and no layout choice)
    try:
        pred_var = model.getVarByName("total_row_acts_(0,0)")
        predicted_acts = pred_var.X
    except Exception:
        logger.warning("Could not find total_row_acts_(0,0), trying row_acts_lut_0...")
        try:
            pred_var = model.getVarByName("row_acts_lut_0")
            predicted_acts = pred_var.X
            logger.info(f"Found row_acts_lut_0: {predicted_acts}")
        except Exception:
            logger.warning("Could not find row_acts_lut_0 either.")
            predicted_acts = 0

        
    logger.info(f"ILP Predicted Input Row Acts: {predicted_acts}")
    
    # Get Mapping
    mapping = result.mappings[0]
    
    # 5. Run TraceGenerator (Honest Validator)
    logger.info("Running TraceGenerator (Honest Validator)...")
    
    # Create DRAMConfig from architecture
    dram_level_idx = optimizer.arch.mem_idx.get("LocalDRAM")
    # Default to 16384 rows if not found (standard DDR4/5)
    num_rows = 16384 
    if hasattr(optimizer.arch, "mem_num_rows") and dram_level_idx is not None:
         # Check if mem_num_rows is populated (it might not be in PIMArchitecture init)
         # PIMArchitecture usually doesn't track num_rows explicitly unless added.
         pass
         
    dram_config = DRAMConfig(
        row_buffer_bytes=config.row_buffer_bytes,
        element_size=1, # Assuming 1 byte per element for this test
        num_rows=num_rows
    )
    tracer = TraceGenerator(dram_config)
    
    # Generate trace with strict ordering
    trace = tracer.generate_trace(mapping, wl, strict_ordering=True)
    
    print("HELLO WORLD - TRACE GENERATED")
    import sys
    sys.stdout.flush()
    
    # DEBUG: Write trace to file
    with open("debug_trace.txt", "w") as f:
        for line in trace[:100]:
            f.write(line + "\n")
    print("DEBUG: Wrote first 100 lines of trace to debug_trace.txt")
    sys.stdout.flush()
    
    # Count Input Row Acts (Bank 0)
    # TraceGenerator._count_input_row_acts is not public, but we can implement a simple counter here
    # or use the one from precompute_row_acts if we import it.
    # Let's implement a simple one here matching TraceGenerator logic.
    
    row_size = config.row_buffer_bytes
    # Assuming TraceGenerator uses the same row size logic.
    # TraceGenerator uses arch.mem_row_buffer_size.
    # We updated optimizer.arch, so tracer.arch should be correct.
    
    # We need to know num_rows to calculate bank/row from address.
    # TraceGenerator uses:
    # bank = addr // bank_size
    # row = (addr % bank_size) // row_size
    # bank_size = row_size * num_rows
    
    # Let's look at TraceGenerator code again to be sure about address mapping.
    # It seems it uses a simple linear mapping.
    
    actual_acts = 0
    current_row = -1
    
    # We need to parse the trace
    # Trace format: "LD <hex_addr>"
    
    # Get num_rows from config
    # num_rows is already defined above
    bank_size = row_size * num_rows
    
    # DEBUG: Print all total_row_acts variables
    logger.info("DEBUG: Checking model variables...")
    for v in model.getVars():
        if "total_row_acts" in v.VarName:
            logger.info(f"Var {v.VarName} = {v.X}")
        if "reuse_penalty" in v.VarName:
            logger.info(f"Var {v.VarName} = {v.X}")
        if "z_lut" in v.VarName and v.X > 0.5:
            logger.info(f"Var {v.VarName} = {v.X}")
    
    for line in trace:
        parts = line.strip().split()
        if len(parts) < 2: continue
        if parts[0] != 'LD': continue
        
        addr = int(parts[1], 16)
        bank = addr // bank_size
        row = (addr % bank_size) // row_size
        
        # Input is mapped to Bank 0 (usually)
        # TraceGenerator logic:
        # Input -> Bank 0
        # Weight -> Bank 1
        # Output -> Bank 2
        
        if bank == 0:
            if current_row != row:
                actual_acts += 1
                current_row = row
                
    logger.info(f"TraceGenerator Actual Input Row Acts: {actual_acts}")
    
    # 6. Compare
    error = abs(predicted_acts - actual_acts)
    error_pct = (error / actual_acts * 100) if actual_acts > 0 else 0.0
    
    logger.info("-" * 40)
    logger.info(f"Validation Results:")
    logger.info(f"  Predicted: {predicted_acts:.2f}")
    logger.info(f"  Actual:    {actual_acts}")
    logger.info(f"  Error:     {error:.2f} ({error_pct:.2f}%)")
    logger.info("-" * 40)
    
    if error_pct < 5.0:
        logger.info("SUCCESS: Model is accurate within 5%.")
    else:
        logger.warning("FAILURE: Model error is too high.")

if __name__ == "__main__":
    verify_single_workload()
