#!/usr/bin/env python3
"""
Verify ILP Hybrid Model Accuracy.

This script generates multiple valid mappings and compares the 
Row Activation counts predicted by the ILP Hybrid Model (MicroTrace)
against the Ground Truth (TraceGenerator).
"""

import sys
import random
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from pim_optimizer.workload.conv import ConvWorkload
from validation.dram.trace_generator import TraceGenerator, DRAMConfig
from validation.dram_v2.core.mapping import (
    MappingConfig, MappingEnumerator, 
    to_trace_generator_mapping, WorkloadConfig
)
from pim_optimizer.generator.hybrid_cost_model import MicroTraceGenerator, MicroTraceConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class HybridModelPredictor:
    """Replicates the logic of ILP Hybrid Model (precompute_row_acts.py)."""
    
    def __init__(self, workload: ConvWorkload, row_buffer_bytes: int, element_size: int):
        self.workload = workload
        self.row_buffer_bytes = row_buffer_bytes
        self.element_size = element_size
        
    def predict(self, config: MappingConfig) -> float:
        """
        Predict row acts using MicroTraceGenerator.
        
        Logic adapted from precompute_row_acts.py:
        1. Determine Tile Size (L1 Tile).
        2. Configure MicroTrace.
        3. Simulate sequence of tiles.
        """
        # 1. Determine Tile Size (L1 Tile)
        # In dram_v2, L1 Tile Size = P_l1 * P_l0 (assuming L2/L3 are loops over tiles)
        # But wait, MappingConfig defines factors.
        # Total P = P_l3 * P_l2 * P_l1 * P_l0
        # The "Buffer Tile" loaded from DRAM is usually the L1+L0 content.
        # TraceGenerator iterates L3 and L2. Inside it accesses L1+L0.
        
        p_tile = config.P_l1 * config.P_l0
        q_tile = config.Q_l1 * config.Q_l0
        c_tile = config.C_l1 * config.C_l0
        
        # 2. Configure MicroTrace
        # NOTE: precompute_row_acts.py passes P_tile as tile_h.
        # This assumes stride=1, dilation=1, kernel=1 roughly?
        # Or it assumes MicroTrace simulates the P-loop iteration?
        # Let's stick to what precompute_row_acts.py does to verify THAT logic.
        
        # Map loop order? 
        # precompute_row_acts uses ['C', 'H', 'W'] derived from ['C', 'P', 'Q']
        # We'll use default ['C', 'H', 'W']
        
        mt_config = MicroTraceConfig(
            tile_h=p_tile, 
            tile_w=q_tile,
            tile_c=c_tile,
            tensor_height=self.workload.input_size['H'],
            tensor_width=self.workload.input_size['W'],
            tensor_channels=self.workload.C,
            element_size=self.element_size,
            row_buffer_size=self.row_buffer_bytes,
            loop_order=['C', 'H', 'W'],
            layout_type='tiled', # Matches TraceGenerator's block-major
            block_h=config.block_h,
            block_w=config.block_w
        )
        
        tracer = MicroTraceGenerator(mt_config)
        
        # 3. Simulate Sequence
        # We need to simulate the sequence of tiles as they appear in DRAM access
        # TraceGenerator iterates L3 then L2.
        # We want to simulate the stream of L1 tiles.
        # The order depends on L3/L2 permutation.
        # But precompute_row_acts.py assumes a simple P-Q iteration order for simulation.
        # Let's replicate that.
        
        num_tiles_p = self.workload.P // p_tile
        num_tiles_q = self.workload.Q // q_tile
        
        limit = 64
        tiles_to_sim = []
        for p in range(num_tiles_p):
            for q in range(num_tiles_q):
                tiles_to_sim.append((p, q))
                if len(tiles_to_sim) >= limit:
                    break
            if len(tiles_to_sim) >= limit:
                break
                
        full_trace = []
        stride_h = self.workload.stride[1] # stride_h corresponds to dim 0? 
        # In ConvWorkload: stride is (h, w). stride[0] is h?
        # Let's check ConvWorkload.
        # Usually stride=(stride_h, stride_w).
        
        s_h = self.workload.stride[0]
        s_w = self.workload.stride[1]
        H_in = self.workload.input_size['H']
        W_in = self.workload.input_size['W']
        HW = H_in * W_in
        
        for p_idx, q_idx in tiles_to_sim:
            # Calculate start offset in Input Tensor
            h_start = p_idx * p_tile * s_h
            w_start = q_idx * q_tile * s_w
            c_start = 0
            
            start_offset = c_start * HW + h_start * W_in + w_start
            
            # Get trace
            tile_trace = tracer.get_trace(start_offset)
            full_trace.extend(tile_trace)
            
        # Count acts
        row_size = self.row_buffer_bytes
        bank_size = row_size * 16384
        
        current_row = -1
        activations = 0
        
        for addr in full_trace:
            bank = addr // bank_size
            row = (addr % bank_size) // row_size
            
            if current_row != row:
                activations += 1
                current_row = row
                
        if not tiles_to_sim:
            return 0.0
            
        return activations / len(tiles_to_sim)

def run_verification():
    print("=== Verify ILP Hybrid Model ===")
    
    # 1. Define Workload
    # Use stride=1 to match precompute assumption (for now)
    wl = ConvWorkload(name="ResNet_L2", N=1, C=32, K=32, P=16, Q=16, R=3, S=3, stride=(1,1))
    print(f"Workload: {wl}")
    
    # Convert to WorkloadConfig for Enumerator
    wl_config = WorkloadConfig(
        P=wl.P, Q=wl.Q, C=wl.C, K=wl.K, R=wl.R, S=wl.S, N=wl.N,
        H=wl.input_size['H'], W=wl.input_size['W']
    )
    
    # 2. Define Architecture
    dram_config = DRAMConfig(row_buffer_bytes=1024, element_size=1)
    
    # 3. Generate Mappings
    print("Generating mappings...")
    enumerator = MappingEnumerator(wl_config) 
    
    mappings = []
    # Use sample_total to limit
    for config in enumerator.enumerate(sample_total=20):
        mappings.append(config)
            
    print(f"Generated {len(mappings)} mappings.")
    
    # 4. Run Comparison
    predictor = HybridModelPredictor(wl, dram_config.row_buffer_bytes, dram_config.element_size)
    
    print(f"{'ID':<3} | {'Block':<7} | {'Tile(P,Q,C)':<12} | {'Pred':<8} | {'Actual':<8} | {'Error':<8}")
    print("-" * 60)
    
    for i, config in enumerate(mappings):
        # Predict
        pred_acts = predictor.predict(config)
        
        # Actual (TraceGenerator)
        tg_mapping = to_trace_generator_mapping(config)
        tracer = TraceGenerator(dram_config)
        trace = tracer.generate_trace(tg_mapping, wl)
        
        # Count actual acts (Bank 0)
        row_size = dram_config.row_buffer_bytes
        bank_size = row_size * 16384
        actual_acts = 0
        current_row = -1
        
        # We need to normalize actual acts to "per tile" to match prediction?
        # Predictor returns "acts per tile".
        # TraceGenerator returns "total acts".
        # We should multiply prediction by num_tiles.
        
        p_tile = config.P_l1 * config.P_l0
        q_tile = config.Q_l1 * config.Q_l0
        c_tile = config.C_l1 * config.C_l0
        
        num_tiles = (wl.P // p_tile) * (wl.Q // q_tile) * (wl.C // c_tile)
        
        # Count total actual acts
        for line in trace:
            parts = line.split()
            if len(parts) >= 2 and parts[0] == 'LD':
                addr = int(parts[1], 16)
                bank = addr // bank_size
                row = (addr % bank_size) // row_size
                if bank == 0:
                    if current_row != row:
                        actual_acts += 1
                        current_row = row
                        
        total_pred = pred_acts * num_tiles
        
        error = abs(total_pred - actual_acts)
        error_pct = (error / actual_acts * 100) if actual_acts > 0 else 0.0
        
        tile_str = f"{p_tile},{q_tile},{c_tile}"
        block_str = f"{config.block_h}x{config.block_w}"
        
        print(f"{i:<3} | {block_str:<7} | {tile_str:<12} | {total_pred:<8.1f} | {actual_acts:<8} | {error_pct:<6.1f}%")

if __name__ == "__main__":
    run_verification()
