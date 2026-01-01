"""
Verify that the Hybrid Cost Model (MicroTraceGenerator) used in ILP 
is aligned with Ground Truth (TraceGenerator).

This script directly uses the same cost calculation logic as the ILP model
to ensure we're validating what will actually be used in optimization.
"""

import sys
import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer.mapping import Mapping
from pim_optimizer.generator.hybrid_cost_model import MicroTraceGenerator, MicroTraceConfig
from validation.dram.trace_generator import TraceGenerator, DRAMConfig, DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N

class ILPAlignmentVerifier:
    def __init__(self):
        self.dram_config = DRAMConfig(row_buffer_bytes=1024, element_size=1)
        self.trace_gen = TraceGenerator(self.dram_config)
        self.results = []

    def calculate_ilp_hybrid_cost(self, workload, p_l2, q_l2, c_l2, block_h, block_w):
        """
        Calculate cost using the EXACT logic from ILP model:
        - Use MicroTraceGenerator (same as precompute_row_acts.py)
        - Use the formula: Cost = N_tiles * [1 + (AvgCost - 1) * ReusePenalty]
        
        For this validation:
        - ReusePenalty = 1.0 (no thrashing from irrelevant loops)
        - Packed Mode: tensor dimensions = tile dimensions (Tiled Wise hardware)
        """
        # Calculate Input Tile Size (same as ILP)
        stride_h = workload.stride[1]
        stride_w = workload.stride[0]
        dilation_h = workload.dilation[1]
        dilation_w = workload.dilation[0]
        
        tile_h_in = (p_l2 - 1) * stride_h + (workload.R - 1) * dilation_h + 1
        tile_w_in = (q_l2 - 1) * stride_w + (workload.S - 1) * dilation_w + 1
        
        # MicroTraceConfig (same as used in ILP)
        config = MicroTraceConfig(
            tile_h=tile_h_in,
            tile_w=tile_w_in,
            tile_c=c_l2,
            tensor_height=tile_h_in,  # Packed (Tiled Wise)
            tensor_width=tile_w_in,   # Packed (Tiled Wise)
            tensor_channels=workload.C,
            element_size=1,
            row_buffer_size=1024,
            loop_order=['C', 'H', 'W'],  # Standard planar layout
            layout_type='linear',
            channel_last=False,
            block_h=block_h,
            block_w=block_w
        )
        
        tracer = MicroTraceGenerator(config)
        
        # Simulate a sequence of tiles (same as precompute logic)
        num_tiles_p = (workload.P + p_l2 - 1) // p_l2
        num_tiles_q = (workload.Q + q_l2 - 1) // q_l2
        limit = 64  # Sample tiles for average
        tiles_to_sim = []
        for p in range(num_tiles_p):
            for q in range(num_tiles_q):
                tiles_to_sim.append((p, q))
                if len(tiles_to_sim) >= limit: break
            if len(tiles_to_sim) >= limit: break
            
        full_trace = []
        HW = tile_h_in * tile_w_in
        
        for (p_idx, q_idx) in tiles_to_sim:
            # Offset for distinct tiles
            start_offset = (p_idx * num_tiles_q + q_idx) * HW * workload.C
            
            # Get trace from MicroTraceGenerator
            tile_accesses = tracer.get_trace(0)  # Relative addresses
            tile_accesses = [a + start_offset for a in tile_accesses]
            
            # CRITICAL: DMA Linear Transfer (Block-Major)
            # Sort to simulate smart DMA controller optimizing burst access
            tile_accesses.sort()
            
            full_trace.extend(tile_accesses)
            
        # Count Row Activations (same as ILP logic)
        row_size = 1024
        bank_size = row_size * 16384
        current_row = None
        activations = 0
        
        for addr in full_trace:
            bank = addr // bank_size
            row = (addr % bank_size) // row_size
            if current_row != row:
                activations += 1
                current_row = row
                
        avg_cost = activations / len(tiles_to_sim) if tiles_to_sim else 0
        
        # ILP Formula: Cost = N_tiles * [1 + (AvgCost - 1) * ReusePenalty]
        total_tiles = ((workload.P + p_l2 - 1) // p_l2) * \
                      ((workload.Q + q_l2 - 1) // q_l2) * \
                      ((workload.C + c_l2 - 1) // c_l2)
        
        reuse_penalty = 1.0  # For this validation
        predicted_cost = total_tiles * (1 + (avg_cost - 1) * reuse_penalty)
        
        return predicted_cost, avg_cost

    def calculate_gt_cost(self, workload, p_l2, q_l2, c_l2):
        """Calculate Ground Truth cost using TraceGenerator."""
        mapping = Mapping()
        mapping.loop_bounds = {
            2: {'temporal': {
                DIM_P: (workload.P + p_l2 - 1) // p_l2, 
                DIM_Q: (workload.Q + q_l2 - 1) // q_l2, 
                DIM_C: (workload.C + c_l2 - 1) // c_l2
            }}, 
            1: {'temporal': {
                DIM_P: p_l2, 
                DIM_Q: q_l2, 
                DIM_C: c_l2
            }} 
        }
        mapping.permutation = {
            2: {0: DIM_P, 1: DIM_Q, 2: DIM_C},
            1: {0: DIM_C, 1: DIM_P, 2: DIM_Q}
        }
        mapping.layout = {'input': 'sequential', 'weight': 'sequential', 'output': 'sequential'}
        
        trace = self.trace_gen.generate_trace(mapping, workload)
        
        row_size = 1024
        bank_size = row_size * 16384
        current_row = None
        gt_activations = 0
        
        for line in trace:
            parts = line.strip().split()
            if len(parts) < 2 or parts[0] != 'LD': continue
            addr = int(parts[1], 16)
            bank = addr // bank_size
            row = (addr % bank_size) // row_size
            
            if bank == 0:  # Input
                if current_row != row:
                    gt_activations += 1
                    current_row = row
        return gt_activations

    def run_verification(self):
        """Run verification on multiple workloads and tile sizes."""
        # Workloads
        workloads = [
            ("Conv2_x", ConvWorkload(R=3, S=3, P=56, Q=56, C=64, K=64, N=1, stride=(1, 1), dilation=(1, 1))),
            ("Conv3_x", ConvWorkload(R=3, S=3, P=28, Q=28, C=128, K=64, N=1, stride=(1, 1), dilation=(1, 1))),
            ("Conv4_x", ConvWorkload(R=3, S=3, P=14, Q=14, C=256, K=64, N=1, stride=(1, 1), dilation=(1, 1))),
        ]
        
        # Test configurations
        test_configs = [
            # (P_Tile, Q_Tile, C_Tile, Block_H, Block_W, Type)
            (28, 28, 32, 32, 32, "Aligned"),
            (30, 30, 32, 32, 32, "Misaligned"),
            (32, 32, 32, 32, 32, "Aligned"),
            (48, 48, 32, 32, 32, "Misaligned"),
        ]
        
        print("Starting ILP Alignment Verification...")
        print("=" * 80)
        
        for wl_name, workload in workloads:
            print(f"\n{wl_name}: P={workload.P}, Q={workload.Q}, C={workload.C}")
            print("-" * 80)
            
            for p_tile, q_tile, c_tile, block_h, block_w, cfg_type in test_configs:
                # Skip if tile size exceeds workload size
                if p_tile > workload.P or q_tile > workload.Q:
                    continue
                
                # ILP Model
                ilp_cost, avg_cost = self.calculate_ilp_hybrid_cost(
                    workload, p_tile, q_tile, c_tile, block_h, block_w
                )
                
                # Ground Truth
                gt_cost = self.calculate_gt_cost(workload, p_tile, q_tile, c_tile)
                
                error = abs(ilp_cost - gt_cost) / gt_cost if gt_cost > 0 else 0
                
                self.results.append({
                    'Workload': wl_name,
                    'P_Tile': p_tile,
                    'Q_Tile': q_tile,
                    'C_Tile': c_tile,
                    'Type': cfg_type,
                    'ILP_Cost': ilp_cost,
                    'GT_Cost': gt_cost,
                    'Error': error,
                    'AvgCost': avg_cost
                })
                
                print(f"  {cfg_type:12s} P={p_tile:2d} Q={q_tile:2d}: "
                      f"ILP={ilp_cost:7.1f}, GT={gt_cost:7d}, "
                      f"Error={error:6.2%}, AvgCost={avg_cost:5.2f}")
    
    def analyze(self):
        """Analyze results."""
        df = pd.DataFrame(self.results)
        if df.empty:
            print("No results to analyze.")
            return
        
        print("\n" + "=" * 80)
        print("=== Final Analysis ===")
        print("=" * 80)
        
        # Overall statistics
        print(f"\nTotal Data Points: {len(df)}")
        print(f"Pearson Correlation: {df['ILP_Cost'].corr(df['GT_Cost']):.4f}")
        print(f"Spearman Correlation: {df['ILP_Cost'].corr(df['GT_Cost'], method='spearman'):.4f}")
        
        print("\nError Statistics:")
        print(df['Error'].describe())
        
        print("\nBy Type:")
        print(df.groupby('Type')[['ILP_Cost', 'GT_Cost', 'Error']].mean())
        
        # Save results
        df.to_csv('ilp_alignment_results.csv', index=False)
        print("\nResults saved to ilp_alignment_results.csv")

if __name__ == "__main__":
    verifier = ILPAlignmentVerifier()
    verifier.run_verification()
    verifier.analyze()
