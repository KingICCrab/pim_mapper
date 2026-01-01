
import sys
import os
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer.mapping import Mapping
from pim_optimizer.generator.hybrid_cost_model import MicroTraceGenerator, MicroTraceConfig
from validation.dram.trace_generator import TraceGenerator, DRAMConfig, DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N

class ComprehensiveVerifier:
    def __init__(self):
        self.dram_config = DRAMConfig(row_buffer_bytes=1024, element_size=1)
        self.trace_gen = TraceGenerator(self.dram_config)
        self.results = []

    def get_factors(self, n):
        factors = set()
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                factors.add(i)
                factors.add(n // i)
        return sorted(list(factors))

    def calculate_ilp_cost(self, workload, p_l2, q_l2, c_l2, block_h, block_w):
        """Calculate cost using the ILP Hybrid Model."""
        # Map loop_order (P,Q,C) to MicroTrace (H,W,C)
        # Standard order for sequential filling
        loop_order = ['C', 'P', 'Q'] 
        mt_loop_order = ['C', 'H', 'W']
            
        # Calculate Input Tile Size
        stride_h = workload.stride[1]
        stride_w = workload.stride[0]
        dilation_h = workload.dilation[1]
        dilation_w = workload.dilation[0]
        
        tile_h_in = (p_l2 - 1) * stride_h + (workload.R - 1) * dilation_h + 1
        tile_w_in = (q_l2 - 1) * stride_w + (workload.S - 1) * dilation_w + 1
        
        # To match TraceGenerator's "Packed" simulation assumption,
        # we set the tensor dimensions to the tile dimensions.
        # This eliminates the "Stride/Gap" effect in ILP, aligning it with GT.
        # Note: TraceGenerator with 'sequential' layout simulates a packed buffer transfer.
        # If the hardware supports Strided DMA (reading gaps), ILP can model it by setting
        # tensor_width = workload.input_size['W'].
        # For validation against TraceGenerator, we must use Packed mode.
        config = MicroTraceConfig(
            tile_h=tile_h_in,
            tile_w=tile_w_in,
            tile_c=c_l2,
            tensor_height=tile_h_in, # Packed
            tensor_width=tile_w_in,  # Packed
            tensor_channels=workload.C,
            element_size=1,
            row_buffer_size=1024,
            loop_order=mt_loop_order,
            layout_type='linear',
            channel_last=False,
            block_h=block_h,
            block_w=block_w
        )
        
        tracer = MicroTraceGenerator(config)
        
        # Simulate a sequence of tiles (up to 64)
        # Fix: Use ceiling division to include partial tiles
        num_tiles_p = (workload.P + p_l2 - 1) // p_l2
        num_tiles_q = (workload.Q + q_l2 - 1) // q_l2
        limit = 64
        tiles_to_sim = []
        for p in range(num_tiles_p):
            for q in range(num_tiles_q):
                tiles_to_sim.append((p, q))
                if len(tiles_to_sim) >= limit: break
            if len(tiles_to_sim) >= limit: break
            
        full_trace = []
        HW = tile_h_in * tile_w_in # Packed stride
        W_in = tile_w_in # Packed stride
        
        for (p_idx, q_idx) in tiles_to_sim:
            # For packed simulation, offsets are just sequential blocks
            # But MicroTraceGenerator generates relative addresses 0..Size.
            # We need to offset them to simulate distinct tiles in memory?
            # GT simulates distinct tiles.
            # If we want to match GT, we should offset them.
            # But since we sort, and tiles are likely far apart, it doesn't matter much
            # unless they share rows.
            # Let's assume they are far apart (no reuse between tiles).
            
            start_offset = (p_idx * num_tiles_q + q_idx) * HW * workload.C # Large offset
            
            # Get accesses and SORT for DMA simulation
            # Note: For partial tiles, MicroTrace generates full tile accesses.
            # We should technically clip the accesses if we want to simulate partial tiles accurately.
            # But MicroTraceGenerator doesn't support "Partial Tile" generation easily 
            # without changing config for each tile.
            # For now, we assume MicroTrace simulates the "Worst Case" (Full Tile).
            # If GT simulates partial tiles, ILP will overestimate.
            
            tile_accesses = tracer.get_trace(0) # Always 0 relative
            # Add offset
            tile_accesses = [a + start_offset for a in tile_accesses]
            
            tile_accesses.sort() # DMA Linear Transfer
            full_trace.extend(tile_accesses)
            
        # Count activations
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
        
        # Total Cost Calculation
        # Fix: Use ceiling division
        total_tiles = ((workload.P + p_l2 - 1) // p_l2) * \
                      ((workload.Q + q_l2 - 1) // q_l2) * \
                      ((workload.C + c_l2 - 1) // c_l2)
        # Assuming ReusePenalty=1 for this validation (focus on spatial layout)
        predicted_cost = total_tiles * (1 + (avg_cost - 1) * 1.0)
        
        return predicted_cost, avg_cost

    def calculate_gt_cost(self, workload, p_l2, q_l2, c_l2, block_h, block_w):
        """Calculate Ground Truth cost using TraceGenerator."""
        # Use the REAL workload (ResNet layer)
        # TraceGenerator will handle the tiling and Halo automatically based on the Mapping.
        
        mapping = Mapping()
        # Level 2 (DRAM Loops) - Iterate over Tiles
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
            
            if bank == 0: # Input
                if current_row != row:
                    gt_activations += 1
                    current_row = row
        return gt_activations

    def run_sweep(self):
        # Define Workloads (ResNet-like)
        workloads = [
            # Name, R, S, P, Q, C, Stride
            ("Conv2_x", 3, 3, 56, 56, 64, 1),
            ("Conv3_x", 3, 3, 28, 28, 128, 1),
            ("Conv4_x", 3, 3, 14, 14, 256, 1),
        ]
        
        # Define Block Sizes to test (DRAM Layout Blocks)
        block_sizes = [32, 64] 
        
        print("Starting Comprehensive Sweep...")
        
        for name, R, S, P, Q, C, stride in workloads:
            workload = ConvWorkload(
                R=R, S=S, P=P, Q=Q, C=C, K=64, N=1,
                stride=(stride, stride), dilation=(1, 1)
            )
            
            # Sweep Tile Sizes (Factors of P/Q)
            # p_factors = [f for f in self.get_factors(P) if f >= 4] # Skip tiny tiles
            # q_factors = [f for f in self.get_factors(Q) if f >= 4]
            
            # Use a wider range of tile sizes, not just factors, to test misalignment
            # e.g. P=56, test 28, 30, 32, 56
            p_factors = sorted(list(set([f for f in self.get_factors(P) if f >= 14] + [30, 32, 48])))
            q_factors = sorted(list(set([f for f in self.get_factors(Q) if f >= 14] + [30, 32, 48])))
            
            # Filter out sizes larger than P/Q
            p_factors = [p for p in p_factors if p <= P]
            q_factors = [q for q in q_factors if q <= Q]
            
            c_factors = [32] # Fixed C tile for simplicity
            
            # Limit combinations to avoid explosion
            import random
            random.seed(42)
            
            combinations = []
            for p in p_factors:
                for q in q_factors:
                    for c in c_factors:
                        combinations.append((p, q, c))
            
            # Sample if too many
            if len(combinations) > 10:
                combinations = random.sample(combinations, 10)
            # Use ALL combinations to get more data points
            
            for p_tile, q_tile, c_tile in tqdm(combinations, desc=f"Sweeping {name}"):
                # We test with a fixed internal block size for ILP, 
                # but since we sort the trace (DMA Linear), the block size shouldn't matter.
                # We set GT to simulate the full tile linearly.
                block_size = 32 
                
                # 1. Run ILP Prediction
                pred_cost, avg_cost = self.calculate_ilp_cost(
                    workload, p_tile, q_tile, c_tile, block_size, block_size
                )
                
                # Calculate Input Tile Size for GT
                stride_h = workload.stride[1]
                stride_w = workload.stride[0]
                dilation_h = workload.dilation[1]
                dilation_w = workload.dilation[0]
                
                tile_h_in = (p_tile - 1) * stride_h + (workload.R - 1) * dilation_h + 1
                tile_w_in = (q_tile - 1) * stride_w + (workload.S - 1) * dilation_w + 1

                # 2. Run Ground Truth
                # Use the REAL workload (ResNet layer)
                # TraceGenerator will handle the tiling and Halo automatically based on the Mapping.
                gt_cost = self.calculate_gt_cost(workload, p_tile, q_tile, c_tile, block_size, block_size)
                
                # Determine type for analysis
                if p_tile % 32 == 0 and q_tile % 32 == 0:
                    type_label = "Aligned_32"
                else:
                    type_label = "Misaligned"

                self.results.append({
                    'Workload': name,
                    'P_Tile': p_tile, 'Q_Tile': q_tile, 'Block': block_size,
                    'Type': type_label,
                    'Pred_Cost': pred_cost,
                    'GT_Cost': gt_cost,
                    'Error': abs(pred_cost - gt_cost) / gt_cost if gt_cost > 0 else 0
                })

    def analyze(self):
        df = pd.DataFrame(self.results)
        if df.empty:
            print("No valid results collected.")
            return
            
        print("\n=== Comprehensive Verification Results ===")
        print(f"Total Data Points: {len(df)}")
        
        # Correlation
        corr = df['Pred_Cost'].corr(df['GT_Cost'])
        print(f"Correlation (Pearson): {corr:.4f}")
        
        # Error Stats
        print("\nError Statistics:")
        print(df['Error'].describe())
        
        # Group by Type
        print("\nBy Type:")
        print(df.groupby('Type')[['Pred_Cost', 'GT_Cost', 'Error']].mean())
        
        # Save to CSV
        df.to_csv('validation_results_comprehensive.csv', index=False)
        print("\nResults saved to validation_results_comprehensive.csv")

if __name__ == "__main__":
    verifier = ComprehensiveVerifier()
    verifier.run_sweep()
    verifier.analyze()
