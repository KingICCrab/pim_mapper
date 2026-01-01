
import sys
import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer.mapping import Mapping
from pim_optimizer.generator.hybrid_cost_model import MicroTraceGenerator, MicroTraceConfig
from validation.dram.trace_generator import TraceGenerator, DRAMConfig, DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N

class OptimizationVerifier:
    def __init__(self):
        self.dram_config = DRAMConfig(row_buffer_bytes=1024, element_size=1)
        self.trace_gen = TraceGenerator(self.dram_config)
        self.results = []

    def calculate_ilp_cost(self, workload, p_l2, q_l2, c_l2, block_h, block_w):
        """Calculate cost using the ILP Hybrid Model (Packed Mode)."""
        mt_loop_order = ['C', 'H', 'W']
            
        # Calculate Input Tile Size
        stride_h = workload.stride[1]
        stride_w = workload.stride[0]
        dilation_h = workload.dilation[1]
        dilation_w = workload.dilation[0]
        
        tile_h_in = (p_l2 - 1) * stride_h + (workload.R - 1) * dilation_h + 1
        tile_w_in = (q_l2 - 1) * stride_w + (workload.S - 1) * dilation_w + 1
        
        # Packed Mode: tensor_width = tile_w_in
        config = MicroTraceConfig(
            tile_h=tile_h_in,
            tile_w=tile_w_in,
            tile_c=c_l2,
            tensor_height=tile_h_in, 
            tensor_width=tile_w_in,
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
        
        # Simulate a sequence of tiles
        # Use ceiling division for correct tile count
        num_tiles_p = (workload.P + p_l2 - 1) // p_l2
        num_tiles_q = (workload.Q + q_l2 - 1) // q_l2
        
        # Limit simulation to save time, but ensure we cover enough to see patterns
        limit = 64
        tiles_to_sim = []
        for p in range(num_tiles_p):
            for q in range(num_tiles_q):
                tiles_to_sim.append((p, q))
                if len(tiles_to_sim) >= limit: break
            if len(tiles_to_sim) >= limit: break
            
        full_trace = []
        HW = tile_h_in * tile_w_in
        
        for (p_idx, q_idx) in tiles_to_sim:
            # Offset for distinct tiles (Packed assumption)
            start_offset = (p_idx * num_tiles_q + q_idx) * HW * workload.C
            
            tile_accesses = tracer.get_trace(0)
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
        total_tiles = ((workload.P + p_l2 - 1) // p_l2) * \
                      ((workload.Q + q_l2 - 1) // q_l2) * \
                      ((workload.C + c_l2 - 1) // c_l2)
                      
        predicted_cost = total_tiles * (1 + (avg_cost - 1) * 1.0)
        
        return predicted_cost

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
            
            if bank == 0: # Input
                if current_row != row:
                    gt_activations += 1
                    current_row = row
        return gt_activations

    def run_optimization_sweep(self):
        # Workload: Conv3_x (28x28, 128)
        workload = ConvWorkload(
            R=3, S=3, P=28, Q=28, C=128, K=64, N=1,
            stride=(1, 1), dilation=(1, 1)
        )
        
        # Search Space
        # P, Q from 4 to 28, step 2
        p_range = range(4, 29, 2)
        q_range = range(4, 29, 2)
        c_tile = 32 # Fixed
        block_size = 32 # Fixed
        
        combinations = []
        for p in p_range:
            for q in q_range:
                combinations.append((p, q))
                
        print(f"Sweeping {len(combinations)} configurations for Conv3_x...")
        
        for p_tile, q_tile in tqdm(combinations):
            # ILP
            ilp_cost = self.calculate_ilp_cost(workload, p_tile, q_tile, c_tile, block_size, block_size)
            
            # GT
            gt_cost = self.calculate_gt_cost(workload, p_tile, q_tile, c_tile)
            
            self.results.append({
                'P_Tile': p_tile,
                'Q_Tile': q_tile,
                'ILP_Cost': ilp_cost,
                'GT_Cost': gt_cost
            })
            
    def analyze(self):
        df = pd.DataFrame(self.results)
        if df.empty: return
        
        print("\n=== Optimization Verification Results ===")
        
        # 1. Rank Correlation
        spearman_corr = df['ILP_Cost'].corr(df['GT_Cost'], method='spearman')
        pearson_corr = df['ILP_Cost'].corr(df['GT_Cost'], method='pearson')
        print(f"Spearman Rank Correlation: {spearman_corr:.4f}")
        print(f"Pearson Correlation:       {pearson_corr:.4f}")
        
        # 2. Top-1 Analysis
        # Sort by ILP
        df_ilp_sorted = df.sort_values('ILP_Cost')
        best_ilp_cfg = df_ilp_sorted.iloc[0]
        
        # Sort by GT
        df_gt_sorted = df.sort_values('GT_Cost')
        best_gt_cfg = df_gt_sorted.iloc[0]
        
        print("\n--- Best Configurations ---")
        print(f"ILP Best: P={best_ilp_cfg['P_Tile']}, Q={best_ilp_cfg['Q_Tile']} -> GT Cost: {best_ilp_cfg['GT_Cost']}")
        print(f"GT Best:  P={best_gt_cfg['P_Tile']}, Q={best_gt_cfg['Q_Tile']} -> GT Cost: {best_gt_cfg['GT_Cost']}")
        
        # 3. Regret
        regret = (best_ilp_cfg['GT_Cost'] - best_gt_cfg['GT_Cost']) / best_gt_cfg['GT_Cost']
        print(f"\nOptimization Regret: {regret:.2%}")
        if regret < 0.01:
            print("SUCCESS: ILP found the optimal (or near-optimal) mapping!")
        else:
            print("WARNING: ILP missed the optimal mapping.")
            
        # 4. Top-5 Overlap
        top_k = 5
        top_ilp_indices = set(df_ilp_sorted.head(top_k).index)
        top_gt_indices = set(df_gt_sorted.head(top_k).index)
        overlap = len(top_ilp_indices.intersection(top_gt_indices))
        print(f"\nTop-{top_k} Overlap: {overlap}/{top_k}")
        
        df.to_csv('optimization_results.csv', index=False)
        print("Results saved to optimization_results.csv")

if __name__ == "__main__":
    verifier = OptimizationVerifier()
    verifier.run_optimization_sweep()
    verifier.analyze()
