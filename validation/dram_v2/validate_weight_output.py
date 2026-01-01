
import math
import json
import os
import logging
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer.mapping import Mapping
from validation.dram.trace_generator import TraceGenerator, DRAMConfig, DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N
from pim_optimizer.generator.hybrid_cost_model import MicroTraceGenerator, MicroTraceConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeightOutputValidator:
    """
    Validates Row Activation Cost Model for Weight and Output tensors.
    Compares ILP model predictions against TraceGenerator (Ground Truth).
    """
    
    def __init__(self):
        self.dram_config = DRAMConfig(row_buffer_bytes=1024, element_size=1)
        self.trace_gen = TraceGenerator(self.dram_config)
        
    def calculate_ilp_cost_weight(self, workload, k_l2, c_l2, r_l2, s_l2):
        """
        Calculate ILP predicted cost for Weight tensor.
        Weight is usually packed and sequential.
        Cost = TotalBytes / RowSize (if aligned) or similar.
        
        In ILP model (row_activation.py):
        Weight is usually treated as sequential.
        If packed: Cost = NumTiles * (1 + (AvgCost - 1) * Reuse)
        For Weight, usually Reuse=1 (read once per tile).
        
        Let's use the MicroTrace logic to be precise, assuming Packed layout.
        """
        # Weight Tile Size
        tile_size = k_l2 * c_l2 * r_l2 * s_l2
        
        # If tile_size >= row_size, it's just streaming.
        # If tile_size < row_size, we might have multiple tiles per row.
        
        # Total Weight Size
        total_weight = workload.K * workload.C * workload.R * workload.S
        
        # Number of tiles
        num_tiles = (total_weight + tile_size - 1) // tile_size
        
        # For Weight, we assume Packed layout in DRAM.
        # So we can use the simple formula:
        # Cost = ceil(TotalBytes / RowSize) + overhead?
        
        # Let's use MicroTrace to be consistent with Input validation.
        # Weight dimensions: K, C, R, S
        # We can map them to H, W, C for MicroTrace.
        # e.g. H=K, W=C*R*S (flattened)
        
        config = MicroTraceConfig(
            tile_h=k_l2,
            tile_w=c_l2 * r_l2 * s_l2,
            tile_c=1,
            tensor_height=k_l2, # Packed
            tensor_width=c_l2 * r_l2 * s_l2, # Packed
            tensor_channels=1,
            element_size=1,
            row_buffer_size=1024,
            loop_order=['C', 'H', 'W'], # Must include all dims
            layout_type='linear'
        )
        
        tracer = MicroTraceGenerator(config)
        
        # Simulate a sequence of tiles to capture inter-tile locality
        # Just like precompute_row_acts.py does for Input
        num_sim_tiles = 16
        total_acts = 0
        
        # Generate trace for multiple sequential tiles
        full_trace = []
        tile_size_elements = k_l2 * c_l2 * r_l2 * s_l2
        
        for i in range(num_sim_tiles):
            start_offset = i * tile_size_elements
            # Get trace for this tile
            # Note: get_trace returns byte addresses relative to start_offset?
            # No, get_trace takes start_offset and returns absolute addresses
            tile_trace = tracer.get_trace(start_offset)
            
            # Sort for DMA linear transfer (Packed)
            tile_trace.sort()
            full_trace.extend(tile_trace)
            
        # Count activations for the full sequence
        row_size = 1024
        current_row = None
        
        for addr in full_trace:
            row = addr // row_size
            if current_row != row:
                total_acts += 1
                current_row = row
                
        avg_cost = total_acts / num_sim_tiles
        
        predicted_cost = num_tiles * avg_cost
        return predicted_cost

    def calculate_ilp_cost_output(self, workload, n_l2, k_l2, p_l2, q_l2):
        """
        Calculate ILP predicted cost for Output tensor.
        Output is usually packed and sequential.
        """
        # Output Tile Size
        tile_size = n_l2 * k_l2 * p_l2 * q_l2
        
        # Total Output Size
        total_output = workload.N * workload.K * workload.P * workload.Q
        
        # Number of tiles
        num_tiles = (total_output + tile_size - 1) // tile_size
        
        # MicroTrace for Output
        # Map N, K, P, Q -> H, W
        # e.g. H=N*K, W=P*Q
        
        config = MicroTraceConfig(
            tile_h=n_l2 * k_l2,
            tile_w=p_l2 * q_l2,
            tile_c=1,
            tensor_height=n_l2 * k_l2, # Packed
            tensor_width=p_l2 * q_l2, # Packed
            tensor_channels=1,
            element_size=1,
            row_buffer_size=1024,
            loop_order=['C', 'H', 'W'], # Must include all dims
            layout_type='linear'
        )
        
        tracer = MicroTraceGenerator(config)
        
        # Simulate a sequence of tiles to capture inter-tile locality
        num_sim_tiles = 16
        total_acts = 0
        
        full_trace = []
        tile_size_elements = n_l2 * k_l2 * p_l2 * q_l2
        
        for i in range(num_sim_tiles):
            start_offset = i * tile_size_elements
            tile_trace = tracer.get_trace(start_offset)
            tile_trace.sort()
            full_trace.extend(tile_trace)
            
        row_size = 1024
        current_row = None
        
        for addr in full_trace:
            row = addr // row_size
            if current_row != row:
                total_acts += 1
                current_row = row
                
        avg_cost = total_acts / num_sim_tiles
        
        predicted_cost = num_tiles * avg_cost
        return predicted_cost

    def calculate_gt_cost(self, workload, tile_sizes, tensor_type='weight'):
        """
        Calculate Ground Truth cost using TraceGenerator.
        """
        mapping = Mapping()
        
        # Set up Mapping based on tensor type
        if tensor_type == 'weight':
            # Weight: K, C, R, S
            k_l2, c_l2, r_l2, s_l2 = tile_sizes
            
            # We want to iterate over Weight Tiles.
            # To ensure no reuse (read once), we put Weight loops at the outer level?
            # Or just have 1 iteration of other loops.
            # Let's set P=1, Q=1, N=1 to minimize other loops.
            # But workload has fixed P, Q, N.
            # We can set the mapping such that we iterate K, C, R, S once.
            
            # Actually, TraceGenerator generates trace based on the mapping.
            # If we want to validate Weight reading cost, we should ensure
            # we read the whole Weight tensor exactly once.
            # This happens if K, C, R, S loops are the only ones, or if they are innermost
            # and we only run one iteration of outer loops.
            
            # Let's define a mapping where we iterate over K, C tiles.
            # And P, Q, N are 1 (or handled such that we don't repeat Weight).
            # Wait, if P, Q > 1, and we have Output Stationary, Weight is read P*Q times.
            # To validate "Unit Cost", we want to read it ONCE.
            # So we can set P=1, Q=1, N=1 in the Workload for this test?
            # Yes, let's modify the workload temporarily for GT calculation.
            
            workload_mod = ConvWorkload(
                C=workload.C, K=workload.K, 
                R=workload.R, S=workload.S, P=1, Q=1, stride=workload.stride, dilation=workload.dilation, N=1
            )
            
            mapping.loop_bounds = {
                2: {'temporal': {
                    DIM_K: (workload.K + k_l2 - 1) // k_l2,
                    DIM_C: (workload.C + c_l2 - 1) // c_l2,
                    DIM_R: (workload.R + r_l2 - 1) // r_l2,
                    DIM_S: (workload.S + s_l2 - 1) // s_l2
                }},
                1: {'temporal': {
                    DIM_K: k_l2,
                    DIM_C: c_l2,
                    DIM_R: r_l2,
                    DIM_S: s_l2
                }}
            }
            # Simple permutation
            mapping.permutation = {
                2: {0: DIM_K, 1: DIM_C, 2: DIM_R, 3: DIM_S},
                1: {0: DIM_K, 1: DIM_C, 2: DIM_R, 3: DIM_S}
            }
            mapping.layout = {'input': 'sequential', 'weight': 'sequential', 'output': 'sequential'}
            
            trace = self.trace_gen.generate_trace(mapping, workload_mod)
            target_bank = 1 # Weight Bank
            
        elif tensor_type == 'output':
            # Output: N, K, P, Q
            n_l2, k_l2, p_l2, q_l2 = tile_sizes
            
            # Modify workload to read/write Output once.
            # C=1, R=1, S=1
            workload_mod = ConvWorkload(
                C=1, K=workload.K, 
                R=1, S=1, P=workload.P, Q=workload.Q, stride=workload.stride, dilation=workload.dilation, N=workload.N
            )
            
            mapping.loop_bounds = {
                2: {'temporal': {
                    DIM_N: (workload.N + n_l2 - 1) // n_l2,
                    DIM_K: (workload.K + k_l2 - 1) // k_l2,
                    DIM_P: (workload.P + p_l2 - 1) // p_l2,
                    DIM_Q: (workload.Q + q_l2 - 1) // q_l2
                }},
                1: {'temporal': {
                    DIM_N: n_l2,
                    DIM_K: k_l2,
                    DIM_P: p_l2,
                    DIM_Q: q_l2
                }}
            }
            mapping.permutation = {
                2: {0: DIM_N, 1: DIM_K, 2: DIM_P, 3: DIM_Q},
                1: {0: DIM_N, 1: DIM_K, 2: DIM_P, 3: DIM_Q}
            }
            mapping.layout = {'input': 'sequential', 'weight': 'sequential', 'output': 'sequential'}
            
            trace = self.trace_gen.generate_trace(mapping, workload_mod)
            target_bank = 2 # Output Bank
            
        # Count Activations
        row_size = 1024
        bank_size = row_size * 16384
        current_row = None
        activations = 0
        
        for line in trace:
            parts = line.strip().split()
            if len(parts) < 2: continue
            # Count both LD and ST for Output? 
            # Weight is LD only. Output is ST (and LD if partial).
            # TraceGenerator generates ST for Output.
            
            addr = int(parts[1], 16)
            bank = addr // bank_size
            row = (addr % bank_size) // row_size
            
            if bank == target_bank:
                if current_row != row:
                    activations += 1
                    current_row = row
                    
        return activations

    def run_validation(self):
        print("Starting Weight & Output Validation...")
        print(f"{'Tensor':<10} | {'Workload':<10} | {'Tile Size':<20} | {'ILP Cost':<10} | {'GT Cost':<10} | {'Error %':<10}")
        print("-" * 80)
        
        # Define Workloads
        workloads = [
            ("Conv2_x", ConvWorkload(C=64, K=64, R=3, S=3, P=56, Q=56, stride=(1,1), dilation=(1,1), N=1)),
            ("Conv3_x", ConvWorkload(C=128, K=128, R=3, S=3, P=28, Q=28, stride=(1,1), dilation=(1,1), N=1)),
        ]
        
        # 1. Validate Weight
        # Tile sizes: (K, C, R, S)
        weight_tiles = [
            (32, 32, 3, 3),
            (64, 64, 3, 3),
            (16, 64, 3, 3)
        ]
        
        for name, wl in workloads:
            for k, c, r, s in weight_tiles:
                # Ensure tile fits
                k = min(k, wl.K)
                c = min(c, wl.C)
                
                ilp = self.calculate_ilp_cost_weight(wl, k, c, r, s)
                gt = self.calculate_gt_cost(wl, (k, c, r, s), 'weight')
                
                error = abs(ilp - gt) / gt * 100 if gt > 0 else 0
                print(f"{'Weight':<10} | {name:<10} | {f'{k}x{c}x{r}x{s}':<20} | {ilp:<10.1f} | {gt:<10} | {error:<10.2f}")

        print("-" * 80)
        
        # 2. Validate Output
        # Tile sizes: (N, K, P, Q)
        output_tiles = [
            (1, 32, 14, 14),
            (1, 64, 7, 7),
            (1, 16, 28, 28)
        ]
        
        for name, wl in workloads:
            for n, k, p, q in output_tiles:
                # Ensure tile fits
                k = min(k, wl.K)
                p = min(p, wl.P)
                q = min(q, wl.Q)
                
                ilp = self.calculate_ilp_cost_output(wl, n, k, p, q)
                gt = self.calculate_gt_cost(wl, (n, k, p, q), 'output')
                
                error = abs(ilp - gt) / gt * 100 if gt > 0 else 0
                print(f"{'Output':<10} | {name:<10} | {f'{n}x{k}x{p}x{q}':<20} | {ilp:<10.1f} | {gt:<10} | {error:<10.2f}")

if __name__ == "__main__":
    validator = WeightOutputValidator()
    validator.run_validation()
