
import itertools
import math
import json
import os
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, field
import logging

from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer.mapping import Mapping
from validation.dram.trace_generator import TraceGenerator, DRAMConfig, DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N
from pim_optimizer.generator.hybrid_cost_model import MicroTraceGenerator, MicroTraceConfig
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PrecomputeConfig:
    """Configuration for the precomputation run."""
    # Options to sweep for block_h and block_w
    # If None, will generate factors of H/W automatically
    block_h_options: List[int] = None 
    block_w_options: List[int] = None
    
    row_buffer_bytes: int = 1024
    element_size: int = 1
    
    # If True, only explore powers of 2 or specific factors to save time
    fast_mode: bool = True 

class RowActivationPrecomputer:
    """
    Precomputes Row Activation costs for Input Tensor using TraceGenerator.
    Generates a Lookup Table for the ILP optimizer.
    """
    
    def __init__(self, workload: ConvWorkload, config: PrecomputeConfig):
        self.workload = workload
        self.config = config
        self.dram_config = DRAMConfig(
            row_buffer_bytes=config.row_buffer_bytes,
            element_size=config.element_size
        )
        
    def get_factors(self, n: int) -> List[int]:
        """Get all factors of n."""
        factors = set()
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                factors.add(i)
                factors.add(n // i)
        return sorted(list(factors))

    def _count_input_row_acts(self, trace: List[str]) -> int:
        """
        Parse trace and count Row Activations for Input Tensor (Bank 0).
        """
        row_size = self.dram_config.row_buffer_bytes
        bank_size = row_size * self.dram_config.num_rows
        
        current_row = None
        activations = 0
        
        for line in trace:
            parts = line.strip().split()
            if len(parts) < 2: continue
            if parts[0] != 'LD': continue # Only count Loads
            
            addr = int(parts[1], 16)
            bank = addr // bank_size
            row = (addr % bank_size) // row_size
            
            # Input is mapped to Bank 0
            if bank == 0:
                if current_row != row:
                    activations += 1
                    current_row = row
        return activations

    def _calculate_hybrid_cost(self, p_l2: int, q_l2: int, c_l2: int, 
                             block_h: int, block_w: int, 
                             loop_order: List[str]) -> float:
        """
        Calculate Average Unit Row Activation Cost using Hybrid Model (Sampling).
        Replaces full trace simulation with Micro-Trace sampling.
        """
        # 1. Configure MicroTrace
        # Note: loop_order passed here is ['P', 'Q', 'C'] or similar.
        # MicroTrace expects ['C', 'H', 'W'] or similar.
        # We need to map P->H, Q->W.
        # And we need to handle the fact that P, Q are output dimensions, but for Input they map to H, W.
        # Input H = P * stride + ...
        # But here we are simulating the *access pattern* of the tile.
        # The tile iterates over p_l2, q_l2, c_l2.
        # The physical addresses are generated based on these.
        
        # Map loop_order to MicroTrace format
        # e.g. ['C', 'P', 'Q'] -> ['C', 'H', 'W']
        mt_loop_order = []
        for dim in loop_order:
            if dim == 'P': mt_loop_order.append('H')
            elif dim == 'Q': mt_loop_order.append('W')
            elif dim == 'C': mt_loop_order.append('C')
            
        # Calculate Input Tile Size based on Convolution formula
        # H_in = (H_out - 1) * stride + kernel_size
        stride_h = self.workload.stride[1]
        stride_w = self.workload.stride[0]
        dilation_h = self.workload.dilation[1]
        dilation_w = self.workload.dilation[0]
        
        tile_h_in = (p_l2 - 1) * stride_h + (self.workload.R - 1) * dilation_h + 1
        tile_w_in = (q_l2 - 1) * stride_w + (self.workload.S - 1) * dilation_w + 1
            
        config = MicroTraceConfig(
            tile_h=tile_h_in, # Use effective Input Tile Height
            tile_w=tile_w_in, # Use effective Input Tile Width
            tile_c=c_l2,
            tensor_height=self.workload.input_size['H'], # Use Input H/W
            tensor_width=self.workload.input_size['W'],
            tensor_channels=self.workload.C,
            element_size=self.config.element_size,
            row_buffer_size=self.config.row_buffer_bytes,
            loop_order=mt_loop_order,
            layout_type='tiled', # Use 'tiled' (dense) instead of 'row_aligned' (padded) for sequential cost
            block_h=block_h,
            block_w=block_w
        )
        
        tracer = MicroTraceGenerator(config)
        
        # 2. Simulate a sequence of tiles to capture row buffer reuse
        # We assume the execution follows the data layout (best case / optimized).
        # For 'tiled' layout, this means we iterate tiles in block-major order.
        
        num_tiles_p = self.workload.P // p_l2
        num_tiles_q = self.workload.Q // q_l2
        
        # We want to simulate a contiguous sequence of tiles.
        # Let's simulate up to 64 tiles.
        limit = 64
        
        tiles_to_sim = []
        for p in range(num_tiles_p):
            for q in range(num_tiles_q):
                tiles_to_sim.append((p, q))
                if len(tiles_to_sim) >= limit:
                    break
            if len(tiles_to_sim) >= limit:
                break
                
        # Collect full trace
        full_trace = []
        
        stride_h = self.workload.stride[1]
        stride_w = self.workload.stride[0]
        H_in = self.workload.input_size['H']
        W_in = self.workload.input_size['W']
        HW = H_in * W_in
        
        for p_idx, q_idx in tiles_to_sim:
            h_start = p_idx * p_l2 * stride_h
            w_start = q_idx * q_l2 * stride_w
            c_start = 0
            
            start_offset = c_start * HW + h_start * W_in + w_start
            
            # Get trace for this tile
            tile_trace = tracer.get_trace(start_offset)
            full_trace.extend(tile_trace)
            
        # Count activations for the full trace
        row_size = self.config.row_buffer_bytes
        bank_size = row_size * 16384
        
        current_row = None
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

    def _generate_full_trace(self, p_l2: int, q_l2: int, c_l2: int, 
                             block_h: int, block_w: int, 
                             loop_order: List[str]) -> float:
        """
        Legacy method. Replaced by _calculate_hybrid_cost.
        """
        return self._calculate_hybrid_cost(p_l2, q_l2, c_l2, block_h, block_w, loop_order)

    def compute_table(self) -> List[Dict[str, Any]]:
        """
        Main entry point. Iterates search space and returns the lookup table.
        """
        # 1. Define Search Space
        p_factors = self.get_factors(self.workload.P)
        q_factors = self.get_factors(self.workload.Q)
        c_factors = self.get_factors(self.workload.C)
        
        # Block H/W options
        # Ensure block sizes are factors of Input H/W for valid tiling
        H_in = self.workload.input_size['H']
        W_in = self.workload.input_size['W']
        
        if self.config.block_h_options is None:
            bh_opts = self.get_factors(H_in)
        else:
            bh_opts = self.config.block_h_options
            
        if self.config.block_w_options is None:
            bw_opts = self.get_factors(W_in)
        else:
            bw_opts = self.config.block_w_options
            
        # Loop Orders for P, Q, C (L2)
        dims = ['P', 'Q', 'C']
        orders = list(itertools.permutations(dims))
        
        # Calculate total size for progress
        total_combinations = len(p_factors) * len(q_factors) * len(c_factors) * len(orders) * len(bh_opts) * len(bw_opts)
        logger.info(f"Starting precomputation. Max search space size: {total_combinations}")
        
        results = []
        count = 0
        
        for p in p_factors:
            for q in q_factors:
                for c in c_factors:
                    
                    for order_tuple in orders:
                        order_list = list(order_tuple)
                        
                        best_cost = float('inf')
                        best_config = None
                        
                        # Sweep Block Sizes
                        for bh in bh_opts:
                            for bw in bw_opts:
                                # Constraint: Block size <= Row Buffer
                                if bh * bw * self.config.element_size > self.config.row_buffer_bytes:
                                    continue
                                
                                # Run FULL Trace Simulation
                                cost = self._generate_full_trace(p, q, c, bh, bw, order_list)
                                
                                if cost < best_cost:
                                    best_cost = cost
                                    best_config = {'block_h': bh, 'block_w': bw}
                        
                        if best_config:
                            entry = {
                                'P': p, 'Q': q, 'C': c,
                                'order': "-".join(order_list),
                                'row_acts': best_cost,
                                'best_block_h': best_config['block_h'],
                                'best_block_w': best_config['block_w']
                            }
                            results.append(entry)
                        
                        count += 1
                        if count % 100 == 0: # Log more frequently as full trace is slower
                            logger.info(f"Processed {count} combinations...")
                            
        logger.info(f"Precomputation complete. Generated {len(results)} entries.")
        return results

    def save_table(self, table: List[Dict[str, Any]], filepath: str):
        """Save table to JSON."""
        with open(filepath, 'w') as f:
            json.dump(table, f, indent=2)
        logger.info(f"Saved lookup table to {filepath}")

if __name__ == "__main__":
    # Example Usage
    wl = ConvWorkload(name="TestLayer", N=1, C=64, K=64, P=56, Q=56, R=3, S=3, stride=(1,1))
    config = PrecomputeConfig(
        block_h_options=[1, 2, 4, 8],
        block_w_options=[4, 8, 16, 32, 64],
        row_buffer_bytes=1024
    )
    
    precomputer = RowActivationPrecomputer(wl, config)
    table = precomputer.compute_table()
    precomputer.save_table(table, "row_activation_cost_table.json")
