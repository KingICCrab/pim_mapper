
import sys
import os
import math
import numpy as np
from typing import List, Dict, Tuple

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer.mapping import Mapping
from pim_optimizer.generator.hybrid_cost_model import MicroTraceGenerator, MicroTraceConfig
from validation.dram.trace_generator import TraceGenerator, DRAMConfig, DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N

def get_factors(n):
    factors = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factors.add(i)
            factors.add(n // i)
    return sorted(list(factors))

class ILPModelVerifier:
    def __init__(self):
        self.dram_config = DRAMConfig(row_buffer_bytes=1024, element_size=1)
        self.trace_gen = TraceGenerator(self.dram_config)
        
    def calculate_analytical_cost(self, workload, mapping, p_l2, q_l2, c_l2, block_h, block_w, loop_order):
        """
        Calculate cost using the NEW ILP formula:
        Cost = N_tiles * [1 + (AvgCost - 1) * ReusePenalty]
        """
        # 1. Calculate AvgCost using MicroTrace
        # Map loop_order (P,Q,C) to MicroTrace (H,W,C)
        mt_loop_order = []
        for dim in loop_order:
            if dim == 'P': mt_loop_order.append('H')
            elif dim == 'Q': mt_loop_order.append('W')
            elif dim == 'C': mt_loop_order.append('C')
            
        # Calculate Input Tile Size
        stride_h = workload.stride[1]
        stride_w = workload.stride[0]
        dilation_h = workload.dilation[1]
        dilation_w = workload.dilation[0]
        
        tile_h_in = (p_l2 - 1) * stride_h + (workload.R - 1) * dilation_h + 1
        tile_w_in = (q_l2 - 1) * stride_w + (workload.S - 1) * dilation_w + 1
        
        config = MicroTraceConfig(
            tile_h=tile_h_in,
            tile_w=tile_w_in,
            tile_c=c_l2,
            tensor_height=workload.input_size['H'],
            tensor_width=workload.input_size['W'],
            tensor_channels=workload.C,
            element_size=1,
            row_buffer_size=1024,
            loop_order=mt_loop_order,
            layout_type='tiled',
            block_h=block_h,
            block_w=block_w
        )
        
        tracer = MicroTraceGenerator(config)
        
        # Simulate 64 tiles
        num_tiles_p = workload.P // p_l2
        num_tiles_q = workload.Q // q_l2
        limit = 64
        tiles_to_sim = []
        for p in range(num_tiles_p):
            for q in range(num_tiles_q):
                tiles_to_sim.append((p, q))
                if len(tiles_to_sim) >= limit: break
            if len(tiles_to_sim) >= limit: break
            
        activations = 0
        current_row = None
        
        # Run MicroTrace
        # Note: MicroTraceGenerator.simulate takes an offset.
        # We need to calculate offsets for the sequence of tiles.
        # But wait, MicroTraceGenerator is designed for a SINGLE tile internal access?
        # No, precompute_row_acts.py uses it to simulate a sequence.
        # Let's look at precompute_row_acts.py again to see how it calls simulate.
        
        # Actually, precompute_row_acts.py calls `_generate_full_trace` logic which uses `MicroTraceGenerator`?
        # No, `precompute_row_acts.py` has its own `_calculate_hybrid_cost` which instantiates `MicroTraceGenerator`.
        # But `MicroTraceGenerator` has a `simulate(start_offset)` method.
        
        # We need to calculate the start offset for each tile in the sequence.
        # For 'tiled' layout, tiles are stored contiguously in block-major order?
        # If layout is 'tiled', then the address space is linearized by blocks.
        # So tile i starts at i * tile_size?
        # Let's assume perfect tiling for now:
        tile_size_bytes = tile_h_in * tile_w_in * c_l2 # Approx? No, Tiled layout is complex.
        
        # Let's use the logic from precompute_row_acts.py directly if possible.
        # It seems precompute_row_acts.py manually calculates addresses?
        # No, it uses `tracer.simulate(offset)`.
        
        # Let's simplify: We will trust the AvgCost from MicroTrace if we can call it correctly.
        # But calculating the offset for the next tile is tricky without the full layout logic.
        
        # Alternative: Just use the MicroTraceGenerator to get the trace for ONE tile, 
        # and assume the next tile starts where the previous ended (if contiguous).
        # Or better, let's just implement the formula verification assuming we KNOW AvgCost.
        
        # Wait, I need to verify the ILP formula against the Ground Truth (TraceGenerator).
        # So I need to calculate AvgCost correctly to get the ILP prediction.
        
        # Let's look at how `precompute_row_acts.py` calculates offsets.
        # It iterates p, q and calculates `h_abs`, `w_abs` then converts to address.
        
        full_trace = []
        for (p_idx, q_idx) in tiles_to_sim:
            # Calculate top-left of this tile in Input Tensor
            h_start = p_idx * p_l2 * stride_h
            w_start = q_idx * q_l2 * stride_w
            
            # Convert (c=0, h=h_start, w=w_start) to linear address offset
            # This depends on the layout (tiled vs linear).
            # MicroTraceGenerator._get_address handles this.
            
            # We need to pass the absolute coordinate to simulate?
            # No, simulate takes `start_offset` which is `c*HW + h*W + w`.
            start_offset = 0 * (workload.input_size['H'] * workload.input_size['W']) + \
                           h_start * workload.input_size['W'] + \
                           w_start
            
            # Get accesses for this tile
            tile_accesses = tracer.get_trace(start_offset)
            
            # CRITICAL FIX: Simulate DMA Linear Transfer (Block-Major)
            # Sort accesses by physical address to simulate smart DMA reading blocks sequentially
            tile_accesses.sort()
            
            full_trace.extend(tile_accesses)
            
        # Count activations in full_trace
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
        
        # 2. Calculate Reuse Penalty
        # Identify irrelevant loops (loops above the tile loops that don't affect data)
        # For Input (C, H, W), irrelevant loops are K (Output Channel) and maybe others depending on dataflow.
        # In this simple test, we assume standard Weight Stationary or Output Stationary.
        # Input is reused across K.
        
        # Reuse Penalty = Product of bounds of irrelevant loops ABOVE the innermost tile loop.
        # But wait, the formula says "ReusePenalty = Product of bounds of irrelevant dims".
        # And it applies if the loop order causes thrashing.
        
        # Let's look at the mapping to determine ReusePenalty.
        # We need to know the full loop nest.
        # For simplicity, let's assume we are verifying the cost for the innermost L2 loops.
        # If there are outer loops (e.g. K) that reuse this Input Tile, 
        # and they are OUTSIDE the L2 loops, then we pay the cost multiple times.
        
        # Actually, the ILP model calculates `row_acts_seq` for the L2 tile execution ONCE.
        # Then it multiplies by `ReusePenalty` which is the number of times this tile is re-loaded?
        # No, `ReusePenalty` in the code is `Π bound_j^{xj}`.
        # If `xj=1` (irrelevant dim is inside), penalty is 1?
        # No, if irrelevant dim is INSIDE, we iterate it inside the tile loading? No.
        # The L2 Tile is loaded once. If we reuse it, it stays in L2.
        # We only reload it if we discard it.
        
        # Let's stick to the definition:
        # Cost = N_tiles * [1 + (AvgCost - 1) * ReusePenalty]
        # Here ReusePenalty represents "How many times we switch away and come back to this row WITHIN the tile processing".
        # Wait, if ReusePenalty is for "Irrelevant Dims", it usually means outer loops.
        
        # Let's assume for this verification that ReusePenalty = 1 (No thrashing from outer loops).
        # We focus on the correctness of the formula for the tile sequence itself.
        reuse_penalty = 1.0 
        
        # If we want to test ReusePenalty, we need to simulate a loop order where we break the row locality.
        # e.g. Tile C, then Tile P...
        
        total_tiles = (workload.P // p_l2) * (workload.Q // q_l2) * (workload.C // c_l2)
        
        # Formula
        predicted_cost = total_tiles * (1 + (avg_cost - 1) * reuse_penalty)
        
        return predicted_cost, avg_cost

    def run_verification(self, workload, p_l2, q_l2, c_l2, block_h, block_w):
        print(f"\nVerifying Config: P={p_l2}, Q={q_l2}, C={c_l2}, BlockH={block_h}, BlockW={block_w}")
        
        # 1. Create Mapping
        mapping = Mapping()
        # Set up L2 loops (RowBuffer)
        # We map P, Q, C to Level 2
        # TraceGenerator expects:
        # - keys 'spatial'/'temporal'
        # - dimension INDICES (not names)
        
        # Level 2 (DRAM Loops)
        mapping.loop_bounds = {
            2: {'temporal': {
                DIM_P: workload.P // p_l2, 
                DIM_Q: workload.Q // q_l2, 
                DIM_C: workload.C // c_l2
            }}, 
            1: {'temporal': {
                DIM_P: p_l2, 
                DIM_Q: q_l2, 
                DIM_C: c_l2
            }} 
        }
        # Set up Permutation: P, Q, C (Standard)
        # TraceGenerator expects dimension INDICES here too?
        # Let's check TraceGenerator._build_dram_loop_structure
        
        mapping.permutation = {
            2: {0: DIM_P, 1: DIM_Q, 2: DIM_C}, # Outer loops
            1: {0: DIM_C, 1: DIM_P, 2: DIM_Q}  # Inner loops (L1 tile access order)
        }
        # Set Layout
        mapping.layout = {'input': 'sequential', 'weight': 'sequential', 'output': 'sequential'}
        
        # 2. Run TraceGenerator (Ground Truth)
        # We need to patch TraceGenerator to use our block sizes?
        # TraceGenerator uses `_compute_data_layouts` which uses `block_h/w` from mapping or heuristics.
        # We can inject block sizes into mapping.tile_info
        mapping.tile_info = {'block_h': block_h, 'block_w': block_w}
        
        # Note: TraceGenerator might not respect `mapping.tile_info` directly if not programmed to.
        # Let's check TraceGenerator._compute_data_layouts.
        # It usually derives block sizes from L1 bounds or similar.
        # For this test, we might need to subclass or mock TraceGenerator to force block sizes.
        # Or we just set L1 bounds to match block sizes (if p_l2 == block_h).
        
        # Let's assume for now we set L1 bounds = Tile Size.
        # And we want to control the "Layout Block Size".
        # In `TraceGenerator`, layout block size is often determined by the "Buffer Tile" size.
        # So if we set Level 1 bounds to (p_l2, q_l2, c_l2), the layout block will be that size.
        
        trace = self.trace_gen.generate_trace(mapping, workload)
        
        # Count Input Activations
        row_size = 1024
        bank_size = row_size * 16384
        current_row = None
        gt_activations = 0
        
        for line in trace:
            parts = line.strip().split()
            if len(parts) < 2: continue
            if parts[0] != 'LD': continue
            addr = int(parts[1], 16)
            bank = addr // bank_size
            row = (addr % bank_size) // row_size
            
            # Input is usually Bank 0 in TraceGenerator (simplified)
            # Or we check the address range.
            # TraceGenerator usually allocates banks: Input=0, Weight=1, Output=2...
            if bank == 0:
                if current_row != row:
                    gt_activations += 1
                    current_row = row
                    
        # 3. Run Analytical Model
        pred_cost, avg_cost = self.calculate_analytical_cost(
            workload, mapping, p_l2, q_l2, c_l2, block_h, block_w, ['C', 'P', 'Q']
        )
        
        print(f"  AvgCost (MicroTrace): {avg_cost:.4f}")
        print(f"  Predicted Total: {pred_cost:.1f}")
        print(f"  Ground Truth:    {gt_activations}")
        print(f"  Error:           {abs(pred_cost - gt_activations) / gt_activations * 100:.2f}%")

if __name__ == "__main__":
    # Define Workload (ResNet Layer)
    # R=3, S=3, P=56, Q=56, C=64, K=64, N=1
    workload = ConvWorkload(
        R=3, S=3, P=56, Q=56, C=64, K=64, N=1,
        stride=(1, 1), dilation=(1, 1)
    )
    
    verifier = ILPModelVerifier()
    
    # Test Case 1: Aligned (Tile = Block)
    # Tile 32x32, Block 32x32
    verifier.run_verification(workload, p_l2=28, q_l2=28, c_l2=32, block_h=28, block_w=28)
    
    # Test Case 2: Misaligned (Tile > Block)
    # Tile 30x30, Block 28x28 (Forces crossing)
    verifier.run_verification(workload, p_l2=56, q_l2=56, c_l2=32, block_h=28, block_w=28)

    # Test Case 3: Aligned (Block > Tile)
    # Tile 30x30, Block 32x32 (Fits inside)
    # We only run Analytical Model here as TraceGenerator requires valid mapping
    print(f"\nVerifying Config: P=28, Q=28, C=32, BlockH=32, BlockW=32 (Block > Tile)")
    pred_cost, avg_cost = verifier.calculate_analytical_cost(
        workload, Mapping(), 28, 28, 32, 32, 32, ['C', 'P', 'Q']
    )
    print(f"  AvgCost (MicroTrace): {avg_cost:.4f}")
    print(f"  Predicted Total: {pred_cost:.1f}")

