import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass
class MicroTraceConfig:
    """Configuration for a single tile micro-trace."""
    tile_h: int
    tile_w: int
    tile_c: int = 1 # Added C dimension
    
    tensor_width: int = 0 # Total width (W)
    tensor_height: int = 0 # Total height (H)
    tensor_channels: int = 0 # Total channels (C)
    
    element_size: int = 1
    row_buffer_size: int = 1024
    loop_order: List[str] = None  # e.g., ['C', 'H', 'W']
    
    # Layout Configuration
    layout_type: str = 'linear' # 'linear', 'tiled', 'row_aligned'
    channel_last: bool = False # If True, use HWC layout (h*W*C + w*C + c)
    block_h: int = 1 
    block_w: int = 1 

class MicroTraceGenerator:
    """
    Simulates memory accesses for a SINGLE tile to determine Row Activation Cost.
    Captures the interaction between Loop Order, Data Layout, and Tile Alignment.
    """
    def __init__(self, config: MicroTraceConfig):
        self.config = config
        self.row_buffer_size = config.row_buffer_size
        self.element_size = config.element_size
        # Default loop order: Channel-Row-Major (C, H, W)
        self.loop_order = config.loop_order if config.loop_order else ['C', 'H', 'W']
        
        # Precompute strides for row_aligned layout
        if self.config.layout_type == 'row_aligned':
            self._init_row_aligned_strides()

    def _init_row_aligned_strides(self):
        """Calculate strides for row_aligned layout matching TraceGenerator."""
        # In row_aligned layout:
        # 1. Data is organized in blocks of (block_h, block_w)
        # 2. Each block is stored contiguously
        # 3. Blocks are ordered by (N, C, H_blk, W_blk)
        # 4. Strides between C/N are aligned to row boundaries
        
        # Block size in bytes
        self.block_size_bytes = self.config.block_h * self.config.block_w * self.element_size
        
        # Stride for C (between channels)
        # In TraceGenerator, C stride is padded to row_buffer_size if it's a "Level 3" dim.
        # Here we assume the layout is fixed:
        # [N][C][H_blk][W_blk][h_in][w_in]
        
        # Number of blocks in W and H
        self.num_blk_w = (self.config.tensor_width + self.config.block_w - 1) // self.config.block_w
        self.num_blk_h = (self.config.tensor_height + self.config.block_h - 1) // self.config.block_h
        
        # Size of one channel plane (in blocks)
        self.channel_size_blocks = self.num_blk_h * self.num_blk_w
        
        # Stride between blocks (usually just block_size)
        # But if we want to match "row_aligned" where C starts at new row?
        # TraceGenerator logic:
        # "For Input, row_aligned means each (N, C) slice is padded to start at a row boundary."
        
        # So, stride_c should be a multiple of row_buffer_size
        raw_c_stride = self.channel_size_blocks * self.block_size_bytes
        self.stride_c_bytes = ((raw_c_stride + self.row_buffer_size - 1) // self.row_buffer_size) * self.row_buffer_size

    def _get_address(self, c_abs: int, h_abs: int, w_abs: int) -> int:
        """Calculate linear byte address based on layout."""
        if self.config.layout_type == 'linear':
            if self.config.channel_last:
                # HWC: h * W * C + w * C + c
                addr_idx = (h_abs * self.config.tensor_width * self.config.tensor_channels +
                           w_abs * self.config.tensor_channels + c_abs)
            else:
                # CHW: C * H * W + h * W + w
                addr_idx = (c_abs * self.config.tensor_height * self.config.tensor_width + 
                           h_abs * self.config.tensor_width + w_abs)
            return addr_idx * self.element_size
            
        elif self.config.layout_type == 'tiled':
            # Simple Tiled: Block-Major
            # We assume Tiled layout is also affected by channel_last
            
            blk_h = h_abs // self.config.block_h
            blk_w = w_abs // self.config.block_w
            in_blk_h = h_abs % self.config.block_h
            in_blk_w = w_abs % self.config.block_w
            
            num_blk_w = (self.config.tensor_width + self.config.block_w - 1) // self.config.block_w
            num_blk_h = (self.config.tensor_height + self.config.block_h - 1) // self.config.block_h
            blk_id = blk_h * num_blk_w + blk_w
            
            if self.config.channel_last:
                # Tiled HWC: Blocks of (H,W) with C interleaved inside?
                # Or Blocks of (H,W,C)?
                # Usually "Tiled" means we store a block of HxW contiguously.
                # If Channel Last, maybe we store Block(H,W) x C?
                # Let's assume "Tiled" means we reorder H,W into blocks, but C is still inner/outer.
                
                # If Channel Last (HWC):
                # Outer: Blocks (H,W)
                # Inner: Block Content.
                # Content: For each pixel in block, all C?
                # Or For each C, whole block?
                # Standard HWC Tiled: Block(h,w) -> Pixel(h,w) -> C
                
                blk_size = self.config.block_h * self.config.block_w * self.config.tensor_channels
                in_blk_offset = (in_blk_h * self.config.block_w + in_blk_w) * self.config.tensor_channels + c_abs
                
                addr_idx = blk_id * blk_size + in_blk_offset
            else:
                # Tiled CHW (Planar Tiled)
                # C is outer. Inside each C, we have H,W tiled.
                blk_size = self.config.block_h * self.config.block_w
                channel_size = num_blk_h * num_blk_w * blk_size
                c_offset = c_abs * channel_size
                
                in_blk_offset = in_blk_h * self.config.block_w + in_blk_w
                
                addr_idx = c_offset + blk_id * blk_size + in_blk_offset
            return addr_idx * self.element_size
            
        elif self.config.layout_type == 'row_aligned':
            # Row Aligned: Blocks + C-padding
            blk_h = h_abs // self.config.block_h
            blk_w = w_abs // self.config.block_w
            in_blk_h = h_abs % self.config.block_h
            in_blk_w = w_abs % self.config.block_w
            
            # Block Index within Channel
            blk_idx_in_channel = blk_h * self.num_blk_w + blk_w
            
            # Address = C_Base + Block_Base + In_Block_Offset
            c_base = c_abs * self.stride_c_bytes
            block_base = blk_idx_in_channel * self.block_size_bytes
            in_blk_offset = (in_blk_h * self.config.block_w + in_blk_w) * self.element_size
            
            return c_base + block_base + in_blk_offset
            
        else:
            raise ValueError(f"Unknown layout type: {self.config.layout_type}")

    def get_trace(self, start_offset: int) -> List[int]:
        """
        Get the sequence of byte addresses for the tile starting at `start_offset`.
        """
        # Decode start_offset
        H = self.config.tensor_height
        W = self.config.tensor_width
        HW = H * W
        
        c_start = start_offset // HW
        rem = start_offset % HW
        h_start = rem // W
        w_start = rem % W
        
        accesses = []
        
        # Create ranges
        ranges = {
            'C': range(self.config.tile_c),
            'H': range(self.config.tile_h),
            'W': range(self.config.tile_w)
        }
        
        def iterate(loop_idx, current_coords):
            if loop_idx == len(self.loop_order):
                c = current_coords['C']
                h = current_coords['H']
                w = current_coords['W']
                
                c_abs = c_start + c
                h_abs = h_start + h
                w_abs = w_start + w
                
                addr_byte = self._get_address(c_abs, h_abs, w_abs)
                accesses.append(addr_byte)
                return

            dim = self.loop_order[loop_idx]
            for i in ranges[dim]:
                current_coords[dim] = i
                iterate(loop_idx + 1, current_coords)
                
        iterate(0, {})
        return accesses

    def simulate(self, start_offset: int, strict_order: bool = True) -> int:
        """
        Simulate the tile access starting at `start_offset`.
        
        Args:
            start_offset: Encoded coordinate `c * (H*W) + h * W + w`
        """
        # Decode start_offset
        H = self.config.tensor_height
        W = self.config.tensor_width
        HW = H * W
        
        c_start = start_offset // HW
        rem = start_offset % HW
        h_start = rem // W
        w_start = rem % W
        
        # 1. Generate all accesses in strict Loop Order
        accesses = []
        
        # Create ranges
        ranges = {
            'C': range(self.config.tile_c),
            'H': range(self.config.tile_h),
            'W': range(self.config.tile_w)
        }
        
        def iterate(loop_idx, current_coords):
            if loop_idx == len(self.loop_order):
                c = current_coords['C']
                h = current_coords['H']
                w = current_coords['W']
                
                c_abs = c_start + c
                h_abs = h_start + h
                w_abs = w_start + w
                
                addr_byte = self._get_address(c_abs, h_abs, w_abs)
                row_id = addr_byte // self.row_buffer_size
                accesses.append(row_id)
                return

            dim = self.loop_order[loop_idx]
            for i in ranges[dim]:
                current_coords[dim] = i
                iterate(loop_idx + 1, current_coords)
                
        iterate(0, {})
        
        # 2. Optimize Order (Block-Major) if requested
        if not strict_order:
            accesses.sort()
        
        # 3. Count Row Activations
        active_row = -1
        row_activations = 0
        
        for row_id in accesses:
            if row_id != active_row:
                row_activations += 1
                active_row = row_id
                
        return row_activations
        row_activations = 0
        
        for row_id in accesses:
            if row_id != active_row:
                row_activations += 1
                active_row = row_id
                
        return row_activations

class HybridCostModel:
    """
    Hybrid Analytical/Simulation model for Input Row Activation Cost.
    
    1. Quantity: Uses Geometric Series (GCD) to count Exact Crossing/Non-Crossing tiles.
    2. Cost: Uses Micro-Trace to estimate Unit Cost for Crossing/Non-Crossing tiles.
    """
    def __init__(self, 
                 H_total: int, 
                 W_total: int, 
                 tile_h: int, 
                 tile_w: int, 
                 stride_h: int,
                 element_size: int = 1,
                 row_buffer_size: int = 1024):
        self.H_total = H_total
        self.W_total = W_total
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.stride_h = stride_h
        self.element_size = element_size
        self.row_buffer_size = row_buffer_size
        
        # Derived parameters
        self.elements_per_row = row_buffer_size // element_size
        self.period = self.elements_per_row
        # Step size in elements (how much the start address moves per vertical step)
        self.step = (W_total * stride_h) % self.period

    def compute_expected_cost(self) -> float:
        """
        Compute the expected (average) cost per tile by sampling valid offsets.
        """
        # 1. Determine the Period and Step
        P = self.elements_per_row
        S = self.step
        
        # 2. Calculate GCD and the number of unique offsets
        g = math.gcd(S, P)
        num_unique_offsets = P // g
        
        # 3. Select offsets to simulate
        # We sample from the set {0, g, 2g, ..., (N-1)g}
        offsets_to_sim = []
        # Increase threshold to 1000 to ensure we cover most cases exhaustively
        if num_unique_offsets <= 1000:
            offsets_to_sim = [k * g for k in range(num_unique_offsets)]
        else:
            # Sample 100 offsets evenly spaced
            step_size = num_unique_offsets // 100
            offsets_to_sim = [k * step_size * g for k in range(100)]
            
        # 4. Run Micro-Trace for each offset
        total_weighted_cost = 0
        total_weight = 0
        
        tracer = MicroTraceGenerator(MicroTraceConfig(
            tile_h=self.tile_h,
            tile_w=self.tile_w,
            tensor_width=self.W_total,
            element_size=self.element_size,
            row_buffer_size=self.row_buffer_size
        ))
        
        for offset in offsets_to_sim:
            # Use strict_order=True to capture realistic thrashing costs
            cost = tracer.simulate(start_offset=offset, strict_order=True)
            total_weighted_cost += cost
            total_weight += 1
            
        return total_weighted_cost / total_weight if total_weight > 0 else 0

def main():
    # Example Usage
    H_total = 224
    W_total = 224
    tile_h = 3
    tile_w = 3
    stride = 1
    
    model = HybridCostModel(H_total, W_total, tile_h, tile_w, stride)
    avg_cost = model.compute_expected_cost()
    
    print(f"Average Row Acts per Tile: {avg_cost:.4f}")

if __name__ == "__main__":
    main()
