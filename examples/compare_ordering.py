import sys
import os
from pim_optimizer.generator.hybrid_cost_model import MicroTraceGenerator, MicroTraceConfig

def compare_ordering_impact():
    print("\n=== Impact of Loop Ordering on Crossing Cost ===")
    
    # Configuration: 3x3 Tile, Row Buffer = 1024 bytes
    # We want to create a scenario with "Double Crossing" or at least "W-Crossing".
    # Let's set W_total such that a row boundary cuts through the tile vertically.
    
    # Case: W-Crossing with Tiled Layout
    # This forces Block A and Block B to be far apart in memory, causing thrashing.
    
    tile_h = 3
    tile_w = 3
    element_size = 1
    row_buffer_size = 1024 # Standard buffer
    
    # Tiled Layout: 4x4 Blocks
    block_h = 4
    block_w = 4
    tensor_width = 100 # Width of tensor
    
    tracer = MicroTraceGenerator(MicroTraceConfig(
        tile_h=tile_h,
        tile_w=tile_w,
        tensor_width=tensor_width,
        element_size=element_size,
        row_buffer_size=row_buffer_size,
        layout_type='tiled',
        block_h=block_h,
        block_w=block_w
    ))
    
    # Start at (0, 2).
    # Tile covers w=2,3,4.
    # Block Boundary is at w=4 (since block_w=4, indices 0,1,2,3 are Block 0).
    # So w=2,3 are in Block 0. w=4 is in Block 1.
    # Row 0: (0,2)[B0], (0,3)[B0], (0,4)[B1]
    # Row 1: (1,2)[B0], (1,3)[B0], (1,4)[B1]
    # Row 2: (2,2)[B0], (2,3)[B0], (2,4)[B1]
    
    # Block 0 Address: 0. Block 1 Address: 16 (4x4).
    # If Row Buffer is small (e.g. 8 bytes), B0 and B1 might be in different rows.
    # Let's force them to be in different rows by making Block Size large or Buffer small.
    # Actually, let's just rely on the fact that Block 0 and Block 1 are distinct.
    # If Block 0 is at 0x0000 and Block 1 is at 0x1000 (simulated by large block size?)
    # No, address calculation is `blk_id * blk_size`.
    # If blk_size < row_buffer_size, they might share a row.
    # We need `blk_size >= row_buffer_size` to guarantee they are in different rows.
    
    # Let's make blocks HUGE.
    block_h = 32
    block_w = 32 # 1024 elements = 1024 bytes = 1 Row Buffer exactly.
    # So each Block is exactly 1 DRAM Row.
    
    tracer = MicroTraceGenerator(MicroTraceConfig(
        tile_h=tile_h,
        tile_w=tile_w,
        tensor_width=tensor_width,
        element_size=element_size,
        row_buffer_size=1024,
        layout_type='tiled',
        block_h=32,
        block_w=32
    ))
    
    # Start at (0, 30).
    # w=30, 31 (Block 0). w=32 (Block 1).
    start_offset = 0 * tensor_width + 30
    
    print(f"Scenario: 3x3 Tile crossing a vertical block boundary (Tiled Layout).")
    print(f"Block Size = 32x32 (1024 bytes = 1 Row).")
    print(f"Tile spans Block 0 (Row 0) and Block 1 (Row 1).")
    print(f"Row 0: [B0, B0, B1]")
    print(f"Row 1: [B0, B0, B1]")
    print(f"Row 2: [B0, B0, B1]")
    
    # 1. Optimistic (Block-Major)
    cost_opt = tracer.simulate(start_offset, strict_order=False)
    print(f"\n[Optimistic / Block-Major]")
    print(f"Logic: Visit all B0, then all B1.")
    print(f"Cost: {cost_opt} Row Activations")
    
    # 2. Pessimistic (Row-Major)
    cost_pess = tracer.simulate(start_offset, strict_order=True)
    print(f"\n[Realistic / Row-Major]")
    print(f"Logic: Row0(B0->B1) -> Row1(B0->B1) -> Row2(B0->B1)")
    print(f"Cost: {cost_pess} Row Activations")
    
    diff = cost_pess - cost_opt
    print(f"\nDifference: {diff} extra activations ({diff/cost_opt*100:.1f}% increase)")
    
    if diff > 0:
        print("\nCONCLUSION: Yes, fine-grained handling IS needed if hardware follows strict loop order.")
    else:
        print("\nCONCLUSION: No difference found.")

if __name__ == "__main__":
    compare_ordering_impact()
