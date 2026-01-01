import sys
import os
import math

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from pim_optimizer.generator.hybrid_cost_model import HybridCostModel, MicroTraceGenerator, MicroTraceConfig

def run_validation(H_total, W_total, tile_h, tile_w, stride_h, element_size=1, row_buffer_size=1024):
    print(f"\n--- Validation: H={H_total}, W={W_total}, Tile={tile_h}x{tile_w}, Stride={stride_h} ---")
    
    # 1. Ground Truth: Exhaustive Simulation
    num_tiles_y = (H_total - tile_h) // stride_h + 1
    total_cost_gt = 0
    
    tracer = MicroTraceGenerator(MicroTraceConfig(
        tile_h=tile_h,
        tile_w=tile_w,
        tensor_width=W_total,
        element_size=element_size,
        row_buffer_size=row_buffer_size
    ))
    
    print(f"Simulating {num_tiles_y} tiles exhaustively...")
    for i in range(num_tiles_y):
        # Calculate exact offset for this tile
        # Address = i * stride_h * W_total
        offset = i * stride_h * W_total
        # Use strict_order=True to match the Hybrid Model's new default
        cost = tracer.simulate(start_offset=offset, strict_order=True)
        total_cost_gt += cost
        
    avg_cost_gt = total_cost_gt / num_tiles_y
    print(f"Ground Truth: Total={total_cost_gt}, Avg={avg_cost_gt:.4f}")
    
    # 2. Hybrid Model Prediction
    model = HybridCostModel(H_total, W_total, tile_h, tile_w, stride_h, element_size, row_buffer_size)
    avg_cost_pred = model.compute_expected_cost()
    total_cost_pred = avg_cost_pred * num_tiles_y
    
    print(f"Hybrid Model: Total={total_cost_pred:.2f}, Avg={avg_cost_pred:.4f}")
    
    # 3. Comparison
    error = abs(total_cost_pred - total_cost_gt)
    error_pct = (error / total_cost_gt) * 100 if total_cost_gt > 0 else 0
    print(f"Error: {error:.2f} ({error_pct:.2f}%)")
    
    return error_pct

def main():
    # Test Case 1: Small aligned case
    # W=1024 (matches row buffer), Stride=1. All offsets should be 0.
    run_validation(H_total=100, W_total=1024, tile_h=3, tile_w=3, stride_h=1)
    
    # Test Case 2: Misaligned case (Prime width)
    # W=100, RowBuffer=1024. Periodicity will be complex.
    run_validation(H_total=224, W_total=224, tile_h=3, tile_w=3, stride_h=1)
    
    # Test Case 3: Large Stride
    run_validation(H_total=224, W_total=224, tile_h=3, tile_w=3, stride_h=2)
    
    # Test Case 4: Large Tile
    run_validation(H_total=224, W_total=224, tile_h=7, tile_w=7, stride_h=2)

if __name__ == "__main__":
    main()
