
import sys
import os
import gurobipy as gp
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer.mapping import Mapping
from pim_optimizer.model.row_activation import compute_dram_row_crossing_count

def debug_ilp_medium_calc():
    # 1. Parameters
    tile_bytes = 576
    row_bytes = 1024
    num_tiles = 16
    
    # 2. GCD Calculation
    crossing = compute_dram_row_crossing_count(tile_bytes, row_bytes, num_tiles)
    print(f"Tile={tile_bytes}, Row={row_bytes}, Num={num_tiles}")
    print(f"Computed Crossing Count: {crossing}")
    print(f"Computed Non-Crossing Count: {num_tiles - crossing}")
    
    # 3. Formula Check
    # Reuse = 2 (Q=2)
    # Outer = 14 (P=14)
    reuse = 2
    outer = 14
    
    # Case A: Reuse Penalty Applied
    # Cost = (NC * 1 + C * 2 * Reuse) * Outer
    cost_a = ((num_tiles - crossing) * 1 + crossing * 2 * reuse) * outer
    print(f"Prediction A (Reuse Penalty): {cost_a}")
    
    # Case B: No Reuse Penalty (Reuse=1 assumption)
    # Cost = (NC * 1 + C * 1) * Outer = Num * Outer
    cost_b = num_tiles * outer
    print(f"Prediction B (No Reuse Penalty): {cost_b}")
    
    # Case C: ILP Result Reverse Engineer
    # ILP = 126. Outer = 14.
    # Base = 9.
    # How to get 9 from 16 tiles?
    # Maybe NumTiles is not 16?
    # Weight Size = 9216.
    # If Tile = 1024 (Row Size)? 9216/1024 = 9.
    # Did the ILP select a different Tile Size?
    
    print("\n--- Reverse Engineering 126 ---")
    print(f"Target ILP Cost: 126")
    print(f"Implied Base Acts (div 14): {126/14}")
    
    # Check if Tile Size could be different
    # L2 Factors: K=8. L0+L1 K=2. Total K=16.
    # C=4. R=3. S=3.
    # 16*4*3*3 = 576.
    # Is it possible L2 C factor is different?
    # Analysis says: (L2, C): 9 ??
    # Wait, let's check analysis.txt again.
    
if __name__ == "__main__":
    debug_ilp_medium_calc()
