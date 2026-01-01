
import math
from dataclasses import dataclass
from typing import Tuple

@dataclass
class MockConfig:
    block_h: int
    tile_h: int
    step_h: int
    num_tiles: int

def compute_block_crossing_1d(block_size, tile_size, step, num_tiles):
    if block_size <= 0 or tile_size <= 0 or step <= 0 or num_tiles <= 0:
        return (0, 0)
    if tile_size > block_size:
        return (num_tiles, num_tiles)
    if tile_size == block_size:
        return (0, num_tiles)
    
    g = math.gcd(step, block_size)
    period = block_size // g
    crossing_in_period = 0
    for k in range(period):
        start_pos = (k * step) % block_size
        if start_pos + tile_size > block_size:
            crossing_in_period += 1
            
    num_complete_periods = num_tiles // period
    remainder = num_tiles % period
    crossing_in_remainder = 0
    for k in range(remainder):
        start_pos = (k * step) % block_size
        if start_pos + tile_size > block_size:
            crossing_in_remainder += 1
            
    total_crossing = num_complete_periods * crossing_in_period + crossing_in_remainder
    return (total_crossing, num_tiles)

def simulate_1d_cost(config: MockConfig):
    # Simulation
    blocks = set()
    crossing_tiles = 0
    
    print(f"\n--- Simulation: Block={config.block_h}, Tile={config.tile_h}, Step={config.step_h}, N={config.num_tiles} ---")
    
    total_acts_sim = 0
    
    for i in range(config.num_tiles):
        start = i * config.step_h
        end = start + config.tile_h
        
        start_block = start // config.block_h
        end_block = (end - 1) // config.block_h
        
        # Track blocks touched
        for b in range(start_block, end_block + 1):
            blocks.add(b)
            
        # Check crossing
        if start_block != end_block:
            crossing_tiles += 1
            # Cost: 2 acts (simplified)
            total_acts_sim += 2
        else:
            # Cost: 1 act (simplified, assuming perfect reuse for non-crossing in same block)
            # But wait, simulation should count unique blocks for non-crossing?
            pass

    # Formula Logic
    # Non-crossing acts = Unique Blocks * 1
    # Crossing acts = Crossing Tiles * 2
    
    # Let's calculate "True Cost" based on the formula's philosophy:
    # 1. Open every touched block once (Baseline)
    # 2. Add penalty for crossing tiles
    
    true_baseline = len(blocks)
    formula_crossing_penalty = crossing_tiles * 2
    
    # But wait, the formula adds (Unique Blocks) + (Crossing * 2).
    # My hypothesis: This double counts.
    # If a block is ONLY touched by crossing tiles, it is counted in Unique Blocks AND in Crossing Penalty.
    
    # Let's see if we can find such a block.
    blocks_with_non_crossing = set()
    for i in range(config.num_tiles):
        start = i * config.step_h
        end = start + config.tile_h
        start_block = start // config.block_h
        end_block = (end - 1) // config.block_h
        
        if start_block == end_block:
            blocks_with_non_crossing.add(start_block)
            
    blocks_only_crossing = blocks - blocks_with_non_crossing
    
    print(f"Total Blocks Touched: {len(blocks)}")
    print(f"Blocks with Non-Crossing: {len(blocks_with_non_crossing)}")
    print(f"Blocks ONLY Crossing: {len(blocks_only_crossing)}")
    print(f"Crossing Tiles: {crossing_tiles}")
    
    # Formula Calculation
    h_crossing, h_total = compute_block_crossing_1d(config.block_h, config.tile_h, config.step_h, config.num_tiles)
    
    # Calculate num_blocks based on extent
    max_extent = config.step_h * (config.num_tiles - 1) + config.tile_h
    num_blocks_formula = math.ceil(max_extent / config.block_h)
    
    print(f"Formula h_crossing: {h_crossing}")
    print(f"Formula num_blocks: {num_blocks_formula}")
    
    formula_cost = num_blocks_formula * 1 + h_crossing * 2
    
    # "Ideal" Cost (if we remove double counting)
    # We pay 1 for every block (baseline).
    # For crossing tiles, we pay EXTRA.
    # How much extra?
    # If crossing tile T touches A and B.
    # We paid 1 for A, 1 for B.
    # T needs A and B.
    # If T causes thrashing, we pay 2.
    # So we paid 1(A)+1(B) + 2(T) = 4.
    # But we only needed 2.
    # So the "Extra" cost of crossing should be... 0?
    # If we assume the baseline covers the opening.
    # But thrashing means we open A, then B.
    # If A was already open (baseline), we close it, open B.
    # So we pay 1 extra for B? And maybe 1 extra for A if we come back?
    
    # Let's look at the "Only T2" case again.
    # T2 crosses A and B.
    # Formula: 1(A) + 1(B) + 2(T2) = 4.
    # Actual: 2.
    # Overcharge: 2.
    
    # Case: T1 (in A), T2 (cross A, B).
    # Formula: 1(A) + 1(B) + 2(T2) = 4.
    # Actual: T1(A) -> 1. T2(A, B) -> 2. Total 3.
    # Overcharge: 1. (Because B was charged in baseline but only used in crossing).
    
    # It seems the overcharge is exactly equal to the number of blocks that are touched by the crossing tile?
    # No.
    
    return

# Case 1: Dense
simulate_1d_cost(MockConfig(block_h=10, tile_h=6, step_h=2, num_tiles=10))

# Case 2: Sparse (The "Only Crossing" case)
# Block=10, Tile=4, Step=9.
# T0: [0, 4] (In B0)
# T1: [9, 13] (Cross B0, B1)
# T2: [18, 22] (Cross B1, B2)
simulate_1d_cost(MockConfig(block_h=10, tile_h=4, step_h=9, num_tiles=3))

# Case 3: Tile > Block
simulate_1d_cost(MockConfig(block_h=5, tile_h=6, step_h=2, num_tiles=5))
