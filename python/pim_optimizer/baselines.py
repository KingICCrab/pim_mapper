"""
Baseline optimizers for comparison.
Includes Heuristic (Rule-based) and Random optimizers.
"""

import random
import math
from typing import List, Dict, Tuple
import copy

from pim_optimizer.workload import ConvWorkload
from pim_optimizer.arch import PIMArchitecture
from pim_optimizer.mapping import Mapping, OptimizationResult

class HeuristicOptimizer:
    """
    A rule-based heuristic optimizer.
    Implements common strategies like "Weight Stationary" or "Output Stationary".
    """
    
    def __init__(self, arch: PIMArchitecture, strategy: str = "weight_stationary"):
        self.arch = arch
        self.strategy = strategy
        
    def optimize(self, workloads: List[ConvWorkload]) -> OptimizationResult:
        mappings = []
        for workload in workloads:
            mapping = self._optimize_single(workload)
            mappings.append(mapping)
            
        return OptimizationResult(
            mappings=mappings,
            solve_time=0.0
        )
        
    def _optimize_single(self, workload: ConvWorkload) -> Mapping:
        """
        Generate a mapping based on the selected strategy.
        """
        # 1. Spatial Tiling (Parallelism)
        # Heuristic: Parallelize the largest dimension across banks
        # Usually Output Channels (K) or Input Channels (C)
        
        num_banks = self.arch.dram_num_banks
        
        # Default: No spatial splitting
        spatial_p = {dim: 1 for dim in ["N", "K", "C", "P", "Q", "R", "S"]}
        
        # Get workload dimensions
        # workload.bounds is [R, S, P, Q, C, K, N]
        # dim_name_idx_dict maps "R" -> 0, etc.
        dim_map = workload.dim_name_idx_dict
        
        # Helper to get dim size by name
        def get_dim(name):
            return workload.bounds[dim_map[name]]
        
        if get_dim("K") >= num_banks:
            spatial_p["K"] = num_banks
        elif get_dim("C") >= num_banks:
            spatial_p["C"] = num_banks
        elif get_dim("P") * get_dim("Q") >= num_banks:
            # Split spatial dimensions if channels are small
            # Simple logic: try to split P
            spatial_p["P"] = min(get_dim("P"), num_banks)
            rem = num_banks // spatial_p["P"]
            spatial_p["Q"] = min(get_dim("Q"), rem)
            
        # 2. Temporal Tiling (Row Buffer Fitting)
        # Heuristic: Fill the row buffer as much as possible
        # Strategy: Weight Stationary -> Maximize Input/Output reuse
        
        row_buffer_elements = self.arch.dram_bank_row_buffer_size // self.arch.default_element_bits * 8 # Assuming size is in bytes?
        # Wait, arch.dram_bank_row_buffer_size is usually in bytes.
        # arch.default_element_bits is bits.
        # So elements = bytes * 8 / bits
        row_buffer_elements = (self.arch.dram_bank_row_buffer_size * 8) // self.arch.default_element_bits
        
        # Start with full workload (divided by spatial)
        temporal_t = {dim: get_dim(dim) // spatial_p[dim] for dim in spatial_p}
        
        # Shrink dimensions until it fits in Row Buffer
        # This is a simplified greedy approach
        while True:
            # Calculate required buffer size
            # Input: N * C * P * Q (simplified)
            # Weight: K * C * R * S
            # Output: N * K * P * Q
            
            # Note: This calculation depends on the exact data layout and loop order
            # Here we use a rough estimate of the "Working Set"
            
            # For Weight Stationary: We need to load a tile of Weights
            weight_size = temporal_t["K"] * temporal_t["C"] * temporal_t["R"] * temporal_t["S"]
            
            # And some Inputs/Outputs
            input_size = temporal_t["N"] * temporal_t["C"] * temporal_t["P"] * temporal_t["Q"] # Rough
            output_size = temporal_t["N"] * temporal_t["K"] * temporal_t["P"] * temporal_t["Q"] # Rough
            
            total_size = weight_size + input_size + output_size
            
            if total_size <= row_buffer_elements:
                break
                
            # Shrink the largest dimension
            # Strategy-aware shrinking to preserve locality
            
            if self.strategy == "output_stationary":
                # For Output Stationary:
                # We want large P, Q tiles to maximize Input Spatial Reuse (sliding window)
                # Since we iterate K, C in inner loops, we should shrink K, C first to fit in buffer
                # Shrink sequence: K, C -> N, P, Q
                
                # Try shrinking K or C first
                candidates_primary = ["K", "C"]
                best_cand = max(candidates_primary, key=lambda x: temporal_t[x])
                
                if temporal_t[best_cand] > 1:
                    temporal_t[best_cand] = math.ceil(temporal_t[best_cand] / 2)
                else:
                    # If K, C are 1, then shrink P, Q (N is usually 1)
                    candidates_secondary = ["P", "Q", "N"]
                    best_cand_sec = max(candidates_secondary, key=lambda x: temporal_t[x])
                    if temporal_t[best_cand_sec] > 1:
                        temporal_t[best_cand_sec] = math.ceil(temporal_t[best_cand_sec] / 2)
                    else:
                        break # Can't shrink
            
            else:
                # Default / Weight Stationary:
                # We want large Weight tiles? Or small Weight tiles to fit?
                # Actually, for WS, maximizing P, Q is also good for Input Reuse.
                # But typically WS prioritizes Weight reuse.
                # The original heuristic (Shrink P, Q first) works well for WS because 
                # small P, Q tiles mean we load a Weight Tile and use it for a small Output chunk,
                # then proceed to next Weight Tile? No.
                # WS loop: Outer K, C. Inner P, Q.
                # We load Weight(K, C). Then loop P, Q.
                # If P, Q tile is small, we process small spatial chunk.
                # But we want to process ALL P, Q for this Weight.
                # So cutting P, Q just means more Inner Loop iterations.
                # Original heuristic seemed to favor keeping K, C large.
                # Let's keep original heuristic for WS.
                
                # Shrink N, P, Q first
                candidates = ["N", "P", "Q"]
                best_cand = max(candidates, key=lambda x: temporal_t[x])
                
                if temporal_t[best_cand] > 1:
                    temporal_t[best_cand] = math.ceil(temporal_t[best_cand] / 2)
                else:
                    # If N, P, Q are 1, try shrinking K or C
                    candidates = ["K", "C"]
                    best_cand = max(candidates, key=lambda x: temporal_t[x])
                    if temporal_t[best_cand] > 1:
                        temporal_t[best_cand] = math.ceil(temporal_t[best_cand] / 2)
                    else:
                        # Can't shrink anymore
                        break
                    
        # 3. Loop Ordering
        # Heuristic: 
        # Weight Stationary: Loop K, C, R, S inner-most (reuse weights)
        # Output Stationary: Loop N, P, Q inner-most (accumulate outputs)
        
        if self.strategy == "weight_stationary":
            # Outer -> Inner
            # Loop over tiles (Temporal loops)
            # We want to reuse Weights, so we should change Inputs/Outputs while keeping Weights constant.
            # So Weight-related loops (K, C) should be OUTER.
            # Wait, "Stationary" means it stays in the buffer.
            # If we want Weights to stay, we should loop over N, P, Q (Inputs) while K, C are fixed.
            # So K, C should be OUTER loops in the Tiling schedule?
            # No, "Loop Order" usually refers to the execution order.
            # If K is the outermost loop, then for a fixed k, we iterate everything else.
            # This means we load weight(k), then use it for all inputs. This is Weight Stationary.
            loop_order = ["K", "C", "R", "S", "N", "P", "Q"]
        else:
            # Output Stationary
            # Fix Output (N, K, P, Q), iterate over reduction (C, R, S)
            loop_order = ["N", "K", "P", "Q", "C", "R", "S"]
            
        # Initialize Mapping
        mapping = Mapping()
        mapping.workload_name = workload.name
        mapping.workload_bounds = list(workload.bounds)
        
        # Initialize loop_bounds structure
        # Level 0 (PE), 1 (GlobalBuffer), 2 (RowBuffer), 3 (DRAM)
        # Assuming 4 levels for now, or check arch.num_mems
        num_mems = self.arch.num_mems # Usually 4
        
        for m in range(num_mems):
            if m == 0:
                mapping.loop_bounds[m] = {"H": {}, "W": {}, "Internal": {}, "temporal": {}}
            else:
                mapping.loop_bounds[m] = {"spatial": {}, "temporal": {}}
            mapping.permutation[m] = {}
            mapping.bypass[m] = {t: False for t in range(3)} # No bypass
            
        # Map heuristic results to Mapping structure
        
        # Level 2: RowBuffer (Bank)
        # spatial_p -> loop_bounds[2]["spatial"]
        # temporal_t -> loop_bounds[2]["temporal"]
        
        # Level 3: DRAM
        # remaining -> loop_bounds[3]["temporal"]
        
        for dim_name, dim_idx in dim_map.items():
            total_dim = workload.bounds[dim_idx]
            
            s_p = spatial_p.get(dim_name, 1)
            t_t = temporal_t.get(dim_name, 1)
            
            # Level 2 Spatial (Bank Parallelism)
            mapping.loop_bounds[2]["spatial"][dim_idx] = s_p
            
            # Level 2 Temporal (Tile Size in RowBuffer)
            mapping.loop_bounds[2]["temporal"][dim_idx] = t_t
            
            # Level 3 Temporal (Outer Loops)
            # Ensure integer division
            outer = math.ceil(total_dim / (s_p * t_t))
            mapping.loop_bounds[3]["temporal"][dim_idx] = outer
            
            # Level 0/1: Set to 1 (Not optimizing PE level in this heuristic)
            mapping.loop_bounds[0]["temporal"][dim_idx] = 1
            mapping.loop_bounds[1]["temporal"][dim_idx] = 1
            
        # Permutation
        # Apply loop_order to Level 3 (DRAM loops)
        # loop_order is list of names ["N", "K", ...]
        # mapping.permutation[m][priority] = dim_idx
        # priority 0 is outermost? Or innermost?
        # Usually 0 is outermost.
        
        for i, dim_name in enumerate(loop_order):
            dim_idx = dim_map[dim_name]
            mapping.permutation[3][i] = dim_idx
            
        # For other levels, just use default order (0..6)
        for m in range(num_mems):
            if m != 3:
                for i in range(7):
                    mapping.permutation[m][i] = i
        
        # Set default layout and tile info
        # Sequential layout is the standard row-major layout
        for t in range(3):
            mapping.layout[t] = "sequential"
            
        # Tile info for data layout (needed by TraceGenerator if layout is row_aligned, but good to have)
        # For heuristic, we don't do complex data layout optimization, so block_h/w = 1
        mapping.tile_info['block_h'] = 1
        mapping.tile_info['block_w'] = 1
                    
        return mapping

class RandomOptimizer:
    """
    A random search optimizer.
    Generates N random valid mappings and picks the best one (based on a simple cost model).
    """
    
    def __init__(self, arch: PIMArchitecture, num_samples: int = 100):
        self.arch = arch
        self.num_samples = num_samples
        
    def optimize(self, workloads: List[ConvWorkload]) -> OptimizationResult:
        # Placeholder for random optimization logic
        # For now, just return a heuristic mapping
        heuristic = HeuristicOptimizer(self.arch)
        return heuristic.optimize(workloads)
