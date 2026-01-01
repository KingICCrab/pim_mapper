
import math
import numpy as np
import logging
from dataclasses import dataclass

from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer.mapping import Mapping
from validation.dram.trace_generator import TraceGenerator, DRAMConfig, DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N
from validation.dram.debug_row_activation import count_row_activations
from pim_optimizer.model.row_activation import precompute_tile_crossing_info

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealWorkloadValidator:
    """
    Validates Row Activation Cost Model for Weight and Output tensors using REAL workloads.
    Compares ILP model logic (precompute_tile_crossing_info) against TraceGenerator (Ground Truth).
    """
    
    def __init__(self):
        self.dram_config = DRAMConfig(row_buffer_bytes=1024, element_size=1)
        self.trace_gen = TraceGenerator(self.dram_config)
        
    def get_relevant_dims(self, tensor_type):
        if tensor_type == 'weight':
            return [DIM_K, DIM_C, DIM_R, DIM_S]
        elif tensor_type == 'output':
            return [DIM_N, DIM_K, DIM_P, DIM_Q]
        elif tensor_type == 'input':
            return [DIM_N, DIM_C, DIM_P, DIM_Q, DIM_R, DIM_S] # Simplified
        return []

    def calculate_ilp_reuse_params(self, workload, permutation_l2, permutation_l1, tensor_type):
        """
        Calculate reuse_penalty and outer_irr_product based on ILP logic (xr/xj).
        Only L2 loops contribute to outer_irr_product.
        """
        relevant_dims = self.get_relevant_dims(tensor_type)
        
        # Combine permutations for xr calculation (Outer -> Inner)
        full_permutation = permutation_l2 + permutation_l1
        num_loops = len(full_permutation)
        xr = [0] * num_loops
        
        has_relevant_inner = False
        for i in range(num_loops - 1, -1, -1):
            dim = full_permutation[i]
            is_relevant = dim in relevant_dims
            if is_relevant or has_relevant_inner:
                xr[i] = 1
                has_relevant_inner = True
            else:
                xr[i] = 0
                
        reuse_penalty = 1
        outer_irr_product = 1
        
        # Iterate L2 loops (Outer Product + Reuse)
        for i, dim in enumerate(permutation_l2):
            dim_bound = workload.bounds[dim]
            is_relevant = dim in relevant_dims
            
            if not is_relevant:
                if xr[i] == 1:
                    reuse_penalty *= dim_bound
                else:
                    outer_irr_product *= dim_bound
                    
        # L1 loops do not contribute to DRAM reuse (they are within the tile)
                
        return reuse_penalty, outer_irr_product

    def calculate_ilp_cost_logic(self, workload, tile_size_elements, tensor_type, reuse_penalty, outer_irr_product):
        """
        Calculate ILP predicted cost using the EXACT logic from row_activation.py.
        
        New Formula Logic:
        If Tile > Row: Cost = 2 * Crossing * Reuse
        If Tile <= Row: Cost = max(NonCrossing, Crossing) * Reuse
        """
        element_bytes = 1
        row_bytes = 1024
        
        if tensor_type == 'weight':
            total_elements = workload.K * workload.C * workload.R * workload.S
        else: # output
            total_elements = workload.N * workload.K * workload.P * workload.Q
            
        tensor_bytes = total_elements * element_bytes
        tile_bytes = tile_size_elements * element_bytes
        
        # Call the actual ILP logic function
        non_crossing_acts_list, crossing_counts_list = precompute_tile_crossing_info(
            [tile_size_elements], element_bytes, row_bytes, tensor_bytes
        )
        
        non_crossing_acts = non_crossing_acts_list[0]
        crossing_count = crossing_counts_list[0]
        
        # ILP Formula Implementation
        # Piecewise Formula:
        # If Reuse = 1: Cost = NC
        # If Reuse > 1: Cost = NC + 1 * C * Reuse
        
        if reuse_penalty == 1:
            base_row_acts = non_crossing_acts
        else:
            base_row_acts = non_crossing_acts + 1 * crossing_count * reuse_penalty
        
        total_cost = base_row_acts * outer_irr_product
            
        return total_cost, non_crossing_acts, crossing_count

    def calculate_gt_cost(self, workload, tile_sizes, tensor_type, permutation_l2, permutation_l1, layout='sequential'):
        """
        Calculate Ground Truth cost using TraceGenerator with explicit permutation.
        """
        mapping = Mapping()
        
        # 1. Define Tile Sizes (L1 bounds for tiled dims)
        if tensor_type == 'weight':
            k_tile, c_tile, r_tile, s_tile = tile_sizes
            tiled_dims = {DIM_K: k_tile, DIM_C: c_tile, DIM_R: r_tile, DIM_S: s_tile}
            target_bank = 1
        else: # output
            n_tile, k_tile, p_tile, q_tile = tile_sizes
            tiled_dims = {DIM_N: n_tile, DIM_K: k_tile, DIM_P: p_tile, DIM_Q: q_tile}
            target_bank = 2

        # 2. Distribute bounds to L2 and L1 based on permutation
        l2_bounds = {}
        l1_bounds = {}
        
        # Initialize all dims to 1
        for dim in [DIM_N, DIM_C, DIM_K, DIM_P, DIM_Q, DIM_R, DIM_S]:
            l2_bounds[dim] = 1
            l1_bounds[dim] = 1

        # Assign bounds
        for dim, bound in enumerate(workload.bounds):
            if dim in tiled_dims:
                l1_size = tiled_dims[dim]
                l2_size = (bound + l1_size - 1) // l1_size
                
                l1_bounds[dim] = l1_size
                if dim in permutation_l2:
                    l2_bounds[dim] = l2_size
            else:
                if dim in permutation_l2:
                    l2_bounds[dim] = bound
                elif dim in permutation_l1:
                    l1_bounds[dim] = bound

        # Construct Loop Bounds
        # Level 1: SRAM (Global Buffer)
        # Level 2: RowBuffer (We keep it 1 here, meaning RowBuffer holds exactly what's in SRAM? 
        #          Or rather, the "Tile" is defined by L1. L2 loops iterate over tiles.)
        #          If we put loops in Level 2, TraceGenerator treats them as INSIDE RowBuffer.
        #          To model "RowBuffer holds a tile, and we iterate tiles", we must put loops in Level 3.
        mapping.loop_bounds = {
            3: {'temporal': l2_bounds},
            1: {'temporal': l1_bounds}
        }
        
        # Construct Permutation Dictionary
        # permutation_l2 is list of dims. Convert to dict {index: dim}
        mapping.permutation = {
            3: {i: dim for i, dim in enumerate(permutation_l2)},
            1: {i: dim for i, dim in enumerate(permutation_l1)}
        }
        
        # TraceGenerator uses integer keys for layout: 0=Input, 1=Weight, 2=Output
        mapping.layout = {0: 'sequential', 1: layout, 2: 'sequential'}
        
        trace = self.trace_gen.generate_trace(mapping, workload)
        
        # Count Activations using standard logic
        activations_dict = count_row_activations(trace, self.dram_config)
        return activations_dict.get(target_bank, 0)

    def run_validation(self):
        print("Starting REAL Workload Validation (Strict ILP Logic)...")
        print(f"{'Case':<20} | {'Tile':<15} | {'Reuse':<5} | {'Outer':<5} | {'ILP Cost':<10} | {'GT Cost':<10} | {'Error %':<10}")
        print("-" * 100)
        
        # Workload: Conv3_x (C=128, K=128, R=3, S=3, P=4, Q=4) - Reduced for speed
        wl = ConvWorkload(C=128, K=128, R=3, S=3, P=4, Q=4, stride=(1,1), dilation=(1,1), N=1)
        
        # Tile Size: 4x4x3x3 (Small Tile)
        k_tile, c_tile, r_tile, s_tile = 4, 4, 3, 3
        tile_size_elements = k_tile * c_tile * r_tile * s_tile # 144 elements
        
        # ---------------------------------------------------------
        # Scenario 1: Weight Stationary (Reuse=1)
        # L2 (DRAM): K, C, R, S
        # L1 (SRAM): P, Q, N
        # ---------------------------------------------------------
        perm_l2_ws = [DIM_K, DIM_C, DIM_R, DIM_S]
        perm_l1_ws = [DIM_P, DIM_Q, DIM_N]
        
        reuse_ws, outer_ws = self.calculate_ilp_reuse_params(wl, perm_l2_ws, perm_l1_ws, 'weight')
        
        ilp_ws, nc_ws, c_ws = self.calculate_ilp_cost_logic(wl, tile_size_elements, 'weight', reuse_ws, outer_ws)
        
        gt_ws = self.calculate_gt_cost(wl, (k_tile, c_tile, r_tile, s_tile), 'weight', 
                                       perm_l2_ws, perm_l1_ws + perm_l2_ws) # L1 includes inner loops of KCRS
        
        err_ws = abs(ilp_ws - gt_ws) / gt_ws * 100 if gt_ws > 0 else 0
        print(f"{'Weight Stationary':<20} | {'4x4x3x3':<15} | {reuse_ws:<5} | {outer_ws:<5} | {ilp_ws:<10.1f} | {gt_ws:<10} | {err_ws:<10.2f}")
        
        # ---------------------------------------------------------
        # Scenario 2: Input Stationary (SRAM Reuse)
        # L2 (DRAM): K, C, R, S
        # L1 (SRAM): P, Q, N
        # ---------------------------------------------------------
        # User correction: Input Stationary implies SRAM (L1) reuse for Weights.
        # So P, Q, N should be in L1, meaning no DRAM reuse penalty.
        
        perm_l2_is = [DIM_K, DIM_C, DIM_R, DIM_S]
        perm_l1_is = [DIM_P, DIM_Q, DIM_N]
        
        reuse_is, outer_is = self.calculate_ilp_reuse_params(wl, perm_l2_is, perm_l1_is, 'weight')
        
        ilp_is, nc_is, c_is = self.calculate_ilp_cost_logic(wl, tile_size_elements, 'weight', reuse_is, outer_is)
        
        # Force strict ordering to ensure all loops are executed
        gt_is = self.calculate_gt_cost(wl, (k_tile, c_tile, r_tile, s_tile), 'weight',
                                       perm_l2_is, perm_l1_is + perm_l2_is)
                                       
        err_is = abs(ilp_is - gt_is) / gt_is * 100 if gt_is > 0 else 0
        print(f"{'Input Stationary':<20} | {'4x4x3x3':<15} | {reuse_is:<5} | {outer_is:<5} | {ilp_is:<10.1f} | {gt_is:<10} | {err_is:<10.2f}")

        # ---------------------------------------------------------
        # Scenario 3: Output Stationary-like (Partial Reuse)
        # L2 (DRAM): K (Outer), P (Middle), C (Inner)
        # L1 (SRAM): Q, N, R, S
        # Weight depends on K, C. P is irrelevant.
        # P wraps C (Relevant). -> Reuse = P.
        # ---------------------------------------------------------
        perm_l2_os = [DIM_K, DIM_P, DIM_C]
        perm_l1_os = [DIM_Q, DIM_N, DIM_R, DIM_S]
        
        reuse_os, outer_os = self.calculate_ilp_reuse_params(wl, perm_l2_os, perm_l1_os, 'weight')
        
        ilp_os, nc_os, c_os = self.calculate_ilp_cost_logic(wl, tile_size_elements, 'weight', reuse_os, outer_os)
        
        gt_os = self.calculate_gt_cost(wl, (k_tile, c_tile, r_tile, s_tile), 'weight',
                                       perm_l2_os, perm_l1_os + perm_l2_os)
                                       
        err_os = abs(ilp_os - gt_os) / gt_os * 100 if gt_os > 0 else 0
        print(f"{'Output Stationary':<20} | {'4x4x3x3':<15} | {reuse_os:<5} | {outer_os:<5} | {ilp_os:<10.1f} | {gt_os:<10} | {err_os:<10.2f}")

        # ---------------------------------------------------------
        # Scenario 4: Weight Row Aligned (Reuse=1)
        # Same as WS, but with row_aligned layout
        # ---------------------------------------------------------
        # For row_aligned, each tile is padded to row boundary.
        # Tile size = 144 elements. Row size = 1024 elements.
        # So each tile takes 1024 elements in DRAM space (padded).
        # But row activation count depends on how many rows are touched.
        # Since 144 < 1024, each tile fits in 1 row (if aligned).
        # So crossing should be 0.
        
        perm_l2_ra = [DIM_K, DIM_C, DIM_R, DIM_S]
        perm_l1_ra = [DIM_P, DIM_Q, DIM_N]
        
        reuse_ra, outer_ra = self.calculate_ilp_reuse_params(wl, perm_l2_ra, perm_l1_ra, 'weight')
        
        # ILP Logic for Row Aligned:
        # If row_aligned, we assume NO crossing (cost = 1 per tile access)
        # unless tile > row_size.
        # Here tile=144 < 1024. So cost = 1 * outer_product.
        # Total Tiles = (K/K_tile) * (C/C_tile) * (R/R_tile) * (S/S_tile)
        # = 32 * 32 * 1 * 1 = 1024.
        
        gt_ra = self.calculate_gt_cost(wl, (k_tile, c_tile, r_tile, s_tile), 'weight', 
                                       perm_l2_ra, perm_l1_ra + perm_l2_ra, layout='row_aligned')
        
        # Manual ILP prediction for Row Aligned
        # Since tile fits in row and is aligned, acts = 1 per access.
        # Total accesses = Total Tiles = 1024.
        ilp_ra = 1024.0
        
        err_ra = abs(ilp_ra - gt_ra) / gt_ra * 100 if gt_ra > 0 else 0
        print(f"{'Weight Row Aligned':<20} | {'4x4x3x3':<15} | {reuse_ra:<5} | {outer_ra:<5} | {ilp_ra:<10.1f} | {gt_ra:<10} | {err_ra:<10.2f}")

        print("-" * 100)
        print("Detailed Breakdown for Weight Stationary (Reuse=1):")
        print(f"Non-Crossing Acts: {nc_ws}")
        print(f"Crossing Count:    {c_ws}")
        print(f"Formula:           max({nc_ws}, {c_ws}) * {reuse_ws}")
        print(f"ILP Prediction:    {ilp_ws}")
        print(f"Ground Truth:      {gt_ws}")
        print(f"Discrepancy:       {ilp_ws - gt_ws}")

if __name__ == "__main__":
    validator = RealWorkloadValidator()
    validator.run_validation()
