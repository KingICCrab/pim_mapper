
import math
import numpy as np
import logging
from dataclasses import dataclass
import pandas as pd

from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer.mapping import Mapping
from validation.dram.trace_generator import TraceGenerator, DRAMConfig, DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N
from validation.dram.debug_row_activation import count_row_activations
from pim_optimizer.model.row_activation import precompute_tile_crossing_info

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveValidator:
    """
    Validates Row Activation Cost Model across multiple workloads and mappings.
    """
    
    def __init__(self):
        self.dram_config = DRAMConfig(row_buffer_bytes=1024, element_size=1)
        self.trace_gen = TraceGenerator(self.dram_config)
        
    def get_relevant_dims(self, tensor_type):
        if tensor_type == 'weight':
            return [DIM_K, DIM_C, DIM_R, DIM_S]
        elif tensor_type == 'output':
            return [DIM_N, DIM_K, DIM_P, DIM_Q]
        return []

    def calculate_ilp_reuse_params(self, workload, permutation_l2, permutation_l1, tensor_type):
        relevant_dims = self.get_relevant_dims(tensor_type)
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
        
        for i, dim in enumerate(permutation_l2):
            dim_bound = workload.bounds[dim]
            is_relevant = dim in relevant_dims
            
            if not is_relevant:
                if xr[i] == 1:
                    reuse_penalty *= dim_bound
                else:
                    outer_irr_product *= dim_bound
                    
        return reuse_penalty, outer_irr_product

    def calculate_ilp_cost(self, workload, tile_size_elements, tensor_type, reuse_penalty, outer_irr_product):
        element_bytes = 1
        row_bytes = 1024
        
        if tensor_type == 'weight':
            total_elements = workload.K * workload.C * workload.R * workload.S
        else: # output
            total_elements = workload.N * workload.K * workload.P * workload.Q
            
        tensor_bytes = total_elements * element_bytes
        
        non_crossing_acts_list, crossing_counts_list = precompute_tile_crossing_info(
            [tile_size_elements], element_bytes, row_bytes, tensor_bytes
        )
        
        non_crossing_acts = non_crossing_acts_list[0]
        crossing_count = crossing_counts_list[0]
        
        # Piecewise Formula:
        # If Reuse = 1: Cost = NC
        # If Reuse > 1: Cost = NC + 1 * C * Reuse
        
        if reuse_penalty == 1:
            base_row_acts = non_crossing_acts
        else:
            base_row_acts = non_crossing_acts + 1 * crossing_count * reuse_penalty
        
        total_cost = base_row_acts * outer_irr_product
            
        return total_cost, non_crossing_acts, crossing_count

    def calculate_gt_cost(self, workload, tile_sizes, tensor_type, permutation_l2, permutation_l1):
        mapping = Mapping()
        
        if tensor_type == 'weight':
            k_tile, c_tile, r_tile, s_tile = tile_sizes
            tiled_dims = {DIM_K: k_tile, DIM_C: c_tile, DIM_R: r_tile, DIM_S: s_tile}
            target_bank = 1
        else: # output
            n_tile, k_tile, p_tile, q_tile = tile_sizes
            tiled_dims = {DIM_N: n_tile, DIM_K: k_tile, DIM_P: p_tile, DIM_Q: q_tile}
            target_bank = 2

        l2_bounds = {}
        l1_bounds = {}
        
        for dim in [DIM_N, DIM_C, DIM_K, DIM_P, DIM_Q, DIM_R, DIM_S]:
            l2_bounds[dim] = 1
            l1_bounds[dim] = 1

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

        mapping.loop_bounds = {
            2: {'temporal': l2_bounds},
            1: {'temporal': l1_bounds}
        }
        
        mapping.permutation = {
            2: {i: dim for i, dim in enumerate(permutation_l2)},
            1: {i: dim for i, dim in enumerate(permutation_l1)}
        }
        
        mapping.layout = {'input': 'sequential', 'weight': 'sequential', 'output': 'sequential'}
        
        trace = self.trace_gen.generate_trace(mapping, workload)
        activations_dict = count_row_activations(trace, self.dram_config)
        return activations_dict.get(target_bank, 0)

    def run_comprehensive_validation(self):
        print("Starting Comprehensive Validation...")
        
        # Define Workloads
        workloads = {
            "ResNet18_Conv2": ConvWorkload(C=64, K=64, R=3, S=3, P=56, Q=56, stride=(1,1), dilation=(1,1), N=1),
            "ResNet18_Conv3": ConvWorkload(C=64, K=128, R=3, S=3, P=28, Q=28, stride=(2,2), dilation=(1,1), N=1),
            "VGG16_Conv1_2":  ConvWorkload(C=64, K=64, R=3, S=3, P=224, Q=224, stride=(1,1), dilation=(1,1), N=1),
            # "MobileNet_DW":   ConvWorkload(C=32, K=32, R=3, S=3, P=112, Q=112, stride=(1,1), dilation=(1,1), N=1, groups=32), # Depthwise - Removed due to API mismatch
        }
        
        # Define Mappings (Permutations)
        # Format: (L2_Perm, L1_Perm, Description)
        mappings = [
            # Weight Stationary (Reuse=1)
            ([DIM_K, DIM_C, DIM_R, DIM_S], [DIM_P, DIM_Q, DIM_N], "WS (Seq)"),
            
            # Output Stationary (Reuse > 1)
            ([DIM_K, DIM_P, DIM_C], [DIM_Q, DIM_N, DIM_R, DIM_S], "OS (Thrash)"),
            
            # Input Stationary (SRAM Reuse, DRAM Seq)
            ([DIM_K, DIM_C, DIM_R, DIM_S], [DIM_P, DIM_Q, DIM_N], "IS (SRAM)"),
            
            # Mixed: K outer, P middle, C inner (Thrashing)
            ([DIM_K, DIM_P, DIM_C, DIM_R, DIM_S], [DIM_Q, DIM_N], "Mixed (Thrash)"),
        ]
        
        results = []
        
        for wl_name, wl in workloads.items():
            # Tile Size: 4x4x3x3 (Small Tile)
            k_tile, c_tile, r_tile, s_tile = 4, 4, 3, 3
            tile_size_elements = k_tile * c_tile * r_tile * s_tile
            
            for l2_perm, l1_perm, map_name in mappings:
                # Skip invalid mappings (e.g. IS mapping for WS test)
                # For simplicity, we test Weight tensor for all
                
                # Calculate ILP
                reuse, outer = self.calculate_ilp_reuse_params(wl, l2_perm, l1_perm, 'weight')
                ilp_cost, nc, c = self.calculate_ilp_cost(wl, tile_size_elements, 'weight', reuse, outer)
                
                # Calculate GT
                # Note: For IS (SRAM), we need to adjust permutation for GT to match ILP logic
                # But here we just use the provided permutations
                gt_cost = self.calculate_gt_cost(wl, (k_tile, c_tile, r_tile, s_tile), 'weight', l2_perm, l1_perm)
                
                error = abs(ilp_cost - gt_cost) / gt_cost * 100 if gt_cost > 0 else 0
                
                results.append({
                    "Workload": wl_name,
                    "Mapping": map_name,
                    "Reuse": reuse,
                    "Outer": outer,
                    "ILP": ilp_cost,
                    "GT": gt_cost,
                    "Error%": error
                })
        
        # Print Results
        df = pd.DataFrame(results)
        print(df.to_string(index=False, float_format="%.1f"))
        
        # Summary
        print("\nSummary:")
        print(f"Mean Error: {df['Error%'].mean():.2f}%")
        print(f"Max Error:  {df['Error%'].max():.2f}%")

if __name__ == "__main__":
    validator = ComprehensiveValidator()
    validator.run_comprehensive_validation()
