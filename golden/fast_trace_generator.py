
import sys
import os
from typing import List, Dict, Tuple, Set
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from golden.trace_generator import TraceGenerator, DRAMConfig

# Dimension indices: R, S, P, Q, C, K, N
DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6

class FastTraceGenerator(TraceGenerator):
    """
    A faster version of TraceGenerator that counts row activations on-the-fly
    without generating string traces, avoiding memory and CPU overhead.
    """
    
    def __init__(self, dram_config: DRAMConfig = None, 
                 track_input: bool = True, 
                 track_weight: bool = True, 
                 track_output: bool = True):
        super().__init__(dram_config)
        self.row_size = self.dram.row_buffer_bytes
        self.bank_size = self.row_size * self.dram.num_rows
        self.track_input = track_input
        self.track_weight = track_weight
        self.track_output = track_output
        self.stats = {
            0: {"row_acts": 0, "last_row": None},
            1: {"row_acts": 0, "last_row": None},
            2: {"row_acts": 0, "last_row": None},
        }

    def _update_stat(self, addr):
        bank = addr // self.bank_size
        if bank in self.stats:
            row = (addr % self.bank_size) // self.row_size
            if self.stats[bank]["last_row"] != row:
                self.stats[bank]["row_acts"] += 1
                self.stats[bank]["last_row"] = row

    def generate_trace(self, mapping, workload, strict_ordering: bool = False) -> Dict[int, int]:
        # Reset stats
        self.stats = {
            0: {"row_acts": 0, "last_row": None},
            1: {"row_acts": 0, "last_row": None},
            2: {"row_acts": 0, "last_row": None},
        }
        
        # Copying generate_trace logic from parent but returning stats
        
        # Compute buffer tile size (level 0 + level 1)
        buffer_tile = self._compute_buffer_tile_size(mapping)
        
        # Compute DRAM loop structure (level 2+)
        dram_loops = self._build_dram_loop_structure(mapping, workload, buffer_tile)
        
        # Build tile_strides dict
        tile_strides = {d: 1 for d in range(7)}
        for loop_info in dram_loops:
            tile_strides[loop_info['dim']] = loop_info['tile_stride']
            
        dram_loop_dims = [(loop_info['level'], loop_info['dim'], loop_info['bound']) for loop_info in dram_loops]
        
        H_in = workload.input_size['H']
        W_in = workload.input_size['W']
        stride_h, stride_w = workload.stride[1], workload.stride[0]
        dilation_h, dilation_w = workload.dilation[1], workload.dilation[0]
        
        layout_info = self._compute_data_layouts(mapping, workload, H_in, W_in, buffer_tile)
        layout_info['dram_loop_dims'] = dram_loop_dims
        
        # Track previous tile indices
        prev_indices = [None]
        
        # Recursive loop function
        def iterate_dram_loops(level_idx: int, indices: Dict[Tuple[int, int], int]):
            if level_idx >= len(dram_loops):
                input_changed = True
                weight_changed = True
                output_changed = True
                
                if prev_indices[0] is not None:
                    prev = prev_indices[0]
                    input_dims = [DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_N]
                    input_changed = any(
                        indices.get(key, 0) != prev.get(key, 0)
                        for key in indices.keys()
                        if key[1] in input_dims
                    )
                    weight_dims = [DIM_R, DIM_S, DIM_C, DIM_K]
                    weight_changed = any(
                        indices.get(key, 0) != prev.get(key, 0)
                        for key in indices.keys()
                        if key[1] in weight_dims
                    )
                    output_dims = [DIM_P, DIM_Q, DIM_K, DIM_N]
                    output_changed = any(
                        indices.get(key, 0) != prev.get(key, 0)
                        for key in indices.keys()
                        if key[1] in output_dims
                    )
                
                prev_indices[0] = indices.copy()
                
                # Pass None as trace, we won't use it
                self._generate_tile_accesses(
                    None, indices, buffer_tile, workload,
                    H_in, W_in, stride_h, stride_w, dilation_h, dilation_w,
                    layout_info, tile_strides,
                    generate_input=input_changed and self.track_input,
                    generate_weight=weight_changed and self.track_weight,
                    generate_output=output_changed and self.track_output,
                    strict_ordering=strict_ordering
                )
                return
            
            loop_info = dram_loops[level_idx]
            level = loop_info['level']
            dim = loop_info['dim']
            bound = loop_info['bound']
            key = (level, dim)
            
            for tile_idx in range(bound):
                new_indices = indices.copy()
                new_indices[key] = tile_idx
                iterate_dram_loops(level_idx + 1, new_indices)
        
        initial_indices = {}
        iterate_dram_loops(0, initial_indices)
        
        return {k: v["row_acts"] for k, v in self.stats.items()}

    def _generate_tile_accesses(
        self, trace, tile_indices, tile_size,
        workload, H_in, W_in, stride_h, stride_w, dilation_h, dilation_w,
        layout_info, tile_strides,
        generate_input=True, generate_weight=True, generate_output=True,
        strict_ordering=False
    ):
        # Copying logic from parent but using _update_stat instead of trace.append
        
        input_base = layout_info['input_base']
        weight_base = layout_info['weight_base']
        output_base = layout_info['output_base']
        
        input_tile_size = (layout_info['block_h'] * layout_info['block_w'] * 
                          tile_size[DIM_C] * tile_size[DIM_N])
        weight_tile_size = tile_size[DIM_K] * tile_size[DIM_C] * tile_size[DIM_R] * tile_size[DIM_S]
        output_tile_size = tile_size[DIM_N] * tile_size[DIM_K] * tile_size[DIM_P] * tile_size[DIM_Q]
        
        dram_loop_dims = layout_info.get('dram_loop_dims', [])

        def get_accumulated_tile_index(dim: int) -> int:
            total = 0
            level_bounds = {}
            for level, d, bound in dram_loop_dims:
                if d == dim:
                    level_bounds[level] = bound
            sorted_levels = sorted(level_bounds.keys(), reverse=True)
            multiplier = 1
            for level in reversed(sorted_levels):
                idx = tile_indices.get((level, dim), 0)
                total += idx * multiplier
                multiplier *= level_bounds[level]
            return total
        
        def compute_flat_tile_index(tile_indices, relevant_dims) -> int:
            flat_idx = 0
            stride = 1
            for level, dim, bound in reversed(dram_loop_dims):
                if dim in relevant_dims:
                    flat_idx += tile_indices.get((level, dim), 0) * stride
                    stride *= bound
            return flat_idx
        
        def compute_flat_tile_index_l3_only(tile_indices, relevant_dims) -> int:
            flat_idx = 0
            stride = 1
            for level, dim, bound in reversed(dram_loop_dims):
                if level == 3 and dim in relevant_dims:
                    flat_idx += tile_indices.get((level, dim), 0) * stride
                    stride *= bound
            return flat_idx

        if generate_input:
            H_per_tile = layout_info['H_per_tile']
            W_per_tile = layout_info['W_per_tile']
            C_per_tile = layout_info['C_per_tile']
            
            input_strides = layout_info.get('input_strides', {})
            input_layout = layout_info.get('input_layout', 'sequential')
            
            stride_p_l3 = input_strides.get((3, DIM_P), 1024)
            stride_q_l3 = input_strides.get((3, DIM_Q), 1024)
            stride_c_l3 = input_strides.get((3, DIM_C), 1024)
            stride_n_l3 = input_strides.get((3, DIM_N), 1024)

            stride_p_l2 = input_strides.get((2, DIM_P), 0)
            stride_q_l2 = input_strides.get((2, DIM_Q), 0)
            stride_c_l2 = input_strides.get((2, DIM_C), 0)
            stride_n_l2 = input_strides.get((2, DIM_N), 0)

            c_l3_idx = tile_indices.get((3, DIM_C), 0)
            n_l3_idx = tile_indices.get((3, DIM_N), 0)
            c_l2_idx = tile_indices.get((2, DIM_C), 0)
            n_l2_idx = tile_indices.get((2, DIM_N), 0)
            
            p_tile = get_accumulated_tile_index(DIM_P)
            q_tile = get_accumulated_tile_index(DIM_Q)
            c_tile = get_accumulated_tile_index(DIM_C)
            n_tile = get_accumulated_tile_index(DIM_N)
            r_tile = get_accumulated_tile_index(DIM_R)
            s_tile = get_accumulated_tile_index(DIM_S)
            
            p_size = tile_size[DIM_P]
            q_size = tile_size[DIM_Q]
            r_size = tile_size[DIM_R]
            s_size = tile_size[DIM_S]
            c_size = tile_size[DIM_C]
            n_size = tile_size[DIM_N]
            
            p_start = p_tile * p_size
            q_start = q_tile * q_size
            c_start = c_tile * c_size
            n_start = n_tile * n_size
            r_start = r_tile * r_size
            s_start = s_tile * s_size
            
            h_offset = r_start * dilation_h
            w_offset = s_start * dilation_w
            
            h_start = p_start * stride_h + h_offset
            h_end = min((p_start + p_size - 1) * stride_h + h_offset + (r_size - 1) * dilation_h + 1, H_in)
            w_start = q_start * stride_w + w_offset
            w_end = min((q_start + q_size - 1) * stride_w + w_offset + (s_size - 1) * dilation_w + 1, W_in)
            
            block_h = layout_info['block_h']
            block_w = layout_info['block_w']
            
            for n_local in range(min(n_size, workload.N - n_start)):
                for c_local in range(min(c_size, workload.C - c_start)):
                    coords_iter = (
                        (h, w, h // block_h, w // block_w, h % block_h, w % block_w)
                        for h in range(h_start, h_end)
                        for w in range(w_start, w_end)
                    )
                    for h, w, h_block, w_block, h_in_block, w_in_block in coords_iter:
                        self._generate_input_access_body(
                            None, input_layout, tile_indices, 
                            h, w, h_block, w_block, h_in_block, w_in_block,
                            c_local, n_local,
                            stride_p_l3, stride_q_l3, stride_c_l3, stride_n_l3,
                            stride_p_l2, stride_q_l2, stride_c_l2, stride_n_l2,
                            c_l3_idx, n_l3_idx, c_l2_idx, n_l2_idx,
                            H_per_tile, W_per_tile, dram_loop_dims,
                            input_base, c_tile, n_tile,
                            w_start, w_end, block_w
                        )

        if generate_weight:
            weight_relevant = [DIM_R, DIM_S, DIM_C, DIM_K]
            weight_layout = layout_info.get('weight_layout', 'sequential')
            row_size = layout_info.get('row_size', 1024)
            
            if weight_layout == 'row_aligned':
                weight_tile_idx = compute_flat_tile_index_l3_only(tile_indices, weight_relevant)
                l2_tile_size = weight_tile_size
                for level, dim, bound in dram_loop_dims:
                    if level == 2 and dim in weight_relevant:
                        l2_tile_size *= bound
                weight_aligned_tile_size = ((l2_tile_size + row_size - 1) // row_size) * row_size
                
                l2_offset = 0
                l2_stride = 1
                for level, dim, bound in reversed(dram_loop_dims):
                    if level == 2 and dim in weight_relevant:
                        l2_offset += tile_indices.get((level, dim), 0) * l2_stride * weight_tile_size
                        l2_stride *= bound
                
                weight_tile_base = weight_base + weight_tile_idx * weight_aligned_tile_size * self.element_size + l2_offset * self.element_size
            else:
                weight_tile_idx = compute_flat_tile_index(tile_indices, weight_relevant)
                weight_aligned_tile_size = weight_tile_size
                weight_tile_base = weight_base + weight_tile_idx * weight_aligned_tile_size * self.element_size
            
            # Fast path for weight: if tile is within one row, just count 1
            start_row = (weight_tile_base % self.bank_size) // self.row_size
            end_row = ((weight_tile_base + weight_tile_size * self.element_size - 1) % self.bank_size) // self.row_size
            
            if start_row == end_row:
                self._update_stat(weight_tile_base) # Just one access to trigger row activation
            else:
                for offset in range(weight_tile_size):
                    addr = weight_tile_base + offset * self.element_size
                    self._update_stat(addr)

        if generate_output:
            output_relevant = [DIM_P, DIM_Q, DIM_K, DIM_N]
            output_layout = layout_info.get('output_layout', 'sequential')
            row_size = layout_info.get('row_size', 1024)
            
            if output_layout == 'row_aligned':
                output_tile_idx = compute_flat_tile_index_l3_only(tile_indices, output_relevant)
                l2_tile_size = output_tile_size
                for level, dim, bound in dram_loop_dims:
                    if level == 2 and dim in output_relevant:
                        l2_tile_size *= bound
                output_aligned_tile_size = ((l2_tile_size + row_size - 1) // row_size) * row_size
                
                l2_offset = 0
                l2_stride = 1
                for level, dim, bound in reversed(dram_loop_dims):
                    if level == 2 and dim in output_relevant:
                        l2_offset += tile_indices.get((level, dim), 0) * l2_stride * output_tile_size
                        l2_stride *= bound
                
                output_tile_base = output_base + output_tile_idx * output_aligned_tile_size * self.element_size + l2_offset * self.element_size
            else:
                output_tile_idx = compute_flat_tile_index(tile_indices, output_relevant)
                output_aligned_tile_size = output_tile_size
                output_tile_base = output_base + output_tile_idx * output_aligned_tile_size * self.element_size
            
            # Fast path for output
            start_row = (output_tile_base % self.bank_size) // self.row_size
            end_row = ((output_tile_base + output_tile_size * self.element_size - 1) % self.bank_size) // self.row_size
            
            if start_row == end_row:
                self._update_stat(output_tile_base)
            else:
                for offset in range(output_tile_size):
                    addr = output_tile_base + offset * self.element_size
                    self._update_stat(addr)

    def _generate_input_access_body(self, trace, input_layout, tile_indices, 
                               h, w, h_block, w_block, h_in_block, w_in_block,
                               c_local, n_local,
                               stride_p_l3, stride_q_l3, stride_c_l3, stride_n_l3,
                               stride_p_l2, stride_q_l2, stride_c_l2, stride_n_l2,
                               c_l3_idx, n_l3_idx, c_l2_idx, n_l2_idx,
                               H_per_tile, W_per_tile, dram_loop_dims,
                               input_base, c_tile, n_tile,
                               w_start, w_end, block_w):
        if input_layout == 'row_aligned':
            block_base = (h_block * stride_p_l3 + 
                         w_block * stride_q_l3 + 
                         c_l3_idx * stride_c_l3 + 
                         n_l3_idx * stride_n_l3)
            
            c_l2_factor = 1
            n_l2_factor = 1
            for level, dim, bound in dram_loop_dims:
                if level == 2 and dim == 4: # DIM_C
                    c_l2_factor = bound
                elif level == 2 and dim == 6: # DIM_N
                    n_l2_factor = bound
            
            l2_tile_base = (c_l2_idx * H_per_tile * W_per_tile +
                           n_l2_idx * H_per_tile * W_per_tile * c_l2_factor)
            
            offset_in_block = (
                h_in_block * stride_p_l2 +
                w_in_block * stride_q_l2 +
                c_local * stride_c_l2 +
                n_local * stride_n_l2
            )
            
            addr = input_base + block_base + l2_tile_base + offset_in_block
            self._update_stat(addr)
        else:
            block_base = (h_block * stride_p_l3 + 
                         w_block * stride_q_l3 + 
                         c_tile * stride_c_l3 + 
                         n_tile * stride_n_l3)
            
            offset_in_block = (
                h_in_block * stride_p_l2 +
                w_in_block * stride_q_l2 +
                c_local * stride_c_l2 +
                n_local * stride_n_l2
            )
            
            addr = input_base + block_base + offset_in_block
            self._update_stat(addr)
