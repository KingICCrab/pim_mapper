"""
Trace generator for Ramulator2 validation.

Generates LD/ST address traces from pim_optimizer mappings,
respecting the loop ordering (permutation) specified in the mapping.

Memory hierarchy (matching arch.yaml):
- Level 0: PE (register)        ─┬─ buffer_tile (SRAM内复用，无DRAM访问)
- Level 1: GlobalBuffer (SRAM)  ─┘
- Level 2: RowBuffer            ─┬─ dram_loops (DRAM访问循环)
- Level 3: LocalDRAM            ─┘

Data reuse:
- Level 0+1: Data reused within SRAM buffer, no DRAM access generated
- Level 2+3: Generate DRAM accesses (RowBuffer and LocalDRAM loops)

Data layout: Follows loop permutation for efficiency.
- Most inner loop dimension has contiguous addresses.
- For Input: H/W order determined by Q/S vs P/R permutation positions.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
from tqdm import tqdm


@dataclass
class DRAMConfig:
    """DRAM configuration for address mapping.
    
    Important: row_buffer_bytes should match arch.yaml's row_buffer_size (default 1024 bytes).
    The ILP model uses this value for row activation counting.
    element_size should match arch.yaml's data_pr (default 8 bits = 1 byte).
    """
    num_channels: int = 1
    num_ranks: int = 1
    num_banks: int = 4
    num_rows: int = 16384
    row_buffer_bytes: int = 1024  # Row buffer size in bytes (matches arch.yaml)
    element_size: int = 1  # Bytes per element (arch.yaml data_pr / 8)
    
    @property
    def transaction_size(self) -> int:
        """Bytes per transaction."""
        return 64  # DDR5: 64 bytes per burst
    
    @property
    def row_size_bytes(self) -> int:
        """Row size in bytes."""
        return self.row_buffer_bytes
    
    @property
    def row_size_elements(self) -> int:
        """Row size in elements."""
        return self.row_buffer_bytes // self.element_size


# Dimension indices: R, S, P, Q, C, K, N
DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6
DIM_NAMES = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']


class TraceGenerator:
    """
    Generate Ramulator2 traces from pim_optimizer mappings.
    
    Only generates DRAM accesses - data reuse within buffer is not traced.
    """
    
    def __init__(self, dram_config: DRAMConfig = None):
        self.dram = dram_config or DRAMConfig()
        self.element_size = self.dram.element_size
    
    def generate_trace(self, mapping, workload) -> List[str]:
        """
        Generate DRAM access trace from mapping and workload.
        
        Iterates DRAM-level loops, generating accesses for each tile load.
        """
        trace = []
        
        # Compute buffer tile size (level 0 + level 1)
        buffer_tile = self._compute_buffer_tile_size(mapping)
        
        # Compute DRAM loop structure (level 2+)
        dram_loops = self._build_dram_loop_structure(mapping, workload, buffer_tile)
        
        # Get tensor sizes for address calculation
        H_in = workload.input_size['H']
        W_in = workload.input_size['W']
        stride_h, stride_w = workload.stride[1], workload.stride[0]
        dilation_h, dilation_w = workload.dilation[1], workload.dilation[0]
        
        # Compute data layouts based on permutation
        layout_info = self._compute_data_layouts(mapping, workload, H_in, W_in, buffer_tile)
        
        # Calculate total iterations for progress bar
        total_iters = 1
        for loop_info in dram_loops:
            total_iters *= loop_info['bound']
        
        # Progress bar
        pbar = tqdm(total=total_iters, desc="Generating trace", unit="tiles", leave=False)
        
        # Track previous tile indices to detect relevant dimension changes
        # Using list to allow mutation in nested function
        prev_indices = [None]
        
        # Iterate DRAM-level loops
        def iterate_dram_loops(level_idx: int, indices: Dict[int, int]):
            if level_idx >= len(dram_loops):
                # Determine which tensors need new accesses based on relevancy
                # Only generate accesses when relevant dimensions change
                input_changed = True
                weight_changed = True
                output_changed = True
                
                if prev_indices[0] is not None:
                    prev = prev_indices[0]
                    # Input relevant: R, S, P, Q, C, N (not K)
                    input_changed = any(
                        indices.get(d, 0) != prev.get(d, 0)
                        for d in [DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_N]
                    )
                    # Weight relevant: R, S, C, K (not P, Q, N)
                    weight_changed = any(
                        indices.get(d, 0) != prev.get(d, 0)
                        for d in [DIM_R, DIM_S, DIM_C, DIM_K]
                    )
                    # Output relevant: P, Q, K, N (not R, S, C)
                    output_changed = any(
                        indices.get(d, 0) != prev.get(d, 0)
                        for d in [DIM_P, DIM_Q, DIM_K, DIM_N]
                    )
                
                prev_indices[0] = indices.copy()
                
                # Generate DRAM accesses for this tile
                self._generate_tile_accesses(
                    trace, indices, buffer_tile, workload,
                    H_in, W_in, stride_h, stride_w, dilation_h, dilation_w,
                    layout_info,
                    generate_input=input_changed,
                    generate_weight=weight_changed,
                    generate_output=output_changed
                )
                pbar.update(1)
                return
            
            loop_info = dram_loops[level_idx]
            dim = loop_info['dim']
            bound = loop_info['bound']
            stride = loop_info['stride']
            
            base = indices.get(dim, 0)
            for i in range(bound):
                new_indices = indices.copy()
                new_indices[dim] = base + i * stride
                iterate_dram_loops(level_idx + 1, new_indices)
        
        initial_indices = {d: 0 for d in range(7)}
        iterate_dram_loops(0, initial_indices)
        pbar.close()
        
        return trace
    
    def _compute_buffer_tile_size(self, mapping) -> Dict[int, int]:
        """
        Compute buffer tile size = product of level 0 and level 1 tiling factors.
        
        Memory hierarchy:
        - Level 0: PE (register)      ─┬─ buffer_tile (SRAM内复用)
        - Level 1: GlobalBuffer (SRAM) ─┘
        - Level 2: RowBuffer           ─┬─ dram_loops (DRAM访问循环)
        - Level 3: LocalDRAM           ─┘
        
        Data within Level 0+1 is reused in SRAM without DRAM access.
        Level 2+3 loops generate DRAM accesses.
        
        Note: Level 0 uses 'H', 'W', 'Internal', 'temporal' format.
              Level 1+ uses 'spatial', 'temporal' format.
        """
        tile = {d: 1 for d in range(7)}
        
        # Include Level 0 and 1 only (SRAM levels)
        for level in [0, 1]:
            if level not in mapping.loop_bounds:
                continue
            
            level_bounds = mapping.loop_bounds[level]
            
            if level == 0:
                # Level 0 (PE): Uses 'H', 'W', 'Internal', 'temporal' keys
                # Each key maps dimension -> bound
                for key in ['H', 'W', 'Internal', 'temporal']:
                    if key in level_bounds:
                        for d, bound in level_bounds[key].items():
                            tile[d] *= bound
            else:
                # Level 1+: Uses 'spatial', 'temporal' keys
                for key in ['spatial', 'temporal']:
                    if key in level_bounds:
                        for d, bound in level_bounds[key].items():
                            tile[d] *= bound
        
        return tile
    
    def _compute_data_layouts(self, mapping, workload, H_in, W_in, buffer_tile: Dict[int, int]) -> Dict:
        """
        Compute data layouts based on loop permutation and layout mode.
        
        Args:
            buffer_tile: Buffer tile sizes (Level 0+1) for each dimension
        
        Returns layout info for address calculation.
        """
        # Build loop orders for each tensor
        weight_dims = [DIM_K, DIM_C, DIM_R, DIM_S]
        output_dims = [DIM_N, DIM_K, DIM_P, DIM_Q]
        input_dims = [DIM_N, DIM_C, DIM_Q, DIM_S, DIM_P, DIM_R]
        
        weight_loop_order = self._build_loop_order(mapping, weight_dims)
        output_loop_order = self._build_loop_order(mapping, output_dims)
        input_loop_order = self._build_loop_order(mapping, input_dims)
        
        # Get dim orders, ensuring all required dims are included
        weight_order = self._get_dim_order(weight_loop_order, weight_dims)
        output_order = self._get_dim_order(output_loop_order, output_dims)
        input_order = self._get_dim_order(input_loop_order, input_dims)
        
        # Get input block_h and block_w from mapping
        block_h = mapping.tile_info.get('block_h', 1)
        block_w = mapping.tile_info.get('block_w', 1)
        
        # Number of blocks in each dimension (ceiling division)
        num_blocks_h = (H_in + block_h - 1) // block_h
        num_blocks_w = (W_in + block_w - 1) // block_w
        
        # Get layout modes from mapping
        # row_aligned 语义：把 RowBuffer level 的 tile padding 到占满 row buffer
        # 这样 DRAM level 的 stride 是 row_buffer_size 的整数倍，避免 row crossing
        input_layout = mapping.layout.get(0, "sequential")
        weight_layout = mapping.layout.get(1, "sequential")
        output_layout = mapping.layout.get(2, "sequential")
        
        # Row size in elements for alignment (using element_size from config)
        row_size = self.dram.row_size_elements
        
        # =======================================================================
        # Compute RowBuffer tile sizes (Level 0+1+2 factors)
        # Note: mapping.get_tile_size doesn't handle Level 0's special format,
        # so we use buffer_tile (Level 0+1) and multiply by Level 2 factors.
        # =======================================================================
        
        # Get Level 2 factors
        level2_factors = {d: 1 for d in range(7)}
        if 2 in mapping.loop_bounds:
            for key in ['spatial', 'temporal']:
                if key in mapping.loop_bounds[2]:
                    for d, bound in mapping.loop_bounds[2][key].items():
                        level2_factors[d] *= bound
        
        # RowBuffer tile = buffer_tile (Level 0+1) × Level 2 factors
        input_rb_tile_c = buffer_tile[DIM_C] * level2_factors[DIM_C]
        input_rb_tile_n = buffer_tile[DIM_N] * level2_factors[DIM_N]
        
        weight_rb_tile_k = buffer_tile[DIM_K] * level2_factors[DIM_K]
        weight_rb_tile_c = buffer_tile[DIM_C] * level2_factors[DIM_C]
        weight_rb_tile_r = buffer_tile[DIM_R] * level2_factors[DIM_R]
        weight_rb_tile_s = buffer_tile[DIM_S] * level2_factors[DIM_S]
        
        output_rb_tile_n = buffer_tile[DIM_N] * level2_factors[DIM_N]
        output_rb_tile_k = buffer_tile[DIM_K] * level2_factors[DIM_K]
        output_rb_tile_p = buffer_tile[DIM_P] * level2_factors[DIM_P]
        output_rb_tile_q = buffer_tile[DIM_Q] * level2_factors[DIM_Q]
        
        # =======================================================================
        # Compute strides with row_aligned padding
        # row_aligned: RowBuffer tile padding to row buffer capacity
        # =======================================================================
        
        # Block size for Input (one spatial block in H×W)
        block_size = block_h * block_w
        
        # Input strides based on tile-wise layout (using N, C, P, Q → N, C, H, W)
        # Map: P controls W direction, Q controls H direction
        # Input dims for layout: [N, C, P, Q] where P→num_blocks_w, Q→num_blocks_h
        input_layout_dims = [DIM_N, DIM_C, DIM_P, DIM_Q]
        # For Input: Level 2 tile uses block_h/block_w (data layout), not P/Q tiling factors (data access)
        # Because sliding window causes tile_h ≠ block_h
        input_rb_n = buffer_tile[DIM_N] * level2_factors[DIM_N]
        input_rb_c = buffer_tile[DIM_C] * level2_factors[DIM_C]
        input_tile_sizes_for_stride = {
            2: {
                DIM_N: input_rb_n,
                DIM_C: input_rb_c,
                DIM_P: block_w,  # data layout tile in W direction
                DIM_Q: block_h,  # data layout tile in H direction
            },
            # Level 3: tile count (total / level2_tile), NOT total size
            3: {
                DIM_N: (workload.N + input_rb_n - 1) // input_rb_n,
                DIM_C: (workload.C + input_rb_c - 1) // input_rb_c,
                DIM_P: num_blocks_w,  # W_in / block_w
                DIM_Q: num_blocks_h,  # H_in / block_h
            },
        }
        # Build input loop order with only [N, C, P, Q] (not R, S)
        input_layout_loop_order = self._build_loop_order(mapping, input_layout_dims)
        input_strides = self._compute_tile_wise_strides(
            input_layout_loop_order, input_tile_sizes_for_stride, input_layout, row_size
        )
        
        # Weight strides based on tile-wise layout
        # Each (level, dim) has its own stride
        weight_sizes = {DIM_K: workload.K, DIM_C: workload.C, DIM_R: workload.R, DIM_S: workload.S}
        weight_level2_tiles = {d: buffer_tile[d] * level2_factors[d] for d in weight_dims}
        weight_tile_sizes = {
            2: weight_level2_tiles,  # Level 2 tile
            # Level 3: tile count (total / level2_tile), NOT total size
            3: {d: (weight_sizes[d] + weight_level2_tiles[d] - 1) // weight_level2_tiles[d] for d in weight_dims},
        }
        weight_strides = self._compute_tile_wise_strides(
            weight_loop_order, weight_tile_sizes, weight_layout, row_size
        )
        
        # Output strides based on tile-wise layout
        output_sizes = {DIM_N: workload.N, DIM_K: workload.K, DIM_P: workload.P, DIM_Q: workload.Q}
        output_level2_tiles = {d: buffer_tile[d] * level2_factors[d] for d in output_dims}
        output_tile_sizes = {
            2: output_level2_tiles,  # Level 2 tile
            # Level 3: tile count (total / level2_tile), NOT total size
            3: {d: (output_sizes[d] + output_level2_tiles[d] - 1) // output_level2_tiles[d] for d in output_dims},
        }
        output_strides = self._compute_tile_wise_strides(
            output_loop_order, output_tile_sizes, output_layout, row_size
        )
        
        # Base addresses: place each tensor in a different bank
        # 
        # DRAM address mapping: | bank | row | column offset |
        #                       | 2b   |     | 10 bits      |
        #
        # row_buffer_bytes = 1024 = 0x400 (10 bits for column)
        # bank_bits = 2 (4 banks)
        # num_rows = 16384 (14 bits for row)
        # 
        # Bank extraction: bank = addr >> (col_bits + row_bits)
        # Row extraction:  row = (addr >> col_bits) & row_mask
        # Col extraction:  col = addr & col_mask
        #
        # Each bank has its own address space:
        # Bank 0: 0x00000000 - 0x00FFFFFF (16MB)
        # Bank 1: 0x01000000 - 0x01FFFFFF (16MB)
        # Bank 2: 0x02000000 - 0x02FFFFFF (16MB)
        # Bank 3: 0x03000000 - 0x03FFFFFF (16MB)
        #
        bank_size = self.dram.row_buffer_bytes * self.dram.num_rows  # 1024 * 16384 = 16MB
        input_base = 0 * bank_size    # Bank 0: 0x00000000
        weight_base = 1 * bank_size   # Bank 1: 0x01000000
        output_base = 2 * bank_size   # Bank 2: 0x02000000
        
        return {
            'block_h': block_h,
            'block_w': block_w,
            'block_size': block_size,
            'num_blocks_h': num_blocks_h,
            'num_blocks_w': num_blocks_w,
            'input_layout': input_layout,
            'weight_layout': weight_layout,
            'output_layout': output_layout,
            'row_size': row_size,
            'input_order': input_order,  # outer to inner for iteration
            'weight_order': weight_order,
            'output_order': output_order,
            'input_strides': input_strides,
            'weight_strides': weight_strides,
            'output_strides': output_strides,
            'input_base': input_base,
            'weight_base': weight_base,
            'output_base': output_base,
            # Tile-wise loop order for stride calculation
            'input_loop_order': input_layout_loop_order,
            'weight_loop_order': weight_loop_order,
            'output_loop_order': output_loop_order,
            'input_tile_sizes': input_tile_sizes_for_stride,
            'weight_tile_sizes': {
                2: {d: buffer_tile[d] * level2_factors[d] for d in weight_dims},
            },
            'output_tile_sizes': {
                2: {d: buffer_tile[d] * level2_factors[d] for d in output_dims},
            },
        }
    
    def _compute_tile_wise_strides(self, loop_order: List[Tuple[int, int]], 
                                    tile_sizes: Dict[int, Dict[int, int]],
                                    layout: str, row_size: int) -> Dict[Tuple[int, int], int]:
        """
        Compute strides for tile-wise layout.
        
        Args:
            loop_order: [(level, dim), ...] from outer to inner (from permutation)
            tile_sizes: {level: {dim: tile_size}} - tile size at each level
            layout: "sequential" or "row_aligned"
            row_size: row buffer size in elements
            
        Returns:
            {(level, dim): stride} for each (level, dim) in tile_sizes
            
        row_aligned semantics for Input tensor:
            For Input[N, C, H, W], row_aligned means each (N, C) slice is padded
            to start at a row boundary. This means stride(C) and stride(N) should
            be multiples of row_size.
            
            The innermost block is H×W (or block_h × block_w for tiled case).
            When H×W < row_size, each C channel is padded to row_size.
            When H×W > row_size, each C channel is padded to ceil(H×W / row_size) * row_size.
        """
        strides = {}
        stride = 1  # Start from 1 (element level)
        
        # Build complete loop order: 
        # 1. Start with permutation-defined order
        # 2. Add any missing (level, dim) pairs from tile_sizes
        loop_order_set = set(loop_order)
        complete_order = list(loop_order)
        
        # Add missing dims from tile_sizes (Level 2 first, then Level 3)
        for level in [2, 3]:
            if level in tile_sizes:
                for dim in tile_sizes[level]:
                    if (level, dim) not in loop_order_set:
                        complete_order.append((level, dim))
                        loop_order_set.add((level, dim))
        
        # Process from inner to outer (reverse order)
        reversed_order = list(reversed(complete_order))
        
        # For row_aligned layout, we need to apply alignment at specific boundaries:
        # - For Input: after processing P (W) and Q (H), i.e., before C and N
        # - The spatial dimensions (P, Q for Input) form the "inner block"
        # - Channel and batch dimensions (C, N) are "outer" and should be row-aligned
        #
        # row_aligned semantics:
        # - RowBuffer tile (e.g., H×W×C=10×10×2=200) pads to row_size (1024)
        # - Within the padded tile, Level 2 dims use natural strides (e.g., C stride = H×W = 100)
        # - Between tiles (Level 3), stride = padded_tile_size (1024)
        
        # Identify spatial dims (P, Q for Input; R, S for Weight)
        spatial_dims_2d = {DIM_P, DIM_Q}  # For Input: these form the H×W block
        
        # Track the stride before and after padding for row_aligned
        stride_before_padding = None
        padded_rb_tile_size = None
        
        for i, (level, dim) in enumerate(reversed_order):
            strides[(level, dim)] = stride
            
            # Get tile size for this (level, dim)
            if level in tile_sizes and dim in tile_sizes[level]:
                dim_tile_size = tile_sizes[level][dim]
            else:
                dim_tile_size = 1  # No tiling at this level
            
            stride *= dim_tile_size
            
            # Apply row alignment after processing all spatial dims at Level 2
            # This ensures each RowBuffer tile starts at a row boundary
            if layout == "row_aligned" and level == 2 and dim in spatial_dims_2d:
                # Check if we've processed all spatial dims (P and Q)
                remaining_spatial_at_level2 = False
                for j in range(i + 1, len(reversed_order)):
                    lv, d = reversed_order[j]
                    if lv == 2 and d in spatial_dims_2d:
                        remaining_spatial_at_level2 = True
                        break
                
                if not remaining_spatial_at_level2:
                    # Save stride before padding (this is the actual RowBuffer tile size)
                    stride_before_padding = stride
                    
                    # Align stride to row boundary for Level 3 (between tiles)
                    if stride % row_size != 0:
                        padded_rb_tile_size = ((stride + row_size - 1) // row_size) * row_size
                    else:
                        padded_rb_tile_size = stride
                    
                    # DON'T update stride here - Level 2 dims should still use unpadded strides
            
            # For Level 3 dims: use padded tile size as base stride
            if layout == "row_aligned" and level == 3 and padded_rb_tile_size is not None:
                # Check if this is the first Level 3 dim after padding
                # For these dims, stride should jump by padded_rb_tile_size
                prev_level = reversed_order[i-1][0] if i > 0 else None
                if prev_level == 2:  # First Level 3 dim after Level 2
                    stride = padded_rb_tile_size
                    strides[(level, dim)] = stride
                    stride *= dim_tile_size
        
        return strides
    
    def _build_dram_loop_structure(self, mapping, workload, buffer_tile) -> List[Dict]:
        """
        Build DRAM-level loop structure (level 2+ for RowBuffer and LocalDRAM).
        
        Memory hierarchy:
        - Level 0: PE              ─┬─ buffer_tile (SRAM内复用)
        - Level 1: GlobalBuffer    ─┘
        - Level 2: RowBuffer       ─┬─ dram_loops (DRAM访问循环)
        - Level 3: LocalDRAM       ─┘
        
        Level 2+3 loops generate DRAM accesses.
        """
        loops = []
        num_mems = max(mapping.loop_bounds.keys()) + 1 if mapping.loop_bounds else 0
        
        # Compute cumulative tile sizes for stride calculation
        # Note: Level 0 uses 'H', 'W', 'Internal', 'temporal' keys (PE level special format)
        #       Level 1+ uses 'spatial', 'temporal' keys
        tile_sizes = [{d: 1 for d in range(7)} for _ in range(num_mems + 1)]
        for m in range(num_mems):
            for d in range(7):
                tile_sizes[m + 1][d] = tile_sizes[m][d]
            if m in mapping.loop_bounds:
                level_bounds = mapping.loop_bounds[m]
                if m == 0:
                    # Level 0 (PE): Uses 'H', 'W', 'Internal', 'temporal' keys
                    for key in ['H', 'W', 'Internal', 'temporal']:
                        if key in level_bounds:
                            for d, bound in level_bounds[key].items():
                                tile_sizes[m + 1][d] *= bound
                else:
                    # Level 1+: Uses 'spatial', 'temporal' keys
                    for loop_type in ['spatial', 'temporal']:
                        if loop_type in level_bounds:
                            for d, bound in level_bounds[loop_type].items():
                                tile_sizes[m + 1][d] *= bound
        
        # Build loops from outermost DRAM level to innermost DRAM level
        # DRAM levels are 2+ (skip level 0=PE, level 1=SRAM)
        for m in range(num_mems - 1, 1, -1):  # Start from highest, stop before level 2
            if m not in mapping.loop_bounds:
                continue
            
            perm_order = mapping.get_loop_order(m) if hasattr(mapping, 'get_loop_order') else list(range(7))
            perm_order = list(reversed(perm_order))
            
            level_bounds = {}
            for loop_type in ['spatial', 'temporal']:
                if loop_type in mapping.loop_bounds[m]:
                    for d, bound in mapping.loop_bounds[m][loop_type].items():
                        if bound > 1:
                            level_bounds[d] = level_bounds.get(d, 1) * bound
            
            for dim in perm_order:
                if dim in level_bounds and level_bounds[dim] > 1:
                    loops.append({
                        'dim': dim,
                        'bound': level_bounds[dim],
                        'stride': tile_sizes[m][dim],
                        'level': m,
                    })
        
        # Add level 2 loops (RowBuffer level)
        if 2 in mapping.loop_bounds:
            perm_order = mapping.get_loop_order(2) if hasattr(mapping, 'get_loop_order') else list(range(7))
            perm_order = list(reversed(perm_order))
            
            level_bounds = {}
            for loop_type in ['spatial', 'temporal']:
                if loop_type in mapping.loop_bounds[2]:
                    for d, bound in mapping.loop_bounds[2][loop_type].items():
                        if bound > 1:
                            level_bounds[d] = level_bounds.get(d, 1) * bound
            
            for dim in perm_order:
                if dim in level_bounds and level_bounds[dim] > 1:
                    loops.append({
                        'dim': dim,
                        'bound': level_bounds[dim],
                        'stride': tile_sizes[2][dim],
                        'level': 2,
                    })
        
        return loops
    
    def _generate_tile_accesses(
        self, trace: List[str], tile_start: Dict[int, int], tile_size: Dict[int, int],
        workload, H_in, W_in, stride_h, stride_w, dilation_h, dilation_w,
        layout_info: Dict,
        generate_input: bool = True,
        generate_weight: bool = True,
        generate_output: bool = True
    ):
        """
        Generate DRAM accesses for all unique elements in a buffer tile.
        
        Args:
            generate_input: If True, generate Input tensor accesses
            generate_weight: If True, generate Weight tensor accesses
            generate_output: If True, generate Output tensor accesses
        """
        n0 = tile_start[DIM_N]
        k0 = tile_start[DIM_K]
        c0 = tile_start[DIM_C]
        p0 = tile_start[DIM_P]
        q0 = tile_start[DIM_Q]
        r0 = tile_start[DIM_R]
        s0 = tile_start[DIM_S]
        
        ns = min(tile_size[DIM_N], workload.N - n0)
        ks = min(tile_size[DIM_K], workload.K - k0)
        cs = min(tile_size[DIM_C], workload.C - c0)
        ps = min(tile_size[DIM_P], workload.P - p0)
        qs = min(tile_size[DIM_Q], workload.Q - q0)
        rs = min(tile_size[DIM_R], workload.R - r0)
        ss = min(tile_size[DIM_S], workload.S - s0)
        
        # Extract layout info
        block_h = layout_info['block_h']
        block_w = layout_info['block_w']
        input_strides = layout_info['input_strides']
        weight_strides = layout_info['weight_strides']
        output_strides = layout_info['output_strides']
        input_base = layout_info['input_base']
        weight_base = layout_info['weight_base']
        output_base = layout_info['output_base']
        
        # Input tile: I[n, c, h, w] with block-wise layout
        # Iterate in permutation order (input_order), track visited (n, c, h, w) to avoid duplicates
        input_order = layout_info['input_order']  # outer to inner
        input_tile_starts = {DIM_N: n0, DIM_C: c0, DIM_Q: q0, DIM_S: s0, DIM_P: p0, DIM_R: r0}
        input_tile_sizes = {DIM_N: ns, DIM_C: cs, DIM_Q: qs, DIM_S: ss, DIM_P: ps, DIM_R: rs}
        visited_input: Set[Tuple[int, int, int, int]] = set()  # (n, c, h, w)
        
        def iterate_input(dim_idx: int, indices: Dict[int, int]):
            if dim_idx >= len(input_order):
                # Compute h, w from q, s, p, r
                h = indices[DIM_Q] * stride_h + indices[DIM_S] * dilation_h
                w = indices[DIM_P] * stride_w + indices[DIM_R] * dilation_w
                n = indices[DIM_N]
                c = indices[DIM_C]
                
                # Check bounds and uniqueness
                if h >= H_in or w >= W_in:
                    return
                key = (n, c, h, w)
                if key in visited_input:
                    return
                visited_input.add(key)
                
                # Element-wise address calculation using (h, w) directly
                # Map: P → w, Q → h (element coordinates)
                input_indices = {DIM_N: n, DIM_C: c, DIM_P: w, DIM_Q: h}
                idx = self._compute_tile_wise_address(
                    input_indices, 
                    layout_info['input_loop_order'],
                    input_strides,
                    layout_info['input_tile_sizes']
                )
                addr = input_base + idx * self.element_size
                trace.append(f"LD 0x{addr:08X}")
                return
            
            dim = input_order[dim_idx]
            start = input_tile_starts[dim]
            size = input_tile_sizes[dim]
            for i in range(size):
                new_indices = indices.copy()
                new_indices[dim] = start + i
                iterate_input(dim_idx + 1, new_indices)
        
        if generate_input:
            iterate_input(0, {})
        
        # Weight tile: W[k, c, r, s]
        if generate_weight:
            self._iterate_tensor(
                trace, 'weight',
                dim_order=layout_info['weight_order'],
                loop_order=layout_info['weight_loop_order'],
                strides=weight_strides,
                tile_sizes_info=layout_info['weight_tile_sizes'],
                tile_starts={DIM_K: k0, DIM_C: c0, DIM_R: r0, DIM_S: s0},
                tile_sizes={DIM_K: ks, DIM_C: cs, DIM_R: rs, DIM_S: ss},
                base_addr=weight_base
            )
        
        # Output tile: O[n, k, p, q]
        if generate_output:
            self._iterate_tensor(
                trace, 'output',
                dim_order=layout_info['output_order'],
                loop_order=layout_info['output_loop_order'],
                strides=output_strides,
                tile_sizes_info=layout_info['output_tile_sizes'],
                tile_starts={DIM_N: n0, DIM_K: k0, DIM_P: p0, DIM_Q: q0},
                tile_sizes={DIM_N: ns, DIM_K: ks, DIM_P: ps, DIM_Q: qs},
                base_addr=output_base
            )
    
    def _compute_tile_wise_address(self, indices: Dict[int, int], loop_order: List[Tuple[int, int]], 
                                    strides: Dict[Tuple[int, int], int],
                                    tile_sizes_info: Dict[int, Dict[int, int]]) -> int:
        """Compute address using tile-wise layout.
        
        For Level 3: use tile index (val // level2_tile_size)
        For Level 2: use offset within tile (val % level2_tile_size)
        
        Note: Iterates over all (level, dim) pairs in strides, not just loop_order,
        to handle dimensions that aren't in permutation but have strides.
        """
        idx = 0
        # Iterate over all (level, dim) pairs that have strides
        for (level, dim), stride in strides.items():
            val = indices.get(dim, 0)
            if level == 3 and 2 in tile_sizes_info and dim in tile_sizes_info[2]:
                # Level 3: tile index
                level2_tile = tile_sizes_info[2][dim]
                idx += (val // level2_tile) * stride
            elif level == 2 and 2 in tile_sizes_info and dim in tile_sizes_info[2]:
                # Level 2: offset within tile
                level2_tile = tile_sizes_info[2][dim]
                idx += (val % level2_tile) * stride
            else:
                idx += val * stride
        return idx
    
    def write_trace(self, trace: List[str], filepath: str):
        """Write trace to file."""
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write('\n'.join(trace) + '\n')
        print(f"Wrote {len(trace)} trace lines to {filepath}")
    
    # =========================================================================
    # Helper methods
    # =========================================================================
    
    def _build_loop_order(self, mapping, tensor_dims: List[int]) -> List[Tuple[int, int]]:
        """Build loop order [(level, dim), ...] from outer to inner for given tensor dims."""
        loop_order = []
        for m in [3, 2]:  # Level 3 (outer) then Level 2 (inner)
            if m not in mapping.permutation:
                continue
            perm = mapping.permutation[m]
            sorted_perms = sorted(perm.items(), key=lambda x: x[0], reverse=True)
            for p_level, dim in sorted_perms:
                if dim in tensor_dims:
                    loop_order.append((m, dim))
        return loop_order
    
    def _get_dim_order(self, loop_order: List[Tuple[int, int]], all_dims: List[int] = None) -> List[int]:
        """Extract unique dims from loop_order, preserving outer-to-inner order.
        
        Args:
            loop_order: [(level, dim), ...] from permutation
            all_dims: If provided, ensure all these dims are included (append missing ones at end)
        """
        seen = set()
        result = []
        for level, dim in loop_order:
            if dim not in seen:
                seen.add(dim)
                result.append(dim)
        
        # Append any missing dims from all_dims
        if all_dims:
            for dim in all_dims:
                if dim not in seen:
                    result.append(dim)
        
        return result
    
    def _iterate_tensor(self, trace: List[str], tensor_type: str,
                        dim_order: List[int], loop_order: List[Tuple[int, int]],
                        strides: Dict[Tuple[int, int], int],
                        tile_sizes_info: Dict[int, Dict[int, int]],
                        tile_starts: Dict[int, int], tile_sizes: Dict[int, int],
                        base_addr: int):
        """Generic tensor iteration for Weight and Output."""
        def iterate(dim_idx: int, indices: Dict[int, int]):
            if dim_idx >= len(dim_order):
                idx = self._compute_tile_wise_address(indices, loop_order, strides, tile_sizes_info)
                addr = base_addr + idx * self.element_size
                trace.append(f"LD 0x{addr:08X}")
                if tensor_type == 'output':
                    trace.append(f"ST 0x{addr:08X}")
                return
            dim = dim_order[dim_idx]
            start = tile_starts[dim]
            size = tile_sizes[dim]
            for i in range(size):
                new_indices = indices.copy()
                new_indices[dim] = start + i
                iterate(dim_idx + 1, new_indices)
        
        iterate(0, {})
