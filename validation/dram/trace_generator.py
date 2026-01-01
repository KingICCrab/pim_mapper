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

IMPORTANT PRINCIPLE: 访问顺序和数据布局一致性能好
- Data layout should match access order for optimal performance
- When access order (from loop permutation) matches data layout order,
  memory accesses are sequential → better cache/row buffer utilization
- Mismatch causes random access patterns → poor performance

===============================================================================
INPUT TENSOR ITERATION SCHEME (方案 B)
===============================================================================

Input has a SPECIAL property: Access tile ≠ Data layout tile

Problem:
- Access tile: determined by sliding window (P, Q, R, S) → (H, W)
- Data layout tile: determined by block_h × block_w
- tile_h, tile_w from sliding window are NOT aligned with block_h, block_w
- One access tile may span MULTIPLE data layout blocks
- Constraint: block_h ≥ tile_h, block_w ≥ tile_w

Solution (方案 B): Iterate by data layout blocks, access only needed elements

    def load_input_tile(c_tile, h_start, w_start, ...):
        tile_h_end = h_start + tile_h
        tile_w_end = w_start + tile_w
        
        # 1. Find affected blocks
        for block in affected_blocks:
            
            # 2. Compute local range within this block
            h_local_start = max(0, h_start - block.h_base)
            h_local_end = min(block_h, tile_h_end - block.h_base)
            w_local_start = max(0, w_start - block.w_base)
            w_local_end = min(block_w, tile_w_end - block.w_base)
            
            # 3. Access only the needed elements (NOT entire block)
            for h_local in range(h_local_start, h_local_end):
                for w_local in range(w_local_start, w_local_end):
                    offset = h_local * block_w + w_local
                    addr = base + block.idx * block_size + offset

Key properties:
- Addresses are NOT contiguous within a tile (gaps between rows)
- Tiles may OVERLAP (sliding window causes repeated access to same elements)
- Iteration order follows data layout (block by block), not logical tile

Diagram:
    ┌─────────────────┬─────────────────┐
    │     Block 0     │     Block 1     │
    │      ┌──────────┼────────┐        │
    │      │ Access   │ Tile   │        │  ← Tile spans 2 blocks
    │      │ partial  │ partial│        │
    │      └──────────┼────────┘        │
    └─────────────────┴─────────────────┘

===============================================================================
INPUT TENSOR INTER-TILE ITERATION (方案 A)
===============================================================================

Tile 间迭代方案：按 loop permutation 顺序

    # DRAM loop structure follows permutation (outer to inner)
    for c_tile in range(C_tiles):        # 按 permutation 顺序
      for p_tile in range(P_tiles):      # (如果 P 在 DRAM level)
        for q_tile in range(Q_tiles):    # (如果 Q 在 DRAM level)
          load_input_tile(c_tile, p_tile, q_tile)

Key decisions:
1. 迭代顺序：遵循 permutation（不按数据布局顺序）
2. Overlap：不去重（同一元素可以被多个 tile 重复访问）
3. P/Q tiling：如果 P, Q 在 DRAM level 有 tiling，会导致 H, W 方向的 overlap

Overlap 示例 (P tiling, stride=1, R=3):
    P tile 0: p ∈ [0, 4)  → h ∈ [0, 6)
    P tile 1: p ∈ [4, 8)  → h ∈ [4, 10)
                               ↑
                           h ∈ [4, 6) 被两个 tile 都访问（不去重）

===============================================================================
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
        # Debug tracking
        self._input_debug_count = 0
        self._input_debug_info = []
        self._input_debug_boundary_info = []  # Track boundary crossings specifically
    
    def print_input_debug_info(self, num_entries: int = 50):
        """Print detailed debug info for Input address calculation."""
        if not hasattr(self, '_input_debug_info') or not self._input_debug_info:
            print("No Input debug info available.")
            return
        
        print("\n" + "=" * 100)
        print("INPUT ADDRESS CALCULATION DEBUG INFO")
        print("=" * 100)
        
        entries = self._input_debug_info[:num_entries]
        
        # Print header
        print(f"\n{'idx':>5} | {'h':>3} {'w':>3} | {'h_blk':>5} {'w_blk':>5} | {'h_in':>4} {'w_in':>4} | "
              f"{'block_base':>12} {'l2_base':>8} {'offset':>8} | {'addr':>12} | {'row':>5} {'col':>5}")
        print("-" * 100)
        
        prev_row = None
        for entry in entries:
            l2_base = entry.get('l2_tile_base', 0)
            row_switch = ""
            if prev_row is not None and entry['row'] != prev_row:
                row_switch = " <-- ROW SWITCH"
            
            print(f"{entry['idx']:>5} | {entry['h']:>3} {entry['w']:>3} | "
                  f"{entry['h_block']:>5} {entry['w_block']:>5} | "
                  f"{entry['h_in_block']:>4} {entry['w_in_block']:>4} | "
                  f"{entry['block_base']:>12} {l2_base:>8} {entry['offset_in_block']:>8} | "
                  f"0x{entry['addr']:08X} | {entry['row']:>5} {entry['col']:>5}{row_switch}")
            prev_row = entry['row']
        
        # Analyze row switch pattern
        print("\n" + "-" * 100)
        print("ROW SWITCH ANALYSIS:")
        row_switches = []
        prev_row = None
        for entry in self._input_debug_info:
            if prev_row is not None and entry['row'] != prev_row:
                row_switches.append((entry['idx'], prev_row, entry['row'], entry['h'], entry['w']))
            prev_row = entry['row']
        
        print(f"Total row switches in debug entries: {len(row_switches)}")
        print(f"\nFirst 20 row switches:")
        for idx, from_row, to_row, h, w in row_switches[:20]:
            print(f"  idx={idx}: row {from_row} -> {to_row} (h={h}, w={w})")
    
    def generate_trace(self, mapping, workload, strict_ordering: bool = False) -> List[str]:
        """
        Generate DRAM access trace from mapping and workload.
        
        Iterates DRAM-level loops, generating accesses for each tile load.
        
        Args:
            mapping: Mapping object
            workload: Workload object
            strict_ordering: If True, generates accesses in strict loop order (H, W)
                             instead of optimizing for block-major access.
                             This exposes row thrashing for misaligned tiles.
        """
        trace = []
        
        # Compute buffer tile size (level 0 + level 1)
        buffer_tile = self._compute_buffer_tile_size(mapping)
        
        # Compute DRAM loop structure (level 2+)
        dram_loops = self._build_dram_loop_structure(mapping, workload, buffer_tile)
        
        # Build tile_strides dict: {dim: tile_stride} for converting tile index to element coord
        tile_strides = {d: 1 for d in range(7)}  # default stride=1 for non-tiled dims
        for loop_info in dram_loops:
            tile_strides[loop_info['dim']] = loop_info['tile_stride']
        
        # Build dram_loop_dims: [(level, dim, bound), ...] for flat tile index calculation
        # Include level to distinguish same dim at different levels (e.g., P at level 3 and level 2)
        dram_loop_dims = [(loop_info['level'], loop_info['dim'], loop_info['bound']) for loop_info in dram_loops]
        
        # Get tensor sizes for address calculation
        H_in = workload.input_size['H']
        W_in = workload.input_size['W']
        stride_h, stride_w = workload.stride[1], workload.stride[0]
        dilation_h, dilation_w = workload.dilation[1], workload.dilation[0]
        
        # Compute data layouts based on permutation
        layout_info = self._compute_data_layouts(mapping, workload, H_in, W_in, buffer_tile)
        
        # Add dram_loop_dims for flat tile index calculation
        layout_info['dram_loop_dims'] = dram_loop_dims
        
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
        # indices uses (level, dim) as key to distinguish same dim at different levels
        def iterate_dram_loops(level_idx: int, indices: Dict[Tuple[int, int], int]):
            if level_idx >= len(dram_loops):
                # Determine which tensors need new accesses based on relevancy
                # Only generate accesses when relevant dimensions change
                input_changed = True
                weight_changed = True
                output_changed = True
                
                if prev_indices[0] is not None:
                    prev = prev_indices[0]
                    # Input relevant: R, S, P, Q, C, N (not K)
                    # Check ALL (level, dim) pairs for these dimensions
                    input_dims = [DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_N]
                    input_changed = any(
                        indices.get(key, 0) != prev.get(key, 0)
                        for key in indices.keys()
                        if key[1] in input_dims
                    )
                    
                    # Weight relevant: R, S, C, K (not P, Q, N)
                    weight_dims = [DIM_R, DIM_S, DIM_C, DIM_K]
                    weight_changed = any(
                        indices.get(key, 0) != prev.get(key, 0)
                        for key in indices.keys()
                        if key[1] in weight_dims
                    )
                    # Output relevant: P, Q, K, N (not R, S, C)
                    output_dims = [DIM_P, DIM_Q, DIM_K, DIM_N]
                    output_changed = any(
                        indices.get(key, 0) != prev.get(key, 0)
                        for key in indices.keys()
                        if key[1] in output_dims
                    )
                
                prev_indices[0] = indices.copy()
                
                # Pass tile indices directly to _generate_tile_accesses
                # Address calculation uses tile_strides internally
                self._generate_tile_accesses(
                    trace, indices, buffer_tile, workload,
                    H_in, W_in, stride_h, stride_w, dilation_h, dilation_w,
                    layout_info, tile_strides,
                    generate_input=input_changed,
                    generate_weight=weight_changed,
                    generate_output=output_changed,
                    strict_ordering=strict_ordering
                )
                pbar.update(1)
                return
            
            loop_info = dram_loops[level_idx]
            level = loop_info['level']
            dim = loop_info['dim']
            bound = loop_info['bound']
            # tile_stride stored for address calculation later
            
            # Use (level, dim) as key to distinguish same dim at different levels
            key = (level, dim)
            # Iterate using pure tile index (0, 1, 2, ...)
            # indices stores tile indices, NOT element coordinates
            for tile_idx in range(bound):
                new_indices = indices.copy()
                new_indices[key] = tile_idx  # pure tile index for this (level, dim)
                iterate_dram_loops(level_idx + 1, new_indices)
        
        # Initialize indices with (level, dim) keys
        initial_indices = {}
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
        
        # =======================================================================
        # Compute Input access tile size (H_per_tile × W_per_tile)
        # This is the size of data accessed in one DRAM loop iteration
        # =======================================================================
        # For Input, the access tile is determined by the sliding window over P, Q, R, S
        # H_per_tile = (P_per_tile - 1) * stride_h + (R_per_tile - 1) * dilation_h + 1
        # W_per_tile = (Q_per_tile - 1) * stride_w + (S_per_tile - 1) * dilation_w + 1
        #
        # buffer_tile contains Level 0+1 cumulative factors
        P_per_tile = buffer_tile[DIM_P]
        Q_per_tile = buffer_tile[DIM_Q]
        R_per_tile = buffer_tile[DIM_R]
        S_per_tile = buffer_tile[DIM_S]
        
        # Get convolution parameters
        stride_h = workload.stride[0] if hasattr(workload, 'stride') else 1
        stride_w = workload.stride[1] if hasattr(workload, 'stride') else 1
        dilation_h = workload.dilation[0] if hasattr(workload, 'dilation') else 1
        dilation_w = workload.dilation[1] if hasattr(workload, 'dilation') else 1
        
        # Input access tile dimensions (sliding window)
        H_per_tile = (P_per_tile - 1) * stride_h + (R_per_tile - 1) * dilation_h + 1
        W_per_tile = (Q_per_tile - 1) * stride_w + (S_per_tile - 1) * dilation_w + 1
        
        # Data layout block size (from tile_info, or use H_per_tile/W_per_tile as default)
        block_h = mapping.tile_info.get('block_h', H_per_tile)
        block_w = mapping.tile_info.get('block_w', W_per_tile)
        
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
        
        # Get Level 3 factors (DRAM level loop bounds)
        level3_factors = {d: 1 for d in range(7)}
        if 3 in mapping.loop_bounds:
            for key in ['spatial', 'temporal']:
                if key in mapping.loop_bounds[3]:
                    for d, bound in mapping.loop_bounds[3][key].items():
                        level3_factors[d] *= bound
        
        # Input DRAM tile size = H_per_tile × W_per_tile × C_per_tile
        # This is the access tile size (not data layout block size)
        C_per_tile = buffer_tile[DIM_C]
        input_dram_tile_size = H_per_tile * W_per_tile * C_per_tile
        level3_factors = {d: 1 for d in range(7)}
        if 3 in mapping.loop_bounds:
            for key in ['spatial', 'temporal']:
                if key in mapping.loop_bounds[3]:
                    for d, bound in mapping.loop_bounds[3][key].items():
                        level3_factors[d] *= bound
        
        P_l3 = level3_factors[DIM_P]
        Q_l3 = level3_factors[DIM_Q]
        C_l3 = level3_factors[DIM_C]
        N_l3 = level3_factors[DIM_N]
        
        # Input strides for row_aligned layout
        # Each DRAM tile (P_tile, Q_tile, C_tile, N_tile) is padded to row boundary
        # 
        # For sequential layout: stride = tile_size
        # For row_aligned layout: stride = ceil(tile_size / row_size) * row_size
        if input_layout == "row_aligned":
            input_aligned_tile_size = ((input_dram_tile_size + row_size - 1) // row_size) * row_size
        else:
            input_aligned_tile_size = input_dram_tile_size
        
        # Build strides for Input based on loop permutation
        # For row_aligned: each DRAM tile occupies one row (padded to row_size)
        # Layout order follows loop permutation: inner loop dim has stride=aligned_tile_size
        # 
        # Get Level 3 loop order for Input (determines data layout)
        # input_loop_order is [(level, dim), ...] from outer to inner
        # We need to reverse it for stride calculation (inner to outer)
        input_l3_dims_in_perm = [(lv, d) for (lv, d) in input_loop_order if lv == 3]
        
        # All Input-relevant dims at Level 3: P, Q, C, N (not K)
        input_l3_tile_counts = {DIM_P: P_l3, DIM_Q: Q_l3, DIM_C: C_l3, DIM_N: N_l3}
        
        # Build stride for each dim based on permutation order
        # Start from innermost (last in permutation) and work outward
        input_strides = {}
        stride = input_aligned_tile_size  # Each tile padded to row boundary
        
        # Process dims from inner to outer (reverse of permutation order)
        # Dims in permutation are processed first, then remaining dims
        processed_dims = set()
        
        # First, process dims that are in the permutation (in reverse order = inner to outer)
        for (lv, dim) in reversed(input_l3_dims_in_perm):
            if dim in input_l3_tile_counts:
                input_strides[(3, dim)] = stride
                stride *= input_l3_tile_counts[dim]
                processed_dims.add(dim)
        
        # Then, add remaining dims not in permutation (Q, N typically)
        for dim in [DIM_Q, DIM_P, DIM_C, DIM_N]:  # Default order
            if dim not in processed_dims and dim in input_l3_tile_counts:
                input_strides[(3, dim)] = stride
                stride *= input_l3_tile_counts[dim]
                processed_dims.add(dim)
        
        # Add Level 2 strides: within DRAM tile, use natural strides based on DATA LAYOUT BLOCK
        # Input data layout: [N][C][H][W] where H=P, W=Q
        # Row-major storage: W (Q) is innermost, then H (P), then C, then N
        # IMPORTANT: Strides must be based on the BLOCK size (data layout), not the TILE size (access)
        # If we access small tiles within a larger block, the stride is the block width.
        input_strides[(2, DIM_Q)] = 1  # Q/W is innermost, stride = 1
        input_strides[(2, DIM_P)] = block_w  # P/H stride = block_w
        input_strides[(2, DIM_C)] = block_h * block_w  # C: stride = block_h * block_w
        input_strides[(2, DIM_N)] = block_h * block_w * C_per_tile  # N: full block size * C
        
        # Also store H_per_tile, W_per_tile for access pattern calculation
        # (different from block_h, block_w which are data layout parameters)
        
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
            'block_size': block_h * block_w,  # data layout block size
            'num_blocks_h': num_blocks_h,
            'num_blocks_w': num_blocks_w,
            # Input access tile size (different from data layout block)
            'H_per_tile': H_per_tile,
            'W_per_tile': W_per_tile,
            'C_per_tile': C_per_tile,
            'input_dram_tile_size': input_dram_tile_size,
            'input_aligned_tile_size': input_aligned_tile_size,
            # DRAM factors
            'P_l3': P_l3,
            'Q_l3': Q_l3,
            'C_l3': C_l3,
            'N_l3': N_l3,
            # Layout modes
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
            'weight_loop_order': weight_loop_order,
            'output_loop_order': output_loop_order,
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
        
        # Add missing dims from tile_sizes (Level 3 first, then Level 2)
        # This ensures after reversing: Level 2 dims are inner, Level 3 dims are outer
        # Correct semantic: Level 2 = within DRAM tile, Level 3 = between DRAM tiles
        #
        # IMPORTANT FIX: We must ensure Level 2 loops are ALWAYS inner to Level 3 loops
        # The permutation might have mixed them up if not careful.
        # We will re-sort the complete_order to enforce Level 3 (outer) -> Level 2 (inner)
        
        for level in [3, 2]:
            if level in tile_sizes:
                for dim in tile_sizes[level]:
                    if (level, dim) not in loop_order_set:
                        complete_order.append((level, dim))
                        loop_order_set.add((level, dim))
        
        # Enforce Level 3 -> Level 2 hierarchy
        # Sort key: (level descending, original_index)
        # This keeps Level 3 before Level 2, and preserves relative order within levels
        # Note: We must capture the original index BEFORE sorting, as index() changes during sort?
        # Actually, list.sort is in-place, so index() inside lambda refers to the list being sorted.
        # This is dangerous. Better to create a list of tuples with original index.
        
        indexed_order = []
        for i, item in enumerate(complete_order):
            indexed_order.append((item, i))
            
        # Sort by level (descending) then original index (ascending)
        # Level 3 comes before Level 2. Within same level, preserve original order.
        indexed_order.sort(key=lambda x: (x[0][0], -x[1]), reverse=True)
        
        complete_order = [item for item, idx in indexed_order]
        
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
        spatial_dims_2d = {DIM_P, DIM_Q, DIM_R, DIM_S}  # Spatial dimensions
        
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
        
        Loop iteration uses TILE INDEX (0, 1, 2, ...) not element coordinates:
        - bound: number of tiles (e.g., 8 for C dimension)
        - tile_stride: element stride between tiles (for address calculation)
        
        Example: C=16, buffer_tile_C=2
        - Loop: for c_tile in range(8)  # tile index 0,1,2,...,7
        - Element coord: c_elem = c_tile * tile_stride  # 0,2,4,...,14
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
        # Now using tile index iteration (stride=1), storing tile_stride for address calc
        
        # STRICTLY enforce Level 3 (Outer) -> Level 2 (Inner)
        # We iterate levels explicitly: 3, then 2
        for m in [3, 2]:
            if m not in mapping.loop_bounds:
                continue
            
            # Get loop order for this level from permutation
            # mapping.permutation[m] is {index: dim}
            # We want outer to inner.
            # Convention: index 0 is INNER-most, index N is OUTER-most.
            # So we sort by index descending.
            perm_order = []
            if m in mapping.permutation:
                perm = mapping.permutation[m]
                sorted_perms = sorted(perm.items(), key=lambda x: x[0], reverse=True)
                perm_order = [dim for _, dim in sorted_perms]
            else:
                # Fallback if no permutation defined
                perm_order = list(range(7))
            
            level_bounds = {}
            for loop_type in ['spatial', 'temporal']:
                if loop_type in mapping.loop_bounds[m]:
                    for d, bound in mapping.loop_bounds[m][loop_type].items():
                        if bound > 1:
                            level_bounds[d] = level_bounds.get(d, 1) * bound
            
            # Add loops in permutation order
            for dim in perm_order:
                if dim in level_bounds and level_bounds[dim] > 1:
                    loops.append({
                        'dim': dim,
                        'bound': level_bounds[dim],      # number of tiles
                        'tile_stride': tile_sizes[m][dim],  # element stride between tiles
                        'level': m,
                    })
                    del level_bounds[dim] # Mark as processed
            
            # Add any remaining loops not in permutation
            for dim, bound in level_bounds.items():
                if bound > 1:
                    loops.append({
                        'dim': dim,
                        'bound': bound,
                        'tile_stride': tile_sizes[m][dim],
                        'level': m,
                    })
        
        return loops
    
    def _generate_tile_accesses(
        self, trace: List[str], tile_indices: Dict[Tuple[int, int], int], tile_size: Dict[int, int],
        workload, H_in, W_in, stride_h, stride_w, dilation_h, dilation_w,
        layout_info: Dict, tile_strides: Dict[int, int],
        generate_input: bool = True,
        generate_weight: bool = True,
        generate_output: bool = True,
        strict_ordering: bool = False
    ):
        """
        Generate DRAM accesses using tile-wise addressing.
        
        This function is called once per DRAM loop iteration (tile_indices).
        For each tensor, it generates all element accesses within that tile.
        
        Address formula: base + tile_index × tile_size + offset_in_tile
        
        Args:
            tile_indices: {(level, dim): tile_index} - current position in DRAM loop iteration
                          e.g., {(3, C): 2, (2, K): 1} means we're at C_l3_tile=2, K_l2_tile=1
            tile_size: buffer tile size per dimension (from Level 0+1)
                       e.g., {R:3, S:3, P:8, Q:8, C:2, K:4, N:1}
            tile_strides: {dim: stride} - element stride between tiles (not used here)
            generate_input/weight/output: flags to control which tensors to generate
        
        ===============================================================================
        IMPORTANT: Input has a SPECIAL PROPERTY that differs from Weight and Output!
        ===============================================================================
        
        For Weight and Output:
        - Access tile shape == Data layout tile shape
        - Simple tile-wise addressing works perfectly
        - addr = base + tile_idx * tile_size + offset
        
        For Input (SPECIAL CASE):
        - Access tile shape ≠ Data layout tile shape
        - Access pattern: sliding window over (P,Q,R,S) → (H,W) coordinates
        - Data layout: organized by (N, C, H, W) or block-wise
        - The "tile" we access is determined by convolution window, not data layout
        - Current simplified implementation assumes access tile == layout tile
        - TODO: Proper implementation needs to handle sliding window access pattern
        ===============================================================================
        """
        # Extract base addresses
        input_base = layout_info['input_base']
        weight_base = layout_info['weight_base']
        output_base = layout_info['output_base']
        
        # =========================================================================
        # Buffer Tile Sizes (number of elements per tile for each tensor)
        # =========================================================================
        
        # Input tile size:
        # - SPECIAL: Input access pattern involves sliding window (P,Q,R,S) → (H,W)
        # - For simplicity, we use block_h × block_w × C × N as tile size
        # - This assumes the input tile covers the entire spatial window needed
        input_tile_size = (layout_info['block_h'] * layout_info['block_w'] * 
                          tile_size[DIM_C] * tile_size[DIM_N])
        
        # Weight tile size: K × C × R × S (direct product, no special handling)
        weight_tile_size = tile_size[DIM_K] * tile_size[DIM_C] * tile_size[DIM_R] * tile_size[DIM_S]
        
        # Output tile size: N × K × P × Q (direct product, no special handling)
        output_tile_size = tile_size[DIM_N] * tile_size[DIM_K] * tile_size[DIM_P] * tile_size[DIM_Q]
        
        # =========================================================================
        # Flat Tile Index Calculation
        # =========================================================================
        # Convert multi-dimensional tile indices to a single flat index
        # Only considers dimensions that are relevant to each tensor
        
        dram_loop_dims = layout_info.get('dram_loop_dims', [])  # [(level, dim, bound), ...]
        
        # Helper to get accumulated tile index for a dimension across all levels
        # This is used for element coordinate calculation (e.g., p_start = p_tile * p_size)
        def get_accumulated_tile_index(dim: int) -> int:
            """Get total tile index for a dimension by summing across all levels.
            
            For address/coordinate calculation, we need the overall tile position:
            - If P appears at level 3 (bound=4) and level 2 (bound=2):
            - Total P tiles = P_l3 * P_l2_bound + P_l2 = P_l3 * 2 + P_l2
            """
            total = 0
            # Find bounds for this dim at each level
            level_bounds = {}
            for level, d, bound in dram_loop_dims:
                if d == dim:
                    level_bounds[level] = bound
            
            # Calculate accumulated index: outer_idx * inner_bound + inner_idx
            sorted_levels = sorted(level_bounds.keys(), reverse=True)  # Outer to inner
            multiplier = 1
            for level in reversed(sorted_levels):  # Inner to outer
                idx = tile_indices.get((level, dim), 0)
                total += idx * multiplier
                multiplier *= level_bounds[level]
            return total
        
        def compute_flat_tile_index(tile_indices: Dict[Tuple[int, int], int], relevant_dims: List[int]) -> int:
            """Compute flat tile index for relevant dimensions only.
            
            tile_indices uses (level, dim) as key to distinguish same dim at different levels.
            Only counts dimensions that are BOTH in dram_loop_dims AND relevant to the tensor.
            Irrelevant dimensions don't change the tile we access.
            
            Example: dram_loop_dims = [(3, C, 4), (3, K, 2)]
            - For Input (K irrelevant): flat_idx = c_tile (0,1,2,3)
            - For Weight (all relevant): flat_idx = c_tile * 2 + k_tile (0..7)
            - For Output (C irrelevant): flat_idx = k_tile (0,1)
            
            Example with same dim at multiple levels: dram_loop_dims = [(3, P, 4), (3, C, 8), (2, P, 2), (2, K, 2)]
            - For Output (relevant: P, K): 
              flat_idx = P_l3 * (P_l2_bound * K_bound) + P_l2 * K_bound + K
                       = P_l3 * 4 + P_l2 * 2 + K
            """
            flat_idx = 0
            stride = 1
            # Inner to outer (reverse of loop order)
            for level, dim, bound in reversed(dram_loop_dims):
                if dim in relevant_dims:
                    # Use (level, dim) as key to get the correct tile index
                    flat_idx += tile_indices.get((level, dim), 0) * stride
                    stride *= bound  # Only multiply stride for relevant dims
            return flat_idx
        
        def compute_flat_tile_index_l3_only(tile_indices: Dict[Tuple[int, int], int], relevant_dims: List[int]) -> int:
            """Compute flat tile index using ONLY Level 3 dimensions.
            
            For row_aligned layout, alignment is at Level 3 (DRAM tile).
            Level 2 iterations are within the same aligned region.
            
            Example: dram_loop_dims = [(3, P, 4), (3, C, 8), (2, P, 2), (2, K, 2)]
            - For Weight (relevant: C, K), but only L3 dims: 
              flat_idx = C_l3 (0..7)
            - Level 2 K iterations are within the same L3 tile
            """
            flat_idx = 0
            stride = 1
            # Inner to outer (reverse of loop order), but ONLY Level 3
            for level, dim, bound in reversed(dram_loop_dims):
                if level == 3 and dim in relevant_dims:
                    flat_idx += tile_indices.get((level, dim), 0) * stride
                    stride *= bound
            return flat_idx
        
        # =========================================================================
        # Input Access Generation (方案 B: 按 data layout block 迭代)
        # =========================================================================
        # SPECIAL: Input access tile ≠ data layout tile (due to sliding window)
        # - Access tile: determined by sliding window (P, Q, R, S) → (H, W)
        # - Data layout: organized by blocks (block_h × block_w)
        # - One access tile may span multiple data layout blocks
        # - Iterate by blocks, access only needed elements within each block
        # 
        # For row_aligned layout:
        # - Each DRAM tile (p_tile, q_tile, c_tile, n_tile) is padded to row boundary
        # - Address = tile_base + offset_in_tile
        # - tile_base = p_tile * stride_p + q_tile * stride_q + c_tile * stride_c + n_tile * stride_n
        # 
        # IMPORTANT for row_aligned:
        # - L3 tiles use aligned strides (stride_x_l3 = row_size or multiple)
        # - L2 iterations are WITHIN the aligned L3 tile, so use L2 offsets
        # - We need to distinguish L3 indices from L2 indices
        if generate_input:
            # Get Input access tile parameters
            H_per_tile = layout_info['H_per_tile']
            W_per_tile = layout_info['W_per_tile']
            C_per_tile = layout_info['C_per_tile']
            
            # DEBUG: Track first few Input accesses for debugging
            input_debug_count = getattr(self, '_input_debug_count', 0)
            if input_debug_count == 0:
                self._input_debug_info = []  # Store debug info
            
            # Get strides for row_aligned calculation
            input_strides = layout_info.get('input_strides', {})
            input_layout = layout_info.get('input_layout', 'sequential')
            
            # Level 3 strides (between DRAM tiles) - these include row_aligned padding
            stride_p_l3 = input_strides.get((3, DIM_P), 1024)
            stride_q_l3 = input_strides.get((3, DIM_Q), 1024)
            stride_c_l3 = input_strides.get((3, DIM_C), 1024)
            stride_n_l3 = input_strides.get((3, DIM_N), 1024)

            stride_p_l2 = input_strides.get((2, DIM_P), 0)
            stride_q_l2 = input_strides.get((2, DIM_Q), 0)
            stride_c_l2 = input_strides.get((2, DIM_C), 0)
            stride_n_l2 = input_strides.get((2, DIM_N), 0)

            # For row_aligned layout, get separate L3 and L2 indices for C and N
            # L3 indices determine which aligned region we're in
            # L2 indices determine offset within that aligned region
            c_l3_idx = tile_indices.get((3, DIM_C), 0)
            n_l3_idx = tile_indices.get((3, DIM_N), 0)
            c_l2_idx = tile_indices.get((2, DIM_C), 0)
            n_l2_idx = tile_indices.get((2, DIM_N), 0)
            
            # Current DRAM tile indices (accumulated across levels) - for coordinate calculation
            p_tile = get_accumulated_tile_index(DIM_P)
            q_tile = get_accumulated_tile_index(DIM_Q)
            c_tile = get_accumulated_tile_index(DIM_C)
            n_tile = get_accumulated_tile_index(DIM_N)
            r_tile = get_accumulated_tile_index(DIM_R)
            s_tile = get_accumulated_tile_index(DIM_S)
            
            # Tile sizes in each dimension (buffer tile size, i.e., per-tile element count)
            p_size = tile_size[DIM_P]
            q_size = tile_size[DIM_Q]
            r_size = tile_size[DIM_R]
            s_size = tile_size[DIM_S]
            c_size = tile_size[DIM_C]
            n_size = tile_size[DIM_N]
            
            # Starting coordinates within the full tensor
            p_start = p_tile * p_size
            q_start = q_tile * q_size
            c_start = c_tile * c_size
            n_start = n_tile * n_size
            r_start = r_tile * r_size
            s_start = s_tile * s_size
            
            # Compute H, W range for this access tile (sliding window)
            # h = p * stride + r * dilation, so R contributes h_offset
            # w = q * stride + s * dilation, so S contributes w_offset
            h_offset = r_start * dilation_h
            w_offset = s_start * dilation_w
            
            h_start = p_start * stride_h + h_offset
            h_end = min((p_start + p_size - 1) * stride_h + h_offset + (r_size - 1) * dilation_h + 1, H_in)
            w_start = q_start * stride_w + w_offset
            w_end = min((q_start + q_size - 1) * stride_w + w_offset + (s_size - 1) * dilation_w + 1, W_in)
            
            # Get data layout block size (independent of access tile size)
            block_h = layout_info['block_h']
            block_w = layout_info['block_w']
            
            # Generate accesses for all elements in this tile
            # Data layout: [N][C][H_block][W_block] with block-wise organization
            # Each (n, c, h_block, w_block) combination may be in a different row
            #
            # OPTIMIZATION: Access by block to minimize row switches
            # Instead of: for h: for w (causes row thrashing at block boundaries)
            # Use: for h_blk: for w_blk: for h_in_blk: for w_in_blk
            # This ensures all elements in one block are accessed before moving to next block
            
            # Compute block ranges that this tile covers
            h_block_start = h_start // block_h
            h_block_end = (h_end - 1) // block_h
            w_block_start = w_start // block_w
            w_block_end = (w_end - 1) // block_w
            
            # Strict Row-Major (H, W) iteration
            # This exposes row thrashing if the tile crosses block boundaries
            for n_local in range(min(n_size, workload.N - n_start)):
                for c_local in range(min(c_size, workload.C - c_start)):
                    coords_iter = (
                        (h, w, h // block_h, w // block_w, h % block_h, w % block_w)
                        for h in range(h_start, h_end)
                        for w in range(w_start, w_end)
                    )
                    for h, w, h_block, w_block, h_in_block, w_in_block in coords_iter:
                        self._generate_input_access_body(
                            trace, input_layout, tile_indices, 
                            h, w, h_block, w_block, h_in_block, w_in_block,
                            c_local, n_local,
                            stride_p_l3, stride_q_l3, stride_c_l3, stride_n_l3,
                            stride_p_l2, stride_q_l2, stride_c_l2, stride_n_l2,
                            c_l3_idx, n_l3_idx, c_l2_idx, n_l2_idx,
                            H_per_tile, W_per_tile, dram_loop_dims,
                            input_base, c_tile, n_tile,
                            w_start, w_end, block_w
                        )
            
            # Save debug count for next iteration
            self._input_debug_count = getattr(self, '_input_debug_count', 0)
        
        # =========================================================================
        # Weight Access Generation  
        # =========================================================================
        # Weight: access tile == data layout tile (simple case)
        # Relevant dims: R, S, C, K (P, Q, N are irrelevant - same weight for all)
        if generate_weight:
            weight_relevant = [DIM_R, DIM_S, DIM_C, DIM_K]
            
            # Use aligned tile size for row_aligned layout
            weight_layout = layout_info.get('weight_layout', 'sequential')
            row_size = layout_info.get('row_size', 1024)
            
            if weight_layout == 'row_aligned':
                # For row_aligned: alignment is at Level 3, use only L3 dims for flat_idx
                weight_tile_idx = compute_flat_tile_index_l3_only(tile_indices, weight_relevant)
                # L3 tile size = buffer_tile × L2_factors
                # Need to compute L2 tile size for this tensor
                l2_tile_size = weight_tile_size
                for level, dim, bound in dram_loop_dims:
                    if level == 2 and dim in weight_relevant:
                        l2_tile_size *= bound
                weight_aligned_tile_size = ((l2_tile_size + row_size - 1) // row_size) * row_size
                
                # Compute offset within the L3 tile from L2 indices
                l2_offset = 0
                l2_stride = 1
                for level, dim, bound in reversed(dram_loop_dims):
                    if level == 2 and dim in weight_relevant:
                        l2_offset += tile_indices.get((level, dim), 0) * l2_stride * weight_tile_size
                        l2_stride *= bound
                
                weight_tile_base = weight_base + weight_tile_idx * weight_aligned_tile_size * self.element_size + l2_offset * self.element_size
            else:
                # Sequential: use all levels for flat_idx
                weight_tile_idx = compute_flat_tile_index(tile_indices, weight_relevant)
                weight_aligned_tile_size = weight_tile_size
                weight_tile_base = weight_base + weight_tile_idx * weight_aligned_tile_size * self.element_size
            
            for offset in range(weight_tile_size):
                addr = weight_tile_base + offset * self.element_size
                trace.append(f"LD 0x{addr:08X}")
        
        # =========================================================================
        # Output Access Generation
        # =========================================================================
        # Output: access tile == data layout tile (simple case)
        # Relevant dims: P, Q, K, N (R, S, C are irrelevant - accumulated into same output)
        if generate_output:
            output_relevant = [DIM_P, DIM_Q, DIM_K, DIM_N]
            
            # Use aligned tile size for row_aligned layout
            output_layout = layout_info.get('output_layout', 'sequential')
            row_size = layout_info.get('row_size', 1024)
            
            if output_layout == 'row_aligned':
                # For row_aligned: alignment is at Level 3, use only L3 dims for flat_idx
                output_tile_idx = compute_flat_tile_index_l3_only(tile_indices, output_relevant)
                # L3 tile size = buffer_tile × L2_factors
                l2_tile_size = output_tile_size
                for level, dim, bound in dram_loop_dims:
                    if level == 2 and dim in output_relevant:
                        l2_tile_size *= bound
                output_aligned_tile_size = ((l2_tile_size + row_size - 1) // row_size) * row_size
                
                # Compute offset within the L3 tile from L2 indices
                l2_offset = 0
                l2_stride = 1
                for level, dim, bound in reversed(dram_loop_dims):
                    if level == 2 and dim in output_relevant:
                        l2_offset += tile_indices.get((level, dim), 0) * l2_stride * output_tile_size
                        l2_stride *= bound
                
                output_tile_base = output_base + output_tile_idx * output_aligned_tile_size * self.element_size + l2_offset * self.element_size
            else:
                # Sequential: use all levels for flat_idx
                output_tile_idx = compute_flat_tile_index(tile_indices, output_relevant)
                output_aligned_tile_size = output_tile_size
                output_tile_base = output_base + output_tile_idx * output_aligned_tile_size * self.element_size
            
            for offset in range(output_tile_size):
                addr = output_tile_base + offset * self.element_size
                trace.append(f"ST 0x{addr:08X}")
    
    def _compute_tile_wise_address(self, indices: Dict[int, int], loop_order: List[Tuple[int, int]], 
                                    strides: Dict[Tuple[int, int], int],
                                    tile_sizes_info: Dict[int, Dict[int, int]]) -> int:
        """Compute address using tile-wise layout.
        
        For Level 3: use tile index (val // level2_tile_size)
        For Level 2: use offset within tile (val % level2_tile_size)
        
        Note: Iterates over all (level, dim) pairs in strides, not just loop_order,
        to handle dimensions that are not in permutation but have strides.
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
                            # For row_aligned: L3 indices determine the aligned region
                            # L2 iterations stay within the same aligned region
                            # 
                            # block_base uses L3 indices only (aligned strides)
                            # L2 offset is added separately (sequential within aligned region)
                            
                            # L3 base: aligned to row boundaries
                            # h_block and w_block determine which spatial L3 tile we're in
                            block_base = (h_block * stride_p_l3 + 
                                         w_block * stride_q_l3 + 
                                         c_l3_idx * stride_c_l3 + 
                                         n_l3_idx * stride_n_l3)
                            
                            # L2 offset: sequential within the aligned L3 region
                            # Need to find L2 factor sizes
                            c_l2_factor = 1
                            n_l2_factor = 1
                            for level, dim, bound in dram_loop_dims:
                                if level == 2 and dim == 4: # DIM_C
                                    c_l2_factor = bound
                                elif level == 2 and dim == 6: # DIM_N
                                    n_l2_factor = bound
                            
                            # L2 tile offset within the aligned L3 region
                            l2_tile_base = (c_l2_idx * H_per_tile * W_per_tile +
                                           n_l2_idx * H_per_tile * W_per_tile * c_l2_factor)
                            
                            # Offset within the block: [h_in_block][w_in_block]
                            offset_in_block = (
                                h_in_block * stride_p_l2 +
                                w_in_block * stride_q_l2 +
                                c_local * stride_c_l2 +
                                n_local * stride_n_l2
                            )
                            
                            addr = input_base + block_base + l2_tile_base + offset_in_block
                            
                            # DEBUG: Store first few accesses
                            input_debug_count = getattr(self, '_input_debug_count', 0)
                            if input_debug_count < 200:
                                self._input_debug_info.append({
                                    'idx': input_debug_count,
                                    'layout': input_layout,
                                    'tile_indices': dict(tile_indices),
                                    'h': h, 'w': w,
                                    'h_block': h_block, 'w_block': w_block,
                                    'h_in_block': h_in_block, 'w_in_block': w_in_block,
                                    'block_base': block_base,
                                    'l2_tile_base': l2_tile_base,
                                    'offset_in_block': offset_in_block,
                                    'addr': addr,
                                    'row': addr // 1024,
                                    'col': addr % 1024,
                                    'w_start': w_start,
                                    'w_end': w_end,
                                })
                            
                            # DEBUG: Track boundary crossings specifically
                            # When W crosses block_w boundary
                            if w == block_w - 1 or w == block_w:
                                if len(self._input_debug_boundary_info) < 100:
                                    self._input_debug_boundary_info.append({
                                        'idx': input_debug_count,
                                        'h': h, 'w': w,
                                        'w_start': w_start, 'w_end': w_end,
                                        'h_block': h_block, 'w_block': w_block,
                                        'h_in_block': h_in_block, 'w_in_block': w_in_block,
                                        'block_base': block_base,
                                        'offset_in_block': offset_in_block,
                                        'addr': addr,
                                        'row': addr // 1024,
                                        'col': addr % 1024,
                                    })
                            
                            trace.append(f"LD 0x{addr:08X}")
                            self._input_debug_count = input_debug_count + 1
                        else:
                            # Sequential layout: use accumulated indices
                            block_base = (h_block * stride_p_l3 + 
                                         w_block * stride_q_l3 + 
                                         c_tile * stride_c_l3 + 
                                         n_tile * stride_n_l3)
                            
                            # Offset within the block: [h_in_block][w_in_block]
                            offset_in_block = (
                                h_in_block * stride_p_l2 +
                                w_in_block * stride_q_l2 +
                                c_local * stride_c_l2 +
                                n_local * stride_n_l2
                            )
                            
                            addr = input_base + block_base + offset_in_block
                            
                            # DEBUG: Store first few accesses for sequential layout too
                            input_debug_count = getattr(self, '_input_debug_count', 0)
                            if input_debug_count < 200:
                                self._input_debug_info.append({
                                    'idx': input_debug_count,
                                    'layout': input_layout,
                                    'tile_indices': dict(tile_indices),
                                    'h': h, 'w': w,
                                    'h_block': h_block, 'w_block': w_block,
                                    'h_in_block': h_in_block, 'w_in_block': w_in_block,
                                    'block_base': block_base,
                                    'offset_in_block': offset_in_block,
                                    'addr': addr,
                                    'row': addr // 1024,
                                    'col': addr % 1024,
                                })
                            
                            trace.append(f"LD 0x{addr:08X}")
                            self._input_debug_count = input_debug_count + 1
