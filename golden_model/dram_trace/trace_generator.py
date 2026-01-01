"""
DRAM Row Access Trace Generator

Generate memory access traces from ILP optimization decisions.
This module converts 7-dimensional Conv tiling decisions into
concrete DRAM row access patterns for cycle-accurate validation.

The trace format is designed to be independent of UniNDP's
partition/mapping model, allowing direct validation of row
activation overhead.

Architecture:
    ILP decisions (tiling + loop order)
           |
           v
    Loop Nest Simulation
           |
           v
    DRAM Row Access Trace
           |
           v
    DRAM Timing Simulator (using UniNDP timing params)
           |
           v
    Cycle Count + Row Activation Stats
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Iterator
from enum import Enum
import math
import numpy as np


class AccessType(Enum):
    """Type of memory access."""
    READ = "read"
    WRITE = "write"


class DataType(Enum):
    """Type of data being accessed."""
    INPUT = 0
    WEIGHT = 1
    OUTPUT = 2


@dataclass
class MemoryAccess:
    """
    Single memory access operation.
    
    Attributes:
        timestamp: Logical timestamp (cycle number in compute)
        datatype: INPUT, WEIGHT, or OUTPUT
        access_type: READ or WRITE
        address: Logical address in tensor
        tensor_coords: Original tensor coordinates (N,C,H,W) or (K,C,R,S)
        size_bytes: Size of access in bytes
        row_id: DRAM row ID (computed from address + row_size)
    """
    timestamp: int
    datatype: DataType
    access_type: AccessType
    address: int
    tensor_coords: Tuple[int, ...]
    size_bytes: int
    row_id: int = -1  # Computed by timing simulator


@dataclass
class TileInfo:
    """Information about a single tile in the loop nest."""
    tile_dims: Dict[str, int]  # {R, S, P, Q, C, K, N} tile sizes
    base_coords: Dict[str, int]  # Starting coordinates
    
    def get_input_tile_size(self, stride_h: int = 1, stride_w: int = 1) -> Tuple[int, int, int, int]:
        """
        Compute input tile size from output tile (P,Q) and kernel (R,S).
        
        Returns:
            (N, C, H, W) where H = P*stride_h + (R-1), W = Q*stride_w + (S-1)
        """
        P = self.tile_dims.get('P', 1)
        Q = self.tile_dims.get('Q', 1)
        R = self.tile_dims.get('R', 1)
        S = self.tile_dims.get('S', 1)
        C = self.tile_dims.get('C', 1)
        N = self.tile_dims.get('N', 1)
        
        H = P * stride_h + (R - 1)
        W = Q * stride_w + (S - 1)
        
        return (N, C, H, W)
    
    def get_weight_tile_size(self) -> Tuple[int, int, int, int]:
        """Returns (K, C, R, S)."""
        return (
            self.tile_dims.get('K', 1),
            self.tile_dims.get('C', 1),
            self.tile_dims.get('R', 1),
            self.tile_dims.get('S', 1),
        )
    
    def get_output_tile_size(self) -> Tuple[int, int, int, int]:
        """Returns (N, K, P, Q)."""
        return (
            self.tile_dims.get('N', 1),
            self.tile_dims.get('K', 1),
            self.tile_dims.get('P', 1),
            self.tile_dims.get('Q', 1),
        )


@dataclass
class LoopNestConfig:
    """
    Configuration extracted from ILP optimization result.
    
    Represents the 7-dimensional Conv loop nest with tiling.
    """
    # Workload dimensions
    N: int  # Batch size
    K: int  # Output channels
    C: int  # Input channels
    P: int  # Output height
    Q: int  # Output width
    R: int  # Kernel height
    S: int  # Kernel width
    
    # Stride and dilation
    stride_h: int = 1
    stride_w: int = 1
    dilation_h: int = 1
    dilation_w: int = 1
    
    # Tiling factors (product of all memory level tiles)
    tile_n: int = 1
    tile_k: int = 1
    tile_c: int = 1
    tile_p: int = 1
    tile_q: int = 1
    tile_r: int = 1  # Usually not tiled
    tile_s: int = 1  # Usually not tiled
    
    # Loop order (inner to outer)
    # Example: ['S', 'R', 'Q', 'P', 'C', 'K', 'N']
    loop_order: List[str] = field(default_factory=lambda: ['S', 'R', 'Q', 'P', 'C', 'K', 'N'])
    
    # Element size in bytes
    element_bytes: int = 1
    
    # Row buffer size in bytes
    row_buffer_size: int = 1024
    
    @property
    def input_h(self) -> int:
        """Input height."""
        return self.P * self.stride_h + (self.R - 1) * self.dilation_h
    
    @property
    def input_w(self) -> int:
        """Input width."""
        return self.Q * self.stride_w + (self.S - 1) * self.dilation_w
    
    @classmethod
    def from_mapping(cls, mapping, workload, dram_level: int = 3) -> 'LoopNestConfig':
        """
        Create LoopNestConfig from ILP Mapping result.
        
        The key insight is that we need to find the tiling at the DRAM level,
        which determines how many times we access DRAM.
        
        Memory hierarchy:
        - Level 0: PE (innermost)
        - Level 1: GlobalBuffer 
        - Level 2: RowBuffer
        - Level 3: DRAM (outermost)
        
        The DRAM level's temporal tiling tells us how many DRAM tile accesses
        we need. The product of levels 0-2 gives us the inner tile size.
        
        Args:
            mapping: Mapping object from optimizer
            workload: ConvWorkload object
            dram_level: Memory level for DRAM (default 3)
        """
        # Extract workload dimensions
        R = getattr(workload, 'R', 1)
        S = getattr(workload, 'S', 1)
        P = getattr(workload, 'P', 1)
        Q = getattr(workload, 'Q', 1)
        C = getattr(workload, 'C', 1)
        K = getattr(workload, 'K', 1)
        N = getattr(workload, 'N', 1)
        
        dim_names = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
        total_dims = {'R': R, 'S': S, 'P': P, 'Q': Q, 'C': C, 'K': K, 'N': N}
        
        # Get stride
        stride = getattr(workload, 'stride', (1, 1))
        stride_h = stride[0] if isinstance(stride, tuple) else stride
        stride_w = stride[1] if isinstance(stride, tuple) else stride
        
        # Find inner tile sizes (product of all levels EXCEPT DRAM level)
        # This is what gets loaded from DRAM each time
        inner_tile = {d: 1 for d in dim_names}
        
        print(f"DEBUG: Extracting inner_tile for DRAM Level {dram_level}")
        for m in sorted(mapping.loop_bounds.keys()):
            print(f"DEBUG: Checking Level {m}")
            if m >= dram_level:
                continue  # Skip DRAM level and above
            bounds = mapping.loop_bounds[m]
            for space_type in ['H', 'W', 'Internal', 'spatial', 'temporal']:
                if space_type in bounds:
                    for j, factor in bounds[space_type].items():
                        print(f"DEBUG: Found factor {factor} for dim {j} in {space_type}")
                        if factor > 1 and j < len(dim_names):
                            inner_tile[dim_names[j]] *= factor
        
        print(f"DEBUG: Final inner_tile: {inner_tile}")
        
        # Get DRAM level tiling (number of times to iterate)
        dram_tiling = {d: 1 for d in dim_names}
        if dram_level in mapping.loop_bounds:
            bounds = mapping.loop_bounds[dram_level]
            for space_type in ['spatial', 'temporal']:
                if space_type in bounds:
                    for j, factor in bounds[space_type].items():
                        if factor > 1 and j < len(dim_names):
                            dram_tiling[dim_names[j]] *= factor
        
        # Extract loop order from DRAM level permutation
        loop_order = []
        if mapping.permutation and dram_level in mapping.permutation:
            perm = mapping.permutation[dram_level]
            for p in sorted(perm.keys()):
                dim_idx = perm[p]
                if dim_idx < len(dim_names):
                    loop_order.append(dim_names[dim_idx])
        
        # Fill in remaining dimensions in default order
        for d in dim_names:
            if d not in loop_order:
                loop_order.append(d)
        
        # The tile size we pass is the INNER tile (what we fetch from DRAM)
        # The config.N/K/C/P/Q are the TOTAL dimensions
        # Trace generator will iterate based on total/tile
        return cls(
            R=R, S=S, P=P, Q=Q, C=C, K=K, N=N,
            stride_h=stride_h,
            stride_w=stride_w,
            tile_r=inner_tile.get('R', 1),
            tile_s=inner_tile.get('S', 1),
            tile_p=inner_tile.get('P', 1),
            tile_q=inner_tile.get('Q', 1),
            tile_c=inner_tile.get('C', 1),
            tile_k=inner_tile.get('K', 1),
            tile_n=inner_tile.get('N', 1),
            loop_order=loop_order,
        )


class TraceGenerator:
    """
    Generate DRAM row access traces from loop nest configuration.
    
    This class simulates the loop nest execution and generates
    a sequence of memory accesses, accounting for:
    - Data reuse within tiles
    - Sliding window overlap for input feature maps
    - Sequential vs strided access patterns
    """
    
    def __init__(self, config: LoopNestConfig):
        self.config = config
        self.accesses: List[MemoryAccess] = []
        self.timestamp = 0
        
        # Track which data is in row buffer (simple model)
        self.row_buffer_state: Dict[DataType, Optional[int]] = {
            DataType.INPUT: None,
            DataType.WEIGHT: None,
            DataType.OUTPUT: None,
        }
    
    def generate_trace(self) -> List[MemoryAccess]:
        """
        Generate complete memory access trace by simulating loop nest.
        
        Returns:
            List of MemoryAccess objects in chronological order
        """
        self.accesses = []
        self.timestamp = 0
        
        # Simulate the tiled loop nest
        for tile in self._iterate_tiles():
            self._process_tile(tile)
        
        # Compute row IDs based on address and row buffer size
        self._compute_row_ids()
        
        return self.accesses
    
    def _iterate_tiles(self) -> Iterator[TileInfo]:
        """
        Iterate over all tiles according to loop order.
        
        Yields TileInfo for each tile iteration.
        """
        c = self.config
        
        # Compute number of tiles in each dimension
        num_tiles = {
            'N': math.ceil(c.N / c.tile_n),
            'K': math.ceil(c.K / c.tile_k),
            'C': math.ceil(c.C / c.tile_c),
            'P': math.ceil(c.P / c.tile_p),
            'Q': math.ceil(c.Q / c.tile_q),
            'R': math.ceil(c.R / c.tile_r),
            'S': math.ceil(c.S / c.tile_s),
        }
        
        tile_sizes = {
            'N': c.tile_n,
            'K': c.tile_k,
            'C': c.tile_c,
            'P': c.tile_p,
            'Q': c.tile_q,
            'R': c.tile_r,
            'S': c.tile_s,
        }
        
        # Create iteration ranges based on loop order
        loop_ranges = [(d, range(num_tiles[d])) for d in c.loop_order]
        
        # Generate all tile combinations
        def iterate_nested(idx, current_indices):
            if idx == len(loop_ranges):
                # Create TileInfo from current indices
                tile_dims = {}
                base_coords = {}
                for d, i in current_indices.items():
                    tile_dims[d] = min(tile_sizes[d], 
                                       getattr(c, d) - i * tile_sizes[d]) if hasattr(c, d) else tile_sizes[d]
                    base_coords[d] = i * tile_sizes[d]
                yield TileInfo(tile_dims=tile_dims, base_coords=base_coords)
            else:
                dim, rng = loop_ranges[idx]
                for i in rng:
                    current_indices[dim] = i
                    yield from iterate_nested(idx + 1, current_indices)
        
        yield from iterate_nested(0, {})
    
    def _process_tile(self, tile: TileInfo):
        """
        Process a single tile: generate memory accesses for input, weight, output.
        
        Simulates the data flow for one tile computation:
        1. Read input tile
        2. Read weight tile
        3. Compute (accumulate cycles)
        4. Write output tile (if last reduction dimension)
        """
        c = self.config
        
        # Read input tile
        input_size = tile.get_input_tile_size(c.stride_h, c.stride_w)
        input_bytes = np.prod(input_size) * c.element_bytes
        input_addr = self._compute_input_address(tile)
        
        self.accesses.append(MemoryAccess(
            timestamp=self.timestamp,
            datatype=DataType.INPUT,
            access_type=AccessType.READ,
            address=input_addr,
            tensor_coords=tile.base_coords,
            size_bytes=input_bytes,
        ))
        self.timestamp += 1
        
        # Read weight tile
        weight_size = tile.get_weight_tile_size()
        weight_bytes = np.prod(weight_size) * c.element_bytes
        weight_addr = self._compute_weight_address(tile)
        
        self.accesses.append(MemoryAccess(
            timestamp=self.timestamp,
            datatype=DataType.WEIGHT,
            access_type=AccessType.READ,
            address=weight_addr,
            tensor_coords=tile.base_coords,
            size_bytes=weight_bytes,
        ))
        self.timestamp += 1
        
        # Write output tile (partial sum accumulation)
        output_size = tile.get_output_tile_size()
        output_bytes = np.prod(output_size) * c.element_bytes
        output_addr = self._compute_output_address(tile)
        
        # Read-modify-write for partial sums (except first iteration)
        is_first_reduction = (tile.base_coords.get('C', 0) == 0 and 
                              tile.base_coords.get('R', 0) == 0 and
                              tile.base_coords.get('S', 0) == 0)
        
        if not is_first_reduction:
            self.accesses.append(MemoryAccess(
                timestamp=self.timestamp,
                datatype=DataType.OUTPUT,
                access_type=AccessType.READ,
                address=output_addr,
                tensor_coords=tile.base_coords,
                size_bytes=output_bytes,
            ))
            self.timestamp += 1
        
        self.accesses.append(MemoryAccess(
            timestamp=self.timestamp,
            datatype=DataType.OUTPUT,
            access_type=AccessType.WRITE,
            address=output_addr,
            tensor_coords=tile.base_coords,
            size_bytes=output_bytes,
        ))
        self.timestamp += 1
    
    def _compute_input_address(self, tile: TileInfo) -> int:
        """
        Compute linear address for input tensor access.
        
        Input tensor layout: [N, C, H, W] in row-major order
        """
        c = self.config
        n = tile.base_coords.get('N', 0)
        ch = tile.base_coords.get('C', 0)
        p = tile.base_coords.get('P', 0)
        q = tile.base_coords.get('Q', 0)
        
        # Convert output coords (p,q) to input coords (h,w)
        h = p * c.stride_h
        w = q * c.stride_w
        
        # Row-major linearization
        addr = (((n * c.C + ch) * c.input_h + h) * c.input_w + w)
        return addr * c.element_bytes
    
    def _compute_weight_address(self, tile: TileInfo) -> int:
        """
        Compute linear address for weight tensor access.
        
        Weight tensor layout: [K, C, R, S] in row-major order
        """
        c = self.config
        k = tile.base_coords.get('K', 0)
        ch = tile.base_coords.get('C', 0)
        r = tile.base_coords.get('R', 0)
        s = tile.base_coords.get('S', 0)
        
        addr = (((k * c.C + ch) * c.R + r) * c.S + s)
        return addr * c.element_bytes
    
    def _compute_output_address(self, tile: TileInfo) -> int:
        """
        Compute linear address for output tensor access.
        
        Output tensor layout: [N, K, P, Q] in row-major order
        """
        c = self.config
        n = tile.base_coords.get('N', 0)
        k = tile.base_coords.get('K', 0)
        p = tile.base_coords.get('P', 0)
        q = tile.base_coords.get('Q', 0)
        
        addr = (((n * c.K + k) * c.P + p) * c.Q + q)
        return addr * c.element_bytes
    
    def _compute_row_ids(self):
        """Compute DRAM row ID for each access based on address and row buffer size."""
        row_size = self.config.row_buffer_size
        for access in self.accesses:
            access.row_id = access.address // row_size


def generate_trace_from_mapping(mapping, workload, arch_config: dict = None) -> List[MemoryAccess]:
    """
    Convenience function to generate trace from ILP optimization result.
    
    Args:
        mapping: Mapping object from optimizer
        workload: ConvWorkload object
        arch_config: Optional architecture configuration dict
        
    Returns:
        List of MemoryAccess objects
    """
    config = LoopNestConfig.from_mapping(mapping, workload)
    
    if arch_config:
        config.element_bytes = arch_config.get('element_bytes', 1)
        config.row_buffer_size = arch_config.get('row_buffer_size', 1024)
    
    generator = TraceGenerator(config)
    return generator.generate_trace()


if __name__ == "__main__":
    # Example: Generate trace for a simple Conv layer
    config = LoopNestConfig(
        N=1, K=64, C=64, P=56, Q=56, R=3, S=3,
        stride_h=1, stride_w=1,
        tile_n=1, tile_k=16, tile_c=16, tile_p=14, tile_q=14, tile_r=1, tile_s=1,
        loop_order=['S', 'R', 'Q', 'P', 'C', 'K', 'N'],
        element_bytes=1,
        row_buffer_size=1024,
    )
    
    generator = TraceGenerator(config)
    trace = generator.generate_trace()
    
    print(f"Generated {len(trace)} memory accesses")
    print(f"First 10 accesses:")
    for i, access in enumerate(trace[:10]):
        print(f"  {i}: {access.datatype.name} {access.access_type.name} "
              f"addr={access.address} row={access.row_id} size={access.size_bytes}")
