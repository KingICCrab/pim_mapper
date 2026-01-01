"""
Access Trace generation for Golden Model Simulator.

This module converts mapping decisions into concrete memory access sequences
that can be simulated cycle-by-cycle.

An AccessTrace represents the actual sequence of memory accesses that would
occur when executing a workload with a given mapping.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Generator
import math


class AccessType(Enum):
    """Types of memory accesses."""
    READ = 0
    WRITE = 1


@dataclass
class MemoryAccess:
    """
    A single memory access.
    
    Attributes:
        access_type: READ or WRITE
        bank_id: Target bank
        row_addr: Target row address
        col_addr: Column address (optional, for detailed modeling)
        size: Size of access in bytes
        tensor_name: Name of the tensor being accessed (for debugging)
        iteration: Loop iteration indices (for debugging)
    """
    access_type: AccessType
    bank_id: int
    row_addr: int
    col_addr: int = 0
    size: int = 64  # Default cache line size
    tensor_name: str = ""
    iteration: Tuple[int, ...] = field(default_factory=tuple)
    
    def __repr__(self) -> str:
        op = 'RD' if self.access_type == AccessType.READ else 'WR'
        return f"{op}(bank={self.bank_id}, row={self.row_addr}, {self.tensor_name})"


class AccessTrace:
    """
    A sequence of memory accesses for simulation.
    
    This represents the ground truth of what memory operations would actually
    occur when executing a workload.
    """
    
    def __init__(self):
        self.accesses: List[MemoryAccess] = []
        self._position = 0
    
    def add_access(self, access: MemoryAccess) -> None:
        """Add an access to the trace."""
        self.accesses.append(access)
    
    def add_read(self, bank_id: int, row_addr: int, 
                 tensor_name: str = "", **kwargs) -> None:
        """Add a read access."""
        self.accesses.append(MemoryAccess(
            access_type=AccessType.READ,
            bank_id=bank_id,
            row_addr=row_addr,
            tensor_name=tensor_name,
            **kwargs
        ))
    
    def add_write(self, bank_id: int, row_addr: int,
                  tensor_name: str = "", **kwargs) -> None:
        """Add a write access."""
        self.accesses.append(MemoryAccess(
            access_type=AccessType.WRITE,
            bank_id=bank_id,
            row_addr=row_addr,
            tensor_name=tensor_name,
            **kwargs
        ))
    
    def __len__(self) -> int:
        return len(self.accesses)
    
    def __iter__(self) -> Generator[MemoryAccess, None, None]:
        for access in self.accesses:
            yield access
    
    def __getitem__(self, idx: int) -> MemoryAccess:
        return self.accesses[idx]
    
    def reset(self) -> None:
        """Reset iteration position."""
        self._position = 0
    
    def get_next(self) -> Optional[MemoryAccess]:
        """Get next access in sequence."""
        if self._position >= len(self.accesses):
            return None
        access = self.accesses[self._position]
        self._position += 1
        return access
    
    def get_statistics(self) -> dict:
        """Get trace statistics."""
        reads = sum(1 for a in self.accesses if a.access_type == AccessType.READ)
        writes = sum(1 for a in self.accesses if a.access_type == AccessType.WRITE)
        
        # Count unique rows per bank
        bank_rows = {}
        for access in self.accesses:
            if access.bank_id not in bank_rows:
                bank_rows[access.bank_id] = set()
            bank_rows[access.bank_id].add(access.row_addr)
        
        unique_rows = sum(len(rows) for rows in bank_rows.values())
        
        return {
            'total_accesses': len(self.accesses),
            'reads': reads,
            'writes': writes,
            'unique_banks': len(bank_rows),
            'unique_rows': unique_rows,
            'rows_per_bank': {bid: len(rows) for bid, rows in bank_rows.items()},
        }


class AccessPatternGenerator:
    """
    Generates access traces from mapping decisions.
    
    This converts high-level mapping (tile sizes, loop orders) into
    concrete memory access sequences.
    """
    
    def __init__(self, row_size: int = 8192, col_size: int = 64):
        """
        Initialize the generator.
        
        Args:
            row_size: Size of a DRAM row in bytes
            col_size: Size of a column access (burst) in bytes
        """
        self.row_size = row_size
        self.col_size = col_size
    
    def generate_1d_tile_accesses(
        self,
        tensor_name: str,
        base_bank: int,
        base_row: int,
        total_elements: int,
        tile_size: int,
        element_size: int = 4,  # 4 bytes for float32
        elements_per_row: Optional[int] = None,
    ) -> AccessTrace:
        """
        Generate accesses for 1D tiled traversal.
        
        Args:
            tensor_name: Name of the tensor
            base_bank: Starting bank ID
            base_row: Starting row address
            total_elements: Total number of elements
            tile_size: Size of each tile
            element_size: Size of each element in bytes
            elements_per_row: Elements per DRAM row (computed if not provided)
            
        Returns:
            AccessTrace with all memory accesses
        """
        trace = AccessTrace()
        
        if elements_per_row is None:
            elements_per_row = self.row_size // element_size
        
        num_tiles = math.ceil(total_elements / tile_size)
        
        for tile_idx in range(num_tiles):
            start_elem = tile_idx * tile_size
            end_elem = min(start_elem + tile_size, total_elements)
            
            for elem_idx in range(start_elem, end_elem):
                row_offset = elem_idx // elements_per_row
                row_addr = base_row + row_offset
                
                trace.add_read(
                    bank_id=base_bank,
                    row_addr=row_addr,
                    tensor_name=tensor_name,
                    iteration=(tile_idx, elem_idx - start_elem),
                )
        
        return trace
    
    def generate_2d_tile_accesses(
        self,
        tensor_name: str,
        base_bank: int,
        base_row: int,
        shape: Tuple[int, int],
        tile_shape: Tuple[int, int],
        element_size: int = 4,
        row_major: bool = True,
    ) -> AccessTrace:
        """
        Generate accesses for 2D tiled traversal.
        
        Args:
            tensor_name: Name of the tensor
            base_bank: Starting bank ID  
            base_row: Starting row address
            shape: (height, width) of the tensor
            tile_shape: (tile_h, tile_w) tile dimensions
            element_size: Size of each element in bytes
            row_major: If True, traverse tiles in row-major order
            
        Returns:
            AccessTrace with all memory accesses
        """
        trace = AccessTrace()
        height, width = shape
        tile_h, tile_w = tile_shape
        elements_per_row = self.row_size // element_size
        
        num_tiles_h = math.ceil(height / tile_h)
        num_tiles_w = math.ceil(width / tile_w)
        
        for tile_i in range(num_tiles_h):
            for tile_j in range(num_tiles_w):
                # Tile boundaries
                start_i = tile_i * tile_h
                end_i = min(start_i + tile_h, height)
                start_j = tile_j * tile_w
                end_j = min(start_j + tile_w, width)
                
                # Access elements within tile
                for i in range(start_i, end_i):
                    for j in range(start_j, end_j):
                        # Linear index in row-major layout
                        linear_idx = i * width + j
                        row_offset = linear_idx // elements_per_row
                        row_addr = base_row + row_offset
                        
                        trace.add_read(
                            bank_id=base_bank,
                            row_addr=row_addr,
                            tensor_name=tensor_name,
                            iteration=(tile_i, tile_j, i - start_i, j - start_j),
                        )
        
        return trace
    
    def generate_conv_accesses(
        self,
        input_shape: Tuple[int, int, int],    # (C, H, W)
        weight_shape: Tuple[int, int, int, int],  # (K, C, R, S)
        output_shape: Tuple[int, int, int],   # (K, P, Q)
        tile_config: dict,  # Tiling configuration
        input_bank: int = 0,
        weight_bank: int = 1,
        output_bank: int = 2,
        base_row: int = 0,
        element_size: int = 4,
    ) -> AccessTrace:
        """
        Generate access trace for convolution operation.
        
        Args:
            input_shape: Input activation shape (C, H, W)
            weight_shape: Weight shape (K, C, R, S)
            output_shape: Output shape (K, P, Q)
            tile_config: Dictionary with tile sizes for each dimension
            input_bank, weight_bank, output_bank: Bank assignments
            base_row: Starting row address
            element_size: Element size in bytes
            
        Returns:
            AccessTrace with all memory accesses for the convolution
        """
        trace = AccessTrace()
        
        C, H, W = input_shape
        K, _, R, S = weight_shape
        _, P, Q = output_shape
        
        elements_per_row = self.row_size // element_size
        
        # Get tile sizes (use full dimension if not specified)
        tile_k = tile_config.get('K', K)
        tile_c = tile_config.get('C', C)
        tile_p = tile_config.get('P', P)
        tile_q = tile_config.get('Q', Q)
        
        # Number of tiles
        num_k = math.ceil(K / tile_k)
        num_c = math.ceil(C / tile_c)
        num_p = math.ceil(P / tile_p)
        num_q = math.ceil(Q / tile_q)
        
        # Standard loop nest: K -> C -> P -> Q -> R -> S (output-stationary)
        for k_tile in range(num_k):
            k_start = k_tile * tile_k
            k_end = min(k_start + tile_k, K)
            
            for c_tile in range(num_c):
                c_start = c_tile * tile_c
                c_end = min(c_start + tile_c, C)
                
                for p_tile in range(num_p):
                    p_start = p_tile * tile_p
                    p_end = min(p_start + tile_p, P)
                    
                    for q_tile in range(num_q):
                        q_start = q_tile * tile_q
                        q_end = min(q_start + tile_q, Q)
                        
                        # Inner loops
                        for k in range(k_start, k_end):
                            for c in range(c_start, c_end):
                                for p in range(p_start, p_end):
                                    for q in range(q_start, q_end):
                                        # Read output (accumulate)
                                        out_idx = k * P * Q + p * Q + q
                                        out_row = base_row + out_idx // elements_per_row
                                        trace.add_read(
                                            bank_id=output_bank,
                                            row_addr=out_row,
                                            tensor_name='output',
                                        )
                                        
                                        for r in range(R):
                                            for s in range(S):
                                                # Input access
                                                h = p + r
                                                w = q + s
                                                in_idx = c * H * W + h * W + w
                                                in_row = base_row + in_idx // elements_per_row
                                                trace.add_read(
                                                    bank_id=input_bank,
                                                    row_addr=in_row,
                                                    tensor_name='input',
                                                )
                                                
                                                # Weight access
                                                w_idx = k * C * R * S + c * R * S + r * S + s
                                                w_row = base_row + w_idx // elements_per_row
                                                trace.add_read(
                                                    bank_id=weight_bank,
                                                    row_addr=w_row,
                                                    tensor_name='weight',
                                                )
                                        
                                        # Write output
                                        trace.add_write(
                                            bank_id=output_bank,
                                            row_addr=out_row,
                                            tensor_name='output',
                                        )
        
        return trace
    
    def generate_gemm_accesses(
        self,
        M: int,
        N: int,
        K: int,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        A_bank: int = 0,
        B_bank: int = 1,
        C_bank: int = 2,
        base_row: int = 0,
        element_size: int = 4,
    ) -> AccessTrace:
        """
        Generate access trace for GEMM: C = A @ B.
        
        Args:
            M, N, K: Matrix dimensions (C is MxN, A is MxK, B is KxN)
            tile_m, tile_n, tile_k: Tile sizes
            A_bank, B_bank, C_bank: Bank assignments
            base_row: Starting row address
            element_size: Element size in bytes
            
        Returns:
            AccessTrace with all memory accesses for GEMM
        """
        trace = AccessTrace()
        elements_per_row = self.row_size // element_size
        
        num_m = math.ceil(M / tile_m)
        num_n = math.ceil(N / tile_n)
        num_k = math.ceil(K / tile_k)
        
        for m_tile in range(num_m):
            m_start = m_tile * tile_m
            m_end = min(m_start + tile_m, M)
            
            for n_tile in range(num_n):
                n_start = n_tile * tile_n
                n_end = min(n_start + tile_n, N)
                
                for k_tile in range(num_k):
                    k_start = k_tile * tile_k
                    k_end = min(k_start + tile_k, K)
                    
                    # Inner loops
                    for m in range(m_start, m_end):
                        for n in range(n_start, n_end):
                            # Read C (for accumulation)
                            c_idx = m * N + n
                            c_row = base_row + c_idx // elements_per_row
                            trace.add_read(
                                bank_id=C_bank,
                                row_addr=c_row,
                                tensor_name='C',
                            )
                            
                            for k in range(k_start, k_end):
                                # Read A[m, k]
                                a_idx = m * K + k
                                a_row = base_row + a_idx // elements_per_row
                                trace.add_read(
                                    bank_id=A_bank,
                                    row_addr=a_row,
                                    tensor_name='A',
                                )
                                
                                # Read B[k, n]
                                b_idx = k * N + n
                                b_row = base_row + b_idx // elements_per_row
                                trace.add_read(
                                    bank_id=B_bank,
                                    row_addr=b_row,
                                    tensor_name='B',
                                )
                            
                            # Write C
                            trace.add_write(
                                bank_id=C_bank,
                                row_addr=c_row,
                                tensor_name='C',
                            )
        
        return trace


def analyze_row_crossing(trace: AccessTrace) -> dict:
    """
    Analyze row crossing (row buffer misses) in an access trace.
    
    This is the ground truth for row activation costs.
    
    Returns:
        Dictionary with row crossing statistics per bank and tensor
    """
    bank_last_row = {}  # Last accessed row per bank
    bank_stats = {}     # Statistics per bank
    tensor_stats = {}   # Statistics per tensor
    
    for access in trace:
        bank_id = access.bank_id
        row_addr = access.row_addr
        tensor = access.tensor_name
        
        # Initialize stats
        if bank_id not in bank_stats:
            bank_stats[bank_id] = {'hits': 0, 'misses': 0, 'total': 0}
        if tensor not in tensor_stats:
            tensor_stats[tensor] = {'hits': 0, 'misses': 0, 'total': 0}
        
        # Check row hit/miss
        bank_stats[bank_id]['total'] += 1
        tensor_stats[tensor]['total'] += 1
        
        if bank_id in bank_last_row:
            if bank_last_row[bank_id] == row_addr:
                # Row hit
                bank_stats[bank_id]['hits'] += 1
                tensor_stats[tensor]['hits'] += 1
            else:
                # Row miss (crossing)
                bank_stats[bank_id]['misses'] += 1
                tensor_stats[tensor]['misses'] += 1
        else:
            # First access (cold miss)
            bank_stats[bank_id]['misses'] += 1
            tensor_stats[tensor]['misses'] += 1
        
        bank_last_row[bank_id] = row_addr
    
    # Calculate hit rates
    for stats in [bank_stats, tensor_stats]:
        for key in stats:
            total = stats[key]['total']
            if total > 0:
                stats[key]['hit_rate'] = stats[key]['hits'] / total
                stats[key]['miss_rate'] = stats[key]['misses'] / total
            else:
                stats[key]['hit_rate'] = 0
                stats[key]['miss_rate'] = 0
    
    return {
        'by_bank': bank_stats,
        'by_tensor': tensor_stats,
        'total_misses': sum(s['misses'] for s in bank_stats.values()),
        'total_hits': sum(s['hits'] for s in bank_stats.values()),
        'total_accesses': sum(s['total'] for s in bank_stats.values()),
    }
