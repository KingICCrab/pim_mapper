"""
Main Simulator for Golden Model.

This module provides the cycle-accurate simulator that executes
memory access traces through the DRAM model.

Based on UniNDP sim/sim.py architecture:
- Global tick counter for cycle-accurate timing
- Instruction queue for memory operations  
- Hardware system model for timing validation
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import math

from .dram import DRAMBank, DRAMBankGroup, DRAMTiming, BankState
from .access_trace import AccessTrace, MemoryAccess, AccessType


@dataclass
class SimulatorConfig:
    """
    Configuration for the DRAM simulator.
    
    Attributes:
        num_banks: Number of DRAM banks
        num_bank_groups: Number of bank groups
        banks_per_group: Banks per bank group
        timing: DRAM timing parameters
        row_size: Size of a DRAM row in bytes
        col_size: Size of a column (burst) access in bytes
    """
    num_banks: int = 16
    num_bank_groups: int = 4
    banks_per_group: int = 4
    timing: DRAMTiming = field(default_factory=DRAMTiming)
    row_size: int = 8192   # 8KB row
    col_size: int = 64     # 64B burst


@dataclass
class SimulationResult:
    """
    Results from a simulation run.
    
    Contains cycle-accurate metrics that serve as ground truth.
    """
    # Timing
    total_cycles: int = 0
    
    # Row buffer statistics
    row_hits: int = 0
    row_misses: int = 0
    row_empty_accesses: int = 0
    
    # Access counts
    total_reads: int = 0
    total_writes: int = 0
    
    # Per-bank statistics
    bank_stats: Dict[int, dict] = field(default_factory=dict)
    
    # Per-tensor statistics (if tensor names provided)
    tensor_stats: Dict[str, dict] = field(default_factory=dict)
    
    @property
    def total_accesses(self) -> int:
        return self.total_reads + self.total_writes
    
    @property
    def row_buffer_hit_rate(self) -> float:
        total = self.row_hits + self.row_misses + self.row_empty_accesses
        return self.row_hits / total if total > 0 else 0.0
    
    @property
    def row_activations(self) -> int:
        """Total row activations (misses + empty accesses)."""
        return self.row_misses + self.row_empty_accesses
    
    def to_dict(self) -> dict:
        """Convert to dictionary for comparison."""
        return {
            'total_cycles': self.total_cycles,
            'row_hits': self.row_hits,
            'row_misses': self.row_misses,
            'row_empty_accesses': self.row_empty_accesses,
            'row_activations': self.row_activations,
            'total_reads': self.total_reads,
            'total_writes': self.total_writes,
            'total_accesses': self.total_accesses,
            'row_buffer_hit_rate': self.row_buffer_hit_rate,
        }


class Simulator:
    """
    Cycle-accurate DRAM simulator.
    
    This is the core of the Golden Model - it provides ground truth
    by simulating every memory access through precise timing models.
    
    Based on UniNDP sim/sim.py architecture.
    """
    
    def __init__(self, config: Optional[SimulatorConfig] = None):
        """
        Initialize the simulator.
        
        Args:
            config: Simulator configuration (uses defaults if not provided)
        """
        self.config = config or SimulatorConfig()
        
        # Initialize banks
        self.banks: List[DRAMBank] = []
        for i in range(self.config.num_banks):
            self.banks.append(DRAMBank(bank_id=i, timing=self.config.timing))
        
        # Global cycle counter
        self.global_tick = 0
        
        # Statistics tracking
        self._reset_stats()
    
    def _reset_stats(self) -> None:
        """Reset simulation statistics."""
        self._stats = SimulationResult()
        self._tensor_last_row: Dict[str, Dict[int, int]] = {}  # tensor -> {bank -> last_row}
    
    def reset(self) -> None:
        """Reset simulator to initial state."""
        self.global_tick = 0
        for bank in self.banks:
            bank.reset()
        self._reset_stats()
    
    def simulate(self, trace: AccessTrace, verbose: bool = False) -> SimulationResult:
        """
        Simulate execution of an access trace.
        
        Args:
            trace: The access trace to execute
            verbose: If True, print progress information
            
        Returns:
            SimulationResult with detailed metrics
        """
        self.reset()
        
        if verbose:
            print(f"Simulating {len(trace)} memory accesses...")
        
        for i, access in enumerate(trace):
            self._execute_access(access)
            
            if verbose and (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1}/{len(trace)} accesses, "
                      f"cycle={self.global_tick}")
        
        # Finalize statistics
        result = self._finalize_stats()
        
        if verbose:
            print(f"Simulation complete: {result.total_cycles} cycles, "
                  f"{result.row_buffer_hit_rate:.2%} hit rate")
        
        return result
    
    def _execute_access(self, access: MemoryAccess) -> None:
        """
        Execute a single memory access.
        
        Updates the global tick and bank states.
        """
        bank_id = access.bank_id
        row_addr = access.row_addr
        is_write = access.access_type == AccessType.WRITE
        
        # Bounds check
        if bank_id >= len(self.banks):
            raise ValueError(f"Invalid bank_id {bank_id}, only {len(self.banks)} banks available")
        
        bank = self.banks[bank_id]
        
        # Execute access
        completion_cycle, access_type = bank.issue_access(
            current_cycle=self.global_tick,
            target_row=row_addr,
            is_write=is_write
        )
        
        # Update global tick (model sequential access within same bank)
        self.global_tick = max(self.global_tick, completion_cycle)
        
        # Update statistics
        self._update_stats(access, access_type)
    
    def _update_stats(self, access: MemoryAccess, access_type: str) -> None:
        """Update simulation statistics."""
        # Access type counts
        if access.access_type == AccessType.READ:
            self._stats.total_reads += 1
        else:
            self._stats.total_writes += 1
        
        # Row buffer statistics
        if access_type == 'hit':
            self._stats.row_hits += 1
        elif access_type == 'miss':
            self._stats.row_misses += 1
        else:  # empty
            self._stats.row_empty_accesses += 1
        
        # Per-bank statistics
        bank_id = access.bank_id
        if bank_id not in self._stats.bank_stats:
            self._stats.bank_stats[bank_id] = {
                'hits': 0, 'misses': 0, 'empty': 0, 'reads': 0, 'writes': 0
            }
        
        if access_type == 'hit':
            self._stats.bank_stats[bank_id]['hits'] += 1
        elif access_type == 'miss':
            self._stats.bank_stats[bank_id]['misses'] += 1
        else:
            self._stats.bank_stats[bank_id]['empty'] += 1
        
        if access.access_type == AccessType.READ:
            self._stats.bank_stats[bank_id]['reads'] += 1
        else:
            self._stats.bank_stats[bank_id]['writes'] += 1
        
        # Per-tensor statistics
        tensor = access.tensor_name
        if tensor:
            if tensor not in self._stats.tensor_stats:
                self._stats.tensor_stats[tensor] = {
                    'hits': 0, 'misses': 0, 'empty': 0, 'reads': 0, 'writes': 0
                }
            
            if access_type == 'hit':
                self._stats.tensor_stats[tensor]['hits'] += 1
            elif access_type == 'miss':
                self._stats.tensor_stats[tensor]['misses'] += 1
            else:
                self._stats.tensor_stats[tensor]['empty'] += 1
            
            if access.access_type == AccessType.READ:
                self._stats.tensor_stats[tensor]['reads'] += 1
            else:
                self._stats.tensor_stats[tensor]['writes'] += 1
    
    def _finalize_stats(self) -> SimulationResult:
        """Finalize and return simulation statistics."""
        self._stats.total_cycles = self.global_tick
        
        # Calculate hit rates for banks
        for bank_id, stats in self._stats.bank_stats.items():
            total = stats['hits'] + stats['misses'] + stats['empty']
            stats['hit_rate'] = stats['hits'] / total if total > 0 else 0
            stats['total'] = total
        
        # Calculate hit rates for tensors
        for tensor, stats in self._stats.tensor_stats.items():
            total = stats['hits'] + stats['misses'] + stats['empty']
            stats['hit_rate'] = stats['hits'] / total if total > 0 else 0
            stats['total'] = total
        
        return self._stats
    
    def simulate_with_comparison(
        self, 
        trace: AccessTrace,
        expected_costs: dict,
    ) -> Tuple[SimulationResult, dict]:
        """
        Simulate and compare with expected costs from ILP/formulas.
        
        Args:
            trace: The access trace to execute
            expected_costs: Dictionary with expected cost metrics
            
        Returns:
            Tuple of (SimulationResult, comparison_dict)
        """
        result = self.simulate(trace)
        
        comparison = {}
        actual = result.to_dict()
        
        for key, expected in expected_costs.items():
            if key in actual:
                actual_val = actual[key]
                diff = actual_val - expected
                rel_error = abs(diff) / expected if expected != 0 else 0
                comparison[key] = {
                    'expected': expected,
                    'actual': actual_val,
                    'difference': diff,
                    'relative_error': rel_error,
                    'match': abs(diff) < 1e-6 or rel_error < 0.01,
                }
        
        return result, comparison


class RowCrossingAnalyzer:
    """
    Specialized analyzer for row crossing (row buffer miss) behavior.
    
    This analyzes the exact row crossing patterns to verify the
    row activation cost model.
    """
    
    def __init__(self, row_size: int = 8192, element_size: int = 4):
        """
        Initialize analyzer.
        
        Args:
            row_size: Size of a DRAM row in bytes
            element_size: Size of each element in bytes
        """
        self.row_size = row_size
        self.element_size = element_size
        self.elements_per_row = row_size // element_size
    
    def analyze_1d_tiling(
        self,
        total_elements: int,
        tile_size: int,
        num_iterations: int = 1,
    ) -> dict:
        """
        Analyze row crossing for 1D tiled access pattern.
        
        This provides ground truth for:
        - Number of unique rows touched
        - Number of row crossings per iteration
        - Total row activations
        
        Args:
            total_elements: Total elements in the tensor
            tile_size: Tile size
            num_iterations: Number of times the tensor is fully traversed
            
        Returns:
            Dictionary with ground truth metrics
        """
        # Calculate unique rows
        total_rows = math.ceil(total_elements / self.elements_per_row)
        
        # Number of tiles
        num_tiles = math.ceil(total_elements / tile_size)
        
        # Rows covered by each tile
        rows_per_tile = math.ceil(tile_size / self.elements_per_row)
        
        # For each iteration through all tiles
        row_crossings_per_iteration = 0
        
        for tile_idx in range(num_tiles):
            start_elem = tile_idx * tile_size
            end_elem = min(start_elem + tile_size, total_elements)
            
            start_row = start_elem // self.elements_per_row
            end_row = (end_elem - 1) // self.elements_per_row
            
            # Crossings within this tile
            tile_crossings = end_row - start_row  # transitions between rows
            
            # Crossing from previous tile (if not first)
            if tile_idx > 0:
                prev_end_elem = start_elem - 1
                prev_end_row = prev_end_elem // self.elements_per_row
                if prev_end_row != start_row:
                    row_crossings_per_iteration += 1  # crossing between tiles
            
            row_crossings_per_iteration += tile_crossings
        
        # First access to first tile is also a "crossing" (cold start)
        row_crossings_per_iteration += 1
        
        # Total for all iterations
        total_row_activations = row_crossings_per_iteration * num_iterations
        
        return {
            'total_elements': total_elements,
            'tile_size': tile_size,
            'elements_per_row': self.elements_per_row,
            'total_rows': total_rows,
            'num_tiles': num_tiles,
            'rows_per_tile': rows_per_tile,
            'row_crossings_per_iteration': row_crossings_per_iteration,
            'num_iterations': num_iterations,
            'total_row_activations': total_row_activations,
            # Crossing ratio = row activations / total accesses
            'crossing_ratio': total_row_activations / (total_elements * num_iterations),
        }
    
    def analyze_interleaved_access(
        self,
        tensor_A_elements: int,
        tensor_B_elements: int,
        tile_A: int,
        tile_B: int,
        interleave_pattern: str = 'AABB',  # Pattern like 'AB' or 'AABB'
    ) -> dict:
        """
        Analyze row crossing for interleaved access to multiple tensors.
        
        This is important for understanding how accessing multiple data
        streams affects row buffer locality.
        
        Args:
            tensor_A_elements: Elements in tensor A
            tensor_B_elements: Elements in tensor B
            tile_A: Tile size for A
            tile_B: Tile size for B
            interleave_pattern: How A and B accesses are interleaved
            
        Returns:
            Dictionary with ground truth metrics
        """
        # Analyze each tensor separately
        analysis_A = self.analyze_1d_tiling(tensor_A_elements, tile_A)
        analysis_B = self.analyze_1d_tiling(tensor_B_elements, tile_B)
        
        # When interleaved, we lose row buffer locality between switches
        # Each switch from A to B (or B to A) causes a row miss
        
        num_switches = len(interleave_pattern) - 1  # e.g., 'AABB' has 3 switches
        
        # Estimate extra crossings due to interleaving
        # This is approximate - actual would need simulation
        total_A_tiles = math.ceil(tensor_A_elements / tile_A)
        total_B_tiles = math.ceil(tensor_B_elements / tile_B)
        
        return {
            'tensor_A': analysis_A,
            'tensor_B': analysis_B,
            'interleave_pattern': interleave_pattern,
            'estimated_switch_overhead': num_switches,
            'combined_row_activations': (
                analysis_A['total_row_activations'] + 
                analysis_B['total_row_activations']
            ),
        }


def simulate_mapping(
    mapping_config: dict,
    workload_config: dict,
    dram_config: Optional[SimulatorConfig] = None,
) -> SimulationResult:
    """
    High-level function to simulate a mapping configuration.
    
    This is the main entry point for verifying ILP solutions.
    
    Args:
        mapping_config: Mapping configuration with tile sizes, loop orders, etc.
        workload_config: Workload configuration (tensor shapes, etc.)
        dram_config: DRAM simulator configuration
        
    Returns:
        SimulationResult with ground truth metrics
    """
    from .access_trace import AccessPatternGenerator
    
    config = dram_config or SimulatorConfig()
    generator = AccessPatternGenerator(
        row_size=config.row_size,
        col_size=config.col_size
    )
    
    workload_type = workload_config.get('type', 'gemm')
    
    if workload_type == 'gemm':
        M = workload_config['M']
        N = workload_config['N']
        K = workload_config['K']
        
        tile_m = mapping_config.get('tile_M', M)
        tile_n = mapping_config.get('tile_N', N)
        tile_k = mapping_config.get('tile_K', K)
        
        trace = generator.generate_gemm_accesses(
            M=M, N=N, K=K,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            A_bank=mapping_config.get('A_bank', 0),
            B_bank=mapping_config.get('B_bank', 1),
            C_bank=mapping_config.get('C_bank', 2),
        )
    
    elif workload_type == 'conv':
        input_shape = workload_config['input_shape']
        weight_shape = workload_config['weight_shape']
        output_shape = workload_config['output_shape']
        
        tile_config = {
            'K': mapping_config.get('tile_K', weight_shape[0]),
            'C': mapping_config.get('tile_C', weight_shape[1]),
            'P': mapping_config.get('tile_P', output_shape[1]),
            'Q': mapping_config.get('tile_Q', output_shape[2]),
        }
        
        trace = generator.generate_conv_accesses(
            input_shape=input_shape,
            weight_shape=weight_shape,
            output_shape=output_shape,
            tile_config=tile_config,
            input_bank=mapping_config.get('input_bank', 0),
            weight_bank=mapping_config.get('weight_bank', 1),
            output_bank=mapping_config.get('output_bank', 2),
        )
    
    else:
        raise ValueError(f"Unknown workload type: {workload_type}")
    
    simulator = Simulator(config)
    return simulator.simulate(trace)
