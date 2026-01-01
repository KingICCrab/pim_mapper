"""
DRAM Timing Simulator

Simulate DRAM row activation and data transfer using timing parameters
from UniNDP/HBM-PIM architecture configuration.

This module models:
1. Row buffer hits/misses
2. Row activation latency (tRCDRD, tRCDWR, tRP)
3. Data transfer cycles (read/write latency)
4. Bank-level parallelism

Key DRAM Timing Parameters (from UniNDP):
- tRCDRD: Row-to-Column delay for reads (activate to read command)
- tRCDWR: Row-to-Column delay for writes
- tRP: Row Precharge time (close row before opening new one)
- tCCDS: Column-to-Column delay (same bank)
- RL: Read Latency (CAS latency)
- WL: Write Latency
- BL: Burst Length

Row Activation Cycle:
1. Close current row (tRP) - only if different row
2. Activate new row (tRCDRD/tRCDWR)
3. Column access (RL/WL + BL)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import math

from .trace_generator import MemoryAccess, AccessType, DataType


@dataclass
class DRAMTimingParams:
    """
    DRAM timing parameters in cycles.
    
    Values can be loaded from UniNDP config file.
    """
    # Row activation timing
    tRCDRD: int = 14  # Row-to-Column delay (read)
    tRCDWR: int = 8   # Row-to-Column delay (write)
    tRP: int = 14     # Row Precharge time
    tRAS: int = 33    # Row Active time (min time row must stay open)
    tRC: int = 47     # Row Cycle time = tRAS + tRP
    
    # Column access timing
    tCL: int = 14     # CAS Latency (same as RL)
    tCWL: int = 4     # CAS Write Latency (same as WL)
    tCCDS: int = 4    # Column-to-Column delay (same bank)
    tCCDL: int = 5    # Column-to-Column delay (long, different bank groups)
    tBURST: int = 2   # Burst transfer cycles
    
    # Other timing
    tFAW: int = 16    # Four Activation Window
    tRRDS: int = 4    # Row-to-Row delay (same bank group)
    tRRDL: int = 6    # Row-to-Row delay (long)
    
    # Burst length
    BL: int = 4       # Burst length in transfers
    
    # Row buffer size
    row_buffer_size: int = 1024  # bytes
    
    # Number of banks
    num_banks: int = 4
    
    # Clock frequency (MHz) - for converting to time
    clock_freq_mhz: int = 1600
    
    @property
    def read_latency(self) -> int:
        """Total read latency for column access: tCL + tBURST."""
        return self.tCL + self.tBURST
    
    @property
    def write_latency(self) -> int:
        """Total write latency for column access: tCWL + tBURST."""
        return self.tCWL + self.tBURST
    
    @property
    def row_hit_read(self) -> int:
        """Cycles for row buffer hit read."""
        return self.read_latency
    
    @property
    def row_miss_read(self) -> int:
        """Cycles for row miss read (precharge + activate + read)."""
        return self.tRP + self.tRCDRD + self.read_latency
    
    @property
    def row_hit_write(self) -> int:
        """Cycles for row buffer hit write."""
        return self.write_latency
    
    @property
    def row_miss_write(self) -> int:
        """Cycles for row miss write."""
        return self.tRP + self.tRCDWR + self.write_latency
    
    @classmethod
    def from_unindp_config(cls, config: dict) -> 'DRAMTimingParams':
        """
        Load timing parameters from UniNDP YAML config.
        
        Args:
            config: Dictionary loaded from UniNDP config file
        """
        timing = config.get('timing', config.get('dram', {}))
        mem = config.get('memory', {})
        
        return cls(
            tRCDRD=timing.get('tRCDRD', 14),
            tRCDWR=timing.get('tRCDWR', 8),
            tRP=timing.get('tRP', 14),
            tRAS=timing.get('tRAS', 33),
            tRC=timing.get('tRC', 47),
            tCL=timing.get('tCL', timing.get('RL', 14)),
            tCWL=timing.get('tCWL', timing.get('WL', 4)),
            tCCDS=timing.get('tCCDS', 4),
            tCCDL=timing.get('tCCDL', 5),
            tBURST=timing.get('tBURST', 2),
            BL=timing.get('BL', 4),
            row_buffer_size=mem.get('row_buffer_size', 
                                    mem.get('page_size', 1024)),
            num_banks=mem.get('num_banks', 4),
        )


@dataclass
class BankState:
    """State of a single DRAM bank."""
    open_row: Optional[int] = None  # Currently open row (None if closed)
    last_access_cycle: int = 0      # Cycle of last access
    activation_count: int = 0       # Total row activations
    hit_count: int = 0              # Row buffer hits
    miss_count: int = 0             # Row buffer misses


@dataclass
class SimulationStats:
    """Statistics from timing simulation."""
    total_cycles: int = 0
    total_accesses: int = 0
    row_activations: int = 0
    row_hits: int = 0
    row_misses: int = 0
    
    # Per-datatype stats
    input_cycles: int = 0
    weight_cycles: int = 0
    output_cycles: int = 0
    
    input_activations: int = 0
    weight_activations: int = 0
    output_activations: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Row buffer hit rate."""
        total = self.row_hits + self.row_misses
        return self.row_hits / total if total > 0 else 0.0
    
    @property
    def avg_cycles_per_access(self) -> float:
        """Average cycles per memory access."""
        return self.total_cycles / self.total_accesses if self.total_accesses > 0 else 0.0
    
    def to_dict(self) -> dict:
        return {
            'total_cycles': self.total_cycles,
            'total_accesses': self.total_accesses,
            'row_activations': self.row_activations,
            'row_hits': self.row_hits,
            'row_misses': self.row_misses,
            'hit_rate': self.hit_rate,
            'avg_cycles_per_access': self.avg_cycles_per_access,
            'input_cycles': self.input_cycles,
            'weight_cycles': self.weight_cycles,
            'output_cycles': self.output_cycles,
            'input_activations': self.input_activations,
            'weight_activations': self.weight_activations,
            'output_activations': self.output_activations,
        }


class DRAMTimingSimulator:
    """
    Simulate DRAM timing for a sequence of memory accesses.
    
    Models per-bank row buffer state and computes accurate
    cycle counts based on row hits/misses.
    
    Simplifications:
    - Each datatype uses dedicated banks (no bank conflict between types)
    - No command queue scheduling optimization
    - No refresh overhead
    """
    
    def __init__(self, params: DRAMTimingParams, shared_banks: bool = True):
        self.params = params
        self.shared_banks = shared_banks
        
        if shared_banks:
            # Shared banks for all datatypes (realistic model)
            self.banks = [BankState() for _ in range(params.num_banks)]
        else:
            # Bank state for each datatype (simple model: dedicated banks)
            self.bank_states: Dict[DataType, List[BankState]] = {
                DataType.INPUT: [BankState() for _ in range(params.num_banks)],
                DataType.WEIGHT: [BankState() for _ in range(params.num_banks)],
                DataType.OUTPUT: [BankState() for _ in range(params.num_banks)],
            }
        
        self.current_cycle = 0
        self.stats = SimulationStats()
    
    def reset(self):
        """Reset simulator state."""
        if self.shared_banks:
            self.banks = [BankState() for _ in range(self.params.num_banks)]
        else:
            for dtype in DataType:
                self.bank_states[dtype] = [
                    BankState() for _ in range(self.params.num_banks)
                ]
        self.current_cycle = 0
        self.stats = SimulationStats()
    
    def simulate_trace(self, trace: List[MemoryAccess]) -> SimulationStats:
        """
        Simulate a complete memory access trace.
        
        Args:
            trace: List of MemoryAccess objects in chronological order
            
        Returns:
            SimulationStats with timing results
        """
        self.reset()
        
        for access in trace:
            self._process_access(access)
        
        # Aggregate bank stats
        if self.shared_banks:
            for bank in self.banks:
                self.stats.row_activations += bank.activation_count
                self.stats.row_hits += bank.hit_count
                self.stats.row_misses += bank.miss_count
        else:
            for dtype in DataType:
                for bank in self.bank_states[dtype]:
                    self.stats.row_activations += bank.activation_count
                    self.stats.row_hits += bank.hit_count
                    self.stats.row_misses += bank.miss_count
        
        self.stats.total_cycles = self.current_cycle
        self.stats.total_accesses = len(trace)
        
        return self.stats
    
    def _process_access(self, access: MemoryAccess):
        """Process a single memory access."""
        # Determine which bank handles this access
        
        if self.shared_banks:
            # Smart Partitioning to avoid conflicts
            # Input: Banks 0, 1
            # Weight: Bank 2
            # Output: Bank 3
            if self.params.num_banks >= 3:
                if access.datatype == DataType.INPUT:
                    bank_id = 0
                elif access.datatype == DataType.WEIGHT:
                    bank_id = 1
                else: # OUTPUT
                    bank_id = 2
            else:
                bank_id = access.row_id % self.params.num_banks
                
            bank = self.banks[bank_id]
        else:
            bank_id = access.row_id % self.params.num_banks
            bank = self.bank_states[access.datatype][bank_id]
        
        # Check for row hit/miss
        target_row = access.row_id
        is_row_hit = (bank.open_row == target_row)
        
        # Compute cycles for this access
        if access.access_type == AccessType.READ:
            if is_row_hit:
                cycles = self.params.row_hit_read
                bank.hit_count += 1
            else:
                cycles = self.params.row_miss_read
                bank.miss_count += 1
                bank.activation_count += 1
        else:  # WRITE
            if is_row_hit:
                cycles = self.params.row_hit_write
                bank.hit_count += 1
            else:
                cycles = self.params.row_miss_write
                bank.miss_count += 1
                bank.activation_count += 1
        
        # Handle multi-column accesses for large tiles
        # Each column access after the first costs tCCDS
        columns_needed = math.ceil(access.size_bytes / (self.params.BL * 8))  # 8 bytes per transfer
        if columns_needed > 1:
            cycles += (columns_needed - 1) * self.params.tCCDS
        
        # Update bank state
        bank.open_row = target_row
        bank.last_access_cycle = self.current_cycle + cycles
        
        # Update per-datatype stats
        if access.datatype == DataType.INPUT:
            self.stats.input_cycles += cycles
            if not is_row_hit:
                self.stats.input_activations += 1
        elif access.datatype == DataType.WEIGHT:
            self.stats.weight_cycles += cycles
            if not is_row_hit:
                self.stats.weight_activations += 1
        else:
            self.stats.output_cycles += cycles
            if not is_row_hit:
                self.stats.output_activations += 1
        
        # Advance global cycle counter
        self.current_cycle += cycles
    
    def get_activation_cycles(self) -> int:
        """Get total cycles spent on row activations (not data transfer)."""
        activation_cycles = self.stats.row_misses * (self.params.tRP + self.params.tRCDRD)
        return activation_cycles
    
    def get_data_transfer_cycles(self) -> int:
        """Get cycles spent on actual data transfer (excluding activation)."""
        return self.stats.total_cycles - self.get_activation_cycles()


def simulate_trace_with_unindp_config(
    trace: List[MemoryAccess], 
    config_path: str
) -> SimulationStats:
    """
    Simulate trace using timing parameters from UniNDP config file.
    
    Args:
        trace: Memory access trace
        config_path: Path to UniNDP YAML config file
        
    Returns:
        SimulationStats
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    params = DRAMTimingParams.from_unindp_config(config)
    simulator = DRAMTimingSimulator(params)
    
    return simulator.simulate_trace(trace)


def compare_with_ilp_prediction(
    trace: List[MemoryAccess],
    params: DRAMTimingParams,
    ilp_row_activations: int,
    ilp_crossing_ratio: float,
) -> dict:
    """
    Compare simulation results with ILP model predictions.
    
    Args:
        trace: Memory access trace
        params: DRAM timing parameters
        ilp_row_activations: Row activations predicted by ILP
        ilp_crossing_ratio: Crossing ratio from ILP model
        
    Returns:
        Dictionary with comparison results
    """
    simulator = DRAMTimingSimulator(params)
    stats = simulator.simulate_trace(trace)
    
    # Compute simulated crossing ratio
    total_accesses = stats.row_hits + stats.row_misses
    sim_crossing_ratio = stats.row_misses / total_accesses if total_accesses > 0 else 0.0
    
    return {
        'ilp_row_activations': ilp_row_activations,
        'sim_row_activations': stats.row_activations,
        'activation_error': abs(stats.row_activations - ilp_row_activations) / max(1, ilp_row_activations),
        
        'ilp_crossing_ratio': ilp_crossing_ratio,
        'sim_crossing_ratio': sim_crossing_ratio,
        'crossing_ratio_error': abs(sim_crossing_ratio - ilp_crossing_ratio),
        
        'total_cycles': stats.total_cycles,
        'activation_cycles': simulator.get_activation_cycles(),
        'transfer_cycles': simulator.get_data_transfer_cycles(),
        
        'hit_rate': stats.hit_rate,
        'stats': stats.to_dict(),
    }


if __name__ == "__main__":
    # Example usage
    from .trace_generator import TraceGenerator, LoopNestConfig
    
    # Create a simple Conv workload
    config = LoopNestConfig(
        N=1, K=64, C=64, P=56, Q=56, R=3, S=3,
        stride_h=1, stride_w=1,
        tile_n=1, tile_k=16, tile_c=16, tile_p=14, tile_q=14, tile_r=1, tile_s=1,
        loop_order=['S', 'R', 'Q', 'P', 'C', 'K', 'N'],
        element_bytes=1,
        row_buffer_size=1024,
    )
    
    # Generate trace
    generator = TraceGenerator(config)
    trace = generator.generate_trace()
    print(f"Generated {len(trace)} memory accesses")
    
    # Simulate timing
    params = DRAMTimingParams(
        tRCDRD=14, tRCDWR=8, tRP=14,
        tCL=14, tCWL=4, tCCDS=4,
        BL=4, row_buffer_size=1024, num_banks=4,
    )
    
    simulator = DRAMTimingSimulator(params)
    stats = simulator.simulate_trace(trace)
    
    print(f"\nSimulation Results:")
    print(f"  Total cycles: {stats.total_cycles}")
    print(f"  Row activations: {stats.row_activations}")
    print(f"  Row buffer hit rate: {stats.hit_rate:.2%}")
    print(f"  Avg cycles/access: {stats.avg_cycles_per_access:.1f}")
    print(f"\nPer-datatype:")
    print(f"  Input:  cycles={stats.input_cycles}, activations={stats.input_activations}")
    print(f"  Weight: cycles={stats.weight_cycles}, activations={stats.weight_activations}")
    print(f"  Output: cycles={stats.output_cycles}, activations={stats.output_activations}")
