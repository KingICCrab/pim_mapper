"""
DRAM Trace Validation Module

Generate memory access traces from ILP decisions and
simulate with cycle-accurate DRAM timing model.
"""

from .trace_generator import (
    TraceGenerator,
    LoopNestConfig,
    MemoryAccess,
    AccessType,
    DataType,
    TileInfo,
    generate_trace_from_mapping,
)

from .timing_simulator import (
    DRAMTimingSimulator,
    DRAMTimingParams,
    SimulationStats,
    BankState,
    simulate_trace_with_unindp_config,
    compare_with_ilp_prediction,
)

__all__ = [
    # Trace generation
    'TraceGenerator',
    'LoopNestConfig',
    'MemoryAccess',
    'AccessType',
    'DataType',
    'TileInfo',
    'generate_trace_from_mapping',
    
    # Timing simulation
    'DRAMTimingSimulator',
    'DRAMTimingParams',
    'SimulationStats',
    'BankState',
    'simulate_trace_with_unindp_config',
    'compare_with_ilp_prediction',
]
