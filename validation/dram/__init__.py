"""
DRAM Validation Module for pim_optimizer.

Provides tools to validate pim_optimizer's DRAM model against Ramulator2 simulation.
"""

from .trace_generator import TraceGenerator, DRAMConfig
from .ramulator_runner import RamulatorRunner, RamulatorResult

__all__ = [
    'TraceGenerator',
    'DRAMConfig',
    'RamulatorRunner', 
    'RamulatorResult',
]
