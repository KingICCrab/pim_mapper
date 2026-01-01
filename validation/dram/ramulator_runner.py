"""
Ramulator2 interface for running DRAM simulations.
"""

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Optional, Dict, Any


# Default Ramulator2 binary path
DEFAULT_RAMULATOR_BIN = os.path.join(
    os.path.dirname(__file__), 
    "../../OptiPIM/simulator/build/ramulator2"
)


@dataclass
class RamulatorResult:
    """Result from Ramulator2 simulation."""
    cycles: int
    num_reads: int = 0
    num_writes: int = 0
    row_hits: int = 0
    row_misses: int = 0
    row_conflicts: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Row buffer hit rate."""
        total = self.row_hits + self.row_misses + self.row_conflicts
        return self.row_hits / total if total > 0 else 0.0
    
    @property
    def total_row_activations(self) -> int:
        """Total row activations (misses + conflicts)."""
        return self.row_misses + self.row_conflicts


class RamulatorRunner:
    """Interface to run Ramulator2 simulations."""
    
    def __init__(self, ramulator_bin: str = None, config_dir: str = None):
        self.ramulator_bin = ramulator_bin or DEFAULT_RAMULATOR_BIN
        self.config_dir = config_dir or os.path.join(os.path.dirname(__file__), "configs")
        
        if not os.path.exists(self.ramulator_bin):
            print(f"Warning: Ramulator binary not found at {self.ramulator_bin}")
    
    def is_available(self) -> bool:
        """Check if Ramulator2 is available."""
        return os.path.exists(self.ramulator_bin)
    
    def run(self, trace_path: str, config_file: str = None) -> RamulatorResult:
        """Run Ramulator2 simulation."""
        if config_file is None:
            config_file = os.path.join(self.config_dir, "ddr5_config.yaml")
        
        cmd = [
            self.ramulator_bin,
            "--config_file", config_file,
            "--param", f"Frontend.path={trace_path}",
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Ramulator simulation timed out")
        except FileNotFoundError:
            raise RuntimeError(f"Ramulator binary not found: {self.ramulator_bin}")
        
        if result.returncode != 0:
            raise RuntimeError(f"Ramulator failed:\n{result.stderr}")
        
        return self._parse_output(result.stdout)
    
    def _parse_output(self, output: str) -> RamulatorResult:
        """Parse Ramulator2 output."""
        cycles = 0
        num_reads = 0
        num_writes = 0
        row_hits = 0
        row_misses = 0
        row_conflicts = 0
        
        for line in output.splitlines():
            if m := re.search(r"memory_system_cycles:\s*(\d+)", line):
                cycles = int(m.group(1))
            elif m := re.search(r"total_num_read_requests:\s*(\d+)", line):
                num_reads = int(m.group(1))
            elif m := re.search(r"total_num_write_requests:\s*(\d+)", line):
                num_writes = int(m.group(1))
            elif m := re.search(r"row_hits:\s*(\d+)", line):
                row_hits = int(m.group(1))
            elif m := re.search(r"row_misses:\s*(\d+)", line):
                row_misses = int(m.group(1))
            elif m := re.search(r"row_conflicts:\s*(\d+)", line):
                row_conflicts = int(m.group(1))
        
        return RamulatorResult(
            cycles=cycles,
            num_reads=num_reads,
            num_writes=num_writes,
            row_hits=row_hits,
            row_misses=row_misses,
            row_conflicts=row_conflicts,
        )
