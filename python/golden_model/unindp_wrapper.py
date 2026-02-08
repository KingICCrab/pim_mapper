"""
UniNDP Simulator Wrapper for Golden Model.

This module provides a clean interface to use UniNDP's cycle-accurate
simulator as the ground truth for verifying PIM optimizer results.

UniNDP provides:
- Cycle-accurate DRAM timing simulation  
- Full memory hierarchy: Channel → Rank → Device → Bank
- PIM compute unit modeling
- Proper request scheduling

NOTE: UniNDP has specific workload size requirements due to its HBM-PIM
backend design. The wrapper handles these constraints automatically.
"""

import sys
import os
import subprocess
import tempfile
import csv
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

# UniNDP project path
UNINDP_PATH = Path("/Users/haochenzhao/Projects/UniNDP")


@dataclass
class UniNDPConfig:
    """Configuration for UniNDP simulator."""
    unindp_path: Path = UNINDP_PATH
    config_file: str = "config/hbm-pim.yaml"
    power_of_2: bool = False
    allow_under_utilize: bool = False


@dataclass 
class UniNDPResult:
    """Results from UniNDP simulation."""
    total_cycles: float = 0.0
    workload_size: Tuple[int, int] = (0, 0)
    raw_output: str = ""
    
    # Timing constants (from UniNDP)
    tCK: int = 1
    tREFI: int = 3900
    tRFC: int = 350
    
    @property
    def cycles_with_refresh(self) -> float:
        """Cycles including DRAM refresh overhead."""
        return self.tCK * self.total_cycles * (self.tRFC + self.tREFI) / self.tREFI
    
    def __repr__(self) -> str:
        return (f"UniNDPResult(workload={self.workload_size}, "
                f"cycles={self.total_cycles:.2f}, "
                f"with_refresh={self.cycles_with_refresh:.2f})")


class UniNDPSimulator:
    """
    Wrapper for UniNDP simulator.
    
    This runs UniNDP as a subprocess to get cycle-accurate simulation results.
    This is the recommended approach because:
    1. UniNDP has complex internal state and dependencies
    2. Subprocess isolation prevents state corruption
    3. We get exactly the same results as running UniNDP directly
    """
    
    def __init__(self, config: Optional[UniNDPConfig] = None):
        """
        Initialize the UniNDP simulator wrapper.
        
        Args:
            config: Configuration for UniNDP
        """
        self.config = config or UniNDPConfig()
        self._check_available()
    
    def _check_available(self) -> bool:
        """Check if UniNDP is available."""
        sim_verify = self.config.unindp_path / "sim_verify.py"
        if not sim_verify.exists():
            print(f"⚠ UniNDP not found at {self.config.unindp_path}")
            return False
        return True
    
    @property
    def is_available(self) -> bool:
        """Check if UniNDP is available."""
        return (self.config.unindp_path / "sim_verify.py").exists()
    
    def simulate_gemm(self, M: int, K: int) -> UniNDPResult:
        """
        Simulate GEMM workload: C[M,1] = A[M,K] @ B[K,1] (MVM)
        
        Note: UniNDP's HBM-PIM verify mode expects MVM (N=1) workloads.
        For general GEMM, the workload is batched MVMs.
        
        Args:
            M: Output vector dimension
            K: Inner dimension (weight rows)
            
        Returns:
            UniNDPResult with cycle-accurate simulation results
        """
        # Build command
        cmd = [
            sys.executable,
            str(self.config.unindp_path / "sim_verify.py"),
            "-S", str(M), str(K),
        ]
        
        if self.config.power_of_2:
            cmd.append("-P")
        if self.config.allow_under_utilize:
            cmd.append("-UU")
        
        # Run UniNDP
        result = subprocess.run(
            cmd,
            cwd=str(self.config.unindp_path),
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise RuntimeError(
                f"UniNDP simulation failed:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
        
        # Parse result from output file
        output_csv = self.config.unindp_path / "verify_result" / "csv" / f"[{M}, {K}].csv"
        
        if output_csv.exists():
            with open(output_csv, 'r') as f:
                reader = csv.reader(f)
                row = next(reader)
                cycles = float(row[2])
        else:
            raise RuntimeError(f"Output file not found: {output_csv}")
        
        return UniNDPResult(
            total_cycles=cycles,
            workload_size=(M, K),
            raw_output=result.stdout,
        )
    
    def simulate_batch(
        self, 
        workloads: List[Tuple[int, int]],
    ) -> List[UniNDPResult]:
        """
        Simulate multiple workloads.
        
        Args:
            workloads: List of (M, K) tuples
            
        Returns:
            List of UniNDPResult for each workload
        """
        results = []
        for M, K in workloads:
            try:
                result = self.simulate_gemm(M, K)
                results.append(result)
            except Exception as e:
                print(f"⚠ Failed for ({M}, {K}): {e}")
                results.append(None)
        return results


class UniNDPInProcess:
    """
    In-process UniNDP simulator (more complex, use with caution).
    
    This imports UniNDP directly and runs simulation in the same process.
    Advantages:
    - Faster (no subprocess overhead)
    - Access to internal state
    
    Disadvantages:
    - May have import conflicts
    - State can persist between runs
    - More fragile
    """
    
    def __init__(self, config: Optional[UniNDPConfig] = None):
        """Initialize in-process UniNDP."""
        self.config = config or UniNDPConfig()
        self._initialized = False
        self._init_unindp()
    
    def _init_unindp(self) -> None:
        """Initialize UniNDP modules."""
        unindp_path = str(self.config.unindp_path)
        if unindp_path not in sys.path:
            sys.path.insert(0, unindp_path)
        
        try:
            from tools import SimConfig, LEVEL
            from sim.sim import sim
            
            self.SimConfig = SimConfig
            self.LEVEL = LEVEL
            self.sim = sim
            
            # Load config
            config_path = self.config.unindp_path / self.config.config_file
            SimConfig.read_from_yaml(str(config_path))
            SimConfig.pu_level = LEVEL.DE
            
            self._initialized = True
            print(f"✓ UniNDP initialized (in-process)")
            
        except ImportError as e:
            print(f"✗ Failed to import UniNDP: {e}")
            self._initialized = False
    
    @property
    def is_available(self) -> bool:
        """Check if UniNDP is available."""
        return self._initialized
    
    def simulate_commands(self, commands: List) -> float:
        """
        Run simulation on pre-generated commands.
        
        Args:
            commands: UniNDP command list from codegen
            
        Returns:
            Total cycles
        """
        if not self._initialized:
            raise RuntimeError("UniNDP not initialized")
        
        return self.sim(commands, silent=True, sim_verify=True, use_tqdm=False)


def run_unindp_simulation(M: int, K: int) -> UniNDPResult:
    """
    Convenience function to run UniNDP simulation.
    
    Args:
        M, K: Workload dimensions for MVM (C[M,1] = A[M,K] @ B[K,1])
        
    Returns:
        UniNDPResult with cycle-accurate results
    """
    simulator = UniNDPSimulator()
    return simulator.simulate_gemm(M, K)


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("UniNDP Golden Model Simulator Test")
    print("=" * 60)
    
    simulator = UniNDPSimulator()
    
    if simulator.is_available:
        print("\n1. Testing GEMM simulation (5000x5000)...")
        try:
            result = simulator.simulate_gemm(5000, 5000)
            print(f"   Result: {result}")
            print(f"   Total cycles: {result.total_cycles:.2f}")
            print(f"   With refresh overhead: {result.cycles_with_refresh:.2f}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print("\n2. Testing batch simulation...")
        workloads = [(1000, 1000), (2000, 2000), (5000, 5000)]
        results = simulator.simulate_batch(workloads)
        for wl, r in zip(workloads, results):
            if r:
                print(f"   {wl}: {r.total_cycles:.2f} cycles")
            else:
                print(f"   {wl}: failed")
    else:
        print("UniNDP not available")
