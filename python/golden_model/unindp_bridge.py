"""
UniNDP Bridge: Connect ILP Optimizer with UniNDP Simulator.

This module provides a bridge to:
1. Convert ILP mapping configurations to UniNDP strategy format
2. Run UniNDP simulation with the strategy
3. Compare ILP predictions with UniNDP ground truth

UniNDP Strategy Format:
- compute_level: LEVEL.DE or LEVEL.RA
- pu_num: Number of PUs per device
- partition: ((ch_m, ch_k, ch_l, ch_b), (ra_m, ra_k, ra_l, ra_b), 
              (de_m, de_k, de_l, de_b), (pu_m, pu_k, pu_l, pu_b))
- simd_k: K dimension SIMD
- mkl_Input_to_row: Input memory layout
- simd_l: L dimension SIMD  
- ml_Out_to_row: Output memory layout
"""

import os
import sys
import subprocess
import tempfile
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


# UniNDP project path - use local copy in pim_optimizer
# Path: src/golden_model/unindp_bridge.py -> parent(golden_model) -> parent(src) -> parent(pim_optimizer) -> UniNDP
_THIS_FILE = Path(__file__).resolve()
UNINDP_PATH = _THIS_FILE.parent.parent.parent / "UniNDP"


@dataclass
class UniNDPStrategy:
    """
    UniNDP execution strategy.
    
    This represents how a workload is mapped to the HBM-PIM hardware.
    """
    # Compute configuration
    compute_level: str = "DE"  # DE (Device) or RA (Rank)
    pu_num: int = 8
    
    # Partition across hardware hierarchy
    # Format: (m, k, l, b) for each level
    ch_partition: Tuple[int, int, int, int] = (1, 1, 64, 1)  # Channel
    ra_partition: Tuple[int, int, int, int] = (1, 1, 1, 1)   # Rank
    de_partition: Tuple[int, int, int, int] = (1, 1, 1, 1)   # Device
    pu_partition: Tuple[int, int, int, int] = (1, 8, 1, 1)   # PU
    
    # Memory layout
    simd_k: int = 8
    simd_l: int = 4
    input_to_row: Tuple[Tuple[int, int, int, int], ...] = ((1, 8, 4, 1),)
    output_to_row: Tuple[Tuple[int, int, int, int], ...] = ((1, 1, 2, 1),)
    
    def to_unindp_format(self) -> tuple:
        """Convert to UniNDP's internal format."""
        partition = (
            self.ch_partition,
            self.ra_partition,
            self.de_partition,
            self.pu_partition,
        )
        return (
            self.compute_level,
            self.pu_num,
            partition,
            self.simd_k,
            self.input_to_row,
            self.simd_l,
            self.output_to_row,
        )
    
    def __str__(self) -> str:
        return (
            f"UniNDPStrategy(\n"
            f"  level={self.compute_level}, pu_num={self.pu_num}\n"
            f"  ch={self.ch_partition}, ra={self.ra_partition}\n"
            f"  de={self.de_partition}, pu={self.pu_partition}\n"
            f"  simd_k={self.simd_k}, simd_l={self.simd_l}\n"
            f")"
        )


@dataclass
class ILPMapping:
    """
    ILP optimizer mapping configuration.
    
    This represents the output of your ILP optimizer.
    """
    # Workload
    M: int = 1      # Output dimension (or batch for MVM)
    K: int = 1000   # Inner dimension
    L: int = 1000   # Another dimension (N in GEMM)
    B: int = 1      # Batch
    
    # Tiling configuration
    tile_m: int = 1
    tile_k: int = 8
    tile_l: int = 4
    tile_b: int = 1
    
    # Hardware mapping
    # How tiles are distributed across channels, devices, PUs
    ch_parallel_dim: str = "l"  # Which dimension parallelized across channels
    pu_parallel_dim: str = "k"  # Which dimension parallelized across PUs
    
    # Number of parallel units used
    num_channels: int = 64
    num_pus_per_device: int = 8
    
    # ILP predictions (to be verified)
    predicted_cycles: float = 0.0
    predicted_row_activations: int = 0
    
    def __str__(self) -> str:
        return (
            f"ILPMapping(\n"
            f"  workload=({self.M}, {self.K}, {self.L}, {self.B})\n"
            f"  tiling=({self.tile_m}, {self.tile_k}, {self.tile_l}, {self.tile_b})\n"
            f"  predicted_cycles={self.predicted_cycles:.2f}\n"
            f")"
        )


def ilp_to_unindp_strategy(ilp_mapping: ILPMapping) -> UniNDPStrategy:
    """
    Convert ILP mapping to UniNDP strategy.
    
    This is the core conversion function that bridges ILP output
    to UniNDP simulation input.
    
    Args:
        ilp_mapping: ILP optimizer output
        
    Returns:
        UniNDPStrategy that can be executed by UniNDP
    """
    strategy = UniNDPStrategy()
    
    # Set compute level (typically device level for HBM-PIM)
    strategy.compute_level = "DE"
    strategy.pu_num = ilp_mapping.num_pus_per_device
    
    # Convert parallel dimensions to partition format
    # UniNDP partition: (m, k, l, b) for each hardware level
    
    # Channel level parallelism
    ch_m, ch_k, ch_l, ch_b = 1, 1, 1, 1
    if ilp_mapping.ch_parallel_dim == "l":
        ch_l = ilp_mapping.num_channels
    elif ilp_mapping.ch_parallel_dim == "k":
        ch_k = ilp_mapping.num_channels
    elif ilp_mapping.ch_parallel_dim == "m":
        ch_m = ilp_mapping.num_channels
    strategy.ch_partition = (ch_m, ch_k, ch_l, ch_b)
    
    # Rank level (HBM typically has 1 rank)
    strategy.ra_partition = (1, 1, 1, 1)
    
    # Device level (1 device per channel in HBM-PIM)
    strategy.de_partition = (1, 1, 1, 1)
    
    # PU level parallelism
    pu_m, pu_k, pu_l, pu_b = 1, 1, 1, 1
    if ilp_mapping.pu_parallel_dim == "k":
        pu_k = ilp_mapping.num_pus_per_device
    elif ilp_mapping.pu_parallel_dim == "l":
        pu_l = ilp_mapping.num_pus_per_device
    strategy.pu_partition = (pu_m, pu_k, pu_l, pu_b)
    
    # Memory layout from tiling
    strategy.simd_k = ilp_mapping.tile_k
    strategy.simd_l = ilp_mapping.tile_l
    
    # Input layout: (m, k, l, b) elements per row
    strategy.input_to_row = ((1, ilp_mapping.tile_k, ilp_mapping.tile_l, 1),)
    
    # Output layout
    strategy.output_to_row = ((1, 1, 2, 1),)  # Typical default
    
    return strategy


@dataclass
class UniNDPResult:
    """Result from UniNDP simulation."""
    cycles: float = 0.0
    success: bool = False
    strategy_used: Optional[str] = None
    error_message: Optional[str] = None


class UniNDPBridge:
    """
    Bridge between ILP optimizer and UniNDP simulator.
    
    Provides methods to:
    1. Convert ILP mappings to UniNDP strategies
    2. Run UniNDP simulation with custom strategies
    3. Compare results
    """
    
    def __init__(self, unindp_path: Optional[Path] = None):
        """
        Initialize the bridge.
        
        Args:
            unindp_path: Path to UniNDP project (auto-detected if not provided)
        """
        self.unindp_path = unindp_path or UNINDP_PATH
        
        if not self.unindp_path.exists():
            raise FileNotFoundError(
                f"UniNDP not found at {self.unindp_path}. "
                "Please set the correct path."
            )
    
    def run_simulation(
        self,
        M: int,
        K: int,
        strategy: Optional[UniNDPStrategy] = None,
        timeout: int = 120,
    ) -> UniNDPResult:
        """
        Run UniNDP simulation for a workload.
        
        If strategy is provided, uses sim_with_strategy.py to run with that
        exact strategy. Otherwise, uses sim_verify.py which finds its own strategy.
        
        Args:
            M: Output dimension (K in UniNDP's [M,K] format for MVM)
            K: Inner dimension 
            strategy: Optional custom strategy (uses sim_with_strategy.py if provided)
            timeout: Timeout in seconds
            
        Returns:
            UniNDPResult with cycles and status
        """
        result = UniNDPResult()
        
        if strategy is not None:
            # Use new sim_with_strategy.py with custom strategy
            return self._run_with_custom_strategy(M, K, strategy, timeout)
        else:
            # Use original sim_verify.py (finds baseline strategy internally)
            return self._run_default_simulation(M, K, timeout)
    
    def _run_with_custom_strategy(
        self,
        M: int,
        K: int,
        strategy: UniNDPStrategy,
        timeout: int,
    ) -> UniNDPResult:
        """
        Run UniNDP simulation with a custom strategy.
        
        This uses sim_with_strategy.py which accepts external strategy parameters.
        """
        result = UniNDPResult()
        
        # Build strategy JSON
        strategy_json = json.dumps({
            'ch_partition': list(strategy.ch_partition),
            'ra_partition': list(strategy.ra_partition),
            'de_partition': list(strategy.de_partition),
            'pu_partition': list(strategy.pu_partition),
            'simd_k': strategy.simd_k,
            'simd_l': strategy.simd_l,
            'input_to_row_k': strategy.simd_k,  # Typically same as simd_k
            'output_to_row': 1,
        })
        
        # Build command
        cmd = [
            sys.executable,
            str(self.unindp_path / "sim_with_strategy.py"),
            "-S", str(M), str(K),
            "--strategy_json", strategy_json,
            "--output_json",
            "--silent",
        ]
        
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.unindp_path),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            if proc.returncode == 0:
                # Parse JSON output
                try:
                    output = json.loads(proc.stdout.strip())
                    result.cycles = output['cycles']
                    result.success = True
                    result.strategy_used = str(strategy)
                except json.JSONDecodeError:
                    result.error_message = f"Failed to parse output: {proc.stdout}"
            else:
                result.error_message = proc.stderr or proc.stdout or "Unknown error"
                
        except subprocess.TimeoutExpired:
            result.error_message = f"Timeout after {timeout}s"
        except Exception as e:
            result.error_message = str(e)
        
        return result
    
    def _run_default_simulation(
        self,
        M: int,
        K: int,
        timeout: int,
    ) -> UniNDPResult:
        """
        Run UniNDP simulation with default strategy search.
        
        This uses the original sim_verify.py which finds a baseline strategy.
        """
        result = UniNDPResult()
        
        # Build command
        cmd = [
            sys.executable,
            str(self.unindp_path / "sim_verify.py"),
            "-S", str(M), str(K),
        ]
        
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.unindp_path),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            if proc.returncode == 0:
                # Parse result from log file
                log_file = self.unindp_path / f"verify_result/log/[{M}, {K}].log"
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        content = f.read()
                        
                    # Parse cycles from "result: xxx" line
                    for line in content.split('\n'):
                        if line.startswith('result:'):
                            cycles_str = line.split(':')[1].strip()
                            result.cycles = float(cycles_str)
                            result.success = True
                        elif line.startswith('strategy:'):
                            result.strategy_used = line.split(':', 1)[1].strip()
                else:
                    result.error_message = "Log file not found"
            else:
                result.error_message = proc.stderr or "Unknown error"
                
        except subprocess.TimeoutExpired:
            result.error_message = f"Timeout after {timeout}s"
        except Exception as e:
            result.error_message = str(e)
        
        return result
    
    def verify_ilp_mapping(
        self,
        ilp_mapping: ILPMapping,
        tolerance: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Verify an ILP mapping against UniNDP simulation.
        
        Args:
            ilp_mapping: The ILP mapping to verify
            tolerance: Acceptable relative error
            
        Returns:
            Dictionary with verification results
        """
        # Convert to UniNDP strategy
        strategy = ilp_to_unindp_strategy(ilp_mapping)
        
        # Run simulation
        # Note: UniNDP uses [M, K] format where:
        # - M is typically 1 for MVM
        # - K and L are the matrix dimensions
        sim_result = self.run_simulation(
            M=ilp_mapping.K,  # Inner dim
            K=ilp_mapping.L,  # Output dim
            strategy=strategy,
        )
        
        verification = {
            'ilp_mapping': str(ilp_mapping),
            'unindp_strategy': str(strategy),
            'simulation_success': sim_result.success,
        }
        
        if sim_result.success:
            # Compare predictions with simulation
            predicted = ilp_mapping.predicted_cycles
            actual = sim_result.cycles
            
            if predicted > 0:
                error = abs(predicted - actual) / actual
                is_accurate = error <= tolerance
            else:
                error = float('inf')
                is_accurate = False
            
            verification.update({
                'predicted_cycles': predicted,
                'actual_cycles': actual,
                'relative_error': error,
                'is_accurate': is_accurate,
                'strategy_used': sim_result.strategy_used,
            })
        else:
            verification['error'] = sim_result.error_message
        
        return verification
    
    def explore_design_space(
        self,
        M: int,
        K: int,
        num_samples: int = 10,
    ) -> List[Tuple[UniNDPStrategy, float]]:
        """
        Explore the design space by running multiple strategies.
        
        This can be used to find the true optimal mapping.
        
        Args:
            M: Workload M dimension
            K: Workload K dimension
            num_samples: Maximum number of strategies to try
            
        Returns:
            List of (strategy, cycles) tuples sorted by cycles
        """
        results = []
        
        # Generate different strategies
        strategies = self._generate_strategies(M, K, num_samples)
        
        for strategy in strategies:
            result = self.run_simulation(M, K, strategy)
            if result.success:
                results.append((strategy, result.cycles))
        
        # Sort by cycles
        results.sort(key=lambda x: x[1])
        
        return results
    
    def _generate_strategies(
        self,
        M: int,
        K: int,
        num_samples: int,
    ) -> List[UniNDPStrategy]:
        """Generate diverse strategies to explore."""
        strategies = []
        
        # Default strategy (channel parallel on L)
        default = UniNDPStrategy()
        strategies.append(default)
        
        # Variations
        channel_parallels = ["l", "k"]
        pu_parallels = ["k", "l"]
        simd_ks = [4, 8, 16]
        
        for ch_p in channel_parallels:
            for pu_p in pu_parallels:
                for simd_k in simd_ks:
                    if len(strategies) >= num_samples:
                        break
                    
                    s = UniNDPStrategy()
                    
                    # Set channel partition
                    if ch_p == "l":
                        s.ch_partition = (1, 1, 64, 1)
                    else:
                        s.ch_partition = (1, 64, 1, 1)
                    
                    # Set PU partition
                    if pu_p == "k":
                        s.pu_partition = (1, 8, 1, 1)
                    else:
                        s.pu_partition = (1, 1, 8, 1)
                    
                    s.simd_k = simd_k
                    strategies.append(s)
        
        return strategies[:num_samples]


def verify_ilp_with_unindp(
    ilp_mapping: Dict[str, Any],
    workload: Dict[str, Any],
    ilp_predictions: Dict[str, float],
) -> Dict[str, Any]:
    """
    High-level function to verify ILP solution with UniNDP.
    
    Args:
        ilp_mapping: ILP mapping configuration
        workload: Workload specification
        ilp_predictions: ILP's predicted costs
        
    Returns:
        Verification results dictionary
    """
    bridge = UniNDPBridge()
    
    # Convert to ILPMapping object
    mapping = ILPMapping(
        M=workload.get('M', 1),
        K=workload.get('K', 1000),
        L=workload.get('L', workload.get('N', 1000)),
        B=workload.get('B', 1),
        tile_m=ilp_mapping.get('tile_M', 1),
        tile_k=ilp_mapping.get('tile_K', 8),
        tile_l=ilp_mapping.get('tile_L', ilp_mapping.get('tile_N', 4)),
        tile_b=ilp_mapping.get('tile_B', 1),
        num_channels=ilp_mapping.get('num_channels', 64),
        num_pus_per_device=ilp_mapping.get('num_pus', 8),
        predicted_cycles=ilp_predictions.get('total_cycles', 0),
        predicted_row_activations=ilp_predictions.get('row_activations', 0),
    )
    
    return bridge.verify_ilp_mapping(mapping)


# Example usage
if __name__ == "__main__":
    print("UniNDP Bridge Test")
    print("=" * 60)
    
    # Create bridge
    bridge = UniNDPBridge()
    
    # Test basic simulation
    print("\n1. Testing basic simulation...")
    result = bridge.run_simulation(M=2000, K=2000, timeout=60)
    
    if result.success:
        print(f"   ✓ Simulation successful")
        print(f"   Cycles: {result.cycles:.2f}")
        print(f"   Strategy: {result.strategy_used[:80]}...")
    else:
        print(f"   ✗ Simulation failed: {result.error_message}")
    
    # Test ILP verification
    print("\n2. Testing ILP verification...")
    ilp_mapping = ILPMapping(
        K=2000,
        L=2000,
        tile_k=8,
        tile_l=4,
        predicted_cycles=3000.0,  # Our prediction
    )
    
    verification = bridge.verify_ilp_mapping(ilp_mapping)
    
    print(f"   Predicted: {verification.get('predicted_cycles', 'N/A')}")
    print(f"   Actual: {verification.get('actual_cycles', 'N/A')}")
    print(f"   Error: {verification.get('relative_error', 'N/A'):.2%}")
    print(f"   Accurate: {verification.get('is_accurate', 'N/A')}")
