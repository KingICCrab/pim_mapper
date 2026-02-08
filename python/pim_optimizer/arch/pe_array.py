"""
PE Array definition for PIM architecture.

This module defines the physical parameters of the PE array.
Dataflow and mapping are determined by the ILP optimizer, not predefined here.
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class PhyDim2:
    """
    2D physical dimension (height, width).
    
    Used to represent PE array shape and other 2D structures.
    """
    h: int
    w: int
    
    def size(self) -> int:
        """Total number of elements (h × w)."""
        return self.h * self.w
    
    def __iter__(self):
        return iter((self.h, self.w))
    
    def __mul__(self, other):
        if isinstance(other, PhyDim2):
            return PhyDim2(self.h * other.h, self.w * other.w)
        return PhyDim2(self.h * other, self.w * other)
    
    def __repr__(self) -> str:
        return f"PhyDim2(h={self.h}, w={self.w})"


@dataclass
class ComputeUnit:
    """
    Compute unit configuration within each PE.
    
    Each PE contains a compute unit with:
    - Multiple parallel MAC units
    - A reduction tree to sum the MAC outputs
    
    Example: num_macs=8 means:
    - 8 parallel multipliers computing a[i]*b[i]
    - A 3-stage (log2(8)) adder tree to sum all 8 products
    - Total: 8 MACs per cycle throughput
    
    Attributes:
        unit_type: Type of compute unit ("scalar", "simd", "tensor_core", "reduction_tree", "systolic")
        num_macs: Number of parallel MAC units (must be power of 2)
        mac_energy: Energy per MAC operation (nJ)
        reduction_latency: Cycles for reduction tree (0 = combinational)
        internal_dim: Which dimension is mapped inside PE (e.g., 4=C for tensor core)
    """
    unit_type: str = "scalar"       # "scalar", "simd", "tensor_core", "reduction_tree", "systolic"
    num_macs: int = 1               # Parallel MACs (also reduction tree width)
    mac_energy: float = 0.56e-3     # nJ per MAC (0.56 pJ)
    reduction_latency: int = 0      # Extra cycles for reduction (0 = same cycle)
    internal_dim: Optional[int] = None  # Dimension mapped inside PE (for tensor_core)
    
    def __post_init__(self):
        # Validate num_macs is power of 2
        if self.num_macs > 1 and (self.num_macs & (self.num_macs - 1)) != 0:
            raise ValueError(f"num_macs must be power of 2, got {self.num_macs}")
    
    @property
    def reduction_depth(self) -> int:
        """Number of stages in the reduction tree (log2 of num_macs)."""
        if self.num_macs <= 1:
            return 0
        return int(math.log2(self.num_macs))
    
    @property
    def total_latency(self) -> int:
        """Total compute latency: 1 cycle for MAC + reduction latency."""
        return 1 + self.reduction_latency
    
    @property
    def throughput(self) -> int:
        """MAC operations per cycle (assuming pipelined)."""
        return self.num_macs
    
    def __repr__(self) -> str:
        if self.num_macs == 1:
            return "ComputeUnit(1 MAC)"
        return f"ComputeUnit({self.num_macs} MACs, reduction_depth={self.reduction_depth})"


@dataclass
class PEArray:
    """
    Processing Element Array configuration.
    
    Defines physical hardware parameters that serve as constraints for
    the ILP optimizer. The actual dataflow mapping is determined by
    the optimizer, not predefined here.
    
    Note: PE local storage (register file) size is defined in MemoryHierarchy
    as PELocalBuffer, not here, to avoid duplication.
    
    Attributes:
        dim: Array dimensions (height, width)
        freq_mhz: Operating frequency (MHz)
        compute_unit: Compute unit configuration per PE
    """
    dim: PhyDim2
    freq_mhz: float = 1000.0              # 1 GHz default
    compute_unit: ComputeUnit = None      # Compute unit config
    
    def __post_init__(self):
        if self.compute_unit is None:
            self.compute_unit = ComputeUnit()
    
    @property
    def num_pes(self) -> int:
        """Total number of PEs in the array."""
        return self.dim.size()
    
    @property
    def macs_per_pe(self) -> int:
        """Number of MAC units per PE."""
        return self.compute_unit.num_macs
    
    @property
    def mac_energy(self) -> float:
        """Energy per MAC operation (nJ)."""
        return self.compute_unit.mac_energy
    
    @property
    def peak_mac_per_cycle(self) -> int:
        """Peak MAC operations per cycle."""
        return self.num_pes * self.compute_unit.throughput
    
    @property
    def peak_throughput_gops(self) -> float:
        """Peak throughput in GOPS (Giga Operations Per Second)."""
        return self.peak_mac_per_cycle * self.freq_mhz / 1000.0
    
    def __repr__(self) -> str:
        return (f"PEArray(dim={self.dim}, num_pes={self.num_pes}, "
                f"macs_per_pe={self.macs_per_pe}, "
                f"peak={self.peak_throughput_gops:.1f} GOPS)")


def default_pe_array() -> PEArray:
    """Create a default PE array configuration (16x16 = 256 PEs)."""
    return PEArray(
        dim=PhyDim2(16, 16),
        freq_mhz=1000.0,
        compute_unit=ComputeUnit(num_macs=1, mac_energy=0.56e-3),
    )


# Keep old name for backward compatibility
ReductionTree = ComputeUnit
