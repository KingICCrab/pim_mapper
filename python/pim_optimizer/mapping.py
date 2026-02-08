"""
Mapping result class for storing optimization results.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Mapping:
    """
    Represents a mapping solution for a workload.
    
    Attributes:
        loop_bounds: Nested dictionary of loop bounds
            [memory_level][spatial/temporal][dimension] = bound
        permutation: Nested dictionary of loop permutation
            [memory_level][perm_level] = dimension
        bypass: Memory bypass configuration
            [memory_level][datatype] = True/False
        layout: Data layout mode
            [datatype] = "sequential" or "row_aligned"
            
        metrics: Dictionary of performance metrics
            latency, energy, row_activations, etc.
            
        tile_info: Input tile information for debugging
            tile_h, tile_w, outer_q_factor, etc.
    """
    # Core mapping parameters
    loop_bounds: dict = field(default_factory=dict)
    permutation: dict = field(default_factory=dict)
    bypass: dict = field(default_factory=dict)
    layout: dict = field(default_factory=dict)
    
    # Performance metrics
    metrics: dict = field(default_factory=dict)
    
    # Debug information
    tile_info: dict = field(default_factory=dict)
    solver_info: dict = field(default_factory=dict)
    
    # Workload reference
    workload_name: str = ""
    workload_bounds: list = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize default values."""
        if not self.layout:
            self.layout = {
                0: "sequential",  # input
                1: "sequential",  # weight
                2: "sequential",  # output
            }
    
    @property
    def latency(self) -> float:
        """Get total latency."""
        return self.metrics.get("latency", 0.0)
    
    @property
    def energy(self) -> float:
        """Get total energy."""
        return self.metrics.get("energy", 0.0)
    
    @property
    def row_activations(self) -> float:
        """Get total row activations."""
        return self.metrics.get("row_activations", 0.0)
    
    def get_tile_size(self, memory_level: int, dimension: int) -> int:
        """
        Get the tile size for a dimension at a memory level.
        
        This is the product of spatial and temporal bounds from
        memory level 0 up to and including the specified level.
        """
        tile = 1
        for m in range(memory_level + 1):
            for s in ("spatial", "temporal"):
                if m in self.loop_bounds and s in self.loop_bounds[m]:
                    tile *= self.loop_bounds[m][s].get(dimension, 1)
        return tile
    
    def get_loop_order(self, memory_level: int) -> list:
        """Get the loop order (inner to outer) at a memory level."""
        if memory_level not in self.permutation:
            return []
        
        perm = self.permutation[memory_level]
        return [perm[p] for p in sorted(perm.keys())]
    
    def is_bypassed(self, memory_level: int, datatype: int) -> bool:
        """Check if a datatype is bypassed at a memory level."""
        if memory_level not in self.bypass:
            return False
        return not self.bypass[memory_level].get(datatype, True)
    
    def to_dict(self) -> dict:
        """Convert mapping to dictionary format."""
        return {
            "loop_bounds": self.loop_bounds,
            "permutation": self.permutation,
            "bypass": self.bypass,
            "layout": self.layout,
            "metrics": self.metrics,
            "tile_info": self.tile_info,
            "workload_name": self.workload_name,
            "workload_bounds": self.workload_bounds,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Mapping":
        """Create mapping from dictionary."""
        return cls(
            loop_bounds=data.get("loop_bounds", {}),
            permutation=data.get("permutation", {}),
            bypass=data.get("bypass", {}),
            layout=data.get("layout", {}),
            metrics=data.get("metrics", {}),
            tile_info=data.get("tile_info", {}),
            workload_name=data.get("workload_name", ""),
            workload_bounds=data.get("workload_bounds", []),
        )
    
    def pretty_print(self) -> str:
        """Generate human-readable mapping representation."""
        lines = []
        lines.append(f"Mapping: {self.workload_name}")
        lines.append("=" * 50)
        
        # Loop bounds
        lines.append("\nLoop Bounds:")
        for m in sorted(self.loop_bounds.keys()):
            lines.append(f"  Memory Level {m}:")
            for s in ("spatial", "temporal"):
                if s in self.loop_bounds[m]:
                    bounds = self.loop_bounds[m][s]
                    bounds_str = ", ".join(f"{k}={v}" for k, v in bounds.items())
                    lines.append(f"    {s}: {bounds_str}")
        
        # Permutation
        lines.append("\nPermutation:")
        for m in sorted(self.permutation.keys()):
            order = self.get_loop_order(m)
            lines.append(f"  Level {m}: {order}")
        
        # Bypass
        lines.append("\nBypass:")
        for m in sorted(self.bypass.keys()):
            bp = self.bypass[m]
            stored = [t for t, stored in bp.items() if stored]
            bypassed = [t for t, stored in bp.items() if not stored]
            lines.append(f"  Level {m}: stored={stored}, bypassed={bypassed}")
        
        # Layout
        lines.append("\nLayout:")
        datatype_names = ["Input", "Weight", "Output"]
        for t, mode in self.layout.items():
            lines.append(f"  {datatype_names[t]}: {mode}")
        
        # Metrics
        lines.append("\nMetrics:")
        for name, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"  {name}: {value:.6e}")
            else:
                lines.append(f"  {name}: {value}")
        
        return "\n".join(lines)
    
    def to_yaml_format(self) -> dict:
        """Convert to YAML-compatible format for Timeloop/MAESTRO."""
        # Convert to standard mapping format
        yaml_mapping = {
            "mapping": []
        }
        
        dim_names = ["R", "S", "P", "Q", "C", "K", "N"]
        
        for m in sorted(self.loop_bounds.keys()):
            level_mapping = {
                "target": f"level_{m}",
                "type": "temporal" if m == 0 else "spatial",
                "factors": [],
                "permutation": [],
            }
            
            for s in ("temporal", "spatial"):
                if s in self.loop_bounds[m]:
                    for dim_idx, bound in self.loop_bounds[m][s].items():
                        if bound > 1:
                            dim_name = dim_names[dim_idx] if dim_idx < len(dim_names) else f"D{dim_idx}"
                            level_mapping["factors"].append(f"{dim_name}={bound}")
            
            # Add permutation
            order = self.get_loop_order(m)
            level_mapping["permutation"] = [
                dim_names[d] if d < len(dim_names) else f"D{d}"
                for d in order
            ]
            
            yaml_mapping["mapping"].append(level_mapping)
        
        return yaml_mapping


@dataclass
class OptimizationResult:
    """
    Complete optimization result containing all mappings and summary.
    
    Attributes:
        mappings: List of Mapping objects for each workload
        summary: Summary statistics
        solver_status: Gurobi solver status
        solve_time: Total solve time in seconds
    """
    mappings: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    solver_status: str = "unknown"
    solve_time: float = 0.0
    
    @property
    def total_latency(self) -> float:
        """Get total latency across all workloads."""
        return sum(m.latency for m in self.mappings)
    
    @property
    def total_energy(self) -> float:
        """Get total energy across all workloads."""
        return sum(m.energy for m in self.mappings)
    
    @property
    def is_optimal(self) -> bool:
        """Check if solution is optimal."""
        return self.solver_status.lower() == "optimal"
    
    def print_summary(self):
        """Print optimization summary."""
        print("\n" + "=" * 60)
        print("OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"Solver Status: {self.solver_status}")
        print(f"Solve Time: {self.solve_time:.2f}s")
        print(f"Number of Workloads: {len(self.mappings)}")
        print(f"Total Latency: {self.total_latency:.6e}")
        print(f"Total Energy: {self.total_energy:.6e}")
        print("=" * 60)
