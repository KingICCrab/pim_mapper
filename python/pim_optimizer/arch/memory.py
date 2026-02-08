"""
Memory hierarchy definitions for PIM architecture.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryLevel:
    """
    Definition of a single memory level in the hierarchy.
    
    Attributes:
        name: Name of this memory level (e.g., "PELocalBuffer", "GlobalBuffer")
        entries: Number of entries (capacity). -1 means unlimited.
        blocksize: Block size in bytes
        instances: Number of instances at this level
        latency: Read latency in cycles
        access_cost: Energy cost per access (nJ)
        stores: List of 3 booleans [inputs, weights, outputs] indicating which datatypes are stored
        bypass_defined: Whether bypass is explicitly defined
        num_banks: Number of banks (for DRAM-like levels)
        row_buffer_size: Row buffer size in bytes (for DRAM-like levels)
        read_bandwidth_limit: Maximum read bandwidth in bytes/cycle (None = unlimited)
        write_bandwidth_limit: Maximum write bandwidth in bytes/cycle (None = unlimited)
        num_read_ports: Number of parallel read ports
        num_write_ports: Number of parallel write ports
    """
    name: str
    entries: int = 256
    blocksize: int = 1
    instances: int = 1
    latency: float = 1.0
    access_cost: float = 0.001
    stores: list[bool] = field(default_factory=lambda: [True, True, True])
    bypass_defined: bool = True
    num_banks: Optional[int] = None
    row_buffer_size: Optional[int] = None
    # Bandwidth constraints
    read_bandwidth_limit: Optional[float] = None   # bytes/cycle, None = blocksize/latency
    write_bandwidth_limit: Optional[float] = None  # bytes/cycle
    num_read_ports: int = 1
    num_write_ports: int = 1
    
    @property
    def stores_inputs(self) -> bool:
        return self.stores[0]
    
    @property
    def stores_weights(self) -> bool:
        return self.stores[1]
    
    @property
    def stores_outputs(self) -> bool:
        return self.stores[2]
    
    @property
    def stored_datatypes(self) -> list[int]:
        """Return indices of stored datatypes."""
        return [i for i, s in enumerate(self.stores) if s]
    
    @property
    def stores_multiple_datatypes(self) -> bool:
        return sum(self.stores) > 1
    
    @property
    def read_bandwidth(self) -> float:
        """Compute read bandwidth in bytes/cycle."""
        if self.read_bandwidth_limit is not None:
            return self.read_bandwidth_limit
        if self.latency > 0:
            return self.blocksize / self.latency
        return float('inf')
    
    @property
    def write_bandwidth(self) -> float:
        """Compute write bandwidth in bytes/cycle."""
        if self.write_bandwidth_limit is not None:
            return self.write_bandwidth_limit
        if self.latency > 0:
            return self.blocksize / self.latency
        return float('inf')
    
    @property
    def total_read_bandwidth(self) -> float:
        """Total read bandwidth considering multiple ports."""
        return self.read_bandwidth * self.num_read_ports
    
    @property
    def total_write_bandwidth(self) -> float:
        """Total write bandwidth considering multiple ports."""
        return self.write_bandwidth * self.num_write_ports


class MemoryHierarchy:
    """
    Definition of a complete memory hierarchy.
    
    Memory levels are ordered from innermost (PE-local) to outermost (DRAM).
    """
    
    def __init__(self, levels: Optional[list[MemoryLevel]] = None):
        """
        Initialize memory hierarchy.
        
        Args:
            levels: List of MemoryLevel objects, ordered from innermost to outermost.
                   If None, creates a default PIM hierarchy.
        """
        if levels is None:
            levels = self._default_pim_hierarchy()
        
        self.levels = levels
        self._build_indices()
    
    def _default_pim_hierarchy(self) -> list[MemoryLevel]:
        """Create default PIM memory hierarchy."""
        return [
            MemoryLevel(
                name="PELocalBuffer",
                entries=64,
                blocksize=1,
                instances=64,
                latency=1,
                access_cost=0.001,
                stores=[False, False, True],
                bypass_defined=True,
            ),
            MemoryLevel(
                name="GlobalBuffer",
                entries=256,
                blocksize=32,
                instances=1,
                latency=1,
                access_cost=0.01,
                stores=[True, True, True],
                bypass_defined=True,
            ),
            MemoryLevel(
                name="RowBuffer",
                entries=1024,
                blocksize=16,
                instances=1,
                latency=1,
                access_cost=0.01,
                stores=[True, True, True],
                bypass_defined=True,
                num_banks=4,
                row_buffer_size=1024,
            ),
            MemoryLevel(
                name="LocalDRAM",
                entries=-1,  # Unlimited
                blocksize=1024,
                instances=1,
                latency=25,
                access_cost=0.1,
                stores=[True, True, True],
                bypass_defined=True,
                num_banks=4,
                row_buffer_size=1024,
            ),
        ]
    
    def _build_indices(self):
        """Build name-to-index mapping."""
        self.name_to_idx = {level.name: idx for idx, level in enumerate(self.levels)}
        self.idx_to_name = {idx: level.name for idx, level in enumerate(self.levels)}
    
    @property
    def num_levels(self) -> int:
        return len(self.levels)
    
    def get_level(self, name_or_idx) -> Optional[MemoryLevel]:
        """Get memory level by name or index."""
        if isinstance(name_or_idx, str):
            idx = self.name_to_idx.get(name_or_idx)
            if idx is None:
                return None
            return self.levels[idx]
        elif isinstance(name_or_idx, int):
            if 0 <= name_or_idx < len(self.levels):
                return self.levels[name_or_idx]
        return None
    
    def get_idx(self, name: str) -> Optional[int]:
        """Get level index by name."""
        return self.name_to_idx.get(name)
    
    @property
    def rowbuffer_level(self) -> Optional[int]:
        """Get RowBuffer level index."""
        return self.name_to_idx.get("RowBuffer")
    
    @property
    def dram_level(self) -> Optional[int]:
        """Get LocalDRAM level index."""
        return self.name_to_idx.get("LocalDRAM")
    
    def compute_fanouts(self) -> list[float]:
        """
        Compute fanout for each memory level.
        
        Fanout[i] = inner_instances / instances[i]
        """
        fanouts = []
        inner_instances = 1024  # Total PE count (assumption)
        
        for level in self.levels:
            if level.instances > 0:
                ratio = inner_instances / level.instances
                if ratio < 1:
                    ratio = 1
                fanouts.append(ratio)
                inner_instances = max(level.instances, 1)
            else:
                fanouts.append(inner_instances)
        
        return fanouts
    
    def __len__(self) -> int:
        return len(self.levels)
    
    def __getitem__(self, idx: int) -> MemoryLevel:
        return self.levels[idx]
    
    def __iter__(self):
        return iter(self.levels)
