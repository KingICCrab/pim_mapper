"""
PIM Architecture definition.
"""

from pathlib import Path
from typing import Optional

import yaml

from pim_optimizer.arch.memory import MemoryLevel, MemoryHierarchy
from pim_optimizer.arch.pe_array import PEArray, PhyDim2, default_pe_array


class PIMArchitecture:
    """
    PIM (Processing-In-Memory) Accelerator Architecture definition.
    
    Defines the memory hierarchy ordered from innermost (PE-local) to outermost (DRAM):
    - Level 0: PE local buffer
    - Level 1: Shared global buffer
    - Level 2: Shared row buffer
    - Level 3: Local DRAM
    """
    
    TIMING_KEYS = {
        "RL",           # Read Latency
        "WL",           # Write Latency
        "tCCDL",        # Column-to-Column delay (Long)
        "tRTRS",        # Read-to-Read Same bank
        "tRTPL",        # Read to Precharge (Long)
        "tWTRL",        # Write to Read (Long)
        "tWR",          # Write Recovery
        "tRCDRD",       # Row-to-Column delay (Read)
        "tRCDWR",       # Row-to-Column delay (Write)
        "tRP",          # Row Precharge
        "BL",           # Burst Length
        "co_w",         # Data beat width (bits per beat)
        "data_pr",      # Element precision (bits)
    }
    
    def __init__(
        self,
        vault_count: Optional[int] = None,
        pu_count: Optional[int] = None,
        timing_config: Optional[dict] = None,
        config_path: Optional[str | Path] = None,
        hierarchy: Optional[MemoryHierarchy] = None,
        pe_array: Optional[PEArray] = None,
    ):
        """
        Initialize PIM architecture.
        
        Args:
            vault_count: Number of vaults (HMC/HBM style)
            pu_count: Number of processing units
            timing_config: DRAM timing parameters dict
            config_path: Path to YAML config file
            hierarchy: Custom memory hierarchy. If None, uses default.
            pe_array: PE array configuration. If None, uses default.
        """
        self.vault_count = vault_count if vault_count is not None else 1
        self.pu_count = pu_count
        
        # Load DRAM timing info
        self.dram_timings: dict[str, float] = {}
        if timing_config:
            self._apply_timing_dict(timing_config)
        if config_path:
            self._load_timings_from_yaml(config_path)
        
        # Set up PE array
        self.pe_array = pe_array if pe_array is not None else default_pe_array()
        
        # Set up memory hierarchy
        if hierarchy is None:
            # Update LocalDRAM latency based on timing config
            localdram_latency = self.dram_timings.get("RL", 25.0)
            hierarchy = MemoryHierarchy()
            # Update DRAM level latency
            dram_level = hierarchy.get_level("LocalDRAM")
            if dram_level is not None:
                dram_level.latency = localdram_latency
        
        self.hierarchy = hierarchy
        self._build_flat_attributes()
        
        # DRAM activation latency
        self.dram_activation_latency = self._infer_activation_latency()
        
        # Column datapath width and element precision
        self.data_beat_bits: int = int(self.dram_timings.get("co_w", 256))
        self.default_element_bits: int = int(self.dram_timings.get("data_pr", 8))
        
        # MAC energy cost (nJ) - use PE array's value
        self.mac_energy = self.pe_array.mac_energy
        
        # 1 entry = 1 byte
        self.entry_bytes: int = 1
        
        # Placeholder for mapping compatibility
        self.mapspace_dict = {'mapspace': {'constraints': []}}
        self.compute_unit = None
    
    def _build_flat_attributes(self):
        """Build flat attribute lists for backward compatibility."""
        h = self.hierarchy
        
        self.num_mems = h.num_levels
        self.mem_name = h.idx_to_name
        self.mem_idx = h.name_to_idx
        
        self.mem_entries = [level.entries for level in h]
        self.mem_blocksizes = [level.blocksize for level in h]
        self.mem_instances = [level.instances for level in h]
        self.read_latency = [level.latency for level in h]
        self.mem_access_cost = [level.access_cost for level in h]
        self.read_bandwidth = [level.read_bandwidth for level in h]
        
        self.fanouts = h.compute_fanouts()
        
        # DRAM-specific
        dram_level = h.get_level("LocalDRAM")
        self.dram_num_banks = dram_level.num_banks if dram_level else 4
        self.dram_bank_row_buffer_size = dram_level.row_buffer_size if dram_level else 1024
        
        # Per-level bank info
        self.mem_num_banks = [level.num_banks for level in h]
        self.mem_row_buffer_size = [level.row_buffer_size for level in h]
        
        # Bypass configuration
        self.mem_stores_datatype = [level.stores for level in h]
        self.mem_bypass_defined = [level.bypass_defined for level in h]
        self.mem_stored_datatypes = [level.stored_datatypes for level in h]
        self.mem_stores_multiple_datatypes = [level.stores_multiple_datatypes for level in h]
    
    def _apply_timing_dict(self, timing_dict: dict) -> None:
        """Apply DRAM timing parameters from dict."""
        for key, value in timing_dict.items():
            if key not in self.TIMING_KEYS:
                continue
            try:
                self.dram_timings[key] = float(value)
            except (TypeError, ValueError):
                continue
    
    def _load_timings_from_yaml(self, config_path: str | Path) -> None:
        """Load DRAM timing parameters from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"DRAM config not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if isinstance(data, dict):
            self._apply_timing_dict(data)
    
    def _infer_activation_latency(self) -> float:
        """Infer row activation latency from timing parameters."""
        t_rcd = self.dram_timings.get("tRCDRD")
        t_rp = self.dram_timings.get("tRP")
        if t_rcd is not None and t_rp is not None:
            return float(t_rcd + t_rp)
        return float(self.dram_timings.get("activation_latency", 25.0))
    
    def get_bank_access_latency(self, same_bank: bool = True) -> float:
        """
        Get access latency for DRAM bank access.
        
        Args:
            same_bank: If True, return latency for same-bank access (sequential)
                      If False, return latency for different-bank access (parallel)
        """
        dram_idx = self.mem_idx.get("LocalDRAM")
        if dram_idx is None:
            return self.read_latency[-1]
        if same_bank:
            return self.read_latency[dram_idx]
        return max(self.read_latency[dram_idx] // max(self.dram_num_banks, 1), 1)
    
    def get_rowbuffer_hit_latency(self) -> float:
        """Get latency for row buffer hit (already activated row)."""
        rowbuf_idx = self.mem_idx.get("RowBuffer")
        if rowbuf_idx is None:
            return self.read_latency[0]
        return self.read_latency[rowbuf_idx]
    
    def print_info(self):
        """Print architecture information for quick inspection."""
        print("\n=== PIM Architecture Configuration ===")
        
        # PE Array info
        print(f"\n[PE Array]")
        print(f"  Dimensions: {self.pe_array.dim.h} x {self.pe_array.dim.w}")
        print(f"  Total PEs: {self.pe_array.num_pes}")
        
        # Compute Unit info
        cu = self.pe_array.compute_unit
        print(f"\n[Compute Unit (per PE)]")
        print(f"  Parallel MACs: {cu.num_macs}")
        if cu.num_macs > 1:
            print(f"  Reduction depth: {cu.reduction_depth} stages")
            print(f"  Reduction latency: {cu.reduction_latency} cycles")
        print(f"  MAC energy: {cu.mac_energy} nJ")
        print(f"  Peak Throughput: {self.pe_array.peak_throughput_gops:.1f} GOPS")
        
        print(f"\nMemory Levels: {self.num_mems}")
        
        for idx, level in enumerate(self.hierarchy):
            print(f"\n[L{idx}] {level.name}")
            print(f"  Entries: {level.entries if level.entries >= 0 else 'unlimited'}")
            print(f"  Block Size: {level.blocksize} bytes")
            print(f"  Instances: {level.instances}")
            print(f"  Fanout: {self.fanouts[idx]}")
            print(f"  Read BW: {level.read_bandwidth} bytes/cycle")
            if level.read_bandwidth_limit is not None:
                print(f"  Read BW Limit: {level.read_bandwidth_limit} bytes/cycle")
            if level.write_bandwidth_limit is not None:
                print(f"  Write BW Limit: {level.write_bandwidth_limit} bytes/cycle")
            if level.num_read_ports > 1:
                print(f"  Read Ports: {level.num_read_ports} (total: {level.total_read_bandwidth} bytes/cycle)")
            if level.num_write_ports > 1:
                print(f"  Write Ports: {level.num_write_ports} (total: {level.total_write_bandwidth} bytes/cycle)")
            print(f"  Read Latency: {level.latency} cycles")
            print(f"  Access Energy: {level.access_cost} nJ")
            print(f"  Stored Datatypes: {level.stored_datatypes}")
            print(f"  Bypass fixed: {level.bypass_defined}")
            print(f"  Stores multiple datatypes: {level.stores_multiple_datatypes}")
        
        print(f"\nMAC Energy: {self.mac_energy} nJ")
        print(f"Entry Size: {self.entry_bytes} byte")
        print(f"Data Beat Width: {self.data_beat_bits} bits")
        print(f"Default Element Bits: {self.default_element_bits}")
        print(f"DRAM Activation Latency: {self.dram_activation_latency} cycles")
        print("=== End of Architecture Info ===\n")
    
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "PIMArchitecture":
        """
        Create PIMArchitecture from a YAML configuration file.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            PIMArchitecture instance
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        
        # Handle nested 'architecture' key
        if "architecture" in config:
            config = config["architecture"]
        
        # Extract timing config
        timing_config = config.get("dram_timings", {})
        
        # Extract PE array config
        pe_array = None
        if "pe_array" in config:
            pe_cfg = config["pe_array"]
            
            # Parse compute unit config
            from pim_optimizer.arch.pe_array import ComputeUnit
            compute_unit = ComputeUnit(
                unit_type=pe_cfg.get("unit_type", "scalar"),
                num_macs=pe_cfg.get("num_macs", 1),
                mac_energy=pe_cfg.get("mac_energy", 0.56e-3),
                reduction_latency=pe_cfg.get("reduction_latency", 0),
                internal_dim=pe_cfg.get("internal_dim", None),
            )
            
            pe_array = PEArray(
                dim=PhyDim2(
                    h=pe_cfg.get("dim_h", 16),
                    w=pe_cfg.get("dim_w", 16),
                ),
                freq_mhz=pe_cfg.get("freq_mhz", 1000.0),
                compute_unit=compute_unit,
            )
        elif "compute" in config:
            # Legacy support: extract from 'compute' section
            compute_cfg = config["compute"]
            pe_count = compute_cfg.get("pe_count", 256)
            # Assume square array
            import math
            side = int(math.sqrt(pe_count))
            pe_array = PEArray(
                dim=PhyDim2(h=side, w=pe_count // side),
                freq_mhz=compute_cfg.get("pe_frequency", 1000.0),
            )
        
        # Extract memory hierarchy
        hierarchy = None
        if "memory_hierarchy" in config:
            levels = []
            for level_config in config["memory_hierarchy"]:
                levels.append(MemoryLevel(
                    name=level_config.get("name", "Unknown"),
                    entries=level_config.get("entries", 256),
                    blocksize=level_config.get("blocksize", 1),
                    instances=level_config.get("instances", 1),
                    latency=level_config.get("latency", 1),
                    access_cost=level_config.get("access_cost", 0.001),
                    stores=level_config.get("stores", [True, True, True]),
                    bypass_defined=level_config.get("bypass_defined", True),
                    num_banks=level_config.get("num_banks"),
                    row_buffer_size=level_config.get("row_buffer_size"),
                    # Bandwidth constraints
                    read_bandwidth_limit=level_config.get("read_bandwidth"),
                    write_bandwidth_limit=level_config.get("write_bandwidth"),
                    num_read_ports=level_config.get("num_read_ports", 1),
                    num_write_ports=level_config.get("num_write_ports", 1),
                ))
            hierarchy = MemoryHierarchy(levels)
        
        return cls(
            vault_count=config.get("vault_count"),
            pu_count=config.get("pu_count"),
            timing_config=timing_config,
            hierarchy=hierarchy,
            pe_array=pe_array,
        )
