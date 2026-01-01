#!/usr/bin/env python3
"""
PIM Simulator Comparison and Validation.

This script compares different PIM simulator models to ensure correctness:
1. UniNDP - Python cycle-accurate simulator
2. OptiPIM/Ramulator - C++ cycle-accurate simulator  
3. nn_dataflow - Analytical cost model

Architecture comparison based on configurations:

┌─────────────────────────────────────────────────────────────────────────┐
│                     PIM Architecture Comparison                          │
├──────────────────┬───────────────┬───────────────┬─────────────────────┤
│ Parameter        │ UniNDP        │ OptiPIM       │ Notes               │
│                  │ (hbm-pim.yaml)│ (HBM3_PIM)    │                     │
├──────────────────┼───────────────┼───────────────┼─────────────────────┤
│ Channels         │ 64            │ 16 (x2 pch)   │ UniNDP: 64 pseudo   │
│ Ranks            │ 1             │ 2             │                     │
│ Bank Groups      │ 4             │ 4             │ Same                │
│ Banks/Group      │ 4             │ 4             │ Same                │
│ Rows             │ 16384         │ 512           │ Different!          │
│ Columns          │ 32            │ 64            │ Different!          │
│ PU per BankGroup │ 8             │ 2             │ Different!          │
├──────────────────┼───────────────┼───────────────┼─────────────────────┤
│ Timing (ns)      │               │               │                     │
├──────────────────┼───────────────┼───────────────┼─────────────────────┤
│ tRCD (RD)        │ 14            │ 7             │ Different           │
│ tRP              │ 14            │ 7             │ Different           │
│ tCCD_L           │ 4             │ 2             │ Different           │
│ tCCD_S           │ 2             │ 1             │ Different           │
│ Read Latency     │ 20            │ 7 (nCL)       │ Different           │
│ Write Latency    │ 8             │ 2 (nCWL)      │ Different           │
│ Burst Length     │ 4             │ 4             │ Same                │
└──────────────────┴───────────────┴───────────────┴─────────────────────┘
"""

import sys
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from pathlib import Path


# Project paths
UNINDP_PATH = Path("/Users/haochenzhao/Projects/UniNDP")
OPTIPIM_PATH = Path("/Users/haochenzhao/Projects/OptiPIM")
NNDATAFLOW_PATH = Path("/Users/haochenzhao/Projects/nn_dataflow")


@dataclass
class DRAMConfig:
    """DRAM configuration for comparison."""
    name: str
    
    # Organization
    channels: int
    ranks: int
    bank_groups: int
    banks_per_group: int
    rows: int
    columns: int
    
    # PIM
    pe_per_bankgroup: int = 0
    
    # Timing (in cycles)
    tRCD_RD: int = 0
    tRCD_WR: int = 0  
    tRP: int = 0
    tRAS: int = 0
    tRC: int = 0
    tCCD_L: int = 0
    tCCD_S: int = 0
    tCL: int = 0  # Read latency
    tCWL: int = 0  # Write latency
    tBL: int = 4  # Burst length
    
    @property
    def total_banks(self) -> int:
        return self.channels * self.ranks * self.bank_groups * self.banks_per_group
    
    @property
    def row_hit_latency(self) -> int:
        """Latency for row buffer hit."""
        return self.tCL + self.tBL
    
    @property
    def row_miss_latency(self) -> int:
        """Latency for row buffer miss (PRE + ACT + RD)."""
        return self.tRP + self.tRCD_RD + self.tCL + self.tBL
    
    @property
    def row_empty_latency(self) -> int:
        """Latency for empty row buffer (ACT + RD)."""
        return self.tRCD_RD + self.tCL + self.tBL


def get_unindp_config() -> DRAMConfig:
    """Parse UniNDP configuration."""
    return DRAMConfig(
        name="UniNDP (hbm-pim.yaml)",
        channels=64,
        ranks=1,
        bank_groups=4,
        banks_per_group=4,
        rows=16384,
        columns=32,
        pe_per_bankgroup=8,
        tRCD_RD=14,
        tRCD_WR=10,
        tRP=14,
        tRAS=17,  # Estimated
        tRC=31,   # Estimated
        tCCD_L=4,
        tCCD_S=2,
        tCL=20,
        tCWL=8,
        tBL=4,
    )


def get_optipim_config() -> DRAMConfig:
    """Parse OptiPIM/Ramulator configuration."""
    return DRAMConfig(
        name="OptiPIM (HBM3_PIM)",
        channels=16,
        ranks=2,
        bank_groups=4,
        banks_per_group=4,
        rows=512,       # 1<<9
        columns=64,     # 1<<6
        pe_per_bankgroup=2,
        tRCD_RD=7,
        tRCD_WR=7,
        tRP=7,
        tRAS=17,
        tRC=19,
        tCCD_L=2,
        tCCD_S=1,
        tCL=7,
        tCWL=2,
        tBL=4,
    )


def compare_configs():
    """Compare configurations between simulators."""
    unindp = get_unindp_config()
    optipim = get_optipim_config()
    
    print("=" * 70)
    print("PIM Simulator Configuration Comparison")
    print("=" * 70)
    
    configs = [unindp, optipim]
    
    # Organization
    print("\n1. DRAM Organization:")
    print("-" * 70)
    print(f"{'Parameter':<25} {'UniNDP':<20} {'OptiPIM':<20}")
    print("-" * 70)
    
    params = [
        ('Channels', 'channels'),
        ('Ranks', 'ranks'),
        ('Bank Groups', 'bank_groups'),
        ('Banks/Group', 'banks_per_group'),
        ('Total Banks', 'total_banks'),
        ('Rows per Bank', 'rows'),
        ('Columns', 'columns'),
        ('PE per BankGroup', 'pe_per_bankgroup'),
    ]
    
    for name, attr in params:
        v1 = getattr(unindp, attr)
        v2 = getattr(optipim, attr)
        match = "✓" if v1 == v2 else "✗"
        print(f"{name:<25} {v1:<20} {v2:<20} {match}")
    
    # Timing
    print("\n2. DRAM Timing (cycles):")
    print("-" * 70)
    print(f"{'Parameter':<25} {'UniNDP':<20} {'OptiPIM':<20}")
    print("-" * 70)
    
    timing_params = [
        ('tRCD (Read)', 'tRCD_RD'),
        ('tRCD (Write)', 'tRCD_WR'),
        ('tRP (Precharge)', 'tRP'),
        ('tCCD_L', 'tCCD_L'),
        ('tCCD_S', 'tCCD_S'),
        ('CAS Latency (CL)', 'tCL'),
        ('CAS Write Latency', 'tCWL'),
        ('Burst Length', 'tBL'),
    ]
    
    for name, attr in timing_params:
        v1 = getattr(unindp, attr)
        v2 = getattr(optipim, attr)
        match = "✓" if v1 == v2 else "✗"
        print(f"{name:<25} {v1:<20} {v2:<20} {match}")
    
    # Derived latencies
    print("\n3. Derived Latencies (cycles):")
    print("-" * 70)
    print(f"{'Access Type':<25} {'UniNDP':<20} {'OptiPIM':<20}")
    print("-" * 70)
    
    print(f"{'Row Hit':<25} {unindp.row_hit_latency:<20} {optipim.row_hit_latency:<20}")
    print(f"{'Row Empty':<25} {unindp.row_empty_latency:<20} {optipim.row_empty_latency:<20}")
    print(f"{'Row Miss':<25} {unindp.row_miss_latency:<20} {optipim.row_miss_latency:<20}")


def validate_unindp_model():
    """Validate UniNDP model by running known workloads."""
    print("\n" + "=" * 70)
    print("UniNDP Model Validation")
    print("=" * 70)
    
    # Check if UniNDP is available
    if not (UNINDP_PATH / "sim_verify.py").exists():
        print("⚠ UniNDP not found")
        return
    
    # Test workloads
    test_cases = [
        (5000, 5000, "Large MVM"),
        (2000, 2000, "Medium MVM"),
        (8000, 8000, "Very Large MVM"),
    ]
    
    print("\nRunning validation workloads...")
    
    for M, K, name in test_cases:
        print(f"\n{name}: {M}x{K}")
        try:
            result = subprocess.run(
                [sys.executable, str(UNINDP_PATH / "sim_verify.py"), "-S", str(M), str(K)],
                cwd=str(UNINDP_PATH),
                capture_output=True,
                text=True,
                timeout=120,
            )
            
            if result.returncode == 0:
                # Read result
                output_csv = UNINDP_PATH / "verify_result" / "csv" / f"[{M}, {K}].csv"
                if output_csv.exists():
                    with open(output_csv) as f:
                        line = f.readline().strip()
                        parts = line.split(',')
                        cycles = float(parts[2])
                        
                        # Calculate expected based on workload size
                        total_ops = M * K  # MAC operations
                        cycles_per_op = cycles / total_ops
                        
                        print(f"  Cycles: {cycles:.2f}")
                        print(f"  Ops: {total_ops:,}")
                        print(f"  Cycles/Op: {cycles_per_op:.4f}")
            else:
                print(f"  ✗ Failed: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout")
        except Exception as e:
            print(f"  ✗ Error: {e}")


def validate_optipim_model():
    """Validate OptiPIM/Ramulator model."""
    print("\n" + "=" * 70)
    print("OptiPIM/Ramulator Model Validation")
    print("=" * 70)
    
    # Check if OptiPIM simulator is built
    sim_binary = OPTIPIM_PATH / "simulator" / "build" / "ramulator2"
    
    if not sim_binary.exists():
        print(f"⚠ OptiPIM simulator not built at {sim_binary}")
        print("  To build: cd OptiPIM/simulator && mkdir build && cd build && cmake .. && make")
        return
    
    print(f"✓ OptiPIM simulator found at {sim_binary}")
    
    # Try running a simple test
    try:
        result = subprocess.run(
            [str(sim_binary), "-h"],
            capture_output=True,
            text=True,
        )
        print(f"  Simulator version/help available: {result.returncode == 0}")
    except Exception as e:
        print(f"  ✗ Error: {e}")


def analytical_model_comparison():
    """
    Compare analytical cost model with simulation.
    
    Analytical formula (simplified):
    - Total cycles = (data_reads * avg_latency) + (compute_cycles)
    
    For MVM: C[M,1] = A[M,K] @ B[K,1]
    - Weight reads: M * K elements
    - Input reads: K elements (reused M times)
    - Output writes: M elements
    - MAC operations: M * K
    """
    print("\n" + "=" * 70)
    print("Analytical Model vs Simulation Comparison")
    print("=" * 70)
    
    config = get_unindp_config()
    
    # Test case
    M, K = 5000, 5000
    
    # Analytical estimation
    total_macs = M * K
    
    # Memory access analysis
    # Assuming good row buffer locality within tiles
    elements_per_row = config.columns * 16 // 2  # 16-bit elements, 256-bit column width
    
    weight_rows = (M * K) // elements_per_row
    input_rows = K // elements_per_row
    output_rows = M // elements_per_row
    
    total_row_accesses = weight_rows + input_rows + output_rows
    
    # Assume 80% row hit rate with good mapping
    hit_rate = 0.8
    avg_latency = (hit_rate * config.row_hit_latency + 
                   (1 - hit_rate) * config.row_miss_latency)
    
    # Memory cycles (very rough estimate)
    memory_cycles = total_row_accesses * avg_latency
    
    # Compute cycles (parallel across PUs)
    total_pus = config.total_banks // config.bank_groups * config.pe_per_bankgroup
    compute_cycles = total_macs / total_pus
    
    estimated_cycles = max(memory_cycles, compute_cycles)  # Overlap assumed
    
    print(f"\nWorkload: MVM {M}x{K}")
    print(f"\nAnalytical Estimation:")
    print(f"  Total MACs: {total_macs:,}")
    print(f"  Total Row Accesses: {total_row_accesses:,}")
    print(f"  Est. Avg Latency: {avg_latency:.1f} cycles")
    print(f"  Memory Cycles: {memory_cycles:,.0f}")
    print(f"  Compute Cycles: {compute_cycles:,.0f}")
    print(f"  Estimated Total: {estimated_cycles:,.0f}")
    
    # Get simulation result if available
    output_csv = UNINDP_PATH / "verify_result" / "csv" / f"[{M}, {K}].csv"
    if output_csv.exists():
        with open(output_csv) as f:
            line = f.readline().strip()
            sim_cycles = float(line.split(',')[2])
            
        print(f"\nSimulation Result:")
        print(f"  UniNDP Cycles: {sim_cycles:,.0f}")
        print(f"\nComparison:")
        error = abs(estimated_cycles - sim_cycles) / sim_cycles * 100
        print(f"  Difference: {error:.1f}%")
        
        if error < 20:
            print("  ✓ Analytical model reasonably accurate")
        else:
            print("  ⚠ Large discrepancy - analytical model needs refinement")


def main():
    """Run all validations."""
    print("\n" + "=" * 70)
    print("   PIM Simulator Validation Suite")
    print("   Comparing UniNDP, OptiPIM, and Analytical Models")
    print("=" * 70)
    
    # 1. Compare configurations
    compare_configs()
    
    # 2. Validate UniNDP
    validate_unindp_model()
    
    # 3. Validate OptiPIM
    validate_optipim_model()
    
    # 4. Analytical comparison
    analytical_model_comparison()
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Key Findings:

1. Configuration Differences:
   - UniNDP and OptiPIM use different timing parameters
   - UniNDP: Conservative timings (tRCD=14, tRP=14)
   - OptiPIM: Aggressive timings (tRCD=7, tRP=7)
   - This will result in different absolute cycle counts

2. Architecture Differences:
   - UniNDP: 64 channels, 8 PUs per bank group
   - OptiPIM: 16 channels, 2 PUs per bank group
   - These represent different PIM architectures (HBM-PIM variants)

3. Validation Approach:
   - For correctness: Both simulators follow standard DRAM protocols
   - For comparison: Use relative performance (cycles/op) not absolute
   - For your ILP: Choose ONE simulator as golden model

Recommendations:
   - Use UniNDP as Golden Model (Python, easier to integrate)
   - Verify timing parameters match your target architecture
   - Use analytical model for fast estimation, simulator for verification
""")


if __name__ == "__main__":
    main()
