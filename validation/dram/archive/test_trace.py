#!/usr/bin/env python3
"""
Validation script: Compare ILP model DRAM latency prediction with Ramulator2 simulation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from pim_optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer.arch.pim_arch import PIMArchitecture

from trace_generator import TraceGenerator, DRAMConfig
from ramulator_runner import RamulatorRunner


def main():
    # Test workload
    workload = ConvWorkload(
        name="test_conv",
        N=1, K=8, C=8, P=4, Q=4, R=3, S=3
    )
    
    # Simple architecture - use config file or default
    arch = PIMArchitecture()
    
    print(f"Workload: N={workload.N}, K={workload.K}, C={workload.C}, "
          f"P={workload.P}, Q={workload.Q}, R={workload.R}, S={workload.S}")
    print(f"Input size: H={workload.input_size['H']}, W={workload.input_size['W']}")
    print(f"Total MACs: {workload.N * workload.K * workload.C * workload.P * workload.Q * workload.R * workload.S}")
    
    # Run optimizer
    print("\n" + "="*60)
    print("Running ILP Optimizer...")
    print("="*60)
    optimizer = PIMOptimizer(arch)
    result = optimizer.optimize([workload])
    
    if result is None or not result.mappings:
        print("Optimization failed!")
        return
    
    mapping = result.mappings[0]  # Get first workload's mapping
    
    print(f"\nILP Model Predictions:")
    print(f"  Latency (scaled):     {mapping.metrics.get('latency', 'N/A')}")
    print(f"  Compute cycles (sc):  {mapping.metrics.get('compute_cycles', 'N/A')}")
    print(f"  Row activations:      {mapping.metrics.get('row_activations', 'N/A')}")
    
    # Calculate actual values (unscaled)
    macs = workload.N * workload.K * workload.C * workload.P * workload.Q * workload.R * workload.S
    MAX_BOUND = 1e4
    macs_scale_factor = MAX_BOUND / (1.02 * macs)
    
    scaled_latency = mapping.metrics.get('latency', 0)
    actual_latency = scaled_latency / macs_scale_factor if macs_scale_factor > 0 else 0
    
    scaled_compute = mapping.metrics.get('compute_cycles', 0)
    actual_compute = scaled_compute / macs_scale_factor if macs_scale_factor > 0 else 0
    
    print(f"\n  Unscaled (actual) values:")
    print(f"    MACs:             {macs}")
    print(f"    Scale factor:     {macs_scale_factor:.6f}")
    print(f"    Actual latency:   {actual_latency:.2f} cycles")
    print(f"    Actual compute:   {actual_compute:.2f} cycles")
    
    print(f"\nMapping:")
    print(f"  loop_bounds: {mapping.loop_bounds}")
    print(f"  permutation: {mapping.permutation}")
    print(f"  layout: {mapping.layout}")
    print(f"  tile_info: {mapping.tile_info}")
    
    # Generate trace
    print("\n" + "="*60)
    print("Generating Memory Trace...")
    print("="*60)
    dram_cfg = DRAMConfig(
        num_channels=1,
        num_ranks=1,
        num_banks=4,
        num_rows=1024,
        num_cols=256,
    )
    trace_gen = TraceGenerator(dram_cfg, element_size=1)
    
    # Debug: show buffer tile size
    buffer_tile = trace_gen._compute_buffer_tile_size(mapping)
    print(f"Buffer tile size: {buffer_tile}")
    print(f"  (R={buffer_tile[0]}, S={buffer_tile[1]}, P={buffer_tile[2]}, Q={buffer_tile[3]}, "
          f"C={buffer_tile[4]}, K={buffer_tile[5]}, N={buffer_tile[6]})")
    
    try:
        trace = trace_gen.generate_trace(mapping, workload)
        print(f"Generated {len(trace)} trace lines")
        
        # Count LD/ST
        num_loads = sum(1 for t in trace if t.startswith('LD'))
        num_stores = sum(1 for t in trace if t.startswith('ST'))
        print(f"  Loads:  {num_loads}")
        print(f"  Stores: {num_stores}")
        
        # Write trace
        trace_path = Path(__file__).parent / "results" / "test_trace.txt"
        trace_gen.write_trace(trace, str(trace_path))
        
        # Show first few lines
        print("\nFirst 10 trace lines:")
        for line in trace[:10]:
            print(f"  {line}")
        
    except Exception as e:
        print(f"Trace generation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run Ramulator2
    print("\n" + "="*60)
    print("Running Ramulator2 Simulation...")
    print("="*60)
    runner = RamulatorRunner()
    
    if not runner.is_available():
        print(f"Ramulator2 not found at {runner.ramulator_bin}")
        return
    
    try:
        sim_result = runner.run(str(trace_path))
        print(f"\nRamulator2 Results:")
        print(f"  Cycles:       {sim_result.cycles}")
        print(f"  Reads:        {sim_result.num_reads}")
        print(f"  Writes:       {sim_result.num_writes}")
        print(f"  Row hits:     {sim_result.row_hits}")
        print(f"  Row misses:   {sim_result.row_misses}")
        print(f"  Row conflicts:{sim_result.row_conflicts}")
        print(f"  Hit rate:     {sim_result.hit_rate:.2%}")
        print(f"  Total ACTs:   {sim_result.total_row_activations}")
    except Exception as e:
        print(f"Ramulator2 failed: {e}")
        return
    
    # Compare
    print("\n" + "="*60)
    print("Comparison: ILP vs Ramulator2")
    print("="*60)
    
    # Get actual (unscaled) latency for comparison
    ilp_latency_scaled = mapping.metrics.get('latency', 0)
    
    # Calculate scale factor
    macs = workload.N * workload.K * workload.C * workload.P * workload.Q * workload.R * workload.S
    MAX_BOUND = 1e4
    scale_factor = MAX_BOUND / (1.02 * macs)
    
    ilp_latency_actual = ilp_latency_scaled / scale_factor if scale_factor > 0 else 0
    ram_cycles = sim_result.cycles
    
    if ilp_latency_actual > 0:
        error = abs(ilp_latency_actual - ram_cycles) / ram_cycles * 100
        print(f"  ILP Predicted (scaled):   {ilp_latency_scaled:.0f}")
        print(f"  ILP Predicted (actual):   {ilp_latency_actual:.0f} cycles")
        print(f"  Ramulator2 Cycles:        {ram_cycles} cycles")
        print(f"  Error:                    {error:.2f}%")
    else:
        print(f"  ILP Latency not available")
        print(f"  Ramulator2 Cycles: {ram_cycles}")


if __name__ == "__main__":
    main()
