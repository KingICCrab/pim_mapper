"""
Comprehensive validation of ILP row activation predictions vs trace-based counting.

Tests multiple workloads with varying:
- Input sizes (H, W)
- Channel counts (C)
- Filter counts (K)
- Batch sizes (N)
- Kernel sizes (R, S)
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
from validation.dram.trace_generator import TraceGenerator, DRAMConfig


@dataclass
class TestWorkload:
    """Test workload configuration."""
    name: str
    N: int
    K: int
    C: int
    P: int
    Q: int
    R: int
    S: int
    stride: Tuple[int, int] = (1, 1)
    dilation: Tuple[int, int] = (1, 1)


# Define diverse test workloads (10 representative cases)
TEST_WORKLOADS = [
    # Basic workloads
    TestWorkload("tiny", N=1, K=4, C=4, P=4, Q=4, R=3, S=3),
    TestWorkload("small", N=1, K=16, C=16, P=8, Q=8, R=3, S=3),
    
    # Varying channels/filters
    TestWorkload("channels_32", N=1, K=16, C=32, P=8, Q=8, R=3, S=3),
    TestWorkload("filters_32", N=1, K=32, C=16, P=8, Q=8, R=3, S=3),
    
    # Varying spatial sizes
    TestWorkload("spatial_4x4", N=1, K=16, C=16, P=4, Q=4, R=3, S=3),
    TestWorkload("spatial_16x16", N=1, K=16, C=16, P=16, Q=16, R=3, S=3),
    
    # Varying kernel sizes
    TestWorkload("kernel_1x1", N=1, K=16, C=16, P=8, Q=8, R=1, S=1),
    TestWorkload("kernel_5x5", N=1, K=16, C=16, P=8, Q=8, R=5, S=5),
    
    # Batch size
    TestWorkload("batch_2", N=2, K=16, C=16, P=8, Q=8, R=3, S=3),
    
    # Strided convolution
    TestWorkload("stride_2", N=1, K=16, C=16, P=8, Q=8, R=3, S=3, stride=(2, 2)),
]


def count_row_activations(trace: List[str], dram_config: DRAMConfig) -> Dict[int, int]:
    """Count row activations per bank from trace."""
    row_size = dram_config.row_buffer_bytes
    bank_size = row_size * dram_config.num_rows
    
    # Track current row in row buffer per bank
    current_row = {}  # bank -> current row
    activations = defaultdict(int)  # bank -> activation count
    
    for line in trace:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        addr = int(parts[1], 16)
        
        # Extract bank and row
        bank = addr // bank_size
        row = (addr % bank_size) // row_size
        
        if bank not in current_row or current_row[bank] != row:
            activations[bank] += 1
            current_row[bank] = row
    
    return dict(activations)


def run_validation(workload_config: TestWorkload, dram_config: DRAMConfig) -> Dict:
    """Run validation for a single workload."""
    # Create workload - ConvWorkload computes H/W from P/Q/R/S/stride/dilation
    workload = ConvWorkload(
        name=workload_config.name,
        N=workload_config.N,
        K=workload_config.K,
        C=workload_config.C,
        P=workload_config.P,
        Q=workload_config.Q,
        R=workload_config.R,
        S=workload_config.S,
        stride=workload_config.stride,
        dilation=workload_config.dilation,
    )
    
    # Run optimizer
    optimizer = PIMOptimizer()
    result = optimizer.optimize([workload], objective="latency")
    
    if result is None or result.mappings is None or len(result.mappings) == 0:
        return {"error": "No solution found"}
    
    mapping = result.mappings[0]
    
    # Get ILP predictions
    ilp_input = mapping.metrics.get("row_activations_input", 0)
    ilp_weight = mapping.metrics.get("row_activations_weight", 0)
    ilp_output = mapping.metrics.get("row_activations_output", 0)
    ilp_total = ilp_input + ilp_weight + ilp_output
    
    # Generate trace
    gen = TraceGenerator(dram_config)
    trace = gen.generate_trace(mapping, workload)
    
    # Count activations from trace
    trace_acts = count_row_activations(trace, dram_config)
    trace_input = trace_acts.get(0, 0)
    trace_weight = trace_acts.get(1, 0)
    trace_output = trace_acts.get(2, 0)
    trace_total = sum(trace_acts.values())
    
    # Calculate errors
    def calc_error(ilp, trace):
        if trace == 0:
            return 0.0 if ilp == 0 else 100.0
        return abs(ilp - trace) / trace * 100
    
    return {
        "name": workload_config.name,
        "config": f"N={workload_config.N}, K={workload_config.K}, C={workload_config.C}, "
                  f"P={workload_config.P}, Q={workload_config.Q}, R={workload_config.R}, S={workload_config.S}",
        "layout": mapping.layout,
        "block_h": mapping.tile_info.get("block_h", 0),
        "block_w": mapping.tile_info.get("block_w", 0),
        "ilp": {"input": ilp_input, "weight": ilp_weight, "output": ilp_output, "total": ilp_total},
        "trace": {"input": trace_input, "weight": trace_weight, "output": trace_output, "total": trace_total},
        "error": {
            "input": calc_error(ilp_input, trace_input),
            "weight": calc_error(ilp_weight, trace_weight),
            "output": calc_error(ilp_output, trace_output),
            "total": calc_error(ilp_total, trace_total),
        },
    }


def print_results(results: List[Dict]):
    """Print validation results in a formatted table."""
    print("=" * 120)
    print("Comprehensive Row Activation Validation Results")
    print("=" * 120)
    print()
    
    # Summary statistics
    total_tests = len(results)
    passed_tests = sum(1 for r in results if isinstance(r.get("error"), dict) and r["error"].get("total", 100) < 1.0)
    error_tests = sum(1 for r in results if isinstance(r.get("error"), str))
    
    print(f"Total tests: {total_tests}")
    print(f"Passed (< 1% error): {passed_tests}")
    print(f"Errors: {error_tests}")
    print(f"Pass rate: {passed_tests / max(1, total_tests - error_tests) * 100:.1f}%")
    print()
    
    # Detailed results table
    print("-" * 120)
    print(f"{'Workload':<15} {'Config':<35} {'ILP':<20} {'Trace':<20} {'Error':<15} {'Layout'}")
    print("-" * 120)
    
    for r in results:
        if "error" in r and isinstance(r["error"], str):
            print(f"{r['name']:<15} ERROR: {r['error']}")
            continue
        
        ilp_str = f"I:{r['ilp']['input']:.0f} W:{r['ilp']['weight']:.0f} O:{r['ilp']['output']:.0f}"
        trace_str = f"I:{r['trace']['input']} W:{r['trace']['weight']} O:{r['trace']['output']}"
        error_str = f"{r['error']['total']:.1f}%"
        
        input_layout = "RA" if r['layout'].get(0) == "row_aligned" else "Seq"
        weight_layout = "RA" if r['layout'].get(1) == "row_aligned" else "Seq"
        output_layout = "RA" if r['layout'].get(2) == "row_aligned" else "Seq"
        layout_str = f"I:{input_layout} W:{weight_layout} O:{output_layout}"
        
        status = "✓" if r['error']['total'] < 1.0 else "✗"
        
        # Truncate config if too long
        config = r['config']
        if len(config) > 35:
            config = config[:32] + "..."
        
        print(f"{r['name']:<15} {config:<35} {ilp_str:<20} {trace_str:<20} {error_str:<15} {layout_str} {status}")
    
    print("-" * 120)
    
    # Show failed tests details
    failed = [r for r in results if isinstance(r.get("error"), dict) and r["error"].get("total", 100) >= 1.0]
    if failed:
        print()
        print("Failed Tests Details:")
        print("-" * 80)
        for r in failed:
            if "error" in r and isinstance(r["error"], str):
                continue
            print(f"\n{r['name']}:")
            print(f"  Config: {r['config']}")
            print(f"  Block: {r['block_h']}x{r['block_w']}")
            print(f"  ILP:   Input={r['ilp']['input']:.1f}, Weight={r['ilp']['weight']:.1f}, Output={r['ilp']['output']:.1f}")
            print(f"  Trace: Input={r['trace']['input']}, Weight={r['trace']['weight']}, Output={r['trace']['output']}")
            print(f"  Error: Input={r['error']['input']:.1f}%, Weight={r['error']['weight']:.1f}%, Output={r['error']['output']:.1f}%")


def main():
    dram_config = DRAMConfig(
        row_buffer_bytes=1024,
        num_banks=4,
        num_rows=16384,
        element_size=1,
    )
    
    print(f"DRAM Config: row_buffer={dram_config.row_buffer_bytes}B, banks={dram_config.num_banks}")
    print()
    
    results = []
    for i, wl in enumerate(TEST_WORKLOADS):
        print(f"Testing [{i+1}/{len(TEST_WORKLOADS)}]: {wl.name}...", end=" ", flush=True)
        try:
            result = run_validation(wl, dram_config)
            results.append(result)
            error = result.get("error", {}).get("total", -1)
            if error >= 0:
                status = "✓" if error < 1.0 else f"✗ ({error:.1f}%)"
                print(status)
            else:
                print("ERROR")
        except Exception as e:
            print(f"EXCEPTION: {e}")
            results.append({"name": wl.name, "error": str(e)})
    
    print()
    print_results(results)


if __name__ == "__main__":
    main()
