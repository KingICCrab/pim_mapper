"""
Main validation script for pim_optimizer DRAM model.

Compares pim_optimizer's predicted DRAM metrics with Ramulator2 simulation.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from pim_optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer.arch.pim_arch import PIMArch

from .trace_generator import TraceGenerator, DRAMConfig
from .ramulator_runner import RamulatorRunner


def validate_single_mapping(
    workload: ConvWorkload,
    arch: PIMArch,
    ramulator_path: str,
    ramulator_config: str = None,
    output_dir: str = "validation_output"
) -> dict:
    """
    Validate a single mapping against Ramulator2.
    
    Returns:
        dict with predicted metrics, simulated metrics, and errors
    """
    # Run optimizer
    optimizer = PIMOptimizer(arch, workload)
    result = optimizer.optimize()
    
    if result is None:
        return {"error": "Optimization failed"}
    
    # Extract predicted metrics
    predicted = {
        "dram_accesses": result.get("total_dram_accesses", 0),
        "row_activations": result.get("row_activations", 0),
    }
    
    # Generate trace
    dram_cfg = DRAMConfig()
    trace_gen = TraceGenerator(dram_cfg)
    mapping = result.get("mapping")
    
    if mapping is None:
        return {"error": "No mapping in result"}
    
    trace = trace_gen.generate_trace(mapping, workload)
    
    # Write trace
    os.makedirs(output_dir, exist_ok=True)
    trace_path = os.path.join(output_dir, "trace.txt")
    trace_gen.write_trace(trace, trace_path)
    
    # Run Ramulator2
    runner = RamulatorRunner(ramulator_path)
    sim_result = runner.run(trace_path, ramulator_config)
    
    if sim_result is None:
        return {"error": "Ramulator2 simulation failed", "predicted": predicted}
    
    # Compare
    simulated = {
        "dram_cycles": sim_result.cycles,
        "row_activations": sim_result.row_activations,
        "memory_reads": sim_result.memory_reads,
    }
    
    return {
        "predicted": predicted,
        "simulated": simulated,
        "trace_length": len(trace),
    }


def validate_sweep(
    workload_params_list: list,
    arch: PIMArch,
    ramulator_path: str,
    ramulator_config: str = None,
    output_dir: str = "validation_output"
) -> dict:
    """
    Validate across a sweep of workload parameters.
    
    Returns:
        dict with predictions, simulations, and correlation metrics
    """
    predictions = []
    simulations = []
    
    for i, params in enumerate(workload_params_list):
        print(f"\n[{i+1}/{len(workload_params_list)}] {params}")
        
        workload = ConvWorkload(**params)
        result = validate_single_mapping(
            workload, arch, ramulator_path, ramulator_config,
            os.path.join(output_dir, f"case_{i}")
        )
        
        if "error" not in result:
            predictions.append(result["predicted"])
            simulations.append(result["simulated"])
    
    if len(predictions) < 2:
        return {"error": "Not enough valid results for correlation"}
    
    # Compute correlations
    pred_acts = [p["row_activations"] for p in predictions]
    sim_acts = [s["row_activations"] for s in simulations]
    
    corr = np.corrcoef(pred_acts, sim_acts)[0, 1]
    
    return {
        "num_cases": len(predictions),
        "row_activation_correlation": corr,
        "predictions": predictions,
        "simulations": simulations,
    }


def main():
    """Run validation with default settings."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate pim_optimizer DRAM model")
    parser.add_argument("--ramulator", type=str, required=True,
                        help="Path to Ramulator2 binary")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to Ramulator2 config")
    parser.add_argument("--output", type=str, default="validation_output",
                        help="Output directory")
    args = parser.parse_args()
    
    # Default architecture
    arch = PIMArch()
    
    # Test workloads
    test_cases = [
        {"N": 1, "C": 64, "H": 28, "W": 28, "K": 64, "R": 3, "S": 3},
        {"N": 1, "C": 128, "H": 14, "W": 14, "K": 128, "R": 3, "S": 3},
        {"N": 1, "C": 256, "H": 7, "W": 7, "K": 256, "R": 3, "S": 3},
    ]
    
    result = validate_sweep(
        test_cases, arch, args.ramulator, args.config, args.output
    )
    
    print("\n" + "="*60)
    print("Validation Results")
    print("="*60)
    print(f"Valid cases: {result.get('num_cases', 0)}")
    print(f"Row activation correlation: {result.get('row_activation_correlation', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
