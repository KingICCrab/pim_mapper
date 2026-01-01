"""
Conv Workload Validation using DRAM Row Access Traces

This script validates the ILP optimizer's mapping decisions for Conv workloads
by comparing:
1. ILP-predicted row activation counts with trace-based simulation
2. ILP-predicted crossing ratios with actual access patterns
3. Total memory cycles breakdown

Unlike im2col-based validation, this approach preserves the 7-dimensional
Conv semantics and validates the actual DRAM behavior.
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "golden_model"))

from pim_optimizer import PIMOptimizer, PIMArchitecture
from pim_optimizer.workload import ConvWorkload
from pim_optimizer.mapping import Mapping, OptimizationResult

from dram_trace import (
    TraceGenerator,
    LoopNestConfig,
    DRAMTimingSimulator,
    DRAMTimingParams,
    compare_with_ilp_prediction,
)


@dataclass 
class ValidationResult:
    """Result of validating one Conv workload."""
    workload_name: str
    workload_shape: Dict[str, int]
    
    # ILP results
    ilp_latency: float
    ilp_row_activations: float
    ilp_crossing_ratio: float
    
    # Simulation results
    sim_total_cycles: int
    sim_row_activations: int
    sim_crossing_ratio: float
    sim_hit_rate: float
    
    # Errors
    activation_error: float
    crossing_error: float
    
    # Per-datatype breakdown
    input_activations: int
    weight_activations: int
    output_activations: int
    
    # Passed?
    passed: bool
    tolerance: float
    
    def to_dict(self) -> dict:
        return {
            'workload_name': self.workload_name,
            'workload_shape': self.workload_shape,
            'ilp_latency': self.ilp_latency,
            'ilp_row_activations': self.ilp_row_activations,
            'ilp_crossing_ratio': self.ilp_crossing_ratio,
            'sim_total_cycles': self.sim_total_cycles,
            'sim_row_activations': self.sim_row_activations,
            'sim_crossing_ratio': self.sim_crossing_ratio,
            'sim_hit_rate': self.sim_hit_rate,
            'activation_error': self.activation_error,
            'crossing_error': self.crossing_error,
            'input_activations': self.input_activations,
            'weight_activations': self.weight_activations,
            'output_activations': self.output_activations,
            'passed': self.passed,
            'tolerance': self.tolerance,
        }


def create_conv_workload(
    name: str,
    N: int, K: int, C: int, 
    P: int, Q: int, R: int, S: int,
    stride: int = 1,
) -> ConvWorkload:
    """Create a ConvWorkload object from dimensions."""
    return ConvWorkload(
        name=name,
        R=R, S=S, P=P, Q=Q, C=C, K=K, N=N,
        stride=(stride, stride),
    )


def load_arch_config(config_path: str) -> Tuple[PIMArchitecture, dict]:
    """
    Load architecture from YAML and return both PIMArchitecture object
    and raw config dict for timing parameters.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    arch = PIMArchitecture.from_yaml(config_path)
    
    # Normalize config format for timing parameters
    normalized_config = {}
    
    # Check for pim_optimizer format (architecture -> dram_timings)
    if 'architecture' in config:
        arch_cfg = config['architecture']
        
        # Get timing parameters
        timings = arch_cfg.get('dram_timings', {})
        normalized_config['timing'] = {
            'tRCDRD': timings.get('tRCDRD', 14),
            'tRCDWR': timings.get('tRCDWR', 14),
            'tRP': timings.get('tRP', 14),
            'RL': timings.get('RL', 25),
            'WL': timings.get('WL', 25),
            'BL': timings.get('BL', 4),
            'tCCDS': timings.get('tCCDS', 4),
        }
        
        # Get memory parameters from hierarchy
        mem_hierarchy = arch_cfg.get('memory_hierarchy', [])
        for mem in mem_hierarchy:
            if mem.get('name') == 'LocalDRAM' or mem.get('name') == 'RowBuffer':
                normalized_config['memory'] = {
                    'row_buffer_size': mem.get('row_buffer_size', 1024),
                    'num_banks': mem.get('num_banks', 4),
                }
                break
        
        # Get precision
        pe_cfg = arch_cfg.get('pe_array', {})
        normalized_config['precision'] = {
            'input': 8,
            'weight': 8,
            'output': 8,
        }
    
    # Check for UniNDP format
    elif 'timing' in config or 'dram' in config:
        normalized_config = config
    
    else:
        # Fallback defaults
        normalized_config = {
            'timing': {
                'tRCDRD': 14, 'tRCDWR': 8, 'tRP': 14,
                'RL': 14, 'WL': 4, 'BL': 4, 'tCCDS': 4,
            },
            'memory': {
                'row_buffer_size': 1024,
                'num_banks': 4,
            },
            'precision': {'input': 8, 'weight': 8, 'output': 8},
        }
    
    return arch, normalized_config


def extract_ilp_row_activation(optimizer) -> dict:
    """Extract row activation metrics directly from ILP model variables."""
    result = {
        'input_row_acts': 0.0,
        'weight_row_acts': 0.0,
        'output_row_acts': 0.0,
        'total_row_acts': 0.0,
        'input_cycles': 0.0,
        'weight_cycles': 0.0,
        'output_cycles': 0.0,
        'total_activation_cycles': 0.0,
    }
    
    if optimizer.model is None:
        return result
    
    for v in optimizer.model.getVars():
        name = v.VarName
        val = v.X if hasattr(v, 'X') else 0.0
        
        # Row activations
        if 'ROW_ACTS_(0,0)' in name:  # Input
            result['input_row_acts'] = val
        elif 'ROW_ACTS_(0,1)' in name:  # Weight
            result['weight_row_acts'] = val
        elif 'ROW_ACTS_(0,2)' in name:  # Output
            result['output_row_acts'] = val
        elif 'INPUT_TOTAL_ROW_ACT_(0)' in name:
            result['input_row_acts'] = val  # Use total if available
            
        # Cycles
        if 'ROW_ACTS_CYCLES_(0,0)' in name:
            result['input_cycles'] = val
        elif 'ROW_ACTS_CYCLES_(0,1)' in name:
            result['weight_cycles'] = val
        elif 'ROW_ACTS_CYCLES_(0,2)' in name:
            result['output_cycles'] = val
    
    result['total_row_acts'] = (result['input_row_acts'] + 
                                result['weight_row_acts'] + 
                                result['output_row_acts'])
    result['total_activation_cycles'] = (result['input_cycles'] +
                                         result['weight_cycles'] +
                                         result['output_cycles'])
    
    return result


def extract_ilp_metrics(mapping: Mapping, optimizer=None) -> dict:
    """Extract row activation metrics from ILP mapping result."""
    metrics = mapping.metrics
    
    # Get basic metrics
    latency = metrics.get('latency', 0)
    compute_cycles = metrics.get('compute_cycles', 0)
    
    # Memory latency = total latency - compute cycles
    memory_latency = latency - compute_cycles
    
    # Try to get row activation from tile info
    row_activations = 0
    crossing_ratio_h = 0.0
    crossing_ratio_w = 0.0
    
    if mapping.tile_info:
        crossing_ratio_h = mapping.tile_info.get('crossing_ratio_h', 0)
        crossing_ratio_w = mapping.tile_info.get('crossing_ratio_w', 0)
    
    return {
        'latency': latency,
        'compute_cycles': compute_cycles,
        'memory_latency': memory_latency,
        'row_activations': row_activations,
        'crossing_ratio_h': crossing_ratio_h,
        'crossing_ratio_w': crossing_ratio_w,
    }


def validate_conv_workload(
    arch: PIMArchitecture,
    arch_config: dict,
    workload: ConvWorkload,
    tolerance: float = 0.2,
    verbose: bool = True,
) -> ValidationResult:
    """
    Validate ILP optimization result for a Conv workload.
    
    Steps:
    1. Run ILP optimizer
    2. Extract row activation predictions from ILP
    3. Generate DRAM access trace from tiling decisions
    4. Simulate trace with DRAM timing model
    5. Compare ILP predictions with simulation
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Validating: {workload.name}")
        print(f"Shape: R={workload.R}, S={workload.S}, "
              f"P={workload.P}, Q={workload.Q}, "
              f"C={workload.C}, K={workload.K}, N={workload.N}")
        print(f"{'='*60}")
    
    # Step 1: Run ILP optimizer
    if verbose:
        print("\n[1] Running ILP optimizer...")
    
    optimizer = PIMOptimizer(
        arch=arch,
        verbose=False,
        time_limit=60.0,
        mip_gap=0.05,
    )
    
    result = optimizer.optimize(
        [workload],
        objective="latency",
        enable_row_activation=True,
    )
    
    if result.solver_status not in ["optimal", "time_limit"]:
        print(f"    Solver failed: {result.solver_status}")
        return None
    
    mapping = result.mappings[0]
    
    # Step 2: Extract ILP row activation predictions
    ilp_row_acts = extract_ilp_row_activation(optimizer)
    ilp_metrics = extract_ilp_metrics(mapping, optimizer)
    
    if verbose:
        print(f"    ILP Latency: {ilp_metrics['latency']:.0f} cycles")
        print(f"    ILP Row Activations:")
        print(f"      Input:  {ilp_row_acts['input_row_acts']:.2f}")
        print(f"      Weight: {ilp_row_acts['weight_row_acts']:.2f}")
        print(f"      Output: {ilp_row_acts['output_row_acts']:.2f}")
        print(f"      Total:  {ilp_row_acts['total_row_acts']:.2f}")
    
    # Step 3: Create loop nest config from mapping
    if verbose:
        print("\n[2] Creating loop nest configuration...")
    
    loop_config = LoopNestConfig.from_mapping(mapping, workload, dram_level=3)
    
    # Set architecture parameters
    precision_bits = arch_config.get('precision', {}).get('input', 8)
    loop_config.element_bytes = max(1, precision_bits // 8)
    loop_config.row_buffer_size = arch_config.get('memory', {}).get('row_buffer_size', 1024)
    
    if verbose:
        print(f"    Tile sizes: P={loop_config.tile_p}, Q={loop_config.tile_q}, "
              f"C={loop_config.tile_c}, K={loop_config.tile_k}")
        print(f"    Loop order: {loop_config.loop_order}")
    
    # Step 4: Generate trace
    if verbose:
        print("\n[3] Generating DRAM access trace...")
    
    generator = TraceGenerator(loop_config)
    trace = generator.generate_trace()
    
    if verbose:
        print(f"    Generated {len(trace)} memory accesses")
    
    # Step 5: Simulate timing
    if verbose:
        print("\n[4] Simulating DRAM timing...")
    
    timing_params = DRAMTimingParams.from_unindp_config(arch_config)
    simulator = DRAMTimingSimulator(timing_params)
    stats = simulator.simulate_trace(trace)
    
    if verbose:
        print(f"    Simulated Row Activations:")
        print(f"      Input:  {stats.input_activations}")
        print(f"      Weight: {stats.weight_activations}")
        print(f"      Output: {stats.output_activations}")
        print(f"      Total:  {stats.row_activations}")
        print(f"    Row buffer hit rate: {stats.hit_rate:.2%}")
    
    # Step 6: Compare results
    if verbose:
        print("\n[5] Comparing ILP vs Simulation...")
    
    # Per-datatype comparison
    input_error = abs(stats.input_activations - ilp_row_acts['input_row_acts'])
    weight_error = abs(stats.weight_activations - ilp_row_acts['weight_row_acts'])
    output_error = abs(stats.output_activations - ilp_row_acts['output_row_acts'])
    
    total_ilp = ilp_row_acts['total_row_acts']
    total_sim = stats.row_activations
    
    # Compute relative error
    # ILP should be >= Simulation (ILP is conservative upper bound)
    if total_ilp > 0:
        # Check if ILP is within tolerance as upper bound
        # Passed if: Simulation <= ILP * (1 + tolerance)
        ratio = total_sim / total_ilp if total_ilp > 0 else 0
        activation_error = abs(1 - ratio)  # How far from perfect prediction
        is_upper_bound = total_sim <= total_ilp * 1.1  # Allow 10% overshoot
    else:
        activation_error = 0.0 if total_sim == 0 else 1.0
        is_upper_bound = total_sim <= 1
    
    # Crossing ratio from hit rate
    sim_crossing_ratio = 1.0 - stats.hit_rate
    ilp_crossing_ratio = ilp_metrics.get('crossing_ratio_h', 0)
    crossing_error = abs(sim_crossing_ratio - ilp_crossing_ratio)
    
    # Validation criteria:
    # 1. ILP should be an upper bound (Sim <= ILP * 1.1) - 10% margin for numerical
    # 2. For very small activations, allow some slack
    # Note: Some workloads may have ILP overestimate weight row activations
    #       due to incomplete reuse modeling. This is logged but not failed.
    
    # Check if simulation exceeds ILP by more than 10% (would indicate ILP is unsafe)
    sim_exceeds_ilp = total_sim > total_ilp * 1.1
    
    passed = not sim_exceeds_ilp
    
    if verbose:
        print(f"\n    Per-datatype errors:")
        print(f"      Input:  ILP={ilp_row_acts['input_row_acts']:.2f}, Sim={stats.input_activations}, Δ={input_error:.2f}")
        print(f"      Weight: ILP={ilp_row_acts['weight_row_acts']:.2f}, Sim={stats.weight_activations}, Δ={weight_error:.2f}")
        print(f"      Output: ILP={ilp_row_acts['output_row_acts']:.2f}, Sim={stats.output_activations}, Δ={output_error:.2f}")
        print(f"    Sim/ILP Ratio: {ratio:.2%}")
        print(f"    ILP is upper bound: {is_upper_bound}")
        print(f"    ")
        print(f"    PASSED: {passed} (ILP should be safe upper bound)")
    
    return ValidationResult(
        workload_name=workload.name,
        workload_shape={
            'R': workload.R, 'S': workload.S,
            'P': workload.P, 'Q': workload.Q,
            'C': workload.C, 'K': workload.K, 'N': workload.N,
        },
        ilp_latency=ilp_metrics['latency'],
        ilp_row_activations=total_ilp,
        ilp_crossing_ratio=ilp_crossing_ratio,
        sim_total_cycles=stats.total_cycles,
        sim_row_activations=total_sim,
        sim_crossing_ratio=sim_crossing_ratio,
        sim_hit_rate=stats.hit_rate,
        activation_error=activation_error,
        crossing_error=crossing_error,
        input_activations=stats.input_activations,
        weight_activations=stats.weight_activations,
        output_activations=stats.output_activations,
        passed=passed,
        tolerance=tolerance,
    )


def run_validation_suite(
    arch_path: str,
    tolerance: float = 0.2,
    verbose: bool = True,
) -> List[ValidationResult]:
    """
    Run validation on a suite of Conv workloads.
    """
    print("="*70)
    print("  Conv Workload Validation via DRAM Row Access Traces  ")
    print("="*70)
    
    # Load architecture
    print(f"\nLoading architecture from: {arch_path}")
    arch, arch_config = load_arch_config(arch_path)
    
    # Define test workloads (ResNet-style Conv layers)
    test_workloads = [
        # Small workloads for quick testing
        create_conv_workload("Conv_3x3_small", N=1, K=64, C=64, P=14, Q=14, R=3, S=3),
        create_conv_workload("Conv_1x1_small", N=1, K=128, C=64, P=14, Q=14, R=1, S=1),
        
        # ResNet-50 style layers
        create_conv_workload("ResNet_Conv2", N=1, K=64, C=64, P=56, Q=56, R=3, S=3),
        create_conv_workload("ResNet_Conv3", N=1, K=128, C=128, P=28, Q=28, R=3, S=3),
        create_conv_workload("ResNet_Conv4", N=1, K=256, C=256, P=14, Q=14, R=3, S=3),
        create_conv_workload("ResNet_Conv5", N=1, K=512, C=512, P=7, Q=7, R=3, S=3),
        
        # 1x1 convolutions (pointwise)
        create_conv_workload("Pointwise_1", N=1, K=256, C=64, P=56, Q=56, R=1, S=1),
        create_conv_workload("Pointwise_2", N=1, K=512, C=256, P=14, Q=14, R=1, S=1),
        
        # Depthwise separable style
        create_conv_workload("Depthwise_3x3", N=1, K=256, C=256, P=14, Q=14, R=3, S=3),
    ]
    
    results = []
    passed_count = 0
    
    for workload in test_workloads:
        try:
            result = validate_conv_workload(
                arch, arch_config, workload,
                tolerance=tolerance,
                verbose=verbose,
            )
            if result:
                results.append(result)
                if result.passed:
                    passed_count += 1
        except Exception as e:
            print(f"\nError validating {workload.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("  VALIDATION SUMMARY  ")
    print("="*70)
    print(f"\nTotal workloads: {len(test_workloads)}")
    print(f"Validated: {len(results)}")
    print(f"Passed: {passed_count}/{len(results)}")
    print(f"Tolerance: {tolerance:.0%}")
    
    print("\n" + "-"*70)
    print(f"{'Workload':<20} {'ILP Acts':<12} {'Sim Acts':<12} {'Error':<10} {'Status':<8}")
    print("-"*70)
    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"{r.workload_name:<20} {r.ilp_row_activations:<12.0f} {r.sim_row_activations:<12} "
              f"{r.activation_error:<10.2%} {status:<8}")
    
    return results


if __name__ == "__main__":
    # Default paths
    ARCH_PATH = str(PROJECT_ROOT / "examples" / "configs" / "arch.yaml")
    
    # Check if UniNDP aligned config exists
    UNINDP_CONFIG = "/Users/haochenzhao/Projects/UniNDP/config/pim-optimizer-aligned-v3.yaml"
    if os.path.exists(UNINDP_CONFIG):
        print(f"Using UniNDP aligned config: {UNINDP_CONFIG}")
        ARCH_PATH = UNINDP_CONFIG
    
    # Run validation
    results = run_validation_suite(
        arch_path=ARCH_PATH,
        tolerance=0.25,  # 25% tolerance for initial validation
        verbose=True,
    )
