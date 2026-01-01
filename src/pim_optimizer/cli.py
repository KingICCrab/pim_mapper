"""
Command-line interface for PIM optimizer.
"""

import argparse
import sys
import os
import yaml

from pim_optimizer.optimizer import PIMOptimizer, run_optimization
from pim_optimizer.arch import PIMArchitecture
from pim_optimizer.workload import ConvWorkload


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PIM Optimizer - Dataflow mapping optimization for PIM architectures"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # =========================================
    # optimize command
    # =========================================
    opt_parser = subparsers.add_parser(
        "optimize",
        help="Run optimization for workloads"
    )
    opt_parser.add_argument(
        "-a", "--arch",
        required=True,
        help="Path to architecture YAML file"
    )
    opt_parser.add_argument(
        "-w", "--workloads",
        nargs="+",
        required=True,
        help="Paths to workload YAML files"
    )
    opt_parser.add_argument(
        "-o", "--output",
        help="Output file for results (YAML format)"
    )
    opt_parser.add_argument(
        "--objective",
        choices=["latency", "energy", "blended"],
        default="latency",
        help="Optimization objective (default: latency)"
    )
    opt_parser.add_argument(
        "--time-limit",
        type=float,
        default=300.0,
        help="Solver time limit in seconds (default: 300)"
    )
    opt_parser.add_argument(
        "--mip-gap",
        type=float,
        default=0.01,
        help="MIP optimality gap tolerance (default: 0.01)"
    )
    opt_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    opt_parser.add_argument(
        "--no-row-activation",
        action="store_true",
        help="Disable row activation modeling"
    )
    opt_parser.add_argument(
        "--fix-permutations",
        action="store_true",
        help="Fix permutations across workloads"
    )
    opt_parser.add_argument(
        "--fix-bypass",
        action="store_true",
        help="Fix bypass across workloads"
    )
    opt_parser.add_argument(
        "--optimize-bypass",
        action="store_true",
        help="Optimize bypass decisions"
    )
    
    # =========================================
    # info command
    # =========================================
    info_parser = subparsers.add_parser(
        "info",
        help="Display architecture or workload information"
    )
    info_parser.add_argument(
        "-a", "--arch",
        help="Path to architecture YAML file"
    )
    info_parser.add_argument(
        "-w", "--workload",
        help="Path to workload YAML file"
    )
    
    # =========================================
    # crossing command - analyze crossing ratios
    # =========================================
    cross_parser = subparsers.add_parser(
        "crossing",
        help="Analyze crossing ratios for tile configurations"
    )
    cross_parser.add_argument(
        "--block-h",
        type=int,
        default=64,
        help="Memory block height"
    )
    cross_parser.add_argument(
        "--block-w",
        type=int,
        default=256,
        help="Memory block width"
    )
    cross_parser.add_argument(
        "--tile-h",
        type=int,
        required=True,
        help="Tile height"
    )
    cross_parser.add_argument(
        "--tile-w",
        type=int,
        default=None,
        help="Tile width"
    )
    cross_parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Tile step (for inputs: q_factor × stride)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    # =========================================
    # Execute command
    # =========================================
    if args.command == "optimize":
        return cmd_optimize(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "crossing":
        return cmd_crossing(args)
    else:
        parser.print_help()
        return 1


def cmd_optimize(args) -> int:
    """Execute optimize command."""
    print(f"PIM Optimizer")
    print("=" * 60)
    print(f"Architecture: {args.arch}")
    print(f"Workloads: {args.workloads}")
    print(f"Objective: {args.objective}")
    print(f"Time limit: {args.time_limit}s")
    print("=" * 60)
    
    try:
        result = run_optimization(
            arch_file=args.arch,
            workload_files=args.workloads,
            objective=args.objective,
            verbose=args.verbose,
            time_limit=args.time_limit,
            fix_permutations=args.fix_permutations,
            fix_bypass=args.fix_bypass,
            optimize_bypass=args.optimize_bypass,
            enable_row_activation=not args.no_row_activation,
        )
        
        # Print summary
        result.print_summary()
        
        # Print each mapping
        for i, mapping in enumerate(result.mappings):
            print(f"\n--- Workload {i}: {mapping.workload_name} ---")
            print(mapping.pretty_print())
        
        # Save results if output specified
        if args.output:
            output_data = {
                "status": result.solver_status,
                "solve_time": result.solve_time,
                "summary": result.summary,
                "mappings": [m.to_dict() for m in result.mappings],
            }
            
            with open(args.output, "w") as f:
                yaml.dump(output_data, f, default_flow_style=False)
            
            print(f"\nResults saved to: {args.output}")
        
        return 0 if result.is_optimal else 1
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_info(args) -> int:
    """Execute info command."""
    if args.arch:
        print("Architecture Information")
        print("=" * 60)
        arch = PIMArchitecture.from_yaml(args.arch)
        arch.print_info()
    
    if args.workload:
        print("\nWorkload Information")
        print("=" * 60)
        workload = ConvWorkload.from_yaml(args.workload)
        print(f"Name: {workload.name}")
        print(f"Dimensions: R={workload.R}, S={workload.S}, P={workload.P}, Q={workload.Q}")
        print(f"            C={workload.C}, K={workload.K}, N={workload.N}")
        print(f"Stride: {workload.stride}")
        print(f"Dilation: {workload.dilation}")
        print(f"MACs: {workload.macs:,}")
        
        print("\nDivisors:")
        dim_names = ["R", "S", "P", "Q", "C", "K", "N"]
        for i, divs in enumerate(workload.divisors):
            print(f"  {dim_names[i]}: {divs}")
    
    return 0


def cmd_crossing(args) -> int:
    """Execute crossing analysis command."""
    from pim_optimizer.model.crossing import (
        compute_block_crossing_ratio_gcd,
        analyze_crossing_pattern,
    )
    
    print("Block Crossing Ratio Analysis")
    print("(Layout block boundaries, NOT DRAM row boundaries)")
    print("=" * 60)
    
    crossing_ratio, g, period, cross_count = compute_block_crossing_ratio_gcd(
        args.block_h, args.tile_h, args.step
    )
    
    print(f"Block height: {args.block_h}")
    print(f"Tile height:  {args.tile_h}")
    print(f"Step:         {args.step}")
    print()
    print(f"GCD(step, block_h) = GCD({args.step}, {args.block_h}) = {g}")
    print(f"Period = block_h / GCD = {args.block_h} / {g} = {period}")
    print(f"Cross count = {cross_count}")
    print(f"Crossing ratio = {cross_count}/{period} = {crossing_ratio:.4f}")
    print()
    
    # Detailed analysis
    analysis = analyze_crossing_pattern(
        args.block_h, args.tile_h, args.step, num_iterations=period
    )
    
    print("Positions within one period:")
    for i, (pos, crosses) in enumerate(zip(analysis["positions"], analysis["crossings"])):
        status = "CROSSES" if crosses else "safe"
        print(f"  {i}: start={pos:3d}, end={pos + args.tile_h:3d} -> {status}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
