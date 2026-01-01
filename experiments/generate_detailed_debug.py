"""Generate detailed debug output for each workload."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "validation" / "dram"))

import os
from pim_optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
from trace_generator import TraceGenerator, DRAMConfig


def format_section_header(title, width=100):
    """Format section header."""
    return f"\n{'='*width}\n{title}\n{'='*width}\n"


def generate_trace_for_mapping(workload, mapping, dram_config):
    """Generate memory trace from mapping."""
    config = DRAMConfig(
        row_buffer_bytes=dram_config['row_buffer_bytes'],
        num_rows=dram_config['num_rows'],
        num_banks=dram_config['num_banks'],
        element_size=dram_config['element_size']
    )
    generator = TraceGenerator(workload, mapping, config)
    return generator.generate_trace()


def count_row_activations_from_trace(trace, dram_config):
    """Count row activations from trace."""
    row_buffer_bytes = dram_config['row_buffer_bytes']
    
    # Track current open row per bank
    bank_open_rows = {}  # bank_id -> row_id
    
    # Track per-tensor statistics
    tensor_stats = {
        'Input': {'bank': 0, 'total_accesses': 0, 'unique_addresses': set(), 
                  'unique_rows': set(), 'rows_accessed': set(), 
                  'row_activations': 0, 'row_visit_counts': {}},
        'Weight': {'bank': 1, 'total_accesses': 0, 'unique_addresses': set(),
                   'unique_rows': set(), 'rows_accessed': set(),
                   'row_activations': 0, 'row_visit_counts': {}},
        'Output': {'bank': 2, 'total_accesses': 0, 'unique_addresses': set(),
                   'unique_rows': set(), 'rows_accessed': set(),
                   'row_activations': 0, 'row_visit_counts': {}},
    }
    
    # Process trace
    for entry in trace:
        tensor = entry['tensor']
        addr = entry['address']
        bank_id = tensor_stats[tensor]['bank']
        row_id = addr // row_buffer_bytes
        
        # Update statistics
        stats = tensor_stats[tensor]
        stats['total_accesses'] += 1
        stats['unique_addresses'].add(addr)
        stats['unique_rows'].add(row_id)
        stats['rows_accessed'].add(row_id)
        
        # Check if row activation needed
        if bank_id not in bank_open_rows:
            # First access to this bank
            bank_open_rows[bank_id] = row_id
            stats['row_activations'] += 1
            stats['row_visit_counts'][row_id] = 1
        elif bank_open_rows[bank_id] != row_id:
            # Row switch - activate new row
            bank_open_rows[bank_id] = row_id
            stats['row_activations'] += 1
            stats['row_visit_counts'][row_id] = stats['row_visit_counts'].get(row_id, 0) + 1
    
    # Convert sets to counts for output
    for tensor in tensor_stats:
        stats = tensor_stats[tensor]
        stats['unique_addresses'] = len(stats['unique_addresses'])
        stats['unique_rows'] = len(stats['unique_rows'])
    
    return tensor_stats


def analyze_workload(workload_name, workload_config, output_dir):
    """Generate detailed debug output for a single workload."""
    
    output_file = output_dir / "analysis.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    original_stdout = sys.stdout
    
    try:
        with open(output_file, 'w') as f:
            sys.stdout = f
            
            print(format_section_header("1. WORKLOAD CONFIGURATION"))
            
            # Create workload
            workload = ConvWorkload(**workload_config)
            print(f"\n  Workload: {workload_name}")
            print(f"  Dimensions:")
            for dim, size in workload_config.items():
                print(f"    {dim}: {size}")
            
            # Calculate tensor sizes
            H = workload_config['P'] + workload_config['R'] - 1
            W = workload_config['Q'] + workload_config['S'] - 1
            input_size = workload_config['N'] * workload_config['C'] * H * W
            weight_size = workload_config['K'] * workload_config['C'] * workload_config['R'] * workload_config['S']
            output_size = workload_config['N'] * workload_config['K'] * workload_config['P'] * workload_config['Q']
            
            print(f"\n  Tensor Sizes (elements):")
            print(f"    Input:  N={workload_config['N']} × C={workload_config['C']} × H={H} × W={W} = {input_size}")
            print(f"    Weight: K={workload_config['K']} × C={workload_config['C']} × R={workload_config['R']} × S={workload_config['S']} = {weight_size}")
            print(f"    Output: N={workload_config['N']} × K={workload_config['K']} × P={workload_config['P']} × Q={workload_config['Q']} = {output_size}")
            
            print(f"\n  Relevancy Matrix:")
            print(f"    Input:  [R, S, P, Q, C, N]")
            print(f"    Weight: [R, S, C, K]")
            print(f"    Output: [P, Q, K, N]")
            
            print(f"\n  Dimension Divisors:")
            dim_names = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
            for i, dim in enumerate(dim_names):
                print(f"    {dim}: {workload.divisors[i]}")
            
            # Create optimizer
            print(format_section_header("2. OPTIMIZER SETUP & RESULTS"))
            
            arch_file = Path(__file__).parent / "configs" / "arch.yaml"
            optimizer = PIMOptimizer(str(arch_file))
            
            print(f"\n  Architecture: {arch_file}")
            print(f"  Memory Hierarchy:")
            for idx, level in enumerate(optimizer.arch.hierarchy):
                print(f"    Level {idx}: {level.name}, entries={level.entries}, blocksize={level.blocksize}")
            
            print(f"\n  DRAM Configuration:")
            # Get DRAM level from hierarchy
            dram_level = optimizer.arch.hierarchy.get_level("LocalDRAM")
            dram_config = {
                'row_buffer_bytes': 1024,  # Default row buffer size
                'num_rows': 16384,
                'num_banks': 4,
                'element_size': 1
            }
            print(f"    row_buffer_bytes: {dram_config['row_buffer_bytes']}")
            print(f"    num_rows: {dram_config['num_rows']}")
            print(f"    num_banks: {dram_config['num_banks']}")
            print(f"    element_size: {dram_config['element_size']}")
            
            # Optimize
            print(f"\n  Running Optimization...")
            result = optimizer.optimize(workload)
            
            print(f"\n  Optimization Result:")
            print(f"    Status: {result['status']}")
            print(f"    Solve Time: {result.get('solve_time', 'N/A'):.4f}s")
            
            print(format_section_header("3. MAPPING DETAILS"))
            
            mapping = result['mapping']
            
            print(f"\n  Layout Mode: {mapping['layout']}")
            
            print(f"\n  Tile Information:")
            print(f"    tile_info: {mapping.get('tile_info', {})}")
            
            print(f"\n  Loop Bounds (per level):")
            for level in range(4):
                bounds = mapping['loop_bounds'].get(level, {})
                if bounds:
                    print(f"    Level {level}:")
                    for dim, bound in sorted(bounds.items()):
                        print(f"      {dim}: {bound}")
            
            print(f"\n  Loop Permutation (per level):")
            for level in range(4):
                perm = mapping['loop_permutation'].get(level, {})
                if perm:
                    print(f"    Level {level}:")
                    for dim, pos in sorted(perm.items(), key=lambda x: x[1]):
                        print(f"      {dim}: position {pos}")
            
            print(f"\n  Tile Sizes (per tensor):")
            for tensor in ['Input', 'Weight', 'Output']:
                tile_sizes = mapping['tile_sizes'].get(tensor, {})
                if tile_sizes:
                    print(f"    {tensor}:")
                    for level, size in sorted(tile_sizes.items()):
                        print(f"      Level {level}: {size} bytes")
            
            print(format_section_header("4. DRAM LEVEL ANALYSIS"))
            
            # Analyze Level 2 + Level 3 factors
            print(f"\n  Level 2 (RowBuffer) Factors:")
            level2_bounds = mapping['loop_bounds'].get(2, {})
            for dim in workload_config.keys():
                factor = level2_bounds.get(dim, 1)
                print(f"    {dim}: {factor}")
            
            print(f"\n  Level 3 (LocalDRAM) Factors:")
            level3_bounds = mapping['loop_bounds'].get(3, {})
            for dim in workload_config.keys():
                factor = level3_bounds.get(dim, 1)
                print(f"    {dim}: {factor}")
            
            print(f"\n  Combined Level 2+3 Factors:")
            combined_factors = {}
            for dim in workload_config.keys():
                l2 = level2_bounds.get(dim, 1)
                l3 = level3_bounds.get(dim, 1)
                combined_factors[dim] = l2 * l3
                print(f"    {dim}: {l2} * {l3} = {combined_factors[dim]}")
            
            # Calculate reuse penalty
            relevancy = {
                'Input': ['R', 'S', 'P', 'Q', 'C', 'N'],
                'Weight': ['R', 'S', 'C', 'K'],
                'Output': ['P', 'Q', 'K', 'N']
            }
            print(f"\n  Reuse Penalty per Tensor (irrelevant DRAM factors):")
            for tensor in ['Input', 'Weight', 'Output']:
                relevant_dims = relevancy[tensor]
                irrelevant_dims = [d for d in workload_config.keys() 
                                 if d not in relevant_dims]
                penalty = 1
                for dim in irrelevant_dims:
                    penalty *= combined_factors.get(dim, 1)
                print(f"    {tensor}: irrelevant dims = {irrelevant_dims}, penalty = {penalty}")
            
            print(format_section_header("5. ILP ROW ACTIVATION PREDICTIONS"))
            
            print(f"\n  ILP Predicted Row Activations:")
            metrics = result.get('metrics', {})
            for tensor in ['Input', 'Weight', 'Output']:
                row_acts = metrics.get(f'row_acts_{tensor.lower()}_row_aligned', 0)
                print(f"    {tensor}: {row_acts:.4f}")
            
            total_row_acts = metrics.get('row_acts_row_aligned', 0)
            print(f"    Total: {total_row_acts:.4f}")
            
            print(format_section_header("6. TRACE GENERATION DETAILS"))
            
            print(f"\n  DRAM Config:")
            print(f"    row_buffer_bytes: {dram_config['row_buffer_bytes']}")
            print(f"    num_banks: {dram_config['num_banks']}")
            print(f"    num_rows: {dram_config['num_rows']}")
            print(f"    element_size: {dram_config['element_size']}")
            bank_size = dram_config['num_rows'] * dram_config['row_buffer_bytes']
            print(f"    bank_size: {bank_size} bytes ({bank_size // (1024*1024)} MB)")
            
            # Generate trace
            print(f"\n  Generating trace...")
            trace = generate_trace_for_mapping(workload, mapping, dram_config)
            
            print(f"\n  Trace Statistics:")
            print(f"    Total trace lines: {len(trace)}")
            
            # Count row activations
            row_acts_per_tensor = count_row_activations_from_trace(trace, dram_config)
            
            for tensor in ['Input', 'Weight', 'Output']:
                if tensor in row_acts_per_tensor:
                    info = row_acts_per_tensor[tensor]
                    print(f"\n  {tensor} (Bank {info['bank']}):")
                    print(f"    Total accesses: {info['total_accesses']}")
                    print(f"    Unique addresses: {info['unique_addresses']}")
                    print(f"    Unique rows: {info['unique_rows']}")
                    print(f"    Rows accessed: {sorted(info['rows_accessed'])}")
                    print(f"    Row activations (switches): {info['row_activations']}")
                    
                    # Show row visit pattern
                    print(f"    Row visit pattern:")
                    for row in sorted(info['rows_accessed']):
                        count = info['row_visit_counts'].get(row, 0)
                        if count > 0:
                            print(f"      Row {row}: activated {count} times")
            
            print(format_section_header("7. SAMPLE TRACE ENTRIES (First 100)"))
            
            # Show first 100 trace entries
            prev_row = {}
            for i, entry in enumerate(trace[:100]):
                tensor = entry['tensor']
                addr = entry['address']
                row = addr // dram_config['row_buffer_bytes']
                col = addr % dram_config['row_buffer_bytes']
                
                marker = ""
                if tensor not in prev_row or prev_row[tensor] != row:
                    marker = " <-- NEW ROW"
                    prev_row[tensor] = row
                
                op = "LD" if entry['operation'] == 'read' else "ST"
                print(f"{i:6d}: {op} 0x{addr:08X} -> {tensor:<6s} Row={row:3d} Col={col:4d}{marker}")
            
            print(format_section_header("8. DISCREPANCY ANALYSIS"))
            
            print(f"\n  Comparison:")
            print(f"  {'Tensor':<10s} {'ILP':<15s} {'Trace':<15s} {'Ratio (Trace/ILP)':<20s}")
            print(f"  {'-'*60}")
            
            for tensor in ['Input', 'Weight', 'Output']:
                ilp_val = metrics.get(f'row_acts_{tensor.lower()}_row_aligned', 0)
                trace_val = row_acts_per_tensor.get(tensor, {}).get('row_activations', 0)
                ratio = trace_val / ilp_val if ilp_val > 0 else float('inf')
                print(f"  {tensor:<10s} {ilp_val:<15.2f} {trace_val:<15d} {ratio:<20.2f}")
            
            print(f"\n  Key Observations:")
            print(f"    - ILP computes row_acts based on DRAM level tiling factors")
            print(f"    - Trace counts actual row switches during execution")
            print(f"    - The large discrepancy suggests:")
            print(f"      1. ILP model may only consider Level 3 factors, not Level 2+3")
            print(f"      2. Or trace generator iterates both Level 2 and Level 3 loops")
            print(f"      3. Or the formula for row_aligned mode is incorrect")
            
    finally:
        sys.stdout = original_stdout
        
    print(f"Generated: {output_file}")


def main():
    """Generate debug output for all workloads."""
    
    # Define workloads
    workloads = {
        'tiny': {
            'R': 3, 'S': 3, 'P': 8, 'Q': 8,
            'C': 8, 'K': 4, 'N': 1
        },
        'small': {
            'R': 3, 'S': 3, 'P': 16, 'Q': 16,
            'C': 16, 'K': 16, 'N': 1
        },
        'medium': {
            'R': 3, 'S': 3, 'P': 56, 'Q': 56,
            'C': 64, 'K': 64, 'N': 1
        },
        'ResNet-L1': {
            'R': 7, 'S': 7, 'P': 112, 'Q': 112,
            'C': 3, 'K': 64, 'N': 1
        },
        'ResNet-L2': {
            'R': 3, 'S': 3, 'P': 56, 'Q': 56,
            'C': 64, 'K': 64, 'N': 1
        },
        'ResNet-L3': {
            'R': 3, 'S': 3, 'P': 28, 'Q': 28,
            'C': 128, 'K': 128, 'N': 1
        },
        'VGG-L1': {
            'R': 3, 'S': 3, 'P': 224, 'Q': 224,
            'C': 3, 'K': 64, 'N': 1
        },
        'MobileNet-L1': {
            'R': 3, 'S': 3, 'P': 112, 'Q': 112,
            'C': 32, 'K': 32, 'N': 1
        },
    }
    
    # Create output directory
    output_dir = Path(__file__).parent / "debug_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating detailed debug output for {len(workloads)} workloads...")
    print(f"Output directory: {output_dir}")
    print()
    
    # Process each workload
    for workload_name, config in workloads.items():
        print(f"Processing {workload_name}...")
        workload_dir = output_dir / workload_name
        try:
            analyze_workload(workload_name, config, workload_dir)
        except Exception as e:
            print(f"Error processing {workload_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("Done! Check the debug_output directory for results.")


if __name__ == '__main__':
    main()
