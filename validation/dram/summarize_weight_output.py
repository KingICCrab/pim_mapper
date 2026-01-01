import os
from pathlib import Path
import re

def parse_analysis_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the Comparison section
    match = re.search(r"Comparison:(.*?)Key Observations:", content, re.DOTALL)
    if not match:
        return None
    
    table_text = match.group(1)
    lines = table_text.strip().split('\n')
    
    results = {}
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            tensor = parts[0]
            if tensor in ['Weight', 'Output']:
                try:
                    ilp = float(parts[1])
                    trace = float(parts[2])
                    results[tensor] = (ilp, trace)
                except ValueError:
                    continue
    return results

def main():
    base_dir = Path("/Users/haochenzhao/Projects/pim_optimizer/validation/dram/debug_output")
    workloads = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    
    print(f"{'Workload':<15} | {'Tensor':<8} | {'ILP':<10} | {'Trace':<10} | {'Error %':<10}")
    print("-" * 65)
    
    for workload_dir in workloads:
        workload_name = workload_dir.name
        analysis_file = workload_dir / "analysis.txt"
        
        if not analysis_file.exists():
            continue
            
        results = parse_analysis_file(analysis_file)
        if not results:
            continue
            
        for tensor in ['Weight', 'Output']:
            if tensor in results:
                ilp, trace = results[tensor]
                if trace == 0:
                    error = 0.0 if ilp == 0 else 100.0
                else:
                    error = abs(ilp - trace) / trace * 100
                
                print(f"{workload_name:<15} | {tensor:<8} | {ilp:<10.2f} | {trace:<10.0f} | {error:<10.2f}%")
        print("-" * 65)

if __name__ == "__main__":
    main()
