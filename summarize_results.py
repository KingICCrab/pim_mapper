import os
from pathlib import Path

output_dir = Path('/Users/haochenzhao/Projects/pim_optimizer/validation/dram/debug_output/')

print("=" * 100)
print("SUMMARY (Weight & Output Comparison)")
print("=" * 100)
print(f"{'Workload':<15} | {'ILP Weight':<12} {'Trace Weight':<12} | {'ILP Output':<12} {'Trace Output':<12}")
print("-" * 100)

workloads = sorted([d.name for d in output_dir.iterdir() if d.is_dir()])

for workload in workloads:
    analysis_file = output_dir / workload / "analysis.txt"
    if not analysis_file.exists():
        continue
        
    ilp_weight = 0
    trace_weight = 0
    ilp_output = 0
    trace_output = 0
    
    with open(analysis_file, 'r') as f:
        lines = f.readlines()
        
    in_comparison = False
    for line in lines:
        if "Comparison:" in line:
            in_comparison = True
            continue
        
        if in_comparison:
            parts = line.split()
            if len(parts) >= 3:
                if parts[0] == "Weight":
                    ilp_weight = float(parts[1])
                    trace_weight = int(parts[2])
                elif parts[0] == "Output":
                    ilp_output = float(parts[1])
                    trace_output = int(parts[2])
    
    print(f"{workload:<15} | {ilp_weight:<12.2f} {trace_weight:<12d} | {ilp_output:<12.2f} {trace_output:<12d}")
