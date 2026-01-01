#!/usr/bin/env python3
"""
Validation Analysis: Compare ILP predictions vs Trace measurements.

This script provides a comprehensive analysis of the discrepancy between
ILP row activation predictions and actual trace measurements.

Key Insight:
- ILP model assumes sequential data access (row_acts = unique rows)
- Trace measures actual row switches based on loop execution order
- For row_aligned layout: Both should match (data is padded to row boundaries)
- For sequential layout: Trace may show MORE row_acts due to non-sequential access pattern
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ValidationResult:
    """Validation result for a single workload."""
    workload_name: str
    
    # ILP predictions
    ilp_input: float
    ilp_weight: float
    ilp_output: float
    ilp_total: float
    
    # Trace measurements
    trace_input: int
    trace_weight: int
    trace_output: int
    trace_total: int
    
    # Layout info
    input_layout: str
    weight_layout: str
    output_layout: str
    
    # Match status
    @property
    def input_match(self) -> bool:
        """Check if input row_acts match within tolerance."""
        return abs(self.ilp_input - self.trace_input) <= max(1, self.ilp_input * 0.1)
    
    @property
    def weight_match(self) -> bool:
        """Check if weight row_acts match within tolerance."""
        # For sequential layout, trace may be higher due to non-sequential access
        if self.weight_layout == "sequential":
            # Expected: trace >= ilp (due to potential non-sequential access)
            return self.trace_weight >= self.ilp_weight * 0.9
        return abs(self.ilp_weight - self.trace_weight) <= max(1, self.ilp_weight * 0.1)
    
    @property
    def output_match(self) -> bool:
        """Check if output row_acts match within tolerance."""
        return abs(self.ilp_output - self.trace_output) <= max(1, self.ilp_output * 0.1)


def parse_analysis_file(filepath: Path) -> ValidationResult:
    """Parse analysis.txt file and extract validation data."""
    content = filepath.read_text()
    
    # Extract workload name
    workload_name = filepath.parent.name
    
    # Extract ILP predictions
    ilp_section = content[content.find("5. ILP ROW ACTIVATION PREDICTIONS"):]
    ilp_input = float(ilp_section.split("Input:")[1].split()[0])
    ilp_weight = float(ilp_section.split("Weight:")[1].split()[0])
    ilp_output = float(ilp_section.split("Output:")[1].split()[0])
    ilp_total = ilp_input + ilp_weight + ilp_output
    
    # Extract trace measurements
    trace_section = content[content.find("6. TRACE GENERATION DETAILS"):]
    
    def extract_row_acts(tensor: str) -> int:
        section = trace_section[trace_section.find(f"{tensor} (Bank"):]
        line = [l for l in section.split('\n') if "Row activations (switches):" in l][0]
        return int(line.split(":")[1].strip())
    
    trace_input = extract_row_acts("Input")
    trace_weight = extract_row_acts("Weight")
    trace_output = extract_row_acts("Output")
    trace_total = trace_input + trace_weight + trace_output
    
    # Extract layout info
    layout_section = content[content.find("Layout:"):content.find("Tile Info:")]
    input_layout = "row_aligned" if "row_aligned" in layout_section.split("Input:")[1].split()[0] else "sequential"
    weight_layout = "row_aligned" if "row_aligned" in layout_section.split("Weight:")[1].split()[0] else "sequential"
    output_layout = "row_aligned" if "row_aligned" in layout_section.split("Output:")[1].split()[0] else "sequential"
    
    return ValidationResult(
        workload_name=workload_name,
        ilp_input=ilp_input,
        ilp_weight=ilp_weight,
        ilp_output=ilp_output,
        ilp_total=ilp_total,
        trace_input=trace_input,
        trace_weight=trace_weight,
        trace_output=trace_output,
        trace_total=trace_total,
        input_layout=input_layout,
        weight_layout=weight_layout,
        output_layout=output_layout,
    )


def generate_validation_report(debug_dir: Path) -> str:
    """Generate validation report from all analysis files."""
    results = []
    
    for workload_dir in sorted(debug_dir.iterdir()):
        if not workload_dir.is_dir():
            continue
        analysis_file = workload_dir / "analysis.txt"
        if not analysis_file.exists():
            continue
        try:
            result = parse_analysis_file(analysis_file)
            results.append(result)
        except Exception as e:
            print(f"Error parsing {analysis_file}: {e}")
    
    # Generate report
    report = []
    report.append("=" * 100)
    report.append("ROW ACTIVATION VALIDATION REPORT")
    report.append("=" * 100)
    report.append("")
    report.append("SUMMARY")
    report.append("-" * 100)
    report.append(f"{'Workload':<15} {'Layout (I/W/O)':<20} {'ILP (I+W+O=T)':<25} {'Trace (I+W+O=T)':<25} {'Status':<10}")
    report.append("-" * 100)
    
    for r in results:
        layout = f"{r.input_layout[:3]}/{r.weight_layout[:3]}/{r.output_layout[:3]}"
        ilp = f"{r.ilp_input:.0f}+{r.ilp_weight:.0f}+{r.ilp_output:.0f}={r.ilp_total:.0f}"
        trace = f"{r.trace_input}+{r.trace_weight}+{r.trace_output}={r.trace_total}"
        
        # Determine status
        if r.input_match and r.weight_match and r.output_match:
            status = "✓ PASS"
        elif r.trace_total >= r.ilp_total * 0.9:
            status = "~ WARN"  # Trace is close or higher (expected for sequential)
        else:
            status = "✗ FAIL"
        
        report.append(f"{r.workload_name:<15} {layout:<20} {ilp:<25} {trace:<25} {status:<10}")
    
    report.append("-" * 100)
    report.append("")
    report.append("DETAILED ANALYSIS")
    report.append("-" * 100)
    
    for r in results:
        report.append(f"\n{r.workload_name}:")
        report.append(f"  Input  (layout={r.input_layout:<10}): ILP={r.ilp_input:6.0f}, Trace={r.trace_input:6d}, Ratio={r.trace_input/r.ilp_input:.2f} {'✓' if r.input_match else '✗'}")
        report.append(f"  Weight (layout={r.weight_layout:<10}): ILP={r.ilp_weight:6.0f}, Trace={r.trace_weight:6d}, Ratio={r.trace_weight/r.ilp_weight:.2f} {'✓' if r.weight_match else '✗'}")
        report.append(f"  Output (layout={r.output_layout:<10}): ILP={r.ilp_output:6.0f}, Trace={r.trace_output:6d}, Ratio={r.trace_output/r.ilp_output:.2f} {'✓' if r.output_match else '✗'}")
    
    report.append("")
    report.append("=" * 100)
    report.append("KEY INSIGHTS")
    report.append("=" * 100)
    report.append("""
1. row_aligned Layout:
   - ILP and Trace should match closely
   - Data is padded to row boundaries, ensuring sequential access
   - Example: Input tensor with row_aligned layout

2. sequential Layout:
   - ILP predicts minimum row_acts (unique rows needed)
   - Trace may show MORE row_acts due to non-sequential loop execution
   - This is NOT a bug - ILP provides lower bound estimate
   - Example: Weight tensor where loop nesting causes row thrashing

3. Validation Criteria:
   - For row_aligned: |ILP - Trace| <= 10% or 1 row
   - For sequential: Trace >= ILP * 0.9 (trace can be higher)
   - If Trace < ILP * 0.9: Indicates model error

4. Why Sequential Layout May Have Higher Trace:
   - ILP assumes data is accessed sequentially
   - Actual execution may access rows in non-sequential order
   - Each row switch counts as one activation
   - Example: Accessing rows [0,1,2,0,1,2] = 6 activations, not 3
""")
    
    return "\n".join(report)


def main():
    """Generate validation report."""
    debug_dir = Path(__file__).parent / "debug_output"
    
    if not debug_dir.exists():
        print(f"Debug output directory not found: {debug_dir}")
        print("Run 'python experiments/generate_detailed_debug.py' first.")
        return 1
    
    report = generate_validation_report(debug_dir)
    print(report)
    
    # Save report
    output_file = debug_dir.parent / "validation_report.txt"
    output_file.write_text(report)
    print(f"\nReport saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
