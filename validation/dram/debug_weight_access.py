#!/usr/bin/env python3
"""
Debug Weight access pattern for small workload.

Analyze why ILP predicts 3 row_acts but trace shows 7.
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
import numpy as np


def analyze_weight_access():
    """Analyze Weight access pattern for small workload."""
    
    # Small workload parameters
    K, C, R, S = 16, 16, 3, 3
    
    print("=" * 80)
    print("WEIGHT ACCESS PATTERN ANALYSIS")
    print("=" * 80)
    
    # Weight size
    weight_size = K * C * R * S
    print(f"\nWeight Tensor: K={K} × C={C} × R={R} × S={S} = {weight_size} elements")
    
    # DRAM row size
    row_size = 1024
    num_rows = (weight_size + row_size - 1) // row_size
    print(f"Row size: {row_size} elements")
    print(f"Weight occupies {num_rows} rows: [0, {num_rows-1}]")
    
    # ILP mapping result
    print("\n" + "=" * 80)
    print("ILP MAPPING (from analysis.txt)")
    print("=" * 80)
    print("Level 3 temporal: K=4, C=8")
    print("Permutation: K -> C (C is inner loop)")
    print("Layout: sequential")
    
    # Weight memory layout (standard NCHW order: K x C x R x S)
    # addr = k * (C * R * S) + c * (R * S) + r * S + s
    def weight_addr(k, c, r, s):
        return k * (C * R * S) + c * (R * S) + r * S + s
    
    print("\n" + "=" * 80)
    print("WEIGHT MEMORY LAYOUT")
    print("=" * 80)
    print(f"addr = k * {C * R * S} + c * {R * S} + r * {S} + s")
    print("\nSample addresses:")
    for k in range(min(4, K)):
        for c in range(min(4, C)):
            addr = weight_addr(k, c, 0, 0)
            row = addr // row_size
            print(f"  k={k}, c={c}, r=0, s=0: addr={addr}, row={row}")
    
    # Now simulate Level 3 loop execution
    print("\n" + "=" * 80)
    print("LEVEL 3 LOOP EXECUTION (K=4, C=8)")
    print("=" * 80)
    print("Loop order: for k in range(4): for c in range(8)")
    print()
    
    # Track row activations
    current_row = None
    row_acts = 0
    accesses = []
    
    for k in range(4):  # Level 3 K
        for c in range(8):  # Level 3 C
            for r in range(3):  # All R (not Level 3)
                for s in range(3):  # All S (not Level 3)
                    addr = weight_addr(k, c, r, s)
                    row = addr // row_size
                    
                    if current_row != row:
                        row_acts += 1
                        current_row = row
                    
                    accesses.append((k, c, r, s, addr, row))
    
    print(f"Total accesses: {len(accesses)}")
    print(f"Row activations (switches): {row_acts}")
    
    # Show first few accesses
    print("\nFirst 50 accesses:")
    for i, (k, c, r, s, addr, row) in enumerate(accesses[:50]):
        marker = "<-- NEW ROW" if i == 0 or accesses[i-1][5] != row else ""
        print(f"  k={k}, c={c}, r={r}, s={s}: addr={addr:4d}, row={row} {marker}")
    
    # Unique rows
    unique_rows = set(a[5] for a in accesses)
    print(f"\nUnique rows accessed: {sorted(unique_rows)}")
    print(f"Unique row count: {len(unique_rows)}")
    print(f"Row activations: {row_acts}")
    
    # Count how many times each row is activated
    row_visits = {}
    current_row = None
    for k, c, r, s, addr, row in accesses:
        if current_row != row:
            row_visits[row] = row_visits.get(row, 0) + 1
            current_row = row
    
    print("\nRow visit pattern:")
    for row in sorted(row_visits.keys()):
        print(f"  Row {row}: activated {row_visits[row]} times")
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("ILP predicts row_acts = 3 (unique rows = 3)")
    print(f"Trace shows row_acts = {row_acts} (row switches)")
    print()
    print("The discrepancy is because ILP assumes sequential access,")
    print("but actual loop nesting causes non-sequential access pattern.")
    
    # What if we consider only Level 3 C and K (ignoring inner R, S)?
    print("\n" + "=" * 80)
    print("CONSIDERING ONLY LEVEL 3 LOOPS (ignoring inner R, S)")
    print("=" * 80)
    
    current_row = None
    row_acts = 0
    accesses = []
    
    for k in range(4):  # Level 3 K
        for c in range(8):  # Level 3 C
            # Base address for this tile (r=0, s=0)
            addr = weight_addr(k, c, 0, 0)
            row = addr // row_size
            
            if current_row != row:
                row_acts += 1
                current_row = row
            
            accesses.append((k, c, addr, row))
    
    print(f"Total Level 3 iterations: {len(accesses)}")
    print(f"Row activations: {row_acts}")
    
    print("\nLevel 3 iterations:")
    for i, (k, c, addr, row) in enumerate(accesses):
        marker = "<-- NEW ROW" if i == 0 or accesses[i-1][3] != row else ""
        print(f"  k={k}, c={c}: addr={addr:4d}, row={row} {marker}")
    
    unique_rows = set(a[3] for a in accesses)
    print(f"\nUnique rows: {sorted(unique_rows)}")
    print(f"Row activations: {row_acts}")


if __name__ == "__main__":
    analyze_weight_access()
