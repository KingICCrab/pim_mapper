#!/usr/bin/env python3
"""
重新推导正确的公式

关键观察:
  KCPQ: Exact=16, K 不在最内 → K 每变化一次，(C,P,Q) 全部重新遍历
  CKPQ: Exact=16, K 不在最内 → K 每变化一次，(P,Q) 重新遍历
  CPKQ: Exact=16, K 不在最内 → K 每变化一次，(Q) 重新遍历
  CPQK: Exact=8,  K 在最内  → K 变化不引起 Input 切换

问题: 为什么 KCPQ, CKPQ, CPKQ 的 Exact 都是 16？

分析 KCPQ (K→C→P→Q):
  K=0: C=0,P=0,Q=0 → C=0,P=0,Q=1 → ... → C=1,P=1,Q=1 (8 tiles)
  K=1: C=0,P=0,Q=0 → ... → C=1,P=1,Q=1 (又 8 tiles, 全部重新访问!)
  总计: 16 tiles

分析 CKPQ (C→K→P→Q):
  C=0: K=0: P=0,Q=0 → P=0,Q=1 → P=1,Q=0 → P=1,Q=1 (4 tiles, C=0)
       K=1: P=0,Q=0 → ... → P=1,Q=1 (又 4 tiles, 重新访问!)
  C=1: K=0: P=0,Q=0 → ... → P=1,Q=1 (4 tiles, C=1)
       K=1: P=0,Q=0 → ... → P=1,Q=1 (又 4 tiles, 重新访问!)
  总计: 4 × 4 = 16 tiles

分析 CPKQ (C→P→K→Q):
  C=0,P=0: K=0: Q=0 → Q=1 (2 tiles)
           K=1: Q=0 → Q=1 (又 2 tiles, 重新访问!)
  C=0,P=1: ... (同上, 4 tiles)
  C=1,P=0: ... (4 tiles)
  C=1,P=1: ... (4 tiles)
  总计: 16 tiles

关键发现:
  只要 K 不在最内层，K 的每次变化都会导致其内部的 Input dims 被重新遍历
  
  设 K 之后的 Input dims 乘积为 inner_input_product
  设 K 之前的 Input dims 乘积为 outer_input_product
  
  If K 在最内层:
      tile_changes = outer_input_product × 1 = C_l3 × P_l3 × Q_l3
  Else:
      tile_changes = outer_input_product × inner_input_product × K_l3
      
  因为 outer × inner = C_l3 × P_l3 × Q_l3
  所以 tile_changes = C_l3 × P_l3 × Q_l3 × K_l3 (当 K 不在最内)
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Workload:
    name: str
    K: int
    C: int
    P: int
    Q: int
    R: int
    S: int


@dataclass 
class Mapping:
    P_l3: int
    Q_l3: int
    C_l3: int
    K_l3: int
    perm: Tuple[str, ...]
    block_w: int
    workload: 'Workload'


def compute_exact(m: Mapping) -> int:
    """精确模拟"""
    perm = m.perm
    ranges = {
        'P': range(m.P_l3), 'Q': range(m.Q_l3),
        'C': range(m.C_l3), 'K': range(m.K_l3)
    }
    
    tile_changes = 0
    last_input = None
    
    for i0 in ranges[perm[0]]:
        for i1 in ranges[perm[1]]:
            for i2 in ranges[perm[2]]:
                for i3 in ranges[perm[3]]:
                    indices = {perm[0]: i0, perm[1]: i1, perm[2]: i2, perm[3]: i3}
                    input_tile = (indices['C'], indices['P'], indices['Q'])
                    
                    if input_tile != last_input:
                        tile_changes += 1
                        last_input = input_tile
    
    return tile_changes


def compute_formula_v3(m: Mapping) -> int:
    """
    正确的公式 V3
    
    如果 K 在最内层: tile_changes = P_l3 × Q_l3 × C_l3
    否则: tile_changes = P_l3 × Q_l3 × C_l3 × K_l3
    """
    base = m.P_l3 * m.Q_l3 * m.C_l3
    k_pos = m.perm.index('K')
    
    if k_pos == 3:  # K 在最内层
        return base
    else:
        return base * m.K_l3


def main():
    print("="*70)
    print("验证公式 V3")
    print("="*70)
    
    # 测试不同 workload 和 tiling
    test_cases = [
        # (P_l3, Q_l3, C_l3, K_l3)
        (2, 2, 2, 2),
        (4, 4, 2, 2),
        (2, 2, 4, 4),
        (1, 2, 4, 8),
    ]
    
    all_perms = [
        ('K', 'C', 'P', 'Q'), ('K', 'C', 'Q', 'P'), ('K', 'P', 'C', 'Q'), ('K', 'P', 'Q', 'C'),
        ('K', 'Q', 'C', 'P'), ('K', 'Q', 'P', 'C'),
        ('C', 'K', 'P', 'Q'), ('C', 'K', 'Q', 'P'), ('C', 'P', 'K', 'Q'), ('C', 'P', 'Q', 'K'),
        ('C', 'Q', 'K', 'P'), ('C', 'Q', 'P', 'K'),
        ('P', 'K', 'C', 'Q'), ('P', 'K', 'Q', 'C'), ('P', 'C', 'K', 'Q'), ('P', 'C', 'Q', 'K'),
        ('P', 'Q', 'K', 'C'), ('P', 'Q', 'C', 'K'),
        ('Q', 'K', 'C', 'P'), ('Q', 'K', 'P', 'C'), ('Q', 'C', 'K', 'P'), ('Q', 'C', 'P', 'K'),
        ('Q', 'P', 'K', 'C'), ('Q', 'P', 'C', 'K'),
    ]
    
    w = Workload(name='Test', K=16, C=16, P=16, Q=16, R=3, S=3)
    
    total_tests = 0
    passed = 0
    
    for P_l3, Q_l3, C_l3, K_l3 in test_cases:
        print(f"\n--- Tiling: P_l3={P_l3}, Q_l3={Q_l3}, C_l3={C_l3}, K_l3={K_l3} ---")
        
        for perm in all_perms:
            m = Mapping(
                P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3,
                perm=perm, block_w=512, workload=w
            )
            
            exact = compute_exact(m)
            formula = compute_formula_v3(m)
            
            total_tests += 1
            if exact == formula:
                passed += 1
            else:
                perm_str = ''.join([p[0] for p in perm])
                print(f"  MISMATCH: {perm_str}, Exact={exact}, Formula={formula}")
    
    print(f"\n{'='*70}")
    print(f"Results: {passed}/{total_tests} passed ({100*passed/total_tests:.1f}%)")
    print(f"{'='*70}")
    
    if passed == total_tests:
        print("\n✓ 公式 V3 完全正确!")
        print("""
最终公式:
  If K 在最内层 (位置 3):
      Input_tile_changes = P_l3 × Q_l3 × C_l3
  Else:
      Input_tile_changes = P_l3 × Q_l3 × C_l3 × K_l3
      
Row_switches = Input_tile_changes × rows_per_tile
""")
    else:
        print("\n✗ 公式 V3 有问题，需要继续分析")


if __name__ == '__main__':
    main()
