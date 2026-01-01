#!/usr/bin/env python3
"""
分析公式 V3 的 MISMATCH 情况

MISMATCH: P_l3=1, Q_l3=2, C_l3=4, K_l3=8
  CQKP: Exact=8, Formula=64
  QCKP: Exact=8, Formula=64
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Workload:
    K: int = 16
    C: int = 16
    P: int = 16
    Q: int = 16


@dataclass 
class Mapping:
    P_l3: int
    Q_l3: int
    C_l3: int
    K_l3: int
    perm: Tuple[str, ...]


def simulate_detailed(m: Mapping):
    """详细模拟"""
    perm = m.perm
    ranges = {
        'P': range(m.P_l3), 'Q': range(m.Q_l3),
        'C': range(m.C_l3), 'K': range(m.K_l3)
    }
    
    print(f"\n循环顺序: {perm[0]} → {perm[1]} → {perm[2]} → {perm[3]}")
    print(f"范围: P∈[0,{m.P_l3}), Q∈[0,{m.Q_l3}), C∈[0,{m.C_l3}), K∈[0,{m.K_l3})")
    
    tile_changes = 0
    last_input = None
    iterations = []
    
    for i0 in ranges[perm[0]]:
        for i1 in ranges[perm[1]]:
            for i2 in ranges[perm[2]]:
                for i3 in ranges[perm[3]]:
                    indices = {perm[0]: i0, perm[1]: i1, perm[2]: i2, perm[3]: i3}
                    input_tile = (indices['C'], indices['P'], indices['Q'])
                    
                    is_new = input_tile != last_input
                    if is_new:
                        tile_changes += 1
                        last_input = input_tile
                    
                    iterations.append({
                        'k': indices['K'], 'c': indices['C'], 
                        'p': indices['P'], 'q': indices['Q'],
                        'input': input_tile, 'new': is_new
                    })
    
    # 显示迭代
    print(f"\n迭代序列 (前 30 个):")
    for i, it in enumerate(iterations[:30]):
        marker = "★" if it['new'] else ""
        print(f"  [{it['k']},{it['c']},{it['p']},{it['q']}] Input=({it['c']},{it['p']},{it['q']}) {marker}")
    
    if len(iterations) > 30:
        print(f"  ... (共 {len(iterations)} 次)")
    
    print(f"\n总 tile 切换: {tile_changes}")
    return tile_changes


def main():
    print("="*70)
    print("分析 MISMATCH 情况")
    print("="*70)
    
    # P_l3=1 是关键！因为 P 只有一个值，所以 P 的位置在循环里不会变
    P_l3, Q_l3, C_l3, K_l3 = 1, 2, 4, 8
    
    print(f"\nTiling: P_l3={P_l3}, Q_l3={Q_l3}, C_l3={C_l3}, K_l3={K_l3}")
    print(f"注意: P_l3=1 意味着 P 循环只有一个值!")
    
    # CQKP
    m1 = Mapping(P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3, perm=('C', 'Q', 'K', 'P'))
    print("\n" + "="*70)
    print("Case 1: CQKP")
    simulate_detailed(m1)
    
    # QCKP
    m2 = Mapping(P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3, perm=('Q', 'C', 'K', 'P'))
    print("\n" + "="*70)
    print("Case 2: QCKP")
    simulate_detailed(m2)
    
    # 对比一个正常情况
    m3 = Mapping(P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3, perm=('C', 'Q', 'P', 'K'))
    print("\n" + "="*70)
    print("Case 3: CQPK (K 在最内层)")
    simulate_detailed(m3)
    
    # 分析
    print("\n" + "="*70)
    print("分析")
    print("="*70)
    print("""
关键发现: 当 P_l3=1 时
  
  CQKP 循环: C → Q → K → P
    - P 循环只有 1 个值 (P=0)
    - K 循环有 8 个值
    - 但因为 P 固定，K 每次变化后 P 还是 0
    - 所以 Input tile = (C, P=0, Q) 
    - K 变化不会导致 Input 重新遍历！
    
  正确的公式应该考虑:
    如果 K 后面的 Input dims 的 _l3 值全是 1，则 K 不会导致重复
    
  修正公式 V4:
    inner_input_product = ∏(K之后的Input dims的_l3值)
    如果 inner_input_product == 1:
        tile_changes = P_l3 × Q_l3 × C_l3  (等价于 K 在最内层)
    否则:
        tile_changes = P_l3 × Q_l3 × C_l3 × K_l3
""")


if __name__ == '__main__':
    main()
