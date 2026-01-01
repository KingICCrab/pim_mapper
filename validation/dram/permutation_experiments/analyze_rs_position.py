#!/usr/bin/env python3
"""
深入分析: R 位置对 Row Activation 的影响

发现: 当 R_l2 > 1 时 (R 在 L3 层展开)，R 的位置会影响 Input tile 切换次数

类似 K 的分析:
- R 影响 Input (因为不同 R 访问不同的 Input 行)
- 但 R 与 Weight 的关系更复杂
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

import math
from dataclasses import dataclass
from typing import Tuple, List
from collections import defaultdict


@dataclass
class Workload:
    K: int = 64
    C: int = 64
    P: int = 56
    Q: int = 56
    R: int = 7  # 使用 7x7 kernel 更明显
    S: int = 7


def simulate_5d(P_l3, Q_l3, C_l3, K_l3, R_l3, perm: Tuple[str, ...]) -> dict:
    """
    模拟 5 维度 permutation
    
    Input tile 由 (C, P, Q, R) 确定 (如果 R_l3 > 1)
    """
    ranges = {
        'P': range(P_l3),
        'Q': range(Q_l3),
        'C': range(C_l3),
        'K': range(K_l3),
        'R': range(R_l3),
    }
    
    tile_changes = 0
    last_input = None
    
    for i0 in ranges[perm[0]]:
        for i1 in ranges[perm[1]]:
            for i2 in ranges[perm[2]]:
                for i3 in ranges[perm[3]]:
                    for i4 in ranges[perm[4]]:
                        indices = {perm[0]: i0, perm[1]: i1, perm[2]: i2,
                                   perm[3]: i3, perm[4]: i4}
                        
                        # Input tile: (C, P, Q, R)
                        # 注意: R 变化会访问不同的 Input 行
                        input_tile = (indices['C'], indices['P'], indices['Q'], indices['R'])
                        
                        if input_tile != last_input:
                            tile_changes += 1
                            last_input = input_tile
    
    return {'tile_changes': tile_changes}


def compute_formula(P_l3, Q_l3, C_l3, K_l3, R_l3, perm: Tuple[str, ...]) -> int:
    """
    扩展公式到 5 维度
    
    Input dims = {C, P, Q, R}
    Non-input dims = {K}
    
    与 K 的分析类似:
    - K 后面的 Input dims 的乘积 == 1 时，tile_changes = C_l3 × P_l3 × Q_l3 × R_l3
    - 否则 tile_changes = C_l3 × P_l3 × Q_l3 × R_l3 × K_l3
    """
    input_dims = {'C', 'P', 'Q', 'R'}
    dim_values = {'P': P_l3, 'Q': Q_l3, 'C': C_l3, 'K': K_l3, 'R': R_l3}
    
    k_pos = perm.index('K')
    
    # K 之后的 Input dims 乘积
    inner_input_product = 1
    for i in range(k_pos + 1, 5):
        d = perm[i]
        if d in input_dims:
            inner_input_product *= dim_values[d]
    
    base = P_l3 * Q_l3 * C_l3 * R_l3
    
    if inner_input_product == 1:
        return base
    else:
        return base * K_l3


def main():
    print("="*70)
    print("R 位置对 Row Activation 的影响分析")
    print("="*70)
    
    # 固定 tiling
    P_l3, Q_l3, C_l3, K_l3 = 2, 2, 2, 2
    R_l3 = 7  # R 在 L3 完全展开
    
    print(f"\nTiling: P_l3={P_l3}, Q_l3={Q_l3}, C_l3={C_l3}, K_l3={K_l3}, R_l3={R_l3}")
    print(f"基础 Input tiles (不考虑 K): P×Q×C×R = {P_l3*Q_l3*C_l3*R_l3}")
    
    # 测试不同的 K 位置
    test_perms = [
        # K 在最外层
        ('K', 'C', 'P', 'Q', 'R'),
        ('K', 'R', 'C', 'P', 'Q'),
        # K 在中间
        ('C', 'K', 'P', 'Q', 'R'),
        ('C', 'P', 'K', 'Q', 'R'),
        ('R', 'C', 'K', 'P', 'Q'),
        # K 在最内层
        ('C', 'P', 'Q', 'R', 'K'),
        ('R', 'C', 'P', 'Q', 'K'),
    ]
    
    print("\n" + "="*70)
    print(f"{'Permutation':<15} {'K pos':<6} {'Exact':<8} {'Formula':<8} {'Match':<6}")
    print("="*70)
    
    all_match = True
    for perm in test_perms:
        exact = simulate_5d(P_l3, Q_l3, C_l3, K_l3, R_l3, perm)['tile_changes']
        formula = compute_formula(P_l3, Q_l3, C_l3, K_l3, R_l3, perm)
        
        perm_str = ''.join([p[0] for p in perm])
        k_pos = perm.index('K')
        match = "✓" if exact == formula else "✗"
        if exact != formula:
            all_match = False
        
        print(f"{perm_str:<15} {k_pos:<6} {exact:<8} {formula:<8} {match:<6}")
    
    # 测试不同的 tiling
    print("\n" + "="*70)
    print("更多 Tiling 配置测试")
    print("="*70)
    
    test_tilings = [
        (2, 2, 2, 2, 3),
        (4, 4, 2, 2, 7),
        (1, 2, 4, 4, 3),  # P_l3=1 的特殊情况
    ]
    
    for P_l3, Q_l3, C_l3, K_l3, R_l3 in test_tilings:
        print(f"\n--- Tiling: P={P_l3}, Q={Q_l3}, C={C_l3}, K={K_l3}, R={R_l3} ---")
        
        passed = 0
        total = 0
        for perm in test_perms:
            exact = simulate_5d(P_l3, Q_l3, C_l3, K_l3, R_l3, perm)['tile_changes']
            formula = compute_formula(P_l3, Q_l3, C_l3, K_l3, R_l3, perm)
            
            total += 1
            if exact == formula:
                passed += 1
            else:
                perm_str = ''.join([p[0] for p in perm])
                print(f"  MISMATCH: {perm_str}, Exact={exact}, Formula={formula}")
        
        print(f"  Results: {passed}/{total}")
    
    # 分析
    print("\n" + "="*70)
    print("结论")
    print("="*70)
    print(f"""
1. R 与 Input 的关系:
   - R 变化会访问不同的 Input 行 (不同的 h 偏移)
   - 所以 R 是 Input-relevant 维度

2. 扩展公式:
   - Input dims = {{C, P, Q, R}} (都影响 Input 访问)
   - Non-input dims = {{K}} (只影响 Weight/Output)
   
3. 公式结构不变:
   If K 后面的 Input dims 乘积 == 1:
       tile_changes = C_l3 × P_l3 × Q_l3 × R_l3
   Else:
       tile_changes = C_l3 × P_l3 × Q_l3 × R_l3 × K_l3

4. 实际应用:
   - 如果 R 完全在 L2 层 (R_l3=1)，则 R 不影响 L3 层的 tile 切换
   - 如果 R 部分在 L3 层 (R_l3>1)，则需要考虑 R 的位置
""")
    
    # S 的分析
    print("\n" + "="*70)
    print("S 维度分析")
    print("="*70)
    print("""
S 与 R 完全类似:
- S 变化会访问不同的 Input 列 (不同的 w 偏移)
- 如果 S 在 L3 层展开，也是 Input-relevant 维度

6 维度公式 (如果 R 和 S 都在 L3 层):
   Input dims = {C, P, Q, R, S}
   Non-input dims = {K}
   
   同样的逻辑: K 后面有 Input dims 时乘以 K_l3
""")


if __name__ == '__main__':
    main()
