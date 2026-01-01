#!/usr/bin/env python3
"""
深入分析 Permutation 对 Row Switch 的影响

关键发现:
- 相关系数全是 1.0 → 公式排序完全正确
- 但误差有差异 → 绝对值有偏差

需要分析:
1. 当 K 不在最内层时，为什么误差是 43.8%？
2. 当 K 在最内层时，为什么误差是 0%？
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

import math
from dataclasses import dataclass
from typing import List, Tuple
from collections import defaultdict


@dataclass
class Workload:
    name: str
    N: int
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
    l3_permutation: Tuple[str, ...]
    block_w: int
    workload: 'Workload'


def simulate_exact(m: Mapping) -> dict:
    """模拟精确的 row switches，并返回详细信息"""
    w = m.workload
    
    P_l3, Q_l3, C_l3, K_l3 = m.P_l3, m.Q_l3, m.C_l3, m.K_l3
    P_buffer = w.P // P_l3
    Q_buffer = w.Q // Q_l3
    C_buffer = w.C // C_l3
    K_buffer = w.K // K_l3
    
    # Input tile 大小
    h_tile = P_buffer + w.R - 1
    w_tile = Q_buffer + w.S - 1
    input_tile_size = C_buffer * h_tile * w_tile
    rows_per_tile = math.ceil(input_tile_size / m.block_w)
    
    perm = m.l3_permutation
    ranges = {
        'P': range(P_l3), 'Q': range(Q_l3),
        'C': range(C_l3), 'K': range(K_l3)
    }
    
    # 统计
    row_switches = 0
    input_tile_changes = 0
    last_c, last_p, last_q = -1, -1, -1
    
    # 模拟循环
    for i0 in ranges[perm[0]]:
        for i1 in ranges[perm[1]]:
            for i2 in ranges[perm[2]]:
                for i3 in ranges[perm[3]]:
                    indices = {perm[0]: i0, perm[1]: i1, perm[2]: i2, perm[3]: i3}
                    c = indices['C']
                    p = indices['P']
                    q = indices['Q']
                    
                    if (c != last_c or p != last_p or q != last_q):
                        input_tile_changes += 1
                        row_switches += rows_per_tile
                        last_c, last_p, last_q = c, p, q
    
    return {
        'row_switches': row_switches,
        'input_tile_changes': input_tile_changes,
        'rows_per_tile': rows_per_tile,
        'n_total_iter': P_l3 * Q_l3 * C_l3 * K_l3,
        'n_input_tiles_formula': P_l3 * Q_l3 * C_l3,  # 公式用的
    }


def compute_formula(m: Mapping) -> int:
    """当前公式"""
    w = m.workload
    
    P_buffer = w.P // m.P_l3
    Q_buffer = w.Q // m.Q_l3
    C_buffer = w.C // m.C_l3
    
    h_tile = P_buffer + w.R - 1
    w_tile = Q_buffer + w.S - 1
    input_tile_size = C_buffer * h_tile * w_tile
    
    rows_per_tile = math.ceil(input_tile_size / m.block_w)
    n_input_tiles = m.P_l3 * m.Q_l3 * m.C_l3
    
    return n_input_tiles * rows_per_tile


def main():
    print("="*70)
    print("深入分析: Permutation 如何影响 Input Tile 切换次数")
    print("="*70)
    
    # 简单 workload 便于分析
    w = Workload(name='Simple', N=1, K=4, C=4, P=8, Q=8, R=3, S=3)
    
    # 固定 tiling
    P_l3, Q_l3, C_l3, K_l3 = 2, 2, 2, 2
    
    print(f"\nWorkload: P={w.P}, Q={w.Q}, C={w.C}, K={w.K}")
    print(f"Tiling: P_l3={P_l3}, Q_l3={Q_l3}, C_l3={C_l3}, K_l3={K_l3}")
    print(f"Total L3 iterations: {P_l3 * Q_l3 * C_l3 * K_l3}")
    print(f"Input-relevant tiles: P_l3 × Q_l3 × C_l3 = {P_l3 * Q_l3 * C_l3}")
    
    perms_to_test = [
        ('K', 'C', 'P', 'Q'),  # K 最外层
        ('C', 'K', 'P', 'Q'),  # K 次外层
        ('C', 'P', 'K', 'Q'),  # K 次内层
        ('C', 'P', 'Q', 'K'),  # K 最内层
    ]
    
    print("\n" + "="*70)
    print(f"{'Permutation':<15} {'Exact Tiles':<12} {'Formula':<12} {'Error':>8}")
    print("="*70)
    
    for perm in perms_to_test:
        m = Mapping(
            P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3,
            l3_permutation=perm,
            block_w=512,
            workload=w
        )
        
        exact = simulate_exact(m)
        formula = compute_formula(m)
        
        err = (formula - exact['row_switches']) / exact['row_switches'] * 100
        
        perm_str = ''.join([p[0] for p in perm])
        print(f"{perm_str:<15} {exact['input_tile_changes']:<12} "
              f"{exact['n_input_tiles_formula']:<12} {err:>7.1f}%")
    
    # 详细展示循环行为
    print("\n" + "="*70)
    print("详细循环分析:")
    print("="*70)
    
    for perm in perms_to_test[:2]:  # 只展示两个代表性的
        perm_str = ''.join([p[0] for p in perm])
        print(f"\n--- Permutation: {perm_str} ---")
        print(f"  循环顺序 (外→内): {perm[0]} → {perm[1]} → {perm[2]} → {perm[3]}")
        
        m = Mapping(
            P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3,
            l3_permutation=perm,
            block_w=512,
            workload=w
        )
        
        ranges = {
            'P': range(P_l3), 'Q': range(Q_l3),
            'C': range(C_l3), 'K': range(K_l3)
        }
        
        tile_changes = 0
        total_iter = 0
        last_input_tile = None
        
        print(f"\n  迭代序列 (显示前 20 个):")
        count = 0
        for i0 in ranges[perm[0]]:
            for i1 in ranges[perm[1]]:
                for i2 in ranges[perm[2]]:
                    for i3 in ranges[perm[3]]:
                        indices = {perm[0]: i0, perm[1]: i1, perm[2]: i2, perm[3]: i3}
                        
                        input_tile = (indices['C'], indices['P'], indices['Q'])
                        is_new_tile = input_tile != last_input_tile
                        
                        if count < 20:
                            marker = "★NEW" if is_new_tile else ""
                            print(f"    [{indices['K']},{indices['C']},{indices['P']},{indices['Q']}] "
                                  f"Input=({input_tile[0]},{input_tile[1]},{input_tile[2]}) {marker}")
                        
                        if is_new_tile:
                            tile_changes += 1
                            last_input_tile = input_tile
                        
                        total_iter += 1
                        count += 1
        
        if count > 20:
            print(f"    ... (共 {total_iter} 次迭代)")
        
        print(f"\n  统计:")
        print(f"    实际 Input tile 切换: {tile_changes} 次")
        print(f"    公式 Input tiles: {P_l3 * Q_l3 * C_l3}")
        
        # 解释
        k_pos = perm.index('K')
        if k_pos == 0:
            print(f"\n  解释: K 在最外层，每次 K 变化时，内层 (C,P,Q) 从头遍历")
            print(f"        所以 Input tile 切换 = K_l3 × (P_l3 × Q_l3 × C_l3)")
            print(f"        = {K_l3} × {P_l3 * Q_l3 * C_l3} = {K_l3 * P_l3 * Q_l3 * C_l3}")
        elif k_pos == 3:
            print(f"\n  解释: K 在最内层，K 变化不影响 Input tile")
            print(f"        所以 Input tile 切换 = P_l3 × Q_l3 × C_l3 = {P_l3 * Q_l3 * C_l3}")
    
    # 结论
    print("\n" + "="*70)
    print("关键发现")
    print("="*70)
    print("""
当前公式: Row_switches = (P_l3 × Q_l3 × C_l3) × rows_per_tile

问题:
  - 公式假设 Input tile 只切换 P_l3 × Q_l3 × C_l3 次
  - 但实际上，当 K 不在最内层时，K 循环会导致 Input 被重复访问

正确计算:
  - 设 K 在位置 pos_k (0=最外, 3=最内)
  - K 后面的维度个数 = 3 - pos_k
  - 如果 K 后面有 Input 相关维度 (C, P, Q)，则每次 K 变化会重复遍历这些维度

修正公式:
  设 n_inner = K 后面的 Input 相关维度数量
  If K 在最内层: 
      Input_tile_changes = P_l3 × Q_l3 × C_l3
  Else:
      K 每变化一次，内层 Input 相关循环从头开始
      Input_tile_changes = P_l3 × Q_l3 × C_l3 × K_l3^(1 if n_inner > 0)
""")


if __name__ == '__main__':
    main()
