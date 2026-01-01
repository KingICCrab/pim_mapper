#!/usr/bin/env python3
"""
深入分析 K 位置影响 - 找到正确的公式

问题: V2 在 K 位置 1, 2 时误差反而更大了
原因: 不是简单的 × K_l3，而是要看 K 内部的 Input 相关维度的乘积
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


def compute_exact(m: Mapping) -> dict:
    """精确模拟，返回详细信息"""
    w = m.workload
    perm = m.perm
    
    P_buffer = w.P // m.P_l3
    Q_buffer = w.Q // m.Q_l3
    C_buffer = w.C // m.C_l3
    
    h_tile = P_buffer + w.R - 1
    w_tile = Q_buffer + w.S - 1
    input_tile_size = C_buffer * h_tile * w_tile
    rows_per_tile = math.ceil(input_tile_size / m.block_w)
    
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
    
    return {
        'tile_changes': tile_changes,
        'row_switches': tile_changes * rows_per_tile,
        'rows_per_tile': rows_per_tile,
    }


def compute_formula_correct(m: Mapping) -> dict:
    """
    正确的公式
    
    关键观察:
    - Input tile 由 (C, P, Q) 唯一确定
    - 循环按 permutation 顺序展开
    - Input tile 切换次数 = 有多少次 (C, P, Q) 发生变化
    
    分析:
    设 perm = (d0, d1, d2, d3)，从外到内
    
    Case 1: K 在最内层 (d3 = K)
      Input tile 切换 = C_l3 × P_l3 × Q_l3
      (K 变化不引起切换)
    
    Case 2: K 在次内层 (d2 = K)
      最内层是某个 Input dim (C/P/Q)
      K 每变化一次，最内层从头遍历
      但 Input tile 的外两层还是只遍历一次
      所以 Input tile 切换 = (外两个 Input dims 的乘积) × K_l3 × (最内层 dim)
    
    更一般的规则:
    对于 K 在位置 pos_k:
      - K 之前的维度: 正常遍历
      - K 之后的维度: 每次 K 变化会重新从头遍历
      
    Input tile 切换 = ∏(K之前的Input dims) × ∏(K及之后的所有dims)
                    = ∏(K之前的Input dims) × K_l3 × ∏(K之后的所有dims)
    
    实际上就是: (K之前的Input dims乘积) × (K及之后的所有dims乘积)
    """
    w = m.workload
    perm = m.perm
    
    P_buffer = w.P // m.P_l3
    Q_buffer = w.Q // m.Q_l3
    C_buffer = w.C // m.C_l3
    
    h_tile = P_buffer + w.R - 1
    w_tile = Q_buffer + w.S - 1
    input_tile_size = C_buffer * h_tile * w_tile
    rows_per_tile = math.ceil(input_tile_size / m.block_w)
    
    dim_values = {'P': m.P_l3, 'Q': m.Q_l3, 'C': m.C_l3, 'K': m.K_l3}
    input_dims = {'C', 'P', 'Q'}
    
    k_pos = perm.index('K')
    
    # K 之前的 Input dims
    before_k_input_product = 1
    for i in range(k_pos):
        if perm[i] in input_dims:
            before_k_input_product *= dim_values[perm[i]]
    
    # K 及之后的所有 dims
    k_and_after_product = 1
    for i in range(k_pos, 4):
        d = perm[i]
        if d in input_dims:  # 只统计 Input dims
            k_and_after_product *= dim_values[d]
    
    tile_changes = before_k_input_product * k_and_after_product
    
    return {
        'tile_changes': tile_changes,
        'row_switches': tile_changes * rows_per_tile,
        'rows_per_tile': rows_per_tile,
        'before_k': before_k_input_product,
        'k_and_after': k_and_after_product,
    }


def main():
    print("="*70)
    print("分析 K 位置对 Input tile 切换的影响 - 寻找正确公式")
    print("="*70)
    
    # 简单 workload
    w = Workload(name='Simple', K=4, C=4, P=4, Q=4, R=3, S=3)
    
    # 固定 tiling
    P_l3, Q_l3, C_l3, K_l3 = 2, 2, 2, 2
    
    print(f"\nWorkload: K={w.K}, C={w.C}, P={w.P}, Q={w.Q}")
    print(f"Tiling: P_l3={P_l3}, Q_l3={Q_l3}, C_l3={C_l3}, K_l3={K_l3}")
    print(f"\n基础 Input tiles = P_l3 × Q_l3 × C_l3 = {P_l3 * Q_l3 * C_l3}")
    
    # 测试所有 K 位置
    test_perms = [
        ('K', 'C', 'P', 'Q'),  # K at 0
        ('C', 'K', 'P', 'Q'),  # K at 1
        ('C', 'P', 'K', 'Q'),  # K at 2
        ('C', 'P', 'Q', 'K'),  # K at 3
    ]
    
    print("\n" + "="*70)
    print(f"{'Perm':<10} {'K pos':<6} {'Exact':<8} {'Formula':<8} {'Match':<6}")
    print("="*70)
    
    for perm in test_perms:
        m = Mapping(
            P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3,
            perm=perm, block_w=512, workload=w
        )
        
        exact = compute_exact(m)
        formula = compute_formula_correct(m)
        
        perm_str = ''.join([p[0] for p in perm])
        k_pos = perm.index('K')
        match = "✓" if exact['tile_changes'] == formula['tile_changes'] else "✗"
        
        print(f"{perm_str:<10} {k_pos:<6} {exact['tile_changes']:<8} "
              f"{formula['tile_changes']:<8} {match:<6}")
    
    # 详细分析
    print("\n" + "="*70)
    print("详细分析:")
    print("="*70)
    
    for perm in test_perms:
        m = Mapping(
            P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3,
            perm=perm, block_w=512, workload=w
        )
        
        exact = compute_exact(m)
        formula = compute_formula_correct(m)
        
        perm_str = ''.join([p[0] for p in perm])
        k_pos = perm.index('K')
        
        print(f"\n{perm_str}: K at position {k_pos}")
        print(f"  循环顺序: {perm[0]} → {perm[1]} → {perm[2]} → {perm[3]}")
        
        # 分解公式
        dim_values = {'P': P_l3, 'Q': Q_l3, 'C': C_l3, 'K': K_l3}
        input_dims = {'C', 'P', 'Q'}
        
        before_k = [perm[i] for i in range(k_pos) if perm[i] in input_dims]
        k_and_after = [perm[i] for i in range(k_pos, 4) if perm[i] in input_dims]
        
        before_k_product = math.prod([dim_values[d] for d in before_k]) if before_k else 1
        k_and_after_product = math.prod([dim_values[d] for d in k_and_after]) if k_and_after else 1
        
        print(f"  K 之前的 Input dims: {before_k} → product = {before_k_product}")
        print(f"  K 及之后的 Input dims: {k_and_after} → product = {k_and_after_product}")
        print(f"  公式: {before_k_product} × {k_and_after_product} = {before_k_product * k_and_after_product}")
        print(f"  精确: {exact['tile_changes']}")
    
    # 最终公式
    print("\n" + "="*70)
    print("最终公式")
    print("="*70)
    print("""
设 perm = (d0, d1, d2, d3)，K 在位置 pos_k

Input_tile_changes = ∏(K之前的Input dims) × ∏(K及之后的Input dims)

公式解释:
  - K 之前的维度: 每个 Input dim 只遍历一次
  - K 及之后的维度: 
    - 如果 K 之后没有 Input dims，这部分乘积 = 1
    - 如果 K 之后有 Input dims，它们会被重复遍历 (每次 K 变化都重新开始)
    
示例:
  KCPQ (K at 0): before_K = [], after_K = [C,P,Q]
    → 1 × (C_l3 × P_l3 × Q_l3) = C_l3 × P_l3 × Q_l3
    
  CKPQ (K at 1): before_K = [C], after_K = [P,Q]  
    → C_l3 × (P_l3 × Q_l3)
    
  CPKQ (K at 2): before_K = [C,P], after_K = [Q]
    → (C_l3 × P_l3) × Q_l3
    
  CPQK (K at 3): before_K = [C,P,Q], after_K = []
    → (C_l3 × P_l3 × Q_l3) × 1 = C_l3 × P_l3 × Q_l3

结论: 当 K 在最外或最内层时，Input_tile_changes = C_l3 × P_l3 × Q_l3
      当 K 在中间时，公式不变但分解方式不同
""")


if __name__ == '__main__':
    main()
