#!/usr/bin/env python3
"""
测试 R、S 顺序是否影响 Row Activation

问题：在之前的分析中，我们只考虑了 L3 层的 (K, C, P, Q) permutation
但 R 和 S 也是循环维度，它们的顺序会影响 row activation 吗？

分析：
1. L3 层: 遍历 (K_l3, C_l3, P_l3, Q_l3) 个 tiles
2. L2 层: 在每个 L3 tile 内，遍历 R_l2 (通常 R 在 L2 展开)
3. L1 层: 在 PE 内计算

关键问题：
- Input tile 的粒度是什么？
- R、S 的遍历是在 Input tile 加载之后还是之前？
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

import math
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class Workload:
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
    R_l2: int  # R 在 L2 层的 tiling
    l3_perm: Tuple[str, ...]
    l2_perm: Tuple[str, ...]  # L2 层的顺序，包括 R
    block_w: int
    workload: 'Workload'


def simulate_with_l2_order(m: Mapping) -> dict:
    """
    模拟包含 L2 层 R 遍历的 row switches
    
    L3 循环: 按 l3_perm 遍历 tiles
    L2 循环: 在每个 tile 内，按 l2_perm 遍历 (可能包括 R)
    """
    w = m.workload
    
    P_buffer = w.P // m.P_l3
    Q_buffer = w.Q // m.Q_l3
    C_buffer = w.C // m.C_l3
    K_buffer = w.K // m.K_l3
    R_buffer = w.R // m.R_l2
    
    # Input tile 大小 (整个 tile，包括 R 的 halo)
    h_tile = P_buffer + w.R - 1
    w_tile = Q_buffer + w.S - 1
    input_tile_size = C_buffer * h_tile * w_tile
    rows_per_tile = math.ceil(input_tile_size / m.block_w)
    
    # L3 层循环
    l3_ranges = {
        'P': range(m.P_l3), 'Q': range(m.Q_l3),
        'C': range(m.C_l3), 'K': range(m.K_l3)
    }
    
    # L2 层循环 (在每个 L3 tile 内)
    l2_ranges = {
        'R': range(m.R_l2),
        # P_buffer 内的遍历 (如果有)
        # 这里简化，假设 L2 只有 R
    }
    
    # 统计
    l3_tile_changes = 0
    l2_sub_tile_accesses = 0
    last_input_tile = None
    
    # L3 循环
    for i0 in l3_ranges[m.l3_perm[0]]:
        for i1 in l3_ranges[m.l3_perm[1]]:
            for i2 in l3_ranges[m.l3_perm[2]]:
                for i3 in l3_ranges[m.l3_perm[3]]:
                    l3_indices = {m.l3_perm[0]: i0, m.l3_perm[1]: i1, 
                                  m.l3_perm[2]: i2, m.l3_perm[3]: i3}
                    
                    # L3 Input tile 由 (C, P, Q) 确定
                    input_tile = (l3_indices['C'], l3_indices['P'], l3_indices['Q'])
                    
                    if input_tile != last_input_tile:
                        l3_tile_changes += 1
                        last_input_tile = input_tile
                    
                    # L2 循环 (在这个 L3 tile 内)
                    for r in range(m.R_l2):
                        l2_sub_tile_accesses += 1
    
    return {
        'l3_tile_changes': l3_tile_changes,
        'l2_sub_tile_accesses': l2_sub_tile_accesses,
        'rows_per_tile': rows_per_tile,
        'total_row_switches': l3_tile_changes * rows_per_tile,
    }


def simulate_with_r_in_l3(m: Mapping, include_r_in_l3: bool) -> dict:
    """
    测试如果把 R 也放到 L3 permutation 会怎样
    
    如果 R 在 L3 层:
    - 每次 R 变化，需要加载不同的 Input 子区域
    - 但传统设计中 R 通常在 L2 层
    """
    w = m.workload
    
    P_buffer = w.P // m.P_l3
    Q_buffer = w.Q // m.Q_l3
    C_buffer = w.C // m.C_l3
    
    if include_r_in_l3:
        # R 在 L3 层，每次只加载部分 Input
        h_tile = P_buffer + (w.R // m.R_l2) - 1  # 只加载部分 R 范围
        w_tile = Q_buffer + w.S - 1
        input_tile_size = C_buffer * h_tile * w_tile
        
        # R 也影响 tile 数量
        n_input_tiles = m.P_l3 * m.Q_l3 * m.C_l3 * m.R_l2
    else:
        # R 在 L2 层，一次加载整个 Input tile (包括 R 的 halo)
        h_tile = P_buffer + w.R - 1
        w_tile = Q_buffer + w.S - 1
        input_tile_size = C_buffer * h_tile * w_tile
        
        n_input_tiles = m.P_l3 * m.Q_l3 * m.C_l3
    
    rows_per_tile = math.ceil(input_tile_size / m.block_w)
    
    return {
        'n_input_tiles': n_input_tiles,
        'input_tile_size': input_tile_size,
        'rows_per_tile': rows_per_tile,
        'total_row_switches': n_input_tiles * rows_per_tile,
    }


def main():
    print("="*70)
    print("测试 R、S 顺序对 Row Activation 的影响")
    print("="*70)
    
    # 测试 workload
    w = Workload(K=64, C=64, P=56, Q=56, R=3, S=3)
    
    print(f"\nWorkload: K={w.K}, C={w.C}, P={w.P}, Q={w.Q}, R={w.R}, S={w.S}")
    
    # 固定 L3 tiling
    P_l3, Q_l3, C_l3, K_l3 = 4, 4, 4, 4
    R_l2 = 3  # R 完全在 L2 展开
    
    print(f"L3 Tiling: P_l3={P_l3}, Q_l3={Q_l3}, C_l3={C_l3}, K_l3={K_l3}")
    print(f"L2 Tiling: R_l2={R_l2}")
    
    P_buffer = w.P // P_l3
    Q_buffer = w.Q // Q_l3
    
    print(f"\nP_buffer={P_buffer}, Q_buffer={Q_buffer}")
    print(f"Input tile 覆盖范围: C_buffer × (P_buffer + R - 1) × (Q_buffer + S - 1)")
    print(f"                   = {w.C // C_l3} × {P_buffer + w.R - 1} × {Q_buffer + w.S - 1}")
    
    # 测试不同的 L3 permutation
    test_perms = [
        ('C', 'P', 'Q', 'K'),  # K 最内层
        ('K', 'C', 'P', 'Q'),  # K 最外层
    ]
    
    print("\n" + "="*70)
    print("测试 1: L3 Permutation 影响 (R 在 L2 层)")
    print("="*70)
    
    for perm in test_perms:
        m = Mapping(
            P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3,
            R_l2=R_l2,
            l3_perm=perm,
            l2_perm=('R',),
            block_w=512,
            workload=w
        )
        
        result = simulate_with_l2_order(m)
        perm_str = ''.join([p[0] for p in perm])
        
        print(f"\n{perm_str}:")
        print(f"  L3 tile 切换: {result['l3_tile_changes']}")
        print(f"  L2 访问次数: {result['l2_sub_tile_accesses']}")
        print(f"  Row switches: {result['total_row_switches']}")
    
    # 测试 R 在 L3 vs L2 的影响
    print("\n" + "="*70)
    print("测试 2: R 在 L3 层 vs L2 层")
    print("="*70)
    
    m_base = Mapping(
        P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3,
        R_l2=R_l2,
        l3_perm=('C', 'P', 'Q', 'K'),
        l2_perm=('R',),
        block_w=512,
        workload=w
    )
    
    result_l2 = simulate_with_r_in_l3(m_base, include_r_in_l3=False)
    result_l3 = simulate_with_r_in_l3(m_base, include_r_in_l3=True)
    
    print(f"\nR 在 L2 层 (传统方式):")
    print(f"  Input tiles: {result_l2['n_input_tiles']}")
    print(f"  每个 tile 大小: {result_l2['input_tile_size']}")
    print(f"  每个 tile 行数: {result_l2['rows_per_tile']}")
    print(f"  总 row switches: {result_l2['total_row_switches']}")
    
    print(f"\nR 在 L3 层 (假设):")
    print(f"  Input tiles: {result_l3['n_input_tiles']}")
    print(f"  每个 tile 大小: {result_l3['input_tile_size']}")
    print(f"  每个 tile 行数: {result_l3['rows_per_tile']}")
    print(f"  总 row switches: {result_l3['total_row_switches']}")
    
    # 分析
    print("\n" + "="*70)
    print("分析")
    print("="*70)
    print("""
关键结论:

1. R、S 通常在 L2 层处理:
   - 一个 L3 Input tile 包含整个 (P_buffer + R - 1) × (Q_buffer + S - 1) 范围
   - R、S 的遍历顺序在 L2 层，不影响 DRAM row activation
   - 因为整个 Input tile 已经一次性加载到 Global Buffer

2. 如果 R 在 L3 层:
   - 每个 L3 tile 只包含部分 R 范围
   - R 的遍历会导致额外的 tile 切换
   - 但这通常不是高效的设计选择

3. 当前公式的适用范围:
   - 假设 R、S 在 L2 层或更低
   - 只考虑 L3 层的 (K, C, P, Q) permutation

如果你的设计中 R、S 也在 L3 层的 permutation 中，需要扩展公式!
""")
    
    # 测试如果 R 也在 L3 permutation 中
    print("\n" + "="*70)
    print("测试 3: 如果 R 也在 L3 permutation 中")
    print("="*70)
    
    # 扩展到 5 维度 permutation
    test_5d_perms = [
        ('R', 'C', 'P', 'Q', 'K'),  # R 最外层
        ('C', 'P', 'Q', 'K', 'R'),  # R 最内层
        ('C', 'R', 'P', 'Q', 'K'),  # R 次外层
    ]
    
    for perm in test_5d_perms:
        result = simulate_5d_permutation(w, P_l3, Q_l3, C_l3, K_l3, R_l2, perm)
        perm_str = ''.join([p[0] for p in perm])
        print(f"\n{perm_str}:")
        print(f"  Input tile 切换: {result['tile_changes']}")


def simulate_5d_permutation(w: Workload, P_l3, Q_l3, C_l3, K_l3, R_l2, perm: Tuple[str, ...]) -> dict:
    """
    模拟 5 维度 (K, C, P, Q, R) 的 permutation
    """
    P_buffer = w.P // P_l3
    Q_buffer = w.Q // Q_l3
    C_buffer = w.C // C_l3
    R_buffer = w.R // R_l2 if R_l2 > 1 else w.R
    
    ranges = {
        'P': range(P_l3),
        'Q': range(Q_l3),
        'C': range(C_l3),
        'K': range(K_l3),
        'R': range(R_l2) if R_l2 > 1 else range(1),
    }
    
    tile_changes = 0
    last_input = None
    
    # 5 层循环
    for i0 in ranges[perm[0]]:
        for i1 in ranges[perm[1]]:
            for i2 in ranges[perm[2]]:
                for i3 in ranges[perm[3]]:
                    for i4 in ranges[perm[4]]:
                        indices = {perm[0]: i0, perm[1]: i1, perm[2]: i2,
                                   perm[3]: i3, perm[4]: i4}
                        
                        # Input tile 由 (C, P, Q, R) 确定
                        # 注意: 如果 R_l2=R, 则 R 索引总是 0
                        r_idx = indices.get('R', 0)
                        input_tile = (indices['C'], indices['P'], indices['Q'], r_idx)
                        
                        if input_tile != last_input:
                            tile_changes += 1
                            last_input = input_tile
    
    return {'tile_changes': tile_changes}


if __name__ == '__main__':
    main()
