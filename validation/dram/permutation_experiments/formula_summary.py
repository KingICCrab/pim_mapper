#!/usr/bin/env python3
"""
================================================================================
Row Activation 公式 - 完整总结
================================================================================

关键发现:
1. R、S 的位置确实会影响 Row Activation（如果它们在 L3 层展开）
2. 但传统设计中 R、S 通常完全在 L2 层，所以只需考虑 (K, C, P, Q)
3. 公式的核心规则是: K 是唯一的 Non-Input 维度

================================================================================
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

import math
from dataclasses import dataclass
from typing import Tuple, Set
from itertools import permutations


def compute_input_tile_changes(dim_l3_values: dict, 
                                perm: Tuple[str, ...],
                                input_dims: Set[str] = {'C', 'P', 'Q'}) -> int:
    """
    通用公式: 计算 Input tile 切换次数
    
    参数:
        dim_l3_values: 各维度的 L3 tiling 值, e.g., {'K': 2, 'C': 4, 'P': 4, 'Q': 4}
        perm: 维度遍历顺序 (外→内), e.g., ('K', 'C', 'P', 'Q')
        input_dims: Input-relevant 维度集合
        
    规则:
        1. 找到所有 Non-input dims (只有 K)
        2. 对于每个 Non-input dim，检查它后面是否有 Input dims
        3. 如果有，则该 Non-input dim 的变化会导致 Input 重复访问
        
    公式:
        base = ∏(所有 Input dims 的 l3 值)
        multiplier = ∏(后面有 Input dims 的 Non-input dims 的 l3 值)
        tile_changes = base × multiplier
    """
    non_input_dims = set(dim_l3_values.keys()) - input_dims
    
    # 基础值: 所有 Input dims 的乘积
    base = 1
    for d in input_dims:
        if d in dim_l3_values:
            base *= dim_l3_values[d]
    
    # 乘数: 后面有 Input dims 的 Non-input dims
    multiplier = 1
    for d in non_input_dims:
        if d not in perm:
            continue
        pos = perm.index(d)
        
        # 检查 d 后面是否有 Input dims
        has_input_after = False
        for i in range(pos + 1, len(perm)):
            if perm[i] in input_dims:
                # 还要检查该 Input dim 的 l3 值是否 > 1
                if dim_l3_values.get(perm[i], 1) > 1:
                    has_input_after = True
                    break
        
        if has_input_after:
            multiplier *= dim_l3_values[d]
    
    return base * multiplier


def test_formula():
    """测试通用公式"""
    print("="*70)
    print("通用公式验证")
    print("="*70)
    
    # 测试 1: 标准 4 维度 (K, C, P, Q)
    print("\n--- 测试 1: 标准 4 维度 ---")
    
    dim_values = {'K': 2, 'C': 2, 'P': 2, 'Q': 2}
    input_dims = {'C', 'P', 'Q'}
    
    test_perms = [
        ('K', 'C', 'P', 'Q'),
        ('C', 'K', 'P', 'Q'),
        ('C', 'P', 'K', 'Q'),
        ('C', 'P', 'Q', 'K'),
    ]
    
    print(f"dim_values: {dim_values}")
    print(f"Input dims: {input_dims}")
    print(f"\n{'Perm':<12} {'Formula':<10} {'Expected':<10}")
    print("-"*35)
    
    for perm in test_perms:
        result = compute_input_tile_changes(dim_values, perm, input_dims)
        
        # 手动计算预期值
        k_pos = perm.index('K')
        if k_pos == 3:  # K 最内层
            expected = 8  # 2×2×2
        else:
            expected = 16  # 2×2×2×2
        
        perm_str = ''.join([p[0] for p in perm])
        match = "✓" if result == expected else "✗"
        print(f"{perm_str:<12} {result:<10} {expected:<10} {match}")
    
    # 测试 2: 5 维度 (K, C, P, Q, R)
    print("\n--- 测试 2: 5 维度 (包含 R) ---")
    
    dim_values = {'K': 2, 'C': 2, 'P': 2, 'Q': 2, 'R': 3}
    input_dims = {'C', 'P', 'Q', 'R'}  # R 也是 Input-relevant
    
    test_perms = [
        ('K', 'C', 'P', 'Q', 'R'),
        ('C', 'P', 'Q', 'R', 'K'),
        ('R', 'K', 'C', 'P', 'Q'),
    ]
    
    print(f"dim_values: {dim_values}")
    print(f"Input dims: {input_dims}")
    print(f"\n{'Perm':<15} {'Formula':<10}")
    print("-"*30)
    
    for perm in test_perms:
        result = compute_input_tile_changes(dim_values, perm, input_dims)
        perm_str = ''.join([p[0] for p in perm])
        print(f"{perm_str:<15} {result:<10}")
    
    # 测试 3: 特殊情况 - 某个 dim 的 l3 = 1
    print("\n--- 测试 3: 特殊情况 (P_l3=1) ---")
    
    dim_values = {'K': 4, 'C': 2, 'P': 1, 'Q': 2}
    input_dims = {'C', 'P', 'Q'}
    
    test_perms = [
        ('C', 'Q', 'K', 'P'),  # K 后面的 P_l3=1
        ('C', 'Q', 'P', 'K'),  # K 最内层
        ('K', 'C', 'Q', 'P'),  # K 最外层
    ]
    
    print(f"dim_values: {dim_values}")
    print(f"Input dims: {input_dims}")
    print(f"\n{'Perm':<12} {'Formula':<10} {'Note':<30}")
    print("-"*55)
    
    for perm in test_perms:
        result = compute_input_tile_changes(dim_values, perm, input_dims)
        perm_str = ''.join([p[0] for p in perm])
        
        k_pos = perm.index('K')
        if k_pos == 3:
            note = "K 最内层"
        elif perm.index('P') > k_pos and dim_values['P'] == 1:
            note = "K 后面只有 P(=1), 等价于最内层"
        else:
            note = "K 后面有有效 Input dims"
        
        print(f"{perm_str:<12} {result:<10} {note:<30}")


def main():
    test_formula()
    
    print("\n" + "="*70)
    print("最终公式总结")
    print("="*70)
    print("""
================================================================================
Row Activation 计算公式
================================================================================

定义:
  - Input dims: 影响 Input tensor 访问的维度 = {C, P, Q, R, S}
    (R, S 只在它们在 L3 层展开时才算)
  - Non-input dims: 不影响 Input 访问的维度 = {K}

公式:
  base = ∏(所有参与 L3 的 Input dims 的 _l3 值)
  
  multiplier = 1
  For each non-input dim d (只有 K):
      If d 后面有 _l3 值 > 1 的 Input dim:
          multiplier *= d_l3
          
  Input_tile_changes = base × multiplier
  
  Row_switches = Input_tile_changes × rows_per_tile
  
其中:
  rows_per_tile = ceil(input_tile_size / block_w)
  input_tile_size = C_buffer × (P_buffer + R - 1) × (Q_buffer + S - 1)

================================================================================
常见场景
================================================================================

场景 1: R, S 在 L2 层 (最常见)
  Input dims = {C, P, Q}
  
  If K 在最内层 OR K 后面的 Input dims 的 _l3 值都是 1:
      tile_changes = C_l3 × P_l3 × Q_l3
  Else:
      tile_changes = C_l3 × P_l3 × Q_l3 × K_l3

场景 2: R 在 L3 层
  Input dims = {C, P, Q, R}
  公式类似，只是 base 多了 R_l3

场景 3: R, S 都在 L3 层
  Input dims = {C, P, Q, R, S}
  公式类似，base 多了 R_l3 × S_l3

================================================================================
""")


if __name__ == '__main__':
    main()
