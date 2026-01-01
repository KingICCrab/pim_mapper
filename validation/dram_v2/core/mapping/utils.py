#!/usr/bin/env python3
"""
工具函数
"""

from typing import List, Tuple


def get_divisors(n: int) -> List[int]:
    """获取 n 的所有因子（升序）"""
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted(divisors)


def get_2level_decompositions(n: int) -> List[Tuple[int, int]]:
    """获取 n 的所有 2 层因子组合
    
    L2 和 L3 独立选择因子，约束: l2 × l3 也是 n 的因子
    
    Returns:
        [(l2, l3), ...] 其中 l2, l3, l2×l3 都是 n 的因子
    """
    divisors = set(get_divisors(n))
    return [(d1, d2) for d1 in divisors for d2 in divisors if d1 * d2 in divisors]


def get_factor_decompositions(n: int, num_levels: int = 4) -> List[Tuple[int, ...]]:
    """获取 n 的所有 num_levels 层因子分解
    
    返回所有满足 f0 * f1 * ... * f_{num_levels-1} = n 的组合
    
    Args:
        n: 要分解的数
        num_levels: 层级数量
        
    Returns:
        所有因子分解的列表，每个元素是 (f0, f1, ..., f_{num_levels-1})
    """
    if num_levels == 1:
        return [(n,)]
    
    divisors = get_divisors(n)
    results = []
    
    for d in divisors:
        remaining = n // d
        sub_decomps = get_factor_decompositions(remaining, num_levels - 1)
        for sub in sub_decomps:
            results.append((d,) + sub)
    
    return results
