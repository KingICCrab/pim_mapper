#!/usr/bin/env python3
"""
最终验证: 公式 V4

正确公式:
  inner_input_product = ∏(K之后的 Input dims 的 _l3 值)
  
  如果 inner_input_product == 1:
      tile_changes = P_l3 × Q_l3 × C_l3
  否则:
      tile_changes = P_l3 × Q_l3 × C_l3 × K_l3
      
  Row_switches = tile_changes × rows_per_tile
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

import math
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import List, Tuple
from collections import defaultdict


@dataclass
class Workload:
    name: str
    K: int
    C: int
    P: int
    Q: int
    R: int
    S: int
    
    def get_divisors(self, val: int) -> List[int]:
        return [i for i in range(1, val + 1) if val % i == 0]


@dataclass 
class Mapping:
    P_l3: int
    Q_l3: int
    C_l3: int
    K_l3: int
    perm: Tuple[str, ...]
    block_w: int
    workload: 'Workload'


PE_SIZE = 256
BUFFER = 65536


def compute_exact(m: Mapping) -> int:
    """精确模拟"""
    w = m.workload
    perm = m.perm
    
    P_buf = w.P // m.P_l3
    Q_buf = w.Q // m.Q_l3
    C_buf = w.C // m.C_l3
    
    h_tile = P_buf + w.R - 1
    w_tile = Q_buf + w.S - 1
    input_tile_size = C_buf * h_tile * w_tile
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
    
    return tile_changes * rows_per_tile


def compute_formula_v4(m: Mapping) -> int:
    """
    正确公式 V4
    
    inner_input_product = ∏(K之后的 Input dims 的 _l3 值)
    如果 inner_input_product == 1: tile_changes = P_l3 × Q_l3 × C_l3
    否则: tile_changes = P_l3 × Q_l3 × C_l3 × K_l3
    """
    w = m.workload
    perm = m.perm
    
    P_buf = w.P // m.P_l3
    Q_buf = w.Q // m.Q_l3
    C_buf = w.C // m.C_l3
    
    h_tile = P_buf + w.R - 1
    w_tile = Q_buf + w.S - 1
    input_tile_size = C_buf * h_tile * w_tile
    rows_per_tile = math.ceil(input_tile_size / m.block_w)
    
    # 找 K 位置
    k_pos = perm.index('K')
    
    # K 之后的 Input dims
    input_dims = {'C', 'P', 'Q'}
    dim_values = {'P': m.P_l3, 'Q': m.Q_l3, 'C': m.C_l3, 'K': m.K_l3}
    
    inner_input_product = 1
    for i in range(k_pos + 1, 4):
        d = perm[i]
        if d in input_dims:
            inner_input_product *= dim_values[d]
    
    # 基础 tiles
    base = m.P_l3 * m.Q_l3 * m.C_l3
    
    if inner_input_product == 1:
        tile_changes = base
    else:
        tile_changes = base * m.K_l3
    
    return tile_changes * rows_per_tile


def compute_formula_v1(m: Mapping) -> int:
    """原始公式 (不考虑 permutation)"""
    w = m.workload
    
    P_buf = w.P // m.P_l3
    Q_buf = w.Q // m.Q_l3
    C_buf = w.C // m.C_l3
    
    h_tile = P_buf + w.R - 1
    w_tile = Q_buf + w.S - 1
    input_tile_size = C_buf * h_tile * w_tile
    rows_per_tile = math.ceil(input_tile_size / m.block_w)
    
    n_input_tiles = m.P_l3 * m.Q_l3 * m.C_l3
    return n_input_tiles * rows_per_tile


def generate_mappings(w: Workload, all_perms, max_per_perm=10):
    """生成合法 mappings"""
    mappings = defaultdict(list)
    
    P_divs = w.get_divisors(w.P)
    Q_divs = w.get_divisors(w.Q)
    C_divs = w.get_divisors(w.C)
    K_divs = w.get_divisors(w.K)
    
    for perm in all_perms:
        for P_l3 in P_divs[:6]:
            for Q_l3 in Q_divs[:6]:
                for C_l3 in C_divs[:5]:
                    for K_l3 in K_divs[:5]:
                        if len(mappings[perm]) >= max_per_perm:
                            break
                        
                        P_buf = w.P // P_l3
                        Q_buf = w.Q // Q_l3
                        C_buf = w.C // C_l3
                        K_buf = w.K // K_l3
                        
                        if P_buf * Q_buf > PE_SIZE:
                            continue
                        if C_buf * K_buf > PE_SIZE:
                            continue
                        
                        h = P_buf + w.R - 1
                        ww = Q_buf + w.S - 1
                        inp = C_buf * h * ww
                        wgt = K_buf * C_buf * w.R * w.S
                        out = K_buf * P_buf * Q_buf
                        
                        if inp + wgt + out > BUFFER:
                            continue
                        
                        for bw in [256, 512, 1024]:
                            if len(mappings[perm]) >= max_per_perm:
                                break
                            
                            mappings[perm].append(Mapping(
                                P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3,
                                perm=perm, block_w=bw, workload=w
                            ))
    
    return mappings


def main():
    print("="*80)
    print("最终验证: 公式 V4")
    print("="*80)
    
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
    
    workloads = [
        Workload(name='ResNet-L1', K=64, C=3, P=56, Q=56, R=7, S=7),
        Workload(name='ResNet-3x3', K=64, C=64, P=56, Q=56, R=3, S=3),
        Workload(name='VGG-conv5', K=512, C=512, P=14, Q=14, R=3, S=3),
        Workload(name='MobileNet', K=32, C=32, P=112, Q=112, R=3, S=3),
    ]
    
    total_v1_errs = []
    total_v4_errs = []
    total_exact_matches = 0
    total_mappings = 0
    
    for w in workloads:
        print(f"\n{'='*60}")
        print(f"Workload: {w.name}")
        print(f"  K={w.K}, C={w.C}, P={w.P}, Q={w.Q}, R={w.R}")
        print(f"{'='*60}")
        
        mappings = generate_mappings(w, all_perms, max_per_perm=10)
        
        v1_errs = []
        v4_errs = []
        exact_matches = 0
        n_mappings = 0
        
        for perm in all_perms:
            if perm not in mappings:
                continue
            
            for m in mappings[perm]:
                exact = compute_exact(m)
                v1 = compute_formula_v1(m)
                v4 = compute_formula_v4(m)
                
                err_v1 = abs(v1 - exact) / max(1, exact) * 100
                err_v4 = abs(v4 - exact) / max(1, exact) * 100
                
                v1_errs.append(err_v1)
                v4_errs.append(err_v4)
                
                if v4 == exact:
                    exact_matches += 1
                
                n_mappings += 1
        
        avg_v1 = np.mean(v1_errs) if v1_errs else 0
        avg_v4 = np.mean(v4_errs) if v4_errs else 0
        
        print(f"\n  Mappings: {n_mappings}")
        if n_mappings > 0:
            print(f"  V1 平均误差: {avg_v1:.1f}%")
            print(f"  V4 平均误差: {avg_v4:.1f}%")
            print(f"  V4 精确匹配: {exact_matches}/{n_mappings} ({100*exact_matches/n_mappings:.1f}%)")
        
        total_v1_errs.extend(v1_errs)
        total_v4_errs.extend(v4_errs)
        total_exact_matches += exact_matches
        total_mappings += n_mappings
    
    # 总体汇总
    print("\n" + "="*80)
    print("总体汇总")
    print("="*80)
    
    print(f"\n总 Mappings: {total_mappings}")
    print(f"\nV1 (原公式, 不考虑 permutation):")
    print(f"  平均误差: {np.mean(total_v1_errs):.1f}%")
    
    print(f"\nV4 (新公式, 考虑 K 位置):")
    print(f"  平均误差: {np.mean(total_v4_errs):.1f}%")
    print(f"  精确匹配: {total_exact_matches}/{total_mappings} ({100*total_exact_matches/total_mappings:.1f}%)")
    
    # 相关性分析
    print("\n" + "="*80)
    print("排序能力 (Spearman 相关系数)")
    print("="*80)
    
    # 收集所有数据点
    all_exact = []
    all_v1 = []
    all_v4 = []
    
    for w in workloads:
        mappings = generate_mappings(w, all_perms, max_per_perm=10)
        
        for perm in all_perms:
            if perm not in mappings:
                continue
            
            for m in mappings[perm]:
                all_exact.append(compute_exact(m))
                all_v1.append(compute_formula_v1(m))
                all_v4.append(compute_formula_v4(m))
    
    corr_v1, _ = stats.spearmanr(all_exact, all_v1)
    corr_v4, _ = stats.spearmanr(all_exact, all_v4)
    
    print(f"\n  V1 Spearman ρ: {corr_v1:.4f}")
    print(f"  V4 Spearman ρ: {corr_v4:.4f}")
    
    # 最终公式
    print("\n" + "="*80)
    print("最终公式 V4")
    print("="*80)
    print("""
Input_tile_changes 计算:
  1. 找到 K 在 permutation 中的位置 pos_k
  2. 计算 K 之后的 Input dims (C/P/Q) 的 _l3 值的乘积: inner_product
  3. 如果 inner_product == 1:
       tile_changes = P_l3 × Q_l3 × C_l3
     否则:
       tile_changes = P_l3 × Q_l3 × C_l3 × K_l3

Row_switches = tile_changes × rows_per_tile

其中:
  rows_per_tile = ceil(input_tile_size / block_w)
  input_tile_size = C_buffer × (P_buffer + R - 1) × (Q_buffer + S - 1)

物理解释:
  - K 与 Input 无关，K 循环内的 Input tile 不变
  - 但如果 K 循环内部有 Input 相关维度 (C/P/Q)，每次 K 变化会导致重新遍历
  - 如果 K 内部的 Input dims 的 _l3 值都是 1，则没有重遍历
""")


if __name__ == '__main__':
    main()
