#!/usr/bin/env python3
"""
验证修正后的公式

关键发现: K 位置影响 Input tile 切换次数
- K 在最内层: Input_tiles = P_l3 × Q_l3 × C_l3
- K 不在最内层: 需要考虑 K 内部有多少 Input 相关维度

修正公式:
  设 dims_after_K = K 后面的维度中 Input 相关 (C/P/Q) 的数量
  If dims_after_K == 0 (K 最内层):
      multiplier = 1
  Else:
      multiplier = K_l3 (因为每次 K 变化，内层 Input dims 重新遍历)
  
  Input_tile_changes = P_l3 × Q_l3 × C_l3 × multiplier
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
    N: int
    K: int
    C: int
    P: int
    Q: int
    R: int
    S: int
    
    def get_divisors(self, dim_value: int) -> List[int]:
        return [i for i in range(1, dim_value + 1) if dim_value % i == 0]


@dataclass
class Mapping:
    P_l3: int
    Q_l3: int
    C_l3: int
    K_l3: int
    l3_permutation: Tuple[str, ...]
    block_w: int
    workload: 'Workload'


def compute_exact(m: Mapping) -> int:
    """精确模拟"""
    w = m.workload
    perm = m.l3_permutation
    
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
    
    row_switches = 0
    last_input = None
    
    for i0 in ranges[perm[0]]:
        for i1 in ranges[perm[1]]:
            for i2 in ranges[perm[2]]:
                for i3 in ranges[perm[3]]:
                    indices = {perm[0]: i0, perm[1]: i1, perm[2]: i2, perm[3]: i3}
                    input_tile = (indices['C'], indices['P'], indices['Q'])
                    
                    if input_tile != last_input:
                        row_switches += rows_per_tile
                        last_input = input_tile
    
    return row_switches


def compute_formula_v1(m: Mapping) -> int:
    """原始公式 (不考虑 permutation)"""
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


def compute_formula_v2(m: Mapping) -> int:
    """修正公式 (考虑 K 位置)"""
    w = m.workload
    perm = m.l3_permutation
    
    P_buffer = w.P // m.P_l3
    Q_buffer = w.Q // m.Q_l3
    C_buffer = w.C // m.C_l3
    
    h_tile = P_buffer + w.R - 1
    w_tile = Q_buffer + w.S - 1
    input_tile_size = C_buffer * h_tile * w_tile
    rows_per_tile = math.ceil(input_tile_size / m.block_w)
    
    # 基础 Input tiles
    base_input_tiles = m.P_l3 * m.Q_l3 * m.C_l3
    
    # 找 K 的位置
    k_pos = perm.index('K')
    
    # K 后面的维度中 Input 相关的有哪些
    input_dims = {'C', 'P', 'Q'}
    dims_after_k = [perm[i] for i in range(k_pos + 1, 4)]
    input_dims_after_k = [d for d in dims_after_k if d in input_dims]
    
    # 如果 K 后面有 Input 相关维度，K 每变化一次会导致重复遍历
    if len(input_dims_after_k) > 0:
        multiplier = m.K_l3
    else:
        multiplier = 1
    
    n_input_tiles = base_input_tiles * multiplier
    return n_input_tiles * rows_per_tile


def generate_mappings(workload: Workload, all_perms, max_per_perm=15):
    """生成测试 mappings"""
    mappings = defaultdict(list)
    
    P_divs = workload.get_divisors(workload.P)
    Q_divs = workload.get_divisors(workload.Q)
    C_divs = workload.get_divisors(workload.C)
    K_divs = workload.get_divisors(workload.K)
    
    PE_H = PE_W = 16
    BUFFER = 65536
    
    for perm in all_perms:
        for P_l3 in P_divs[:6]:
            for Q_l3 in Q_divs[:6]:
                for C_l3 in C_divs[:5]:
                    for K_l3 in K_divs[:5]:
                        if len(mappings[perm]) >= max_per_perm:
                            break
                        
                        P_buf = workload.P // P_l3
                        Q_buf = workload.Q // Q_l3
                        C_buf = workload.C // C_l3
                        K_buf = workload.K // K_l3
                        
                        if P_buf * Q_buf > PE_H * PE_W:
                            continue
                        if C_buf * K_buf > PE_H * PE_W:
                            continue
                        
                        h_tile = P_buf + workload.R - 1
                        w_tile = Q_buf + workload.S - 1
                        input_tile = C_buf * h_tile * w_tile
                        weight_tile = K_buf * C_buf * workload.R * workload.S
                        output_tile = K_buf * P_buf * Q_buf
                        
                        if input_tile + weight_tile + output_tile > BUFFER:
                            continue
                        
                        for block_w in [256, 512, 1024]:
                            if len(mappings[perm]) >= max_per_perm:
                                break
                            
                            m = Mapping(
                                P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3,
                                l3_permutation=perm,
                                block_w=block_w,
                                workload=workload
                            )
                            mappings[perm].append(m)
    
    return mappings


def main():
    print("="*80)
    print("验证修正公式 V2")
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
        Workload(name='ResNet-L1', N=1, K=64, C=3, P=56, Q=56, R=7, S=7),
        Workload(name='ResNet-3x3', N=1, K=64, C=64, P=56, Q=56, R=3, S=3),
        Workload(name='VGG-conv5', N=1, K=512, C=512, P=14, Q=14, R=3, S=3),
        Workload(name='MobileNet', N=1, K=32, C=32, P=112, Q=112, R=3, S=3),
    ]
    
    # 按 K 位置统计
    k_pos_v1 = defaultdict(list)  # {k_pos: [err%]}
    k_pos_v2 = defaultdict(list)
    
    all_v1_errs = []
    all_v2_errs = []
    
    for w in workloads:
        print(f"\n{'='*60}")
        print(f"Workload: {w.name}")
        print(f"  P={w.P}, Q={w.Q}, C={w.C}, K={w.K}, R={w.R}")
        print(f"{'='*60}")
        
        mappings = generate_mappings(w, all_perms, max_per_perm=15)
        
        for perm in all_perms:
            if perm not in mappings or len(mappings[perm]) < 3:
                continue
            
            k_pos = perm.index('K')
            perm_str = ''.join([p[0] for p in perm])
            
            exact_list = []
            v1_list = []
            v2_list = []
            
            for m in mappings[perm]:
                exact = compute_exact(m)
                v1 = compute_formula_v1(m)
                v2 = compute_formula_v2(m)
                
                exact_list.append(exact)
                v1_list.append(v1)
                v2_list.append(v2)
            
            # 计算相关系数
            if len(set(exact_list)) >= 2:
                corr_v1, _ = stats.spearmanr(exact_list, v1_list)
                corr_v2, _ = stats.spearmanr(exact_list, v2_list)
            else:
                corr_v1 = corr_v2 = 1.0
            
            # 计算误差
            err_v1 = np.mean([abs(v1 - ex) / max(1, ex) * 100 
                             for ex, v1 in zip(exact_list, v1_list)])
            err_v2 = np.mean([abs(v2 - ex) / max(1, ex) * 100 
                             for ex, v2 in zip(exact_list, v2_list)])
            
            k_pos_v1[k_pos].append(err_v1)
            k_pos_v2[k_pos].append(err_v2)
            all_v1_errs.append(err_v1)
            all_v2_errs.append(err_v2)
        
        # 每个 workload 的简要汇总
        print(f"\n  K 位置  |  V1 Err%  |  V2 Err%")
        print(f"  {'-'*40}")
        for pos in [0, 1, 2, 3]:
            if pos in k_pos_v1:
                # 只取当前 workload 的数据
                n_perms_per_pos = 6  # 每个位置有 6 个 permutation
                start_idx = len(k_pos_v1[pos]) - n_perms_per_pos
                v1_errs = k_pos_v1[pos][start_idx:]
                v2_errs = k_pos_v2[pos][start_idx:]
                if v1_errs:
                    print(f"  pos={pos}   | {np.mean(v1_errs):8.1f}% | {np.mean(v2_errs):8.1f}%")
    
    # 总体汇总
    print("\n" + "="*80)
    print("总体汇总")
    print("="*80)
    
    print(f"\n{'K Position':<12} {'V1 Error':>12} {'V2 Error':>12} {'Cases':>8}")
    print("-"*50)
    for pos in [0, 1, 2, 3]:
        v1_avg = np.mean(k_pos_v1[pos])
        v2_avg = np.mean(k_pos_v2[pos])
        n = len(k_pos_v1[pos])
        print(f"{'K at pos ' + str(pos):<12} {v1_avg:>11.1f}% {v2_avg:>11.1f}% {n:>8}")
    
    print(f"\n{'Overall':<12} {np.mean(all_v1_errs):>11.1f}% {np.mean(all_v2_errs):>11.1f}%")
    
    print("\n" + "="*80)
    print("结论")
    print("="*80)
    print(f"""
原公式 V1: Row_switches = P_l3 × Q_l3 × C_l3 × rows_per_tile
  - 不考虑 permutation
  - 平均误差: {np.mean(all_v1_errs):.1f}%

修正公式 V2: 
  If K 在最内层: Row_switches = P_l3 × Q_l3 × C_l3 × rows_per_tile
  Else:          Row_switches = P_l3 × Q_l3 × C_l3 × K_l3 × rows_per_tile
  - 考虑 K 位置对 Input reuse 的影响
  - 平均误差: {np.mean(all_v2_errs):.1f}%
""")


if __name__ == '__main__':
    main()
