#!/usr/bin/env python3
"""
Permutation 影响分析 - 精简版

目标: 快速分析不同 permutation 对公式准确性的影响
只输出关键汇总信息
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

import math
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import List, Dict, Tuple
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
    
    @property
    def H_in(self) -> int:
        return self.P + self.R - 1
    
    @property
    def W_in(self) -> int:
        return self.Q + self.S - 1
    
    def get_divisors(self, dim_value: int) -> List[int]:
        return [i for i in range(1, dim_value + 1) if dim_value % i == 0]


@dataclass
class Mapping:
    name: str
    P_l3: int
    Q_l3: int
    C_l3: int
    K_l3: int
    R_l2: int
    l3_permutation: Tuple[str, ...]
    block_h: int
    block_w: int
    workload: 'Workload'


# 硬件参数
PE_ARRAY_H = 16
PE_ARRAY_W = 16
GLOBAL_BUFFER_SIZE = 65536
ROW_BUFFER_SIZE = 4096


def compute_exact_row_switches(m: Mapping) -> int:
    """精确计算 row switches"""
    w = m.workload
    
    P_l3, Q_l3, C_l3, K_l3 = m.P_l3, m.Q_l3, m.C_l3, m.K_l3
    P_buffer = w.P // P_l3
    Q_buffer = w.Q // Q_l3
    C_buffer = w.C // C_l3
    K_buffer = w.K // K_l3
    
    n_l3_iter = P_l3 * Q_l3 * C_l3 * K_l3
    
    # 计算 tile 大小
    h_tile = P_buffer + w.R - 1
    w_tile = Q_buffer + w.S - 1
    input_tile_size = C_buffer * h_tile * w_tile
    
    # 根据 block 计算每个 tile 的行数
    rows_per_tile = math.ceil(input_tile_size / m.block_w)
    
    # 计算 Input 相关维度的 L3 tiles 数量
    perm = m.l3_permutation
    
    # 模拟循环计算 row switches
    row_switches = 0
    last_c, last_p, last_q = -1, -1, -1
    
    # 按 permutation 顺序展开循环
    ranges = {
        'P': range(P_l3), 'Q': range(Q_l3),
        'C': range(C_l3), 'K': range(K_l3)
    }
    
    for i0 in ranges[perm[0]]:
        for i1 in ranges[perm[1]]:
            for i2 in ranges[perm[2]]:
                for i3 in ranges[perm[3]]:
                    indices = {perm[0]: i0, perm[1]: i1, perm[2]: i2, perm[3]: i3}
                    c = indices['C']
                    p = indices['P']
                    q = indices['Q']
                    
                    if (c != last_c or p != last_p or q != last_q):
                        row_switches += rows_per_tile
                        last_c, last_p, last_q = c, p, q
    
    return row_switches


def compute_formula_v1(m: Mapping) -> int:
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


def generate_mappings(workload: Workload, all_perms: List[Tuple[str, ...]], max_per_perm: int = 10):
    """为每个 permutation 生成 mappings"""
    mappings = defaultdict(list)
    
    P_divs = workload.get_divisors(workload.P)
    Q_divs = workload.get_divisors(workload.Q)
    C_divs = workload.get_divisors(workload.C)
    K_divs = workload.get_divisors(workload.K)
    R_divs = workload.get_divisors(workload.R)
    
    count = 0
    for perm in all_perms:
        # 为每个 perm 生成一些 mappings
        for P_l3 in P_divs[:5]:
            for Q_l3 in Q_divs[:5]:
                for C_l3 in C_divs[:4]:
                    for K_l3 in K_divs[:4]:
                        for R_l2 in R_divs[:2]:
                            if len(mappings[perm]) >= max_per_perm:
                                break
                            
                            P_buffer = workload.P // P_l3
                            Q_buffer = workload.Q // Q_l3
                            C_buffer = workload.C // C_l3
                            K_buffer = workload.K // K_l3
                            
                            if P_buffer * Q_buffer > PE_ARRAY_H * PE_ARRAY_W:
                                continue
                            if C_buffer * K_buffer > PE_ARRAY_H * PE_ARRAY_W:
                                continue
                            
                            h_tile = P_buffer + workload.R - 1
                            w_tile = Q_buffer + workload.S - 1
                            input_tile = C_buffer * h_tile * w_tile
                            weight_tile = K_buffer * C_buffer * workload.R * workload.S
                            output_tile = K_buffer * P_buffer * Q_buffer
                            
                            total_buffer = input_tile + weight_tile + output_tile
                            if total_buffer > GLOBAL_BUFFER_SIZE:
                                continue
                            
                            for block_h in [1, 4, 16]:
                                for block_w in [256, 512, 1024]:
                                    if len(mappings[perm]) >= max_per_perm:
                                        break
                                    
                                    m = Mapping(
                                        name=f"perm{''.join([p[0] for p in perm])}_P{P_l3}Q{Q_l3}C{C_l3}K{K_l3}",
                                        P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3,
                                        R_l2=R_l2,
                                        l3_permutation=perm,
                                        block_h=block_h, block_w=block_w,
                                        workload=workload
                                    )
                                    mappings[perm].append(m)
    
    return mappings


def analyze_permutation(perm: Tuple[str, ...], mappings: List[Mapping]):
    """分析单个 permutation"""
    exact_vals = [compute_exact_row_switches(m) for m in mappings]
    v1_vals = [compute_formula_v1(m) for m in mappings]
    
    if len(set(exact_vals)) < 2:
        return None
    
    corr_v1, _ = stats.spearmanr(exact_vals, v1_vals)
    avg_err = np.mean([abs(v1 - ex) / max(1, ex) * 100 for ex, v1 in zip(exact_vals, v1_vals)])
    
    return {
        'perm': perm,
        'perm_str': ''.join([p[0] for p in perm]),
        'n': len(mappings),
        'corr': corr_v1,
        'avg_err': avg_err,
        'k_pos': perm.index('K'),
        'inner_dim': perm[-1],
    }


def main():
    print("="*70)
    print("Permutation 影响分析: K 位置 vs 公式准确性")
    print("="*70)
    
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
    ]
    
    # 按 K 位置统计
    k_pos_stats = defaultdict(lambda: {'corrs': [], 'errs': []})
    inner_stats = defaultdict(lambda: {'corrs': [], 'errs': []})
    
    all_results = []
    
    for workload in workloads:
        print(f"\n--- {workload.name} (P={workload.P}, Q={workload.Q}, C={workload.C}, K={workload.K}) ---")
        
        mappings = generate_mappings(workload, all_perms, max_per_perm=10)
        
        for perm in all_perms:
            if perm in mappings and len(mappings[perm]) >= 3:
                result = analyze_permutation(perm, mappings[perm])
                if result:
                    all_results.append(result)
                    k_pos_stats[result['k_pos']]['corrs'].append(result['corr'])
                    k_pos_stats[result['k_pos']]['errs'].append(result['avg_err'])
                    inner_stats[result['inner_dim']]['corrs'].append(result['corr'])
                    inner_stats[result['inner_dim']]['errs'].append(result['avg_err'])
    
    print("\n" + "="*70)
    print("汇总: K 位置 vs 公式准确性")
    print("="*70)
    print(f"\n{'K Position':>12}  {'Avg ρ':>10}  {'Avg Err':>10}  {'#Cases':>8}")
    print("-"*50)
    for pos in sorted(k_pos_stats.keys()):
        stats_data = k_pos_stats[pos]
        avg_corr = np.mean(stats_data['corrs'])
        avg_err = np.mean(stats_data['errs'])
        n = len(stats_data['corrs'])
        print(f"{'K at pos ' + str(pos):>12}  {avg_corr:>10.3f}  {avg_err:>9.1f}%  {n:>8}")
    
    print("\n" + "="*70)
    print("汇总: 最内层维度 vs 公式准确性")
    print("="*70)
    print(f"\n{'Inner Dim':>12}  {'Avg ρ':>10}  {'Avg Err':>10}  {'#Cases':>8}")
    print("-"*50)
    for dim in ['K', 'C', 'P', 'Q']:
        if dim in inner_stats:
            stats_data = inner_stats[dim]
            avg_corr = np.mean(stats_data['corrs'])
            avg_err = np.mean(stats_data['errs'])
            n = len(stats_data['corrs'])
            print(f"{dim:>12}  {avg_corr:>10.3f}  {avg_err:>9.1f}%  {n:>8}")
    
    # 找出问题 permutation
    print("\n" + "="*70)
    print("具体 Permutation 分析 (按 K 位置分组)")
    print("="*70)
    
    # 按 K 位置分组
    by_k_pos = defaultdict(list)
    for r in all_results:
        by_k_pos[r['k_pos']].append(r)
    
    for pos in sorted(by_k_pos.keys()):
        print(f"\n--- K 在位置 {pos} (0=最外层循环) ---")
        results = by_k_pos[pos]
        
        # 按相关系数排序
        results.sort(key=lambda x: x['corr'], reverse=True)
        
        # 显示几个代表性的
        perms_shown = set()
        for r in results:
            if r['perm_str'] not in perms_shown:
                perms_shown.add(r['perm_str'])
                print(f"  {r['perm_str']}: ρ = {r['corr']:.3f}, err = {r['avg_err']:.1f}%")
            if len(perms_shown) >= 6:
                break
    
    # 总结
    print("\n" + "="*70)
    print("结论")
    print("="*70)
    
    # 计算各位置的平均相关系数
    pos_avgs = {pos: np.mean(data['corrs']) for pos, data in k_pos_stats.items()}
    
    best_pos = max(pos_avgs, key=pos_avgs.get)
    worst_pos = min(pos_avgs, key=pos_avgs.get)
    
    print(f"\n1. K 位置影响:")
    print(f"   - K 在位置 {best_pos} 时公式最准确 (ρ = {pos_avgs[best_pos]:.3f})")
    print(f"   - K 在位置 {worst_pos} 时公式最不准确 (ρ = {pos_avgs[worst_pos]:.3f})")
    
    # 内层维度影响
    inner_avgs = {dim: np.mean(data['corrs']) for dim, data in inner_stats.items()}
    
    best_inner = max(inner_avgs, key=inner_avgs.get)
    worst_inner = min(inner_avgs, key=inner_avgs.get)
    
    print(f"\n2. 最内层维度影响:")
    print(f"   - {best_inner} 在最内层时公式最准确 (ρ = {inner_avgs[best_inner]:.3f})")
    print(f"   - {worst_inner} 在最内层时公式最不准确 (ρ = {inner_avgs[worst_inner]:.3f})")
    
    print(f"\n3. 公式改进方向:")
    print(f"   当前公式: N_input_tiles × rows_per_tile")
    print(f"   问题: 当 K 在位置 {worst_pos} 时，K 循环变化不会引起 Input tile 切换")
    print(f"         但公式没有考虑这一点")


if __name__ == '__main__':
    main()
