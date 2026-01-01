#!/usr/bin/env python3
"""
按 Permutation 分组分析

目标: 分别测试每个 permutation 下公式的准确性
- 对每个 permutation，计算 Formula vs Exact 的相关系数
- 分析哪些 permutation 公式预测准确，哪些不准确
- 根据结果思考模型改进方向
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

import math
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
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
    workload: Optional[Workload] = None
    
    @property
    def P_buffer(self): return self.workload.P // self.P_l3 if self.workload else 0
    @property
    def Q_buffer(self): return self.workload.Q // self.Q_l3 if self.workload else 0
    @property
    def C_buffer(self): return self.workload.C // self.C_l3 if self.workload else 0
    @property
    def K_buffer(self): return self.workload.K // self.K_l3 if self.workload else 0
    @property
    def R_buffer(self): return self.workload.R // self.R_l2 if self.workload else 0


def compute_exact_row_switches(mapping: Mapping) -> int:
    """精确模拟"""
    w = mapping.workload
    perm = mapping.l3_permutation
    
    num_h_blocks = math.ceil(w.H_in / mapping.block_h)
    num_w_blocks = math.ceil(w.W_in / mapping.block_w)
    
    ranges = {
        'K': range(mapping.K_l3),
        'C': range(mapping.C_l3),
        'P': range(mapping.P_l3),
        'Q': range(mapping.Q_l3),
    }
    
    row_switches = 0
    last_block = None
    
    def iterate(level, indices):
        nonlocal row_switches, last_block
        
        if level == len(perm):
            k, c, p, q = indices['K'], indices['C'], indices['P'], indices['Q']
            
            for r in range(mapping.R_l2):
                h = p * mapping.P_buffer + r
                w_start = q * mapping.Q_buffer
                w_end = q * mapping.Q_buffer + mapping.Q_buffer + w.S - 2
                
                hb = min(h // mapping.block_h, num_h_blocks - 1)
                wb_start = min(w_start // mapping.block_w, num_w_blocks - 1)
                wb_end = min(w_end // mapping.block_w, num_w_blocks - 1)
                
                for wb in range(wb_start, wb_end + 1):
                    curr_block = (c, hb, wb)
                    if last_block is not None and curr_block != last_block:
                        row_switches += 1
                    last_block = curr_block
        else:
            dim = perm[level]
            for i in ranges[dim]:
                new_indices = indices.copy()
                new_indices[dim] = i
                iterate(level + 1, new_indices)
    
    iterate(0, {'K': 0, 'C': 0, 'P': 0, 'Q': 0})
    return row_switches


def compute_formula_v1(mapping: Mapping) -> int:
    """
    公式 V1: 当前版本
    """
    w = mapping.workload
    perm = mapping.l3_permutation
    
    num_h_blocks = math.ceil(w.H_in / mapping.block_h)
    num_w_blocks = math.ceil(w.W_in / mapping.block_w)
    
    # Q tiles 跨 block 分析
    q_crossing = 0
    total_multi_block = 0
    for q in range(mapping.Q_l3):
        w_start = q * mapping.Q_buffer
        w_end = w_start + mapping.Q_buffer + w.S - 2
        wb_start = w_start // mapping.block_w
        wb_end = min(w_end // mapping.block_w, num_w_blocks - 1)
        blocks = wb_end - wb_start
        total_multi_block += blocks
        if blocks > 0:
            q_crossing += 1
    
    pos = {dim: perm.index(dim) for dim in perm}
    
    # 1. multi_block
    multi_block = total_multi_block * mapping.P_l3 * mapping.R_l2 * mapping.C_l3 * mapping.K_l3
    
    # 2. R switches
    r_switches = q_crossing * mapping.P_l3 * max(0, mapping.R_l2 - 1) * mapping.C_l3 * mapping.K_l3
    
    # 3. Q switches
    q_boundary = 0
    for q in range(1, mapping.Q_l3):
        w_end_prev = (q - 1) * mapping.Q_buffer + mapping.Q_buffer + w.S - 2
        wb_prev = min(w_end_prev // mapping.block_w, num_w_blocks - 1)
        wb_curr = q * mapping.Q_buffer // mapping.block_w
        if wb_prev != wb_curr:
            q_boundary += 1
    
    q_outer_mult = 1
    for dim in perm[:pos['Q']]:
        if dim == 'K': q_outer_mult *= mapping.K_l3
        elif dim == 'C': q_outer_mult *= mapping.C_l3
        elif dim == 'P': q_outer_mult *= mapping.P_l3
    q_switches = q_boundary * q_outer_mult * mapping.R_l2
    
    # 4. P switches
    p_outer_mult = 1
    for dim in perm[:pos['P']]:
        if dim == 'K': p_outer_mult *= mapping.K_l3
        elif dim == 'C': p_outer_mult *= mapping.C_l3
        elif dim == 'Q': p_outer_mult *= mapping.Q_l3
    p_switches = max(0, mapping.P_l3 - 1) * p_outer_mult
    
    # 5. C switches
    c_outer_mult = 1
    for dim in perm[:pos['C']]:
        if dim == 'K': c_outer_mult *= mapping.K_l3
        elif dim == 'P': c_outer_mult *= mapping.P_l3
        elif dim == 'Q': c_outer_mult *= mapping.Q_l3
    c_switches = max(0, mapping.C_l3 - 1) * c_outer_mult
    
    # 6. K switches
    if pos['K'] > 0:
        k_outer_mult = 1
        for dim in perm[:pos['K']]:
            if dim == 'C': k_outer_mult *= mapping.C_l3
            elif dim == 'P': k_outer_mult *= mapping.P_l3
            elif dim == 'Q': k_outer_mult *= mapping.Q_l3
        k_switches = max(0, mapping.K_l3 - 1) * k_outer_mult
    else:
        k_switches = 0
    
    return multi_block + r_switches + p_switches + q_switches + c_switches + k_switches


def compute_formula_v2(mapping: Mapping) -> int:
    """
    公式 V2: 改进版 - 更准确地处理 permutation
    
    核心思想:
    - Input 相关维度: C, P, Q (通过 c, h, w 坐标)
    - K 对 Input 无关
    - row switch 发生在 (c, hb, wb) 变化时
    
    对于每个维度变化:
    - C 变化: 总是 switch (不同 channel)
    - P 变化: 可能 switch (h 变化可能跨 h_block)
    - Q 变化: 可能 switch (w 变化可能跨 w_block)
    - K 变化: 如果回到不同的 (c, hb, wb) 则 switch
    """
    w = mapping.workload
    perm = mapping.l3_permutation
    
    num_h_blocks = math.ceil(w.H_in / mapping.block_h)
    num_w_blocks = math.ceil(w.W_in / mapping.block_w)
    
    pos = {dim: perm.index(dim) for dim in perm}
    
    # 分析每个 Q tile 是否跨 w_block
    q_multi_block = []
    for q in range(mapping.Q_l3):
        w_start = q * mapping.Q_buffer
        w_end = w_start + mapping.Q_buffer + w.S - 2
        wb_start = w_start // mapping.block_w
        wb_end = min(w_end // mapping.block_w, num_w_blocks - 1)
        q_multi_block.append(wb_end - wb_start)
    
    total_multi = sum(q_multi_block)
    q_crossing = sum(1 for x in q_multi_block if x > 0)
    
    # ====== 关键改进：根据 permutation 计算每种 switch 的次数 ======
    
    # 思路：遍历每个维度的每次变化，判断是否会导致 row switch
    # switch 条件: (c, hb, wb) 变化
    
    total_switches = 0
    
    # --- 1. Q tile 内部的 multi-block switches ---
    # 每次访问跨 block 的 Q tile 都会产生 (wb_end - wb_start) 个 switch
    # 总次数 = P_l3 × R_l2 × C_l3 × K_l3
    inner_mult = mapping.P_l3 * mapping.R_l2 * mapping.C_l3 * mapping.K_l3
    multi_block_switches = total_multi * inner_mult
    
    # --- 2. R 变化导致的 switches ---
    # R 变化改变 h，可能改变 hb
    # 只在跨 h_block 时 switch
    # 对于 multi-block Q tiles，R 变化后 wb 会从 wb_end 回到 wb_start
    r_changes_per_pq = max(0, mapping.R_l2 - 1)
    if q_crossing > 0:
        # multi-block Q tiles: R 变化后需要从 wb=wb_end 回到 wb=wb_start
        r_switches = q_crossing * mapping.P_l3 * r_changes_per_pq * mapping.C_l3 * mapping.K_l3
    else:
        r_switches = 0
    
    # --- 3. Q 变化导致的 switches ---
    # Q 变化改变 w，可能改变 wb
    q_boundary_switches = 0
    for q in range(1, mapping.Q_l3):
        # 从 Q=q-1 的最后一个访问 到 Q=q 的第一个访问
        # Q=q-1 最后访问的 wb = wb_end(q-1)
        # Q=q 第一个访问的 wb = wb_start(q) = q * Q_buffer // block_w
        w_end_prev = (q - 1) * mapping.Q_buffer + mapping.Q_buffer + w.S - 2
        wb_prev = min(w_end_prev // mapping.block_w, num_w_blocks - 1)
        wb_curr = q * mapping.Q_buffer // mapping.block_w
        if wb_prev != wb_curr:
            q_boundary_switches += 1
    
    # Q 变化的重复次数: Q 外层所有维度的乘积 × R_l2
    q_outer_mult = mapping.R_l2
    for dim in perm[:pos['Q']]:
        if dim == 'K': q_outer_mult *= mapping.K_l3
        elif dim == 'C': q_outer_mult *= mapping.C_l3
        elif dim == 'P': q_outer_mult *= mapping.P_l3
    q_switches = q_boundary_switches * q_outer_mult
    
    # --- 4. P 变化导致的 switches ---
    # P 变化改变 h 的起始位置
    # 从 P=p-1 到 P=p: h 从 (p-1)*P_buffer + R_l2-1 变到 p*P_buffer
    # 同时 Q 回到 0，所以 w 也变化
    p_changes = max(0, mapping.P_l3 - 1)
    p_outer_mult = 1
    for dim in perm[:pos['P']]:
        if dim == 'K': p_outer_mult *= mapping.K_l3
        elif dim == 'C': p_outer_mult *= mapping.C_l3
        elif dim == 'Q': p_outer_mult *= mapping.Q_l3
    p_switches = p_changes * p_outer_mult
    
    # --- 5. C 变化导致的 switches ---
    # C 变化总是导致 switch (c 不同)
    c_changes = max(0, mapping.C_l3 - 1)
    c_outer_mult = 1
    for dim in perm[:pos['C']]:
        if dim == 'K': c_outer_mult *= mapping.K_l3
        elif dim == 'P': c_outer_mult *= mapping.P_l3
        elif dim == 'Q': c_outer_mult *= mapping.Q_l3
    c_switches = c_changes * c_outer_mult
    
    # --- 6. K 变化导致的 switches ---
    # K 对 Input 无关，但 K 变化后内层循环重新开始
    # 关键问题: K 变化后，(c, hb, wb) 是否变化？
    # 
    # 如果 K 是最外层 (pos[K]=0): K 变化后内层从头开始，(c,hb,wb) 相同，不 switch
    # 如果 K 不是最外层: K 变化后外层维度不变，但内层重新开始
    #   - K 变化前: 处于内层循环的最后一个值
    #   - K 变化后: 内层循环从第一个值开始
    #   - 如果外层是 C/P/Q 任一，则 (c,hb,wb) 可能不同
    
    if pos['K'] == 0:
        # K 最外层: K 变化不导致 switch (内层从头开始，第一个访问和上一个 K 的第一个相同)
        k_switches = 0
    else:
        # K 不是最外层
        # K 变化后，K 外层的维度不变，K 内层的维度从第一个值开始
        # 比较: K 变化前的最后一个 (c,hb,wb) vs K 变化后的第一个 (c,hb,wb)
        # 
        # 简化: 如果 K 内层有 Input 相关维度 (C/P/Q)，则 K 变化会导致 switch
        k_inner_has_input_dim = any(perm[i] in ['C', 'P', 'Q'] for i in range(pos['K']+1, len(perm)))
        
        if k_inner_has_input_dim:
            k_changes = max(0, mapping.K_l3 - 1)
            k_outer_mult = 1
            for dim in perm[:pos['K']]:
                if dim == 'C': k_outer_mult *= mapping.C_l3
                elif dim == 'P': k_outer_mult *= mapping.P_l3
                elif dim == 'Q': k_outer_mult *= mapping.Q_l3
            k_switches = k_changes * k_outer_mult
        else:
            k_switches = 0
    
    return multi_block_switches + r_switches + p_switches + q_switches + c_switches + k_switches


def generate_mappings(workload: Workload, perms: List[Tuple[str, ...]], max_per_perm: int = 20):
    """为每个 permutation 生成多个 mapping"""
    mappings = defaultdict(list)
    
    P_divs = [d for d in workload.get_divisors(workload.P)][:6]
    Q_divs = [d for d in workload.get_divisors(workload.Q)][:6]
    C_divs = [d for d in workload.get_divisors(workload.C)][:5]
    K_divs = [d for d in workload.get_divisors(workload.K)][:5]
    
    buffer_limit = 65536
    
    for perm in perms:
        count = 0
        for P_l3 in P_divs:
            for Q_l3 in Q_divs:
                for C_l3 in C_divs:
                    for K_l3 in K_divs:
                        for R_l2 in [1, workload.R] if workload.R > 1 else [1]:
                            if count >= max_per_perm:
                                break
                            
                            P_buf = workload.P // P_l3
                            Q_buf = workload.Q // Q_l3
                            C_buf = workload.C // C_l3
                            K_buf = workload.K // K_l3
                            R_buf = workload.R // R_l2
                            
                            input_tile = (P_buf + R_buf - 1) * (Q_buf + workload.S - 1) * C_buf
                            weight_tile = R_buf * workload.S * C_buf * K_buf
                            output_tile = P_buf * Q_buf * K_buf
                            
                            if input_tile + weight_tile + output_tile > buffer_limit:
                                continue
                            
                            block_h = workload.H_in
                            block_w = max(1, workload.W_in // 2)
                            
                            perm_str = ''.join([p[0] for p in perm])
                            name = f"P{P_l3}_Q{Q_l3}_C{C_l3}_K{K_l3}_R{R_l2}"
                            
                            m = Mapping(
                                name=name,
                                P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3,
                                R_l2=R_l2,
                                l3_permutation=perm,
                                block_h=block_h, block_w=block_w,
                                workload=workload
                            )
                            
                            mappings[perm].append(m)
                            count += 1
    
    return mappings


def analyze_permutation(perm: Tuple[str, ...], mappings: List[Mapping]):
    """分析单个 permutation 下公式的准确性"""
    results = []
    
    for m in mappings:
        exact = compute_exact_row_switches(m)
        formula_v1 = compute_formula_v1(m)
        formula_v2 = compute_formula_v2(m)
        
        results.append({
            'name': m.name,
            'exact': exact,
            'v1': formula_v1,
            'v2': formula_v2,
            'mapping': m,
        })
    
    # 计算相关系数
    exact_vals = [r['exact'] for r in results]
    v1_vals = [r['v1'] for r in results]
    v2_vals = [r['v2'] for r in results]
    
    if len(set(exact_vals)) < 2:
        corr_v1 = corr_v2 = float('nan')
    else:
        corr_v1, _ = stats.spearmanr(exact_vals, v1_vals)
        corr_v2, _ = stats.spearmanr(exact_vals, v2_vals)
    
    # 平均误差
    avg_err_v1 = np.mean([abs(r['v1'] - r['exact']) / max(1, r['exact']) * 100 for r in results])
    avg_err_v2 = np.mean([abs(r['v2'] - r['exact']) / max(1, r['exact']) * 100 for r in results])
    
    return {
        'perm': perm,
        'perm_str': ''.join([p[0] for p in perm]),
        'n': len(results),
        'corr_v1': corr_v1,
        'corr_v2': corr_v2,
        'avg_err_v1': avg_err_v1,
        'avg_err_v2': avg_err_v2,
        'results': results,
    }


def main():
    print("="*80)
    print("按 Permutation 分组分析 - Formula vs Exact")
    print("="*80)
    
    # 所有可能的 4 维度排列 (4! = 24)
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
    
    # 全局统计
    global_v1_corrs = []
    global_v2_corrs = []
    all_perm_stats = defaultdict(lambda: {'v1': [], 'v2': []})
    
    for workload in workloads:
        print(f"\n{'='*80}")
        print(f"Workload: {workload.name}")
        print(f"  P={workload.P}, Q={workload.Q}, C={workload.C}, K={workload.K}, R={workload.R}, S={workload.S}")
        print(f"{'='*80}")
        
        # 生成 mappings
        mappings_by_perm = generate_mappings(workload, all_perms, max_per_perm=15)
        
        # 分析每个 permutation
        all_analyses = []
        for perm in all_perms:
            if perm in mappings_by_perm and len(mappings_by_perm[perm]) >= 3:
                analysis = analyze_permutation(perm, mappings_by_perm[perm])
                all_analyses.append(analysis)
        
        # 按 V2 相关系数排序显示
        all_analyses.sort(key=lambda x: x['corr_v2'] if not np.isnan(x['corr_v2']) else -1, reverse=True)
        
        print(f"\n{'Permutation':<12} {'#Maps':>6} {'ρ(V1)':>8} {'ρ(V2)':>8} {'Err_V1%':>9} {'Err_V2%':>9}")
        print("-" * 60)
        
        for a in all_analyses:
            print(f"{a['perm_str']:<12} {a['n']:>6} {a['corr_v1']:>8.3f} {a['corr_v2']:>8.3f} "
                  f"{a['avg_err_v1']:>8.1f}% {a['avg_err_v2']:>8.1f}%")
        
        # 找出问题 permutation
        print(f"\n问题分析:")
        
        good_perms = [a for a in all_analyses if a['corr_v2'] > 0.95]
        bad_perms = [a for a in all_analyses if a['corr_v2'] < 0.9]
        
        print(f"  V2 准确 (ρ > 0.95): {len(good_perms)} 个")
        print(f"  V2 不准 (ρ < 0.90): {len(bad_perms)} 个")
        
        if bad_perms:
            print(f"\n  不准确的 permutation 详情:")
            for a in bad_perms[:5]:
                print(f"\n  {a['perm_str']} (ρ={a['corr_v2']:.3f}, err={a['avg_err_v2']:.1f}%):")
                # 显示几个具体例子
                for r in sorted(a['results'], key=lambda x: x['exact'])[:3]:
                    err_v2 = abs(r['v2'] - r['exact']) / max(1, r['exact']) * 100
                    print(f"    {r['name']}: Exact={r['exact']}, V2={r['v2']}, err={err_v2:.1f}%")
        
        # 分析 K 位置的影响
        print(f"\n  K 位置分析:")
        k_pos_analysis = defaultdict(list)
        for a in all_analyses:
            k_pos = a['perm'].index('K')
            k_pos_analysis[k_pos].append(a['corr_v2'])
        
        for pos in sorted(k_pos_analysis.keys()):
            avg_corr = np.mean(k_pos_analysis[pos])
            print(f"    K 在位置 {pos} (0=最外): 平均 ρ = {avg_corr:.3f}")
        
        # 分析最内层维度的影响
        print(f"\n  最内层维度分析:")
        inner_analysis = defaultdict(list)
        for a in all_analyses:
            inner_dim = a['perm'][-1]  # 最内层
            inner_analysis[inner_dim].append(a['corr_v2'])
        
        for dim in ['K', 'C', 'P', 'Q']:
            if dim in inner_analysis:
                avg_corr = np.mean(inner_analysis[dim])
                print(f"    {dim} 在最内层: 平均 ρ = {avg_corr:.3f}")


if __name__ == '__main__':
    main()
