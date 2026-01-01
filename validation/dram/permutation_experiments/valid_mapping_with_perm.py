#!/usr/bin/env python3
"""
合法 Mapping 验证实验 v2 - 包含 Permutation

Mapping 包含:
1. Tiling factors: P_l3, Q_l3, C_l3, K_l3, R_l2, S_l2
2. Permutation: L3 层循环顺序 (对 Input 影响最大的是 K, C, P, Q 的顺序)
3. Data layout: block_h, block_w

Permutation 对 row switches 的影响:
- Input 相关维度: C, P, Q (还有 R 通过 P+r 间接相关)
- 把 Input 相关维度放在内层 → 更好的局部性 → 更少 row switches
- K 对 Input 无关，但 K 在外层会导致 Input 被重复访问 K 次
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

import math
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from itertools import permutations


# ==========================================
# 硬件参数
# ==========================================
@dataclass
class HardwareConfig:
    pe_h: int = 16
    pe_w: int = 16
    global_buffer_entries: int = 65536  # 256KB - 放宽以允许更多 mapping
    row_buffer_entries: int = 1024      # 4KB


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
    stride: int = 1
    
    @property
    def H_in(self) -> int:
        return self.P + self.R - 1
    
    @property
    def W_in(self) -> int:
        return self.Q + self.S - 1
    
    def get_divisors(self, dim_value: int) -> List[int]:
        divisors = []
        for i in range(1, dim_value + 1):
            if dim_value % i == 0:
                divisors.append(i)
        return divisors


@dataclass
class Mapping:
    """完整的 Mapping 配置"""
    name: str
    
    # L3 tiling
    P_l3: int
    Q_l3: int
    C_l3: int
    K_l3: int
    
    # L2 tiling
    R_l2: int = 1
    S_l2: int = 1
    
    # Permutation: L3 层循环顺序 (从外到内)
    # 例如: ['K', 'C', 'P', 'Q'] 表示 for K: for C: for P: for Q: ...
    # R 在 L2 层，总是在最内层
    l3_permutation: Tuple[str, ...] = ('K', 'C', 'P', 'Q')
    
    # 数据布局
    block_h: int = 31
    block_w: int = 31
    
    workload: Optional[Workload] = None
    
    @property
    def P_buffer(self) -> int:
        return self.workload.P // self.P_l3 if self.workload else 0
    
    @property
    def Q_buffer(self) -> int:
        return self.workload.Q // self.Q_l3 if self.workload else 0
    
    @property
    def C_buffer(self) -> int:
        return self.workload.C // self.C_l3 if self.workload else 0
    
    @property
    def K_buffer(self) -> int:
        return self.workload.K // self.K_l3 if self.workload else 0
    
    @property
    def R_buffer(self) -> int:
        return self.workload.R // self.R_l2 if self.workload else 0


def validate_mapping(mapping: Mapping, hw: HardwareConfig) -> Tuple[bool, List[str]]:
    """验证 mapping 合法性"""
    violations = []
    w = mapping.workload
    
    if w is None:
        return False, ["No workload"]
    
    # 维度分解
    if mapping.P_l3 * mapping.P_buffer != w.P:
        violations.append(f"P: {mapping.P_l3}×{mapping.P_buffer}!={w.P}")
    if mapping.Q_l3 * mapping.Q_buffer != w.Q:
        violations.append(f"Q: {mapping.Q_l3}×{mapping.Q_buffer}!={w.Q}")
    if mapping.C_l3 * mapping.C_buffer != w.C:
        violations.append(f"C: {mapping.C_l3}×{mapping.C_buffer}!={w.C}")
    if mapping.K_l3 * mapping.K_buffer != w.K:
        violations.append(f"K: {mapping.K_l3}×{mapping.K_buffer}!={w.K}")
    if mapping.R_l2 * mapping.R_buffer != w.R:
        violations.append(f"R: {mapping.R_l2}×{mapping.R_buffer}!={w.R}")
    
    # Buffer 容量
    input_tile = (mapping.P_buffer + mapping.R_buffer - 1) * \
                 (mapping.Q_buffer + w.S - 1) * mapping.C_buffer
    weight_tile = mapping.R_buffer * w.S * mapping.C_buffer * mapping.K_buffer
    output_tile = mapping.P_buffer * mapping.Q_buffer * mapping.K_buffer
    total = input_tile + weight_tile + output_tile
    
    if total > hw.global_buffer_entries:
        violations.append(f"Buffer overflow: {total}>{hw.global_buffer_entries}")
    
    # Block 大小
    if mapping.block_h < 1 or mapping.block_h > w.H_in:
        violations.append(f"block_h: {mapping.block_h}")
    if mapping.block_w < 1 or mapping.block_w > w.W_in:
        violations.append(f"block_w: {mapping.block_w}")
    
    return len(violations) == 0, violations


def compute_exact_row_switches(mapping: Mapping) -> int:
    """
    精确模拟 - 支持任意 permutation
    """
    w = mapping.workload
    perm = mapping.l3_permutation
    
    num_h_blocks = math.ceil(w.H_in / mapping.block_h)
    num_w_blocks = math.ceil(w.W_in / mapping.block_w)
    
    # 构建循环范围
    ranges = {
        'K': range(mapping.K_l3),
        'C': range(mapping.C_l3),
        'P': range(mapping.P_l3),
        'Q': range(mapping.Q_l3),
    }
    
    row_switches = 0
    last_block = None
    
    # 根据 permutation 生成循环
    def iterate(level, indices):
        nonlocal row_switches, last_block
        
        if level == len(perm):
            # 最内层: R_l2 循环
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


def compute_formula_prediction(mapping: Mapping) -> int:
    """
    简化公式预测 - 考虑 permutation
    
    关键观察:
    - Input 相关维度: C, P, Q (通过 h, w 坐标)
    - K 对 Input 无关，但会导致重复访问
    
    Permutation 影响:
    - 如果 K 在最外层: Input 被完整访问 K 次，每次切换 K 都从头开始
    - 如果 K 在最内层: 一个 Input tile 被访问 K 次后才换下一个
    """
    w = mapping.workload
    perm = mapping.l3_permutation
    
    num_h_blocks = math.ceil(w.H_in / mapping.block_h)
    num_w_blocks = math.ceil(w.W_in / mapping.block_w)
    
    # 计算 Q tiles 跨 w_block 的情况
    q_crossing = 0
    multi_block_per_q = []
    for q in range(mapping.Q_l3):
        w_start = q * mapping.Q_buffer
        w_end = w_start + mapping.Q_buffer + w.S - 2
        wb_start = w_start // mapping.block_w
        wb_end = min(w_end // mapping.block_w, num_w_blocks - 1)
        blocks = wb_end - wb_start
        multi_block_per_q.append(blocks)
        if blocks > 0:
            q_crossing += 1
    
    total_multi_block = sum(multi_block_per_q)
    
    # 找到各维度在 permutation 中的位置 (0=最外层)
    pos = {dim: perm.index(dim) for dim in perm}
    
    # 计算 switches
    # 基本思路: 每次"相关维度变化"都可能导致 switch
    
    # 1. multi_block switches (Q tile 内部跨 block)
    # 发生次数 = P_l3 × Q_l3 × C_l3 × K_l3 × R_l2，但只有跨 block 的 Q 才产生
    multi_block = total_multi_block * mapping.P_l3 * mapping.R_l2 * mapping.C_l3 * mapping.K_l3
    
    # 2. R switches (R 变化导致 h 变化)
    # 只在 multi-block Q tiles 中有意义
    r_switches = q_crossing * mapping.P_l3 * max(0, mapping.R_l2 - 1) * mapping.C_l3 * mapping.K_l3
    
    # 3. Q switches (Q 变化跨 w_block 边界)
    q_boundary_switches = 0
    for q in range(1, mapping.Q_l3):
        w_end_prev = (q - 1) * mapping.Q_buffer + mapping.Q_buffer + w.S - 2
        wb_prev = min(w_end_prev // mapping.block_w, num_w_blocks - 1)
        wb_curr = q * mapping.Q_buffer // mapping.block_w
        if wb_prev != wb_curr:
            q_boundary_switches += 1
    
    # Q switches 的重复次数取决于 Q 外层有哪些维度
    q_outer_mult = 1
    for dim in perm[:pos['Q']]:
        if dim == 'K':
            q_outer_mult *= mapping.K_l3
        elif dim == 'C':
            q_outer_mult *= mapping.C_l3
        elif dim == 'P':
            q_outer_mult *= mapping.P_l3
    q_switches = q_boundary_switches * q_outer_mult * mapping.R_l2
    
    # 4. P switches
    p_switches_base = max(0, mapping.P_l3 - 1)
    p_outer_mult = 1
    for dim in perm[:pos['P']]:
        if dim == 'K':
            p_outer_mult *= mapping.K_l3
        elif dim == 'C':
            p_outer_mult *= mapping.C_l3
        elif dim == 'Q':
            p_outer_mult *= mapping.Q_l3
    p_switches = p_switches_base * p_outer_mult
    
    # 5. C switches
    c_switches_base = max(0, mapping.C_l3 - 1)
    c_outer_mult = 1
    for dim in perm[:pos['C']]:
        if dim == 'K':
            c_outer_mult *= mapping.K_l3
        elif dim == 'P':
            c_outer_mult *= mapping.P_l3
        elif dim == 'Q':
            c_outer_mult *= mapping.Q_l3
    c_switches = c_switches_base * c_outer_mult
    
    # 6. K switches (K 对 Input 无关，但切换 K 后会回到不同的 (C,P,Q) 位置)
    # 只有当 K 不是最外层时，K 切换会导致 row switch
    if pos['K'] > 0:
        k_switches = max(0, mapping.K_l3 - 1)
        k_outer_mult = 1
        for dim in perm[:pos['K']]:
            if dim == 'C':
                k_outer_mult *= mapping.C_l3
            elif dim == 'P':
                k_outer_mult *= mapping.P_l3
            elif dim == 'Q':
                k_outer_mult *= mapping.Q_l3
        k_switches *= k_outer_mult
    else:
        k_switches = 0
    
    return multi_block + r_switches + p_switches + q_switches + c_switches + k_switches


def compute_current_ilp(mapping: Mapping, BC: int = 10) -> int:
    """当前 ILP (不考虑 permutation 和 R_l2)"""
    return (mapping.P_l3 * mapping.Q_l3 * mapping.C_l3 + BC) * mapping.K_l3


def generate_valid_mappings(workload: Workload, hw: HardwareConfig, max_count: int = 50) -> List[Mapping]:
    """生成合法的 mappings，包含不同的 permutation"""
    valid_mappings = []
    
    # 因子 - 取更多选项
    P_divs = [d for d in workload.get_divisors(workload.P)][:8]
    Q_divs = [d for d in workload.get_divisors(workload.Q)][:8]
    C_divs = [d for d in workload.get_divisors(workload.C)][:6]
    K_divs = [d for d in workload.get_divisors(workload.K)][:6]
    
    # 几种典型的 permutation
    perms = [
        ('K', 'C', 'P', 'Q'),  # K 最外
        ('C', 'K', 'P', 'Q'),  # C 最外
        ('P', 'Q', 'K', 'C'),  # 空间维度最外
        ('K', 'P', 'Q', 'C'),  # C 最内
        ('C', 'P', 'Q', 'K'),  # K 最内
        ('Q', 'P', 'C', 'K'),  # Q 最外
    ]
    
    count = 0
    
    for P_l3 in P_divs:
        for Q_l3 in Q_divs:
            for C_l3 in C_divs:
                for K_l3 in K_divs:
                    for R_l2 in [1, workload.R] if workload.R > 1 else [1]:
                        if count >= max_count:
                            return valid_mappings
                        
                        # 检查 buffer 约束
                        P_buf = workload.P // P_l3
                        Q_buf = workload.Q // Q_l3
                        C_buf = workload.C // C_l3
                        K_buf = workload.K // K_l3
                        R_buf = workload.R // R_l2
                        
                        input_tile = (P_buf + R_buf - 1) * (Q_buf + workload.S - 1) * C_buf
                        weight_tile = R_buf * workload.S * C_buf * K_buf
                        output_tile = P_buf * Q_buf * K_buf
                        
                        if input_tile + weight_tile + output_tile > hw.global_buffer_entries:
                            continue
                        
                        # 为这个 tiling 配置尝试不同的 permutation
                        for perm in perms:
                            if count >= max_count:
                                return valid_mappings
                            
                            # Block 大小
                            block_h = workload.H_in
                            block_w = max(1, workload.W_in // 2)
                            
                            perm_str = ''.join([p[0] for p in perm])
                            name = f"P{P_l3}_Q{Q_l3}_C{C_l3}_K{K_l3}_R{R_l2}_{perm_str}"
                            
                            m = Mapping(
                                name=name,
                                P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3,
                                R_l2=R_l2, S_l2=1,
                                l3_permutation=perm,
                                block_h=block_h, block_w=block_w,
                                workload=workload
                            )
                            
                            is_valid, _ = validate_mapping(m, hw)
                            if is_valid:
                                valid_mappings.append(m)
                                count += 1
    
    return valid_mappings


def run_workload_validation(workload: Workload, hw: HardwareConfig):
    """验证单个 workload"""
    print(f"\n{'='*80}")
    print(f"Workload: {workload.name}")
    print(f"  P={workload.P}, Q={workload.Q}, C={workload.C}, K={workload.K}, R={workload.R}, S={workload.S}")
    print(f"{'='*80}")
    
    mappings = generate_valid_mappings(workload, hw, max_count=50)
    print(f"\n生成了 {len(mappings)} 个合法 mapping (包含不同 permutation)")
    
    if len(mappings) < 5:
        print("  ⚠️ mapping 数量太少")
        return None
    
    results = []
    for m in mappings:
        exact = compute_exact_row_switches(m)
        formula = compute_formula_prediction(m)
        ilp = compute_current_ilp(m)
        
        results.append({
            'name': m.name,
            'perm': ''.join([p[0] for p in m.l3_permutation]),
            'exact': exact,
            'formula': formula,
            'ilp': ilp,
        })
    
    # 显示结果
    print(f"\n{'Mapping':<45} {'Perm':>6} {'Exact':>8} {'Formula':>8} {'ILP':>8} {'F_err%':>8}")
    print("-" * 90)
    
    for r in sorted(results, key=lambda x: x['exact'])[:25]:
        f_err = abs(r['formula'] - r['exact']) / max(1, r['exact']) * 100
        print(f"{r['name']:<45} {r['perm']:>6} {r['exact']:>8} {r['formula']:>8} {r['ilp']:>8} {f_err:>7.1f}%")
    
    if len(results) > 25:
        print(f"  ... 还有 {len(results) - 25} 个 mapping")
    
    # 相关系数
    exact_vals = [r['exact'] for r in results]
    formula_vals = [r['formula'] for r in results]
    ilp_vals = [r['ilp'] for r in results]
    
    corr_f, p_f = stats.spearmanr(exact_vals, formula_vals)
    corr_i, p_i = stats.spearmanr(exact_vals, ilp_vals)
    
    print(f"\nSpearman 排序相关系数:")
    print(f"  Formula vs Exact: ρ = {corr_f:.3f}")
    print(f"  Current ILP vs Exact: ρ = {corr_i:.3f}")
    
    # 分析 permutation 的影响
    print(f"\n各 Permutation 的平均 row_switches:")
    perm_stats = {}
    for r in results:
        perm = r['perm']
        if perm not in perm_stats:
            perm_stats[perm] = []
        perm_stats[perm].append(r['exact'])
    
    for perm, vals in sorted(perm_stats.items(), key=lambda x: np.mean(x[1])):
        print(f"  {perm}: avg={np.mean(vals):.0f}, min={min(vals)}, max={max(vals)}")
    
    return {
        'name': workload.name,
        'num_mappings': len(results),
        'corr_formula': corr_f,
        'corr_ilp': corr_i,
    }


def main():
    print("="*80)
    print("合法 Mapping 验证实验 v2 - 包含 Permutation")
    print("="*80)
    
    hw = HardwareConfig()
    
    workloads = [
        Workload(name='ResNet-L1', N=1, K=64, C=3, P=56, Q=56, R=7, S=7),
        Workload(name='ResNet-3x3', N=1, K=64, C=64, P=56, Q=56, R=3, S=3),
        Workload(name='ResNet-1x1', N=1, K=256, C=64, P=56, Q=56, R=1, S=1),
        Workload(name='VGG-conv5', N=1, K=512, C=512, P=14, Q=14, R=3, S=3),
        Workload(name='MobileNet', N=1, K=32, C=32, P=112, Q=112, R=3, S=3),
    ]
    
    all_results = []
    
    for workload in workloads:
        result = run_workload_validation(workload, hw)
        if result:
            all_results.append(result)
    
    # 总结
    if all_results:
        print("\n" + "="*80)
        print("总体结果")
        print("="*80)
        
        print(f"\n{'Workload':<15} {'#Maps':>6} {'ρ(Formula)':>12} {'ρ(ILP)':>10}")
        print("-"*50)
        
        for r in all_results:
            print(f"{r['name']:<15} {r['num_mappings']:>6} {r['corr_formula']:>12.3f} {r['corr_ilp']:>10.3f}")
        
        avg_f = np.mean([r['corr_formula'] for r in all_results])
        avg_i = np.mean([r['corr_ilp'] for r in all_results])
        print("-"*50)
        print(f"{'平均':<15} {'':>6} {avg_f:>12.3f} {avg_i:>10.3f}")


if __name__ == '__main__':
    main()
