#!/usr/bin/env python3
"""
多 Mapping 验证实验 v2

核心问题: 验证模型在不同 mapping 下的 **排序一致性**

实验设计:
1. 对同一个 workload，生成多个不同 mapping（变化 P_l3, Q_l3, R_l2 等）
2. 用精确模拟计算每个 mapping 的 row_switches
3. 用简化公式计算预测值
4. 计算 Spearman 相关系数

重点：同一个 workload 下不同 mapping 的相对排序，
而不是跨 workload 的绝对数值
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

import math
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class TestMapping:
    """测试用的简化 Mapping"""
    name: str
    P_l3: int
    Q_l3: int
    C_l3: int
    K_l3: int
    R_l2: int
    P_buffer: int
    Q_buffer: int
    block_h: int
    block_w: int


def compute_exact_row_switches(
    P_l3, Q_l3, C_l3, K_l3, R_l2, P_buffer, Q_buffer, S,
    block_h, block_w, H_in, W_in
) -> int:
    """精确模拟计算 row switches"""
    num_h_blocks = math.ceil(H_in / block_h)
    num_w_blocks = math.ceil(W_in / block_w)
    
    row_switches = 0
    last_block = None
    
    # 循环顺序: K -> C -> P -> Q -> R (从外到内)
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r in range(R_l2):
                        # 计算当前访问的 h, w 范围
                        h = p * P_buffer + r
                        w_start = q * Q_buffer
                        w_end = q * Q_buffer + Q_buffer + S - 2
                        
                        # 确定访问的 blocks
                        hb = min(h // block_h, num_h_blocks - 1)
                        wb_start = min(w_start // block_w, num_w_blocks - 1)
                        wb_end = min(w_end // block_w, num_w_blocks - 1)
                        
                        # 遍历所有访问的 blocks
                        for wb in range(wb_start, wb_end + 1):
                            curr_block = (c, hb, wb)
                            if last_block is not None and curr_block != last_block:
                                row_switches += 1
                            last_block = curr_block
    
    return row_switches


def compute_formula_prediction(
    P_l3, Q_l3, C_l3, K_l3, R_l2, P_buffer, Q_buffer, S,
    block_h, block_w, H_in, W_in
) -> int:
    """简化公式预测 row switches"""
    num_h_blocks = math.ceil(H_in / block_h)
    num_w_blocks = math.ceil(W_in / block_w)
    
    # 计算 Q tiles 中有多少跨 w_block 边界
    q_crossing = 0
    for q in range(Q_l3):
        w_start = q * Q_buffer
        w_end = w_start + Q_buffer + S - 2
        wb_start = w_start // block_w
        wb_end = min(w_end // block_w, num_w_blocks - 1)
        if wb_end > wb_start:
            q_crossing += 1
    
    # 1. multi_block switches
    multi_block = 0
    for q in range(Q_l3):
        w_start = q * Q_buffer
        w_end = w_start + Q_buffer + S - 2
        wb_start = w_start // block_w
        wb_end = min(w_end // block_w, num_w_blocks - 1)
        multi_block += (wb_end - wb_start)
    multi_block *= P_l3 * R_l2 * C_l3 * K_l3
    
    # 2. R switches (简化估计)
    r_switches = q_crossing * P_l3 * (R_l2 - 1) * C_l3 * K_l3
    
    # 3. P switches
    p_switches = (P_l3 - 1) * C_l3 * K_l3
    
    # 4. Q switches
    q_switches_per_prck = 0
    for q in range(1, Q_l3):
        w_end_prev = (q - 1) * Q_buffer + Q_buffer + S - 2
        wb_prev = min(w_end_prev // block_w, num_w_blocks - 1)
        wb_curr = q * Q_buffer // block_w
        if wb_prev != wb_curr:
            q_switches_per_prck += 1
    q_switches = q_switches_per_prck * P_l3 * R_l2 * C_l3 * K_l3
    
    # 5. C & K switches
    c_switches = (C_l3 - 1) * K_l3
    k_switches = K_l3 - 1
    
    return multi_block + r_switches + p_switches + q_switches + c_switches + k_switches


def compute_current_ilp(P_l3, Q_l3, C_l3, K_l3, BC=10) -> int:
    """当前 ILP 模型 (不含 R_l2)"""
    return (P_l3 * Q_l3 * C_l3 + BC) * K_l3


def generate_mappings_for_workload(
    P: int, Q: int, C: int, K: int, R: int, S: int
) -> List[TestMapping]:
    """为给定 workload 生成多个不同的 mapping"""
    mappings = []
    
    # 生成多种不同的 tiling 配置
    P_l3_options = [p for p in [1, 2, 4, 7, 14, 28, 56] if P % p == 0 and p <= P]
    Q_l3_options = [q for q in [1, 2, 4, 7, 14, 28, 56] if Q % q == 0 and q <= Q]
    K_l3_options = [k for k in [1, 2, 4, 8, 16, 32, 64] if K % k == 0 and k <= K]
    
    # R_l2 选项
    R_l2_options = [1, R] if R > 1 else [1]
    
    # 限制组合数量
    count = 0
    max_mappings = 20
    
    for P_l3 in P_l3_options[:4]:
        for Q_l3 in Q_l3_options[:4]:
            for K_l3 in K_l3_options[:3]:
                for R_l2 in R_l2_options:
                    if count >= max_mappings:
                        break
                    
                    P_buffer = P // P_l3
                    Q_buffer = Q // Q_l3
                    
                    # 估算合理的 block 大小
                    H_in = P + R - 1
                    W_in = Q + S - 1
                    
                    # 尝试几种不同的 block 大小
                    block_configs = [
                        (H_in, W_in),  # 单 block
                        (max(1, H_in // 2), W_in),  # 2 H blocks
                        (H_in, max(1, W_in // 2)),  # 2 W blocks
                        (max(1, H_in // 2), max(1, W_in // 2)),  # 4 blocks
                    ]
                    
                    for block_h, block_w in block_configs[:2]:  # 限制 block 配置数量
                        if count >= max_mappings:
                            break
                        
                        name = f"P{P_l3}_Q{Q_l3}_K{K_l3}_R{R_l2}_bh{block_h}_bw{block_w}"
                        mappings.append(TestMapping(
                            name=name,
                            P_l3=P_l3, Q_l3=Q_l3, C_l3=C, K_l3=K_l3,
                            R_l2=R_l2,
                            P_buffer=P_buffer, Q_buffer=Q_buffer,
                            block_h=block_h, block_w=block_w
                        ))
                        count += 1
    
    return mappings


def validate_workload(name: str, P: int, Q: int, C: int, K: int, R: int, S: int):
    """验证单个 workload 下多个 mapping 的排序一致性"""
    print(f"\n{'='*70}")
    print(f"Workload: {name}")
    print(f"  P={P}, Q={Q}, C={C}, K={K}, R={R}, S={S}")
    H_in = P + R - 1
    W_in = Q + S - 1
    print(f"  H_in={H_in}, W_in={W_in}")
    print(f"{'='*70}")
    
    # 生成 mappings
    mappings = generate_mappings_for_workload(P, Q, C, K, R, S)
    print(f"\n生成了 {len(mappings)} 个不同的 mapping")
    
    results = []
    
    for m in mappings:
        exact = compute_exact_row_switches(
            m.P_l3, m.Q_l3, m.C_l3, m.K_l3, m.R_l2,
            m.P_buffer, m.Q_buffer, S,
            m.block_h, m.block_w, H_in, W_in
        )
        
        formula = compute_formula_prediction(
            m.P_l3, m.Q_l3, m.C_l3, m.K_l3, m.R_l2,
            m.P_buffer, m.Q_buffer, S,
            m.block_h, m.block_w, H_in, W_in
        )
        
        ilp = compute_current_ilp(m.P_l3, m.Q_l3, m.C_l3, m.K_l3)
        
        results.append({
            'name': m.name,
            'mapping': m,
            'exact': exact,
            'formula': formula,
            'ilp': ilp,
        })
    
    # 显示结果
    print(f"\n{'Mapping':<35} {'Exact':>10} {'Formula':>10} {'ILP':>10} {'F_err%':>8} {'I_err%':>8}")
    print("-"*85)
    
    for r in sorted(results, key=lambda x: x['exact']):
        f_err = abs(r['formula'] - r['exact']) / r['exact'] * 100 if r['exact'] > 0 else 0
        i_err = abs(r['ilp'] - r['exact']) / r['exact'] * 100 if r['exact'] > 0 else 0
        print(f"{r['name']:<35} {r['exact']:>10} {r['formula']:>10} {r['ilp']:>10} {f_err:>7.1f}% {i_err:>7.1f}%")
    
    # 计算相关系数
    if len(results) > 2:
        exact_vals = [r['exact'] for r in results]
        formula_vals = [r['formula'] for r in results]
        ilp_vals = [r['ilp'] for r in results]
        
        corr_f, p_f = stats.spearmanr(exact_vals, formula_vals)
        corr_i, p_i = stats.spearmanr(exact_vals, ilp_vals)
        
        print(f"\nSpearman 排序相关系数:")
        print(f"  Formula vs Exact: ρ = {corr_f:.3f} (p = {p_f:.4f})")
        print(f"  Current ILP vs Exact: ρ = {corr_i:.3f} (p = {p_i:.4f})")
        
        # 平均误差
        avg_f_err = np.mean([abs(r['formula'] - r['exact']) / r['exact'] * 100 for r in results if r['exact'] > 0])
        avg_i_err = np.mean([abs(r['ilp'] - r['exact']) / r['exact'] * 100 for r in results if r['exact'] > 0])
        
        print(f"\n平均误差:")
        print(f"  Formula: {avg_f_err:.1f}%")
        print(f"  Current ILP: {avg_i_err:.1f}%")
        
        return {
            'name': name,
            'num_mappings': len(results),
            'corr_formula': corr_f,
            'corr_ilp': corr_i,
            'avg_err_formula': avg_f_err,
            'avg_err_ilp': avg_i_err,
        }
    
    return None


def main():
    print("="*70)
    print("多 Mapping Row Activation 验证实验 v2")
    print("="*70)
    print("\n目标: 验证简化模型在同一 workload 下能否正确排序不同 mapping")
    
    # 测试多个 workload
    workloads = [
        ('ResNet-L1 (7x7)', 56, 56, 3, 64, 7, 7),
        ('ResNet-3x3', 56, 56, 64, 64, 3, 3),
        ('ResNet-1x1', 56, 56, 64, 256, 1, 1),
        ('VGG-small', 112, 112, 64, 64, 3, 3),
        ('VGG-large', 14, 14, 512, 512, 3, 3),
    ]
    
    all_results = []
    
    for name, P, Q, C, K, R, S in workloads:
        result = validate_workload(name, P, Q, C, K, R, S)
        if result:
            all_results.append(result)
    
    # 总结
    print("\n" + "="*70)
    print("总体验证结果汇总")
    print("="*70)
    
    if all_results:
        print(f"\n{'Workload':<20} {'#Maps':>6} {'ρ(Form)':>10} {'ρ(ILP)':>10} {'AvgErr(F)':>10} {'AvgErr(I)':>10}")
        print("-"*70)
        
        for r in all_results:
            print(f"{r['name']:<20} {r['num_mappings']:>6} {r['corr_formula']:>10.3f} {r['corr_ilp']:>10.3f} {r['avg_err_formula']:>9.1f}% {r['avg_err_ilp']:>9.1f}%")
        
        avg_corr_f = np.mean([r['corr_formula'] for r in all_results])
        avg_corr_i = np.mean([r['corr_ilp'] for r in all_results])
        
        print("-"*70)
        print(f"{'平均':>26} {avg_corr_f:>10.3f} {avg_corr_i:>10.3f}")
        
        print("\n结论:")
        if avg_corr_f > 0.8:
            print("✓ Formula 模型与精确值高度相关 (ρ > 0.8)")
            print("  → 可以正确识别更优的 mapping")
        elif avg_corr_f > 0.5:
            print("△ Formula 模型与精确值中度相关 (0.5 < ρ < 0.8)")
            print("  → 排序基本准确，但可能有局部错误")
        else:
            print("✗ Formula 模型与精确值相关性较低 (ρ < 0.5)")
            print("  → 需要改进模型")
        
        if avg_corr_f > avg_corr_i:
            print(f"\n✓ Formula (ρ={avg_corr_f:.3f}) 优于 Current ILP (ρ={avg_corr_i:.3f})")
        else:
            print(f"\n△ Current ILP (ρ={avg_corr_i:.3f}) 与 Formula (ρ={avg_corr_f:.3f}) 相当")


if __name__ == '__main__':
    main()
