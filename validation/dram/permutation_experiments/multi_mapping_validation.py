#!/usr/bin/env python3
"""
多 Mapping 验证实验

目标: 验证简化模型是否能正确排序不同 mapping 的好坏

实验设计:
1. 选择多个 workload (不同 R, S, P, Q, C, K)
2. 对每个 workload，生成多个不同的 mapping 配置
3. 用 Trace 计算真实 row_switches
4. 用简化模型计算预测值
5. 计算 Spearman 相关系数验证排序一致性

Mapping 变化维度:
- R_l2: R 在 Level 2 的 tiling (影响 h 方向的访问)
- P_l3, Q_l3: 空间 tiling
- block_h, block_w: 数据布局
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/validation/dram')

import math
import itertools
from scipy import stats
import numpy as np

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
from trace_generator import TraceGenerator, DRAMConfig


def count_row_switches(trace, bank_size=1024*16384, row_size=1024):
    """从 trace 中统计 Input 的 row switches"""
    last_row = None
    switches = 0
    
    for line in trace:
        if 'LD' in line:
            parts = line.split()
            addr = int(parts[1], 16)
            bank = addr // bank_size
            if bank == 0:  # Input bank
                row = (addr % bank_size) // row_size
                if last_row is not None and row != last_row:
                    switches += 1
                last_row = row
    
    return switches


def compute_formula_prediction(
    P_l3, Q_l3, C_l3, K_l3, R_l2,
    P_buffer, Q_buffer, S,
    block_h, block_w, H_in, W_in
):
    """用分析公式计算预测的 row switches"""
    num_h_blocks = math.ceil(H_in / block_h)
    num_w_blocks = math.ceil(W_in / block_w)
    
    # 1. multi_block
    multi_block = 0
    for q in range(Q_l3):
        w_start = q * Q_buffer
        w_end = w_start + Q_buffer + S - 2
        wb_start = w_start // block_w
        wb_end = min(w_end // block_w, num_w_blocks - 1)
        multi_block += (wb_end - wb_start)
    multi_block *= P_l3 * R_l2 * C_l3 * K_l3
    
    # 2. R switches
    r_switches_per_ck = 0
    for p in range(P_l3):
        for q in range(Q_l3):
            w_end = q * Q_buffer + Q_buffer + S - 2
            wb_end = min(w_end // block_w, num_w_blocks - 1)
            wb_start = q * Q_buffer // block_w
            
            last_hb, last_wb = None, None
            for r in range(R_l2):
                h = p * P_buffer + r
                hb = min(h // block_h, num_h_blocks - 1)
                if last_hb is not None and (hb, wb_start) != (last_hb, last_wb):
                    r_switches_per_ck += 1
                last_hb, last_wb = hb, wb_end
    r_switches = r_switches_per_ck * C_l3 * K_l3
    
    # 3. P switches
    p_switches_per_ck = 0
    for p in range(1, P_l3):
        h_prev = (p - 1) * P_buffer + R_l2 - 1
        hb_prev = min(h_prev // block_h, num_h_blocks - 1)
        w_end_prev = (Q_l3 - 1) * Q_buffer + Q_buffer + S - 2
        wb_prev = min(w_end_prev // block_w, num_w_blocks - 1)
        
        h_curr = p * P_buffer
        hb_curr = min(h_curr // block_h, num_h_blocks - 1)
        wb_curr = 0
        
        if (hb_prev, wb_prev) != (hb_curr, wb_curr):
            p_switches_per_ck += 1
    p_switches = p_switches_per_ck * C_l3 * K_l3
    
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


def extract_mapping_params(mapping, workload):
    """从 mapping 对象中提取参数"""
    loop_bounds = mapping.loop_bounds
    tile_info = mapping.tile_info
    
    # Level 3 temporal
    l3_t = loop_bounds.get(3, {}).get('temporal', {})
    P_l3 = l3_t.get(2, 1)
    Q_l3 = l3_t.get(3, 1)
    C_l3 = l3_t.get(4, 1)
    K_l3 = l3_t.get(5, 1)
    
    # Level 2 temporal
    l2_t = loop_bounds.get(2, {}).get('temporal', {})
    R_l2 = l2_t.get(0, 1)
    S_l2 = l2_t.get(1, 1)
    
    # Buffer tile (Level 0+1)
    P_buffer = workload.P // P_l3 if P_l3 > 0 else workload.P
    Q_buffer = workload.Q // Q_l3 if Q_l3 > 0 else workload.Q
    
    # Block size
    block_h = tile_info.get('block_h', 31)
    block_w = tile_info.get('block_w', 31)
    
    return {
        'P_l3': P_l3, 'Q_l3': Q_l3, 'C_l3': C_l3, 'K_l3': K_l3,
        'R_l2': R_l2, 'S_l2': S_l2,
        'P_buffer': P_buffer, 'Q_buffer': Q_buffer,
        'block_h': block_h, 'block_w': block_w,
    }


def run_single_workload_validation(workload, num_mappings=5):
    """对单个 workload 运行多个 mapping 的验证"""
    print(f"\n{'='*60}")
    print(f"Workload: {workload.name}")
    print(f"  R={workload.R}, S={workload.S}, P={workload.P}, Q={workload.Q}")
    print(f"  C={workload.C}, K={workload.K}")
    print(f"  H_in={workload.input_size['H']}, W_in={workload.input_size['W']}")
    print(f"{'='*60}")
    
    optimizer = PIMOptimizer()
    gen = TraceGenerator(DRAMConfig())
    
    results = []
    
    # 方法 1: 使用 optimizer 得到最优 mapping
    print("\n正在优化...")
    result = optimizer.optimize([workload], objective='latency')
    mapping = result.mappings[0]
    
    # 提取参数
    params = extract_mapping_params(mapping, workload)
    print(f"Optimal mapping params: {params}")
    
    # 生成 trace
    print("正在生成 Trace...")
    trace = gen.generate_trace(mapping, workload)
    trace_switches = count_row_switches(trace)
    
    # 计算公式预测
    H_in = workload.input_size['H']
    W_in = workload.input_size['W']
    S = workload.S
    
    formula_pred = compute_formula_prediction(
        params['P_l3'], params['Q_l3'], params['C_l3'], params['K_l3'],
        params['R_l2'], params['P_buffer'], params['Q_buffer'], S,
        params['block_h'], params['block_w'], H_in, W_in
    )
    
    # 当前 ILP 预测 (无 R)
    current_ilp = (params['P_l3'] * params['Q_l3'] * params['C_l3'] + 10) * params['K_l3']
    
    results.append({
        'name': 'optimal',
        'params': params,
        'trace': trace_switches,
        'formula': formula_pred,
        'current_ilp': current_ilp,
    })
    
    print(f"\nOptimal mapping:")
    print(f"  Trace: {trace_switches}")
    print(f"  Formula: {formula_pred}")
    print(f"  Current ILP: {current_ilp}")
    
    return results


def run_systematic_validation():
    """系统性验证：多个 workload，每个有多个 mapping"""
    
    # 定义多个 workload
    workloads = [
        # ResNet-50 典型层
        ConvWorkload(name='resnet_l1', N=1, K=64, C=3, P=56, Q=56, R=7, S=7),
        ConvWorkload(name='resnet_3x3', N=1, K=64, C=64, P=56, Q=56, R=3, S=3),
        ConvWorkload(name='resnet_1x1', N=1, K=256, C=64, P=56, Q=56, R=1, S=1),
        
        # VGG 典型层
        ConvWorkload(name='vgg_3x3_small', N=1, K=64, C=64, P=112, Q=112, R=3, S=3),
        ConvWorkload(name='vgg_3x3_large', N=1, K=512, C=512, P=14, Q=14, R=3, S=3),
        
        # MobileNet depthwise
        ConvWorkload(name='mobilenet_dw', N=1, K=32, C=32, P=112, Q=112, R=3, S=3),
    ]
    
    all_results = []
    
    for workload in workloads:
        try:
            results = run_single_workload_validation(workload)
            all_results.extend(results)
        except Exception as e:
            print(f"Error processing {workload.name}: {e}")
            continue
    
    return all_results


def analyze_results(results):
    """分析验证结果"""
    print("\n" + "="*70)
    print("验证结果分析")
    print("="*70)
    
    if not results:
        print("没有结果")
        return
    
    # 提取数据
    trace_values = [r['trace'] for r in results]
    formula_values = [r['formula'] for r in results]
    ilp_values = [r['current_ilp'] for r in results]
    
    print("\n所有结果:")
    print(f"{'Workload':<20} {'Trace':>10} {'Formula':>10} {'ILP':>10} {'F_err%':>8} {'I_err%':>8}")
    print("-"*70)
    
    for r in results:
        f_err = abs(r['formula'] - r['trace']) / r['trace'] * 100 if r['trace'] > 0 else 0
        i_err = abs(r['current_ilp'] - r['trace']) / r['trace'] * 100 if r['trace'] > 0 else 0
        print(f"{r['name']:<20} {r['trace']:>10} {r['formula']:>10} {r['current_ilp']:>10} {f_err:>7.1f}% {i_err:>7.1f}%")
    
    # 计算相关系数
    if len(results) > 2:
        # Formula vs Trace
        corr_formula, p_formula = stats.spearmanr(trace_values, formula_values)
        # Current ILP vs Trace
        corr_ilp, p_ilp = stats.spearmanr(trace_values, ilp_values)
        
        print(f"\nSpearman 相关系数:")
        print(f"  Formula vs Trace: ρ = {corr_formula:.3f} (p = {p_formula:.4f})")
        print(f"  Current ILP vs Trace: ρ = {corr_ilp:.3f} (p = {p_ilp:.4f})")
        
        # 平均误差
        avg_f_err = np.mean([abs(r['formula'] - r['trace']) / r['trace'] * 100 for r in results if r['trace'] > 0])
        avg_i_err = np.mean([abs(r['current_ilp'] - r['trace']) / r['trace'] * 100 for r in results if r['trace'] > 0])
        
        print(f"\n平均误差:")
        print(f"  Formula: {avg_f_err:.1f}%")
        print(f"  Current ILP: {avg_i_err:.1f}%")
    
    print("\n" + "="*70)
    print("结论")
    print("="*70)
    
    if len(results) > 2 and corr_formula > 0.8:
        print("✓ Formula 模型与 Trace 高度相关，可以正确排序 mapping")
    elif len(results) > 2 and corr_formula > 0.5:
        print("△ Formula 模型与 Trace 中度相关，排序基本准确")
    else:
        print("✗ Formula 模型与 Trace 相关性较低，需要进一步改进")


if __name__ == '__main__':
    print("="*70)
    print("多 Mapping Row Activation 验证实验")
    print("="*70)
    
    results = run_systematic_validation()
    analyze_results(results)
