#!/usr/bin/env python3
"""
直接验证 Sequential 模式的 crossing ratio（匹配 expressions.py 的逻辑）
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')

import math
import numpy as np
from pim_optimizer.arch.pim_arch import PIMArchitecture
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer.model.row_activation import (
    compute_dram_row_crossing_ratio,
    compute_input_block_crossing_ratio
)
from pim_optimizer.model.expressions import compute_unique_input_size


def direct_verify_v2(name, wl_params):
    """直接验证公式（匹配 expressions.py 的逻辑）"""
    
    print("=" * 80)
    print(f"Workload: {name}")
    print("=" * 80)
    
    workload = ConvWorkload(**wl_params)
    arch = PIMArchitecture()
    
    # 架构参数
    banks = 16
    row_buffer_size = 1024.0
    element_bytes = 2.0  # FP16
    
    stride_h = workload.stride[1]
    stride_w = workload.stride[0]
    dilation_h = workload.dilation[1]
    dilation_w = workload.dilation[0]
    total_S = workload.bounds[1]  # S
    total_R = workload.bounds[0]  # R
    total_Q = workload.bounds[3]  # Q
    total_P = workload.bounds[2]  # P
    
    print(f"\n【Workload 参数】")
    print(f"  R={total_R}, S={total_S}, P={total_P}, Q={total_Q}")
    print(f"  stride=(w={stride_w}, h={stride_h}), dilation=(w={dilation_w}, h={dilation_h})")
    
    h_divisors = getattr(workload, 'hw_divisors', {}).get('H', [1])
    w_divisors = getattr(workload, 'hw_divisors', {}).get('W', [1])
    
    print(f"  H_divisors: {h_divisors}")
    print(f"  W_divisors: {w_divisors}")
    
    # =========================================================================
    # 构建 yh_list 和 yw_list（和 expressions.py 逻辑完全一致）
    # =========================================================================
    # Width: 遍历所有 (P, R) 组合
    yw_list = []
    yw_p_list = []
    yw_r_list = []
    for p_ in workload.divisors[2]:  # P divisors
        for r_ in workload.divisors[0]:  # R divisors
            unique_w = compute_unique_input_size(stride_w, dilation_w, p_, r_)
            yw_list.append(unique_w)
            yw_p_list.append(p_)
            yw_r_list.append(r_)
    
    # Height: 遍历所有 (Q, S) 组合
    yh_list = []
    yh_q_list = []
    yh_s_list = []
    for q_ in workload.divisors[3]:  # Q divisors
        for s_ in workload.divisors[1]:  # S divisors
            unique_h = compute_unique_input_size(stride_h, dilation_h, q_, s_)
            yh_list.append(unique_h)
            yh_q_list.append(q_)
            yh_s_list.append(s_)
    
    print(f"\n【input_tile_info（和 expressions.py 一致）】")
    print(f"  yh_list: {yh_list}")
    print(f"  yh_q_list: {yh_q_list}")
    print(f"  yh_s_list: {yh_s_list}")
    print(f"  yw_list: {yw_list}")
    print(f"  yw_p_list: {yw_p_list}")
    print(f"  yw_r_list: {yw_r_list}")
    
    # =========================================================================
    # 1. Row Crossing Ratio (cr_r)
    # =========================================================================
    seq_row_crossing_ratios = []
    for block_h in h_divisors:
        for tile_h in yh_list:
            for block_w in w_divisors:
                for tile_w in yw_list:
                    tile_bytes = tile_h * tile_w * element_bytes
                    cr = compute_dram_row_crossing_ratio(tile_bytes, row_buffer_size)
                    seq_row_crossing_ratios.append(cr)
    
    cr_r = sum(seq_row_crossing_ratios) / len(seq_row_crossing_ratios) if seq_row_crossing_ratios else 0.0
    
    print(f"\n【Row Crossing】")
    print(f"  Combinations: {len(seq_row_crossing_ratios)}")
    print(f"  avg_row_crossing_ratio (cr_r) = {cr_r:.6f}")
    
    # =========================================================================
    # 2. Block Crossing Ratio (cr_b)
    # =========================================================================
    seq_block_crossing_ratios_h = []
    for block_h in h_divisors:
        for j_idx, tile_h in enumerate(yh_list):
            q_factor = yh_q_list[j_idx]
            s_factor = yh_s_list[j_idx]
            step_h = q_factor * stride_h
            
            cr_h = compute_input_block_crossing_ratio(
                block_h=block_h,
                tile_h=tile_h,
                step=step_h,
                tiler_s=s_factor,
                total_S=total_S,
                dilation=dilation_h
            )
            seq_block_crossing_ratios_h.append(cr_h)
    
    seq_block_crossing_ratios_w = []
    for block_w in w_divisors:
        for jj, tile_w in enumerate(yw_list):
            p_factor = yw_p_list[jj]
            r_factor = yw_r_list[jj]
            step_w = p_factor * stride_w
            
            cr_w = compute_input_block_crossing_ratio(
                block_h=block_w,
                tile_h=tile_w,
                step=step_w,
                tiler_s=r_factor,
                total_S=total_R,
                dilation=dilation_w
            )
            seq_block_crossing_ratios_w.append(cr_w)
    
    avg_block_cr_h = sum(seq_block_crossing_ratios_h) / len(seq_block_crossing_ratios_h) if seq_block_crossing_ratios_h else 0.0
    avg_block_cr_w = sum(seq_block_crossing_ratios_w) / len(seq_block_crossing_ratios_w) if seq_block_crossing_ratios_w else 0.0
    cr_b = 1.0 - (1.0 - avg_block_cr_h) * (1.0 - avg_block_cr_w)
    
    print(f"\n【Block Crossing】")
    print(f"  H combinations: {len(seq_block_crossing_ratios_h)}")
    print(f"  avg_block_cr_h = {avg_block_cr_h:.6f}")
    print(f"  W combinations: {len(seq_block_crossing_ratios_w)}")
    print(f"  avg_block_cr_w = {avg_block_cr_w:.6f}")
    print(f"  cr_b = 1 - (1-cr_h)(1-cr_w) = {cr_b:.6f}")
    
    # =========================================================================
    # 3. Coefficients
    # =========================================================================
    non_crossing_coeff = (1.0 - cr_b) * (1.0 - cr_r)
    crossing_coeff = 2.0 * (cr_b + cr_r)
    
    # tiles_per_row
    avg_tile_bytes_list = []
    for tile_h in yh_list:
        for tile_w in yw_list:
            tile_bytes = tile_h * tile_w * element_bytes
            avg_tile_bytes_list.append(tile_bytes)
    avg_tile_bytes = sum(avg_tile_bytes_list) / len(avg_tile_bytes_list) if avg_tile_bytes_list else element_bytes
    tiles_per_row = max(1.0, row_buffer_size / avg_tile_bytes)
    
    unique_rows_factor = 1.0 / tiles_per_row
    
    print(f"\n【Sequential 参数】")
    print(f"  cr_b = {cr_b:.10f}")
    print(f"  cr_r = {cr_r:.10f}")
    print(f"  non_crossing_coeff = (1-cr_b)(1-cr_r) = {non_crossing_coeff:.10f}")
    print(f"  crossing_coeff = 2(cr_b+cr_r) = {crossing_coeff:.10f}")
    print(f"  avg_tile_bytes = {avg_tile_bytes:.1f}")
    print(f"  tiles_per_row = {tiles_per_row:.10f}")
    print(f"  unique_rows_factor = 1/tpr = {unique_rows_factor:.10f}")
    print(f"  crossing/non_crossing = {crossing_coeff / non_crossing_coeff if non_crossing_coeff > 0 else float('inf'):.6f}")
    
    # =========================================================================
    # 4. 运行 ILP 验证
    # =========================================================================
    from pim_optimizer import PIMOptimizer
    optimizer = PIMOptimizer(arch, verbose=False)
    result = optimizer.optimize([workload])
    model = optimizer.model
    
    rb_tiles = model.getVarByName('RB_TILES_INPUT_(0)').X
    reuse = model.getVarByName('REUSE_INPUT_(0)').X
    nc = model.getVarByName('ROW_ACT_INPUT_NC_(0)').X
    cr = model.getVarByName('ROW_ACT_INPUT_CR_(0)').X
    crossing_base = model.getVarByName('CROSSING_INPUT_BASE_(0)').X
    
    print(f"\n【ILP 变量】")
    print(f"  rb_tiles = {rb_tiles:.10f}")
    print(f"  reuse = {reuse:.10f}")
    print(f"  crossing_base = {crossing_base:.10f} (= rb_tiles × reuse: {rb_tiles * reuse:.10f})")
    print(f"  nc = {nc:.10f}")
    print(f"  cr = {cr:.10f}")
    
    # =========================================================================
    # 5. 手动验证公式
    # =========================================================================
    # nc = rb_tiles × unique_rows_factor × non_crossing_coeff / banks
    expected_nc = rb_tiles * unique_rows_factor * non_crossing_coeff / banks
    # cr = crossing_base × unique_rows_factor × crossing_coeff / banks
    expected_cr = crossing_base * unique_rows_factor * crossing_coeff / banks
    
    print(f"\n【手动公式验证】")
    print(f"  expected_nc = rb_tiles × (1/tpr) × nc_coeff / banks")
    print(f"              = {rb_tiles:.6f} × {unique_rows_factor:.10f} × {non_crossing_coeff:.10f} / {banks}")
    print(f"              = {expected_nc:.10f}")
    print(f"  ILP nc      = {nc:.10f}")
    print(f"  匹配        = {abs(expected_nc - nc) < 1e-6}")
    
    print(f"\n  expected_cr = crossing_base × (1/tpr) × cr_coeff / banks")
    print(f"              = {crossing_base:.6f} × {unique_rows_factor:.10f} × {crossing_coeff:.10f} / {banks}")
    print(f"              = {expected_cr:.10f}")
    print(f"  ILP cr      = {cr:.10f}")
    print(f"  匹配        = {abs(expected_cr - cr) < 1e-6}")
    
    # 验证总的 sequential row activation
    expected_total = expected_nc + expected_cr
    ilp_total = nc + cr
    print(f"\n  expected_total = {expected_total:.10f}")
    print(f"  ILP total      = {ilp_total:.10f}")
    print(f"  匹配           = {abs(expected_total - ilp_total) < 1e-6}")
    
    print()
    return {
        'name': name,
        'cr_b': cr_b,
        'cr_r': cr_r,
        'non_crossing_coeff': non_crossing_coeff,
        'crossing_coeff': crossing_coeff,
        'tiles_per_row': tiles_per_row,
        'rb_tiles': rb_tiles,
        'reuse': reuse,
        'nc_match': abs(expected_nc - nc) < 1e-6,
        'cr_match': abs(expected_cr - cr) < 1e-6,
    }


def main():
    test_workloads = [
        {"name": "tiny", "N": 1, "K": 8, "C": 8, "P": 4, "Q": 4, "R": 3, "S": 3},
        {"name": "small", "N": 1, "K": 32, "C": 32, "P": 14, "Q": 14, "R": 3, "S": 3},
        {"name": "medium_1x1", "N": 1, "K": 64, "C": 64, "P": 14, "Q": 14, "R": 1, "S": 1},
        {"name": "medium_3x3", "N": 1, "K": 64, "C": 64, "P": 14, "Q": 14, "R": 3, "S": 3},
        {"name": "large", "N": 1, "K": 128, "C": 128, "P": 7, "Q": 7, "R": 3, "S": 3},
    ]
    
    results = []
    for wl in test_workloads:
        name = wl.pop('name')
        wl['name'] = name
        r = direct_verify_v2(name, wl)
        results.append(r)
    
    # 汇总
    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)
    print(f"{'Workload':<12} {'cr_b':>10} {'cr_r':>10} {'tpr':>10} {'nc_match':>10} {'cr_match':>10}")
    print("-" * 64)
    for r in results:
        print(f"{r['name']:<12} {r['cr_b']:>10.4f} {r['cr_r']:>10.4f} {r['tiles_per_row']:>10.2f} {str(r['nc_match']):>10} {str(r['cr_match']):>10}")


if __name__ == "__main__":
    main()
