#!/usr/bin/env python3
"""
手动验证 Sequential 模式 Row Activation 公式

公式:
row_acts = rb_tiles / tiles_per_row / banks × [(1-cr_b)(1-cr_r) + 2×reuse×(cr_b+cr_r)]

从 ILP 结果验证计算是否正确
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')

import math
import numpy as np
from pim_optimizer.arch.pim_arch import PIMArchitecture
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer import PIMOptimizer
from pim_optimizer.model.row_activation import (
    compute_input_crossing_ratio,
    precise_crossing_ratio
)


def verify_workload(name, wl_params):
    """验证单个 workload 的 row activation 计算"""
    
    print("=" * 80)
    print(f"验证 Workload: {name}")
    print("=" * 80)
    
    workload = ConvWorkload(**wl_params)
    arch = PIMArchitecture()
    
    optimizer = PIMOptimizer(arch, verbose=False)
    result = optimizer.optimize([workload])
    model = optimizer.model
    
    # 基本参数
    H_in = wl_params['Q'] + wl_params['S'] - 1
    W_in = wl_params['P'] + wl_params['R'] - 1
    R, S = wl_params['R'], wl_params['S']
    
    print(f"\n基本参数:")
    print(f"  Input size: {H_in} × {W_in}")
    print(f"  Kernel: {R} × {S}")
    
    # 架构参数
    row_buffer_bytes = 1024  # 1KB
    element_bytes = 2  # INT16
    banks = 16
    
    print(f"  Row buffer: {row_buffer_bytes} bytes")
    print(f"  Element size: {element_bytes} bytes")
    print(f"  Banks: {banks}")
    
    # 从 ILP 获取结果
    rb_tiles_var = model.getVarByName('RB_TILES_INPUT_(0)')
    reuse_var = model.getVarByName('REUSE_INPUT_(0)')
    seq_var = model.getVarByName('INPUT_ROW_ACT_SEQ_(0)')
    nc_var = model.getVarByName('ROW_ACT_INPUT_NC_(0)')
    cr_var = model.getVarByName('ROW_ACT_INPUT_CR_(0)')
    
    rb_tiles = rb_tiles_var.X if rb_tiles_var else 0
    reuse = reuse_var.X if reuse_var else 1
    seq_result = seq_var.X if seq_var else 0
    nc_result = nc_var.X if nc_var else 0
    cr_result = cr_var.X if cr_var else 0
    
    print(f"\n从 ILP 获取的结果:")
    print(f"  rb_tiles = {rb_tiles:.4f}")
    print(f"  reuse = {reuse:.4f}")
    print(f"  seq_row_act = {seq_result:.4f}")
    print(f"    non-crossing part = {nc_result:.4f}")
    print(f"    crossing part = {cr_result:.4f}")
    
    # 获取 H/W divisors
    h_divisors = getattr(workload, 'hw_divisors', {}).get('H', [1])
    w_divisors = getattr(workload, 'hw_divisors', {}).get('W', [1])
    
    print(f"\n  H divisors: {h_divisors}")
    print(f"  W divisors: {w_divisors}")
    
    # 获取 tile info
    yh_list = []
    yw_list = []
    
    # 从 tile_info 获取可能的 tile 大小
    # 简化：使用 workload 的 Q, P 和 kernel 大小计算
    # input_tile_h = Q_tile + S - 1
    # input_tile_w = P_tile + R - 1
    
    Q_divs = workload.divisors[3] if len(workload.divisors) > 3 else [1]
    P_divs = workload.divisors[2] if len(workload.divisors) > 2 else [1]
    
    print(f"  Q divisors: {Q_divs}")
    print(f"  P divisors: {P_divs}")
    
    for q_div in Q_divs:
        yh_list.append(q_div + S - 1)
    for p_div in P_divs:
        yw_list.append(p_div + R - 1)
    
    print(f"  Possible yh (input tile H): {yh_list}")
    print(f"  Possible yw (input tile W): {yw_list}")
    
    # 计算平均 crossing ratios
    avg_row_cr_list = []
    avg_block_cr_h_list = []
    avg_block_cr_w_list = []
    
    stride_h = getattr(workload, 'stride_h', 1)
    stride_w = getattr(workload, 'stride_w', 1)
    
    for block_h in h_divisors:
        for tile_h in yh_list:
            # Block crossing H
            cr_h = compute_input_crossing_ratio(
                block_h=block_h, tile_h=tile_h, step=1,
                tiler_s=S, total_S=S, dilation=1
            )
            avg_block_cr_h_list.append(cr_h)
    
    for block_w in w_divisors:
        for tile_w in yw_list:
            # Block crossing W
            cr_w = compute_input_crossing_ratio(
                block_h=block_w, tile_h=tile_w, step=1,
                tiler_s=R, total_S=R, dilation=1
            )
            avg_block_cr_w_list.append(cr_w)
    
    for tile_h in yh_list:
        for tile_w in yw_list:
            tile_bytes = tile_h * tile_w * element_bytes
            cr_r = precise_crossing_ratio(tile_bytes, row_buffer_bytes)
            avg_row_cr_list.append(cr_r)
    
    avg_block_cr_h = sum(avg_block_cr_h_list) / len(avg_block_cr_h_list) if avg_block_cr_h_list else 0
    avg_block_cr_w = sum(avg_block_cr_w_list) / len(avg_block_cr_w_list) if avg_block_cr_w_list else 0
    avg_row_cr = sum(avg_row_cr_list) / len(avg_row_cr_list) if avg_row_cr_list else 0
    
    # Combined block crossing
    avg_block_cr = 1.0 - (1.0 - avg_block_cr_h) * (1.0 - avg_block_cr_w)
    
    print(f"\n计算的 crossing ratios:")
    print(f"  avg_block_cr_h = {avg_block_cr_h:.4f}")
    print(f"  avg_block_cr_w = {avg_block_cr_w:.4f}")
    print(f"  avg_block_cr (combined) = {avg_block_cr:.4f}")
    print(f"  avg_row_cr = {avg_row_cr:.4f}")
    
    # 计算 tiles_per_row
    avg_tile_bytes = sum(h * w * element_bytes for h in yh_list for w in yw_list) / (len(yh_list) * len(yw_list))
    tiles_per_row = max(1.0, row_buffer_bytes / avg_tile_bytes)
    
    print(f"\n  avg_tile_bytes = {avg_tile_bytes:.2f}")
    print(f"  tiles_per_row = {tiles_per_row:.4f}")
    
    # 手动计算公式
    cr_b = avg_block_cr
    cr_r = avg_row_cr
    
    non_crossing_coeff = (1.0 - cr_b) * (1.0 - cr_r)
    crossing_coeff = 2.0 * (cr_b + cr_r)
    
    print(f"\n公式系数:")
    print(f"  non_crossing_coeff = (1-{cr_b:.4f})(1-{cr_r:.4f}) = {non_crossing_coeff:.4f}")
    print(f"  crossing_coeff = 2 × ({cr_b:.4f} + {cr_r:.4f}) = {crossing_coeff:.4f}")
    
    # 公式计算
    # row_acts = rb_tiles / tiles_per_row / banks × [(1-cr_b)(1-cr_r) + 2×reuse×(cr_b+cr_r)]
    
    # Non-crossing part
    manual_nc = rb_tiles / tiles_per_row * non_crossing_coeff / banks
    
    # Crossing part
    manual_cr = rb_tiles * reuse / tiles_per_row * crossing_coeff / banks
    
    manual_total = manual_nc + manual_cr
    
    print(f"\n手动计算结果:")
    print(f"  non_crossing = {rb_tiles:.4f} / {tiles_per_row:.4f} × {non_crossing_coeff:.4f} / {banks} = {manual_nc:.4f}")
    print(f"  crossing = {rb_tiles:.4f} × {reuse:.4f} / {tiles_per_row:.4f} × {crossing_coeff:.4f} / {banks} = {manual_cr:.4f}")
    print(f"  total = {manual_total:.4f}")
    
    print(f"\n对比:")
    print(f"  ILP 结果: {seq_result:.4f}")
    print(f"  手动计算: {manual_total:.4f}")
    
    diff = abs(seq_result - manual_total)
    print(f"  差异: {diff:.6f}")
    
    if diff < 0.01:
        print(f"  ✓ 计算一致！")
    else:
        print(f"  ✗ 存在差异，需要检查!")
    
    return {
        'name': name,
        'ilp': seq_result,
        'manual': manual_total,
        'diff': diff
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
        r = verify_workload(name, wl)
        results.append(r)
        print()
    
    # 汇总
    print("\n" + "=" * 80)
    print("验证汇总")
    print("=" * 80)
    print(f"{'Workload':<15} {'ILP结果':>12} {'手动计算':>12} {'差异':>12} {'状态':<8}")
    print("-" * 60)
    for r in results:
        status = "✓" if r['diff'] < 0.01 else "✗"
        print(f"{r['name']:<15} {r['ilp']:>12.4f} {r['manual']:>12.4f} {r['diff']:>12.6f} {status:<8}")


if __name__ == "__main__":
    main()
