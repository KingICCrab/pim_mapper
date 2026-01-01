#!/usr/bin/env python3
"""
直接验证 Sequential 模式使用的 cr_b 和 cr_r 值
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')

import math
from pim_optimizer.arch.pim_arch import PIMArchitecture
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer.model.row_activation import (
    compute_dram_row_crossing_ratio,
    compute_input_block_crossing_ratio
)


def direct_verify(name, wl_params):
    """直接验证公式"""
    
    print("=" * 80)
    print(f"Workload: {name}")
    print("=" * 80)
    
    workload = ConvWorkload(**wl_params)
    arch = PIMArchitecture()
    
    # 架构参数
    banks = 16
    row_buffer_size = 1024.0
    element_bytes = 2.0  # FP16
    
    stride_h = getattr(workload, 'stride_h', 1)
    stride_w = getattr(workload, 'stride_w', 1)
    dilation_h = getattr(workload, 'dilation_h', 1)
    dilation_w = getattr(workload, 'dilation_w', 1)
    total_S = getattr(workload, 'S', 1)
    total_R = getattr(workload, 'R', 1)
    
    h_divisors = getattr(workload, 'hw_divisors', {}).get('H', [1])
    w_divisors = getattr(workload, 'hw_divisors', {}).get('W', [1])
    q_divisors = workload.divisors[2]
    p_divisors = workload.divisors[3]
    
    # 构建 yh_list, yw_list (和 row_activation.py 相同的逻辑)
    yh_list = []
    yh_q_list = []
    yh_s_list = []
    for q_factor in q_divisors:
        s_factor = total_S
        tile_h = q_factor * stride_h + (s_factor - 1) * dilation_h + 1 - stride_h
        yh_list.append(tile_h)
        yh_q_list.append(q_factor)
        yh_s_list.append(s_factor)
    
    yw_list = []
    yw_p_list = []
    yw_r_list = []
    for p_factor in p_divisors:
        r_factor = total_R
        tile_w = p_factor * stride_w + (r_factor - 1) * dilation_w + 1 - stride_w
        yw_list.append(tile_w)
        yw_p_list.append(p_factor)
        yw_r_list.append(r_factor)
    
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
    
    print(f"\n【计算的 Sequential 参数】")
    print(f"  cr_b = {cr_b:.6f}")
    print(f"  cr_r = {cr_r:.6f}")
    print(f"  non_crossing_coeff = (1-cr_b)(1-cr_r) = {non_crossing_coeff:.10f}")
    print(f"  crossing_coeff = 2(cr_b+cr_r) = {crossing_coeff:.10f}")
    print(f"  tiles_per_row = {tiles_per_row:.6f}")
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
    print(f"  rb_tiles = {rb_tiles:.6f}")
    print(f"  reuse = {reuse:.6f}")
    print(f"  crossing_base = {crossing_base:.6f}")
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
    print(f"  差异        = {abs(expected_nc - nc):.10f}")
    
    print(f"\n  expected_cr = crossing_base × (1/tpr) × cr_coeff / banks")
    print(f"              = {crossing_base:.6f} × {unique_rows_factor:.10f} × {crossing_coeff:.10f} / {banks}")
    print(f"              = {expected_cr:.10f}")
    print(f"  ILP cr      = {cr:.10f}")
    print(f"  差异        = {abs(expected_cr - cr):.10f}")
    
    # 验证比例
    if nc > 1e-15:
        ilp_ratio = cr / nc
        expected_ratio = (crossing_coeff / non_crossing_coeff) * reuse if non_crossing_coeff > 0 else float('inf')
        print(f"\n  ILP cr/nc = {ilp_ratio:.6f}")
        print(f"  Expected (cr_coeff/nc_coeff × reuse) = {expected_ratio:.6f}")
        print(f"  差异 = {abs(ilp_ratio - expected_ratio):.6f}")
    
    print()


def main():
    test_workloads = [
        {"name": "tiny", "N": 1, "K": 8, "C": 8, "P": 4, "Q": 4, "R": 3, "S": 3},
        {"name": "small", "N": 1, "K": 32, "C": 32, "P": 14, "Q": 14, "R": 3, "S": 3},
        {"name": "medium_1x1", "N": 1, "K": 64, "C": 64, "P": 14, "Q": 14, "R": 1, "S": 1},
    ]
    
    for wl in test_workloads:
        name = wl.pop('name')
        wl['name'] = name
        direct_verify(name, wl)


if __name__ == "__main__":
    main()
