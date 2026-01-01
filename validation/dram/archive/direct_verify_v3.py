#!/usr/bin/env python3
"""
直接验证 Sequential 模式的 crossing ratio（使用正确的架构参数）
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
from pim_optimizer.model.expressions import compute_unique_input_size


def direct_verify_v3(name, wl_params):
    """直接验证公式（使用正确的架构参数）"""
    
    print("=" * 80)
    print(f"Workload: {name}")
    print("=" * 80)
    
    workload = ConvWorkload(**wl_params)
    arch = PIMArchitecture()
    
    # 从架构获取参数
    dram_level = arch.mem_idx.get("LocalDRAM")
    rowbuf_level = arch.mem_idx.get("RowBuffer")
    
    # 使用实际的架构参数
    dram_num_banks = arch.mem_num_banks[dram_level]
    effective_banks = max(1, dram_num_banks)
    
    # row_buffer_size_bytes
    row_buffer_size_bytes = None
    if hasattr(arch, "mem_row_buffer_size"):
        rb_size = arch.mem_row_buffer_size[rowbuf_level]
        if rb_size not in (None, 0):
            row_buffer_size_bytes = float(rb_size)
        else:
            rb_entries = arch.mem_entries[rowbuf_level]
            if rb_entries not in (None, 0, -1):
                row_buffer_size_bytes = float(rb_entries)
    
    if row_buffer_size_bytes is None:
        row_buffer_size_bytes = 1024.0
    
    # element_bytes
    elem_bits_map = getattr(arch, "element_bits_per_dtype", None)
    if isinstance(elem_bits_map, dict):
        element_bits_input = elem_bits_map.get("input", getattr(arch, "default_element_bits", 8))
    else:
        element_bits_input = getattr(arch, "default_element_bits", 8)
    element_bytes_input = max(1.0, math.ceil(element_bits_input / 8.0))
    
    print(f"\n【架构参数】")
    print(f"  DRAM banks: {effective_banks}")
    print(f"  row_buffer_size: {row_buffer_size_bytes}")
    print(f"  element_bits_input: {element_bits_input}")
    print(f"  element_bytes_input: {element_bytes_input}")
    
    # Workload 参数
    stride_h = workload.stride[1]
    stride_w = workload.stride[0]
    dilation_h = workload.dilation[1]
    dilation_w = workload.dilation[0]
    total_S = workload.bounds[1]
    total_R = workload.bounds[0]
    total_Q = workload.bounds[3]
    total_P = workload.bounds[2]
    
    h_divisors = getattr(workload, 'hw_divisors', {}).get('H', [1])
    w_divisors = getattr(workload, 'hw_divisors', {}).get('W', [1])
    
    # =========================================================================
    # 构建 yh_list 和 yw_list
    # =========================================================================
    yw_list = []
    yw_p_list = []
    yw_r_list = []
    for p_ in workload.divisors[2]:
        for r_ in workload.divisors[0]:
            unique_w = compute_unique_input_size(stride_w, dilation_w, p_, r_)
            yw_list.append(unique_w)
            yw_p_list.append(p_)
            yw_r_list.append(r_)
    
    yh_list = []
    yh_q_list = []
    yh_s_list = []
    for q_ in workload.divisors[3]:
        for s_ in workload.divisors[1]:
            unique_h = compute_unique_input_size(stride_h, dilation_h, q_, s_)
            yh_list.append(unique_h)
            yh_q_list.append(q_)
            yh_s_list.append(s_)
    
    # =========================================================================
    # 1. Row Crossing Ratio (cr_r)
    # =========================================================================
    seq_row_crossing_ratios = []
    for block_h in h_divisors:
        for tile_h in yh_list:
            for block_w in w_divisors:
                for tile_w in yw_list:
                    tile_bytes = tile_h * tile_w * element_bytes_input
                    cr = compute_dram_row_crossing_ratio(tile_bytes, row_buffer_size_bytes)
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
            tile_bytes = tile_h * tile_w * element_bytes_input
            avg_tile_bytes_list.append(tile_bytes)
    avg_tile_bytes = sum(avg_tile_bytes_list) / len(avg_tile_bytes_list) if avg_tile_bytes_list else element_bytes_input
    tiles_per_row = max(1.0, row_buffer_size_bytes / avg_tile_bytes)
    
    unique_rows_factor = 1.0 / tiles_per_row
    
    print(f"\n【Sequential 参数】")
    print(f"  cr_b = {cr_b:.10f}")
    print(f"  cr_r = {cr_r:.10f}")
    print(f"  non_crossing_coeff = {non_crossing_coeff:.10f}")
    print(f"  crossing_coeff = {crossing_coeff:.10f}")
    print(f"  avg_tile_bytes = {avg_tile_bytes:.1f}")
    print(f"  tiles_per_row = {tiles_per_row:.10f}")
    print(f"  unique_rows_factor = {unique_rows_factor:.10f}")
    
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
    print(f"  crossing_base = {crossing_base:.10f}")
    print(f"  nc = {nc:.10f}")
    print(f"  cr = {cr:.10f}")
    
    # =========================================================================
    # 5. 手动验证公式
    # =========================================================================
    # nc = rb_tiles × unique_rows_factor × non_crossing_coeff / banks
    expected_nc = rb_tiles * unique_rows_factor * non_crossing_coeff / effective_banks
    # cr = crossing_base × unique_rows_factor × crossing_coeff / banks
    expected_cr = crossing_base * unique_rows_factor * crossing_coeff / effective_banks
    
    print(f"\n【手动公式验证】")
    print(f"  expected_nc = {rb_tiles:.6f} × {unique_rows_factor:.10f} × {non_crossing_coeff:.10f} / {effective_banks}")
    print(f"              = {expected_nc:.10f}")
    print(f"  ILP nc      = {nc:.10f}")
    nc_match = abs(expected_nc - nc) < 1e-5
    print(f"  匹配        = {nc_match}")
    
    print(f"\n  expected_cr = {crossing_base:.6f} × {unique_rows_factor:.10f} × {crossing_coeff:.10f} / {effective_banks}")
    print(f"              = {expected_cr:.10f}")
    print(f"  ILP cr      = {cr:.10f}")
    cr_match = abs(expected_cr - cr) < 1e-5
    print(f"  匹配        = {cr_match}")
    
    # 验证总的 sequential row activation
    expected_total = expected_nc + expected_cr
    ilp_total = nc + cr
    print(f"\n  expected_total = {expected_total:.10f}")
    print(f"  ILP total      = {ilp_total:.10f}")
    total_match = abs(expected_total - ilp_total) < 1e-5
    print(f"  匹配           = {total_match}")
    
    print()
    return nc_match, cr_match, total_match


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
        nc_m, cr_m, tot_m = direct_verify_v3(name, wl)
        results.append((name, nc_m, cr_m, tot_m))
    
    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)
    print(f"{'Workload':<15} {'nc_match':>10} {'cr_match':>10} {'total_match':>12}")
    print("-" * 50)
    for name, nc_m, cr_m, tot_m in results:
        print(f"{name:<15} {str(nc_m):>10} {str(cr_m):>10} {str(tot_m):>12}")


if __name__ == "__main__":
    main()
