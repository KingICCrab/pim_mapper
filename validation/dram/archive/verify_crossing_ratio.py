#!/usr/bin/env python3
"""
验证 Sequential 模式公式中的 crossing ratio 参数
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


def analyze_crossing_ratios(name, wl_params):
    """分析 workload 的各种 crossing ratio"""
    
    print("=" * 80)
    print(f"Workload: {name}")
    print("=" * 80)
    
    workload = ConvWorkload(**wl_params)
    arch = PIMArchitecture()
    
    # 从 workload 获取参数
    stride_h = getattr(workload, 'stride_h', 1)
    stride_w = getattr(workload, 'stride_w', 1)
    dilation_h = getattr(workload, 'dilation_h', 1)
    dilation_w = getattr(workload, 'dilation_w', 1)
    total_S = getattr(workload, 'S', 1)
    total_R = getattr(workload, 'R', 1)
    
    element_bytes = 2.0  # FP16
    row_buffer_size = 1024.0  # bytes
    
    # H/W divisors
    h_divisors = getattr(workload, 'hw_divisors', {}).get('H', [1])
    w_divisors = getattr(workload, 'hw_divisors', {}).get('W', [1])
    
    print(f"\n【Workload 参数】")
    print(f"  Q={workload.bounds[2]}, P={workload.bounds[3]}, S={total_S}, R={total_R}")
    print(f"  stride_h={stride_h}, stride_w={stride_w}")
    print(f"  H_in = Q * stride + (S-1)*dilation + 1 - stride")
    H_in = workload.bounds[2] * stride_h + (total_S - 1) * dilation_h + 1 - stride_h
    W_in = workload.bounds[3] * stride_w + (total_R - 1) * dilation_w + 1 - stride_w
    print(f"  H_in={H_in}, W_in={W_in}")
    
    print(f"\n【可用 divisors】")
    print(f"  H divisors: {h_divisors}")
    print(f"  W divisors: {w_divisors}")
    
    # 计算所有可能的 tile_h
    Q = workload.bounds[2]
    P = workload.bounds[3]
    
    # 从 expressions.py 逻辑: tile_h = Q_factor × stride + (S_factor - 1) × dilation + 1 - stride
    # 其中 Q_factor 是 Q 的因子, S_factor 是 S 的因子
    
    # 简化: 假设 S_factor = total_S (不分割 kernel)
    # tile_h = Q_factor × stride + (total_S - 1) × dilation + 1 - stride
    
    q_divisors = workload.divisors[2]  # Q 的 divisors
    p_divisors = workload.divisors[3]  # P 的 divisors
    
    print(f"\n【Q/P divisors】")
    print(f"  Q={Q}, divisors: {q_divisors}")
    print(f"  P={P}, divisors: {p_divisors}")
    
    # 计算 yh_list (tile heights)
    yh_list = []
    yh_q_list = []
    yh_s_list = []
    for q_factor in q_divisors:
        s_factor = total_S  # 假设不分割 kernel
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
    
    print(f"\n【Tile sizes】")
    print(f"  yh_list (tile heights): {yh_list}")
    print(f"  yw_list (tile widths): {yw_list}")
    print(f"  yh_q_list (Q factors): {yh_q_list}")
    print(f"  yw_p_list (P factors): {yw_p_list}")
    
    # =====================
    # 1. Row Crossing Ratio (DRAM row boundary crossing for Sequential mode)
    # =====================
    print(f"\n【Row Crossing Ratio (tile 跨 DRAM row)】")
    seq_row_crossing_ratios = []
    for block_h in h_divisors:
        for tile_h in yh_list:
            for block_w in w_divisors:
                for tile_w in yw_list:
                    tile_bytes = tile_h * tile_w * element_bytes
                    cr = compute_dram_row_crossing_ratio(tile_bytes, row_buffer_size)
                    seq_row_crossing_ratios.append(cr)
                    if len(seq_row_crossing_ratios) <= 10:
                        print(f"    tile {tile_h}x{tile_w} = {tile_bytes:.1f} bytes → row_cr = {cr:.4f}")
    
    avg_row_cr = sum(seq_row_crossing_ratios) / len(seq_row_crossing_ratios) if seq_row_crossing_ratios else 0.0
    print(f"  Average Row Crossing Ratio (cr_r): {avg_row_cr:.4f}")
    
    # =====================
    # 2. Block Crossing Ratio (H direction - sliding window pattern)
    # =====================
    print(f"\n【Block Crossing Ratio H (滑动窗口跨 block)】")
    seq_block_crossing_ratios_h = []
    for i, block_h in enumerate(h_divisors):
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
            if len(seq_block_crossing_ratios_h) <= 10:
                print(f"    block_h={block_h}, tile_h={tile_h}, step={step_h} → cr_h = {cr_h:.4f}")
    
    avg_block_cr_h = sum(seq_block_crossing_ratios_h) / len(seq_block_crossing_ratios_h) if seq_block_crossing_ratios_h else 0.0
    print(f"  Average Block Crossing Ratio H: {avg_block_cr_h:.4f}")
    
    # =====================
    # 3. Block Crossing Ratio (W direction)
    # =====================
    print(f"\n【Block Crossing Ratio W】")
    seq_block_crossing_ratios_w = []
    for ii, block_w in enumerate(w_divisors):
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
            if len(seq_block_crossing_ratios_w) <= 10:
                print(f"    block_w={block_w}, tile_w={tile_w}, step={step_w} → cr_w = {cr_w:.4f}")
    
    avg_block_cr_w = sum(seq_block_crossing_ratios_w) / len(seq_block_crossing_ratios_w) if seq_block_crossing_ratios_w else 0.0
    print(f"  Average Block Crossing Ratio W: {avg_block_cr_w:.4f}")
    
    # =====================
    # 4. Combined Block Crossing
    # =====================
    # cr_b = 1 - (1 - cr_h) × (1 - cr_w)
    avg_block_cr = 1.0 - (1.0 - avg_block_cr_h) * (1.0 - avg_block_cr_w)
    print(f"\n【Combined Block Crossing】")
    print(f"  cr_b = 1 - (1 - {avg_block_cr_h:.4f}) × (1 - {avg_block_cr_w:.4f}) = {avg_block_cr:.4f}")
    
    # =====================
    # 5. Formula 系数
    # =====================
    cr_b = avg_block_cr
    cr_r = avg_row_cr
    
    non_crossing_coeff = (1.0 - cr_b) * (1.0 - cr_r)
    crossing_coeff = 2.0 * (cr_b + cr_r)
    
    print(f"\n【公式系数】")
    print(f"  cr_b (Block Crossing) = {cr_b:.6f}")
    print(f"  cr_r (Row Crossing) = {cr_r:.6f}")
    print(f"  non_crossing_coeff = (1-cr_b)(1-cr_r) = {non_crossing_coeff:.6f}")
    print(f"  crossing_coeff = 2×(cr_b+cr_r) = {crossing_coeff:.6f}")
    print(f"  crossing_coeff / non_crossing_coeff = {crossing_coeff / non_crossing_coeff if non_crossing_coeff > 0 else 'inf':.6f}")
    
    # =====================
    # 6. tiles_per_row
    # =====================
    avg_tile_bytes_list = []
    for tile_h in yh_list:
        for tile_w in yw_list:
            tile_bytes = tile_h * tile_w * element_bytes
            avg_tile_bytes_list.append(tile_bytes)
    avg_tile_bytes = sum(avg_tile_bytes_list) / len(avg_tile_bytes_list) if avg_tile_bytes_list else element_bytes
    tiles_per_row = max(1.0, row_buffer_size / avg_tile_bytes)
    
    print(f"\n【tiles_per_row】")
    print(f"  Average tile bytes = {avg_tile_bytes:.1f}")
    print(f"  tiles_per_row = {row_buffer_size} / {avg_tile_bytes:.1f} = {tiles_per_row:.4f}")
    
    print()
    return {
        'name': name,
        'cr_b': cr_b,
        'cr_r': cr_r,
        'tiles_per_row': tiles_per_row,
        'non_crossing_coeff': non_crossing_coeff,
        'crossing_coeff': crossing_coeff,
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
        r = analyze_crossing_ratios(name, wl)
        results.append(r)
    
    # 汇总
    print("\n" + "=" * 80)
    print("汇总: crossing_coeff / non_crossing_coeff 对比 ILP")
    print("=" * 80)
    
    # ILP 中反推的 crossing/non_crossing 比值
    ilp_ratios = {
        'tiny': 13.106720,
        'small': 14.661158,
        'medium_1x1': 3.560509,
        'medium_3x3': 14.661158,
        'large': 12.495572,
    }
    
    print(f"{'Workload':<12} {'cr_b':>10} {'cr_r':>10} {'tpr':>10} {'nc_coeff':>12} {'cr_coeff':>12} {'ratio':>12} {'ILP ratio':>12}")
    print("-" * 100)
    for r in results:
        ratio = r['crossing_coeff'] / r['non_crossing_coeff'] if r['non_crossing_coeff'] > 0 else float('inf')
        ilp_ratio = ilp_ratios.get(r['name'], 0)
        print(f"{r['name']:<12} {r['cr_b']:>10.4f} {r['cr_r']:>10.4f} {r['tiles_per_row']:>10.2f} {r['non_crossing_coeff']:>12.6f} {r['crossing_coeff']:>12.6f} {ratio:>12.6f} {ilp_ratio:>12.6f}")


if __name__ == "__main__":
    main()
