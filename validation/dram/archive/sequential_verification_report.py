#!/usr/bin/env python3
"""
Sequential 模式 Input Row Activation 公式完整验证报告

验证公式:
  row_acts = rb_tiles / tiles_per_row / banks × [(1-cr_b)(1-cr_r) + 2×reuse×(cr_b+cr_r)]

其中:
- rb_tiles: RowBuffer 级别的 unique tile 数量
- tiles_per_row: 一个 DRAM row 能容纳多少个 tile
- banks: DRAM bank 数量
- cr_b: Block Crossing ratio (滑动窗口跨 block 边界的概率)
- cr_r: Row Crossing ratio (tile 数据跨 DRAM row 边界的概率)
- reuse: K dimension 的 reuse factor (当 K 是最内层循环时)
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')

import math
from pim_optimizer.arch.pim_arch import PIMArchitecture
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer.model.row_activation import (
    precise_crossing_ratio,
    compute_input_crossing_ratio
)
from pim_optimizer.model.expressions import compute_unique_input_size
from pim_optimizer import PIMOptimizer


def verify_workload(name, wl_params):
    """验证单个 workload"""
    
    workload = ConvWorkload(**wl_params)
    arch = PIMArchitecture()
    
    # 架构参数
    dram_level = arch.mem_idx.get("LocalDRAM")
    rowbuf_level = arch.mem_idx.get("RowBuffer")
    effective_banks = max(1, arch.mem_num_banks[dram_level])
    row_buffer_size = float(arch.mem_row_buffer_size[rowbuf_level] or arch.mem_entries[rowbuf_level])
    element_bits = getattr(arch, "default_element_bits", 8)
    element_bytes = max(1.0, math.ceil(element_bits / 8.0))
    
    # Workload 参数
    stride_h, stride_w = workload.stride[1], workload.stride[0]
    dilation_h, dilation_w = workload.dilation[1], workload.dilation[0]
    total_S, total_R = workload.bounds[1], workload.bounds[0]
    h_divisors = getattr(workload, 'hw_divisors', {}).get('H', [1])
    w_divisors = getattr(workload, 'hw_divisors', {}).get('W', [1])
    
    # 构建 tile lists
    yh_list, yh_q_list, yh_s_list = [], [], []
    for q_ in workload.divisors[3]:
        for s_ in workload.divisors[1]:
            yh_list.append(compute_unique_input_size(stride_h, dilation_h, q_, s_))
            yh_q_list.append(q_)
            yh_s_list.append(s_)
    
    yw_list, yw_p_list, yw_r_list = [], [], []
    for p_ in workload.divisors[2]:
        for r_ in workload.divisors[0]:
            yw_list.append(compute_unique_input_size(stride_w, dilation_w, p_, r_))
            yw_p_list.append(p_)
            yw_r_list.append(r_)
    
    # 计算 Row Crossing ratio
    row_cr_list = []
    for bh in h_divisors:
        for th in yh_list:
            for bw in w_divisors:
                for tw in yw_list:
                    tile_bytes = th * tw * element_bytes
                    row_cr_list.append(precise_crossing_ratio(tile_bytes, row_buffer_size))
    cr_r = sum(row_cr_list) / len(row_cr_list)
    
    # 计算 Block Crossing ratio
    block_cr_h_list = []
    for bh in h_divisors:
        for j, th in enumerate(yh_list):
            step_h = yh_q_list[j] * stride_h
            block_cr_h_list.append(compute_input_crossing_ratio(bh, th, step_h, yh_s_list[j], total_S, dilation_h))
    
    block_cr_w_list = []
    for bw in w_divisors:
        for j, tw in enumerate(yw_list):
            step_w = yw_p_list[j] * stride_w
            block_cr_w_list.append(compute_input_crossing_ratio(bw, tw, step_w, yw_r_list[j], total_R, dilation_w))
    
    cr_h = sum(block_cr_h_list) / len(block_cr_h_list)
    cr_w = sum(block_cr_w_list) / len(block_cr_w_list)
    cr_b = 1.0 - (1.0 - cr_h) * (1.0 - cr_w)
    
    # 公式系数
    non_crossing_coeff = (1.0 - cr_b) * (1.0 - cr_r)
    crossing_coeff = 2.0 * (cr_b + cr_r)
    
    # tiles_per_row
    tile_bytes_list = [th * tw * element_bytes for th in yh_list for tw in yw_list]
    avg_tile_bytes = sum(tile_bytes_list) / len(tile_bytes_list)
    tiles_per_row = max(1.0, row_buffer_size / avg_tile_bytes)
    unique_rows_factor = 1.0 / tiles_per_row
    
    # 运行 ILP
    optimizer = PIMOptimizer(arch, verbose=False)
    optimizer.optimize([workload])
    model = optimizer.model
    
    rb_tiles = model.getVarByName('RB_TILES_INPUT_(0)').X
    reuse = model.getVarByName('REUSE_INPUT_(0)').X
    crossing_base = model.getVarByName('CROSSING_INPUT_BASE_(0)').X
    ilp_nc = model.getVarByName('ROW_ACT_INPUT_NC_(0)').X
    ilp_cr = model.getVarByName('ROW_ACT_INPUT_CR_(0)').X
    ilp_total = model.getVarByName('INPUT_ROW_ACT_SEQ_(0)').X
    
    # 手动计算
    expected_nc = rb_tiles * unique_rows_factor * non_crossing_coeff / effective_banks
    expected_cr = crossing_base * unique_rows_factor * crossing_coeff / effective_banks
    expected_total = expected_nc + expected_cr
    
    return {
        'name': name,
        'cr_b': cr_b,
        'cr_r': cr_r,
        'rb_tiles': rb_tiles,
        'reuse': reuse,
        'expected_nc': expected_nc,
        'ilp_nc': ilp_nc,
        'nc_match': abs(expected_nc - ilp_nc) < 1e-6,
        'expected_cr': expected_cr,
        'ilp_cr': ilp_cr,
        'cr_match': abs(expected_cr - ilp_cr) < 1e-6,
        'expected_total': expected_total,
        'ilp_total': ilp_total,
        'total_match': abs(expected_total - ilp_total) < 1e-6,
    }


def main():
    print("=" * 80)
    print("Sequential 模式 Input Row Activation 公式验证报告")
    print("=" * 80)
    print("""
公式: row_acts = rb_tiles / tiles_per_row / banks × 
                 [(1-cr_b)(1-cr_r) + 2×reuse×(cr_b+cr_r)]

拆分为:
  nc (Non-Crossing) = rb_tiles × unique_rows_factor × (1-cr_b)(1-cr_r) / banks
  cr (Crossing) = crossing_base × unique_rows_factor × 2(cr_b+cr_r) / banks
  
  其中 crossing_base = rb_tiles × reuse
""")
    
    workloads = [
        {"name": "tiny", "N": 1, "K": 8, "C": 8, "P": 4, "Q": 4, "R": 3, "S": 3},
        {"name": "small", "N": 1, "K": 32, "C": 32, "P": 14, "Q": 14, "R": 3, "S": 3},
        {"name": "medium_1x1", "N": 1, "K": 64, "C": 64, "P": 14, "Q": 14, "R": 1, "S": 1},
        {"name": "medium_3x3", "N": 1, "K": 64, "C": 64, "P": 14, "Q": 14, "R": 3, "S": 3},
        {"name": "large", "N": 1, "K": 128, "C": 128, "P": 7, "Q": 7, "R": 3, "S": 3},
    ]
    
    results = []
    for wl in workloads:
        name = wl.pop('name')
        wl['name'] = name
        r = verify_workload(name, wl)
        results.append(r)
    
    # 打印结果表格
    print("\n【验证结果汇总】\n")
    print(f"{'Workload':<12} {'cr_b':>8} {'cr_r':>8} {'rb_tiles':>12} {'reuse':>8} {'ILP Total':>12} {'Match':>8}")
    print("-" * 76)
    for r in results:
        print(f"{r['name']:<12} {r['cr_b']:>8.4f} {r['cr_r']:>8.4f} {r['rb_tiles']:>12.4f} {r['reuse']:>8.1f} {r['ilp_total']:>12.6f} {'✓' if r['total_match'] else '✗':>8}")
    
    print("\n【详细验证】\n")
    for r in results:
        print(f"Workload: {r['name']}")
        print(f"  Non-Crossing: expected={r['expected_nc']:.10f}, ILP={r['ilp_nc']:.10f}, match={r['nc_match']}")
        print(f"  Crossing:     expected={r['expected_cr']:.10f}, ILP={r['ilp_cr']:.10f}, match={r['cr_match']}")
        print(f"  Total:        expected={r['expected_total']:.10f}, ILP={r['ilp_total']:.10f}, match={r['total_match']}")
        print()
    
    all_match = all(r['total_match'] for r in results)
    print("=" * 80)
    print(f"验证结果: {'✓ 所有 workloads 公式验证通过' if all_match else '✗ 存在验证失败的 workload'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
