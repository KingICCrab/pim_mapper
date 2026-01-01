#!/usr/bin/env python3
"""
从 ILP 结果反推 crossing ratio 和 tiles_per_row
验证公式是否正确实现
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')

from pim_optimizer.arch.pim_arch import PIMArchitecture
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer import PIMOptimizer


def analyze_workload(name, wl_params):
    """从 ILP 结果反推参数"""
    
    print("=" * 80)
    print(f"Workload: {name}")
    print("=" * 80)
    
    workload = ConvWorkload(**wl_params)
    arch = PIMArchitecture()
    
    optimizer = PIMOptimizer(arch, verbose=False)
    result = optimizer.optimize([workload])
    model = optimizer.model
    
    # 获取变量
    rb_tiles = model.getVarByName('RB_TILES_INPUT_(0)').X
    reuse = model.getVarByName('REUSE_INPUT_(0)').X
    nc = model.getVarByName('ROW_ACT_INPUT_NC_(0)').X
    cr = model.getVarByName('ROW_ACT_INPUT_CR_(0)').X
    crossing_base = model.getVarByName('CROSSING_INPUT_BASE_(0)').X
    seq_total = model.getVarByName('INPUT_ROW_ACT_SEQ_(0)').X
    
    banks = 16  # 架构参数
    
    print(f"\n【ILP 变量值】")
    print(f"  rb_tiles = {rb_tiles:.6f}")
    print(f"  reuse = {reuse:.6f}")
    print(f"  crossing_base = rb_tiles × reuse = {crossing_base:.6f}")
    print(f"  non_crossing (nc) = {nc:.6f}")
    print(f"  crossing (cr) = {cr:.6f}")
    print(f"  seq_total = nc + cr = {seq_total:.6f}")
    print(f"  banks = {banks}")
    
    # 验证 crossing_base = rb_tiles × reuse
    print(f"\n【验证 crossing_base】")
    expected_crossing_base = rb_tiles * reuse
    print(f"  rb_tiles × reuse = {expected_crossing_base:.6f}")
    print(f"  crossing_base = {crossing_base:.6f}")
    print(f"  差异 = {abs(expected_crossing_base - crossing_base):.10f}")
    
    # 公式:
    # nc = rb_tiles × (1/tiles_per_row) × non_crossing_coeff / banks
    # cr = crossing_base × (1/tiles_per_row) × crossing_coeff / banks
    #
    # 所以:
    # nc × banks = rb_tiles / tiles_per_row × non_crossing_coeff
    # cr × banks = crossing_base / tiles_per_row × crossing_coeff
    #
    # 设 unique_rows_factor = 1 / tiles_per_row
    # nc × banks = rb_tiles × unique_rows_factor × non_crossing_coeff
    # cr × banks = crossing_base × unique_rows_factor × crossing_coeff
    #
    # 从这两个方程可以解出:
    # cr / nc = (crossing_base / rb_tiles) × (crossing_coeff / non_crossing_coeff)
    #         = reuse × (crossing_coeff / non_crossing_coeff)
    
    print(f"\n【从 ILP 反推参数】")
    
    # cr / nc = reuse × (crossing_coeff / non_crossing_coeff)
    if nc > 1e-10:
        ratio = cr / nc
        expected_coeff_ratio = ratio / reuse
        print(f"  cr / nc = {ratio:.6f}")
        print(f"  cr / nc / reuse = crossing_coeff / non_crossing_coeff = {expected_coeff_ratio:.6f}")
        
        # 公式: 
        # non_crossing_coeff = (1 - cr_b) × (1 - cr_r)
        # crossing_coeff = 2 × (cr_b + cr_r)
        #
        # 设 cr_b + cr_r = x，则:
        # crossing_coeff / non_crossing_coeff = 2x / [(1-cr_b)(1-cr_r)]
        #
        # 如果 cr_b 和 cr_r 都很小，(1-cr_b)(1-cr_r) ≈ 1
        # 则 crossing_coeff / non_crossing_coeff ≈ 2 × (cr_b + cr_r)
        
        # 近似计算 cr_b + cr_r
        approx_cr_sum = expected_coeff_ratio / 2
        print(f"  近似 cr_b + cr_r ≈ {approx_cr_sum:.6f}")
    
    # 从 nc 反推 tiles_per_row × (1 - cr_b)(1 - cr_r)
    # nc × banks = rb_tiles / tiles_per_row × (1-cr_b)(1-cr_r)
    # tiles_per_row / (1-cr_b)(1-cr_r) = rb_tiles / (nc × banks)
    
    if nc > 1e-10:
        tpr_div_nc_coeff = rb_tiles / (nc * banks)
        print(f"  tiles_per_row / non_crossing_coeff = {tpr_div_nc_coeff:.6f}")
    
    # 从 cr 反推 tiles_per_row / crossing_coeff
    # cr × banks = crossing_base / tiles_per_row × crossing_coeff
    # tiles_per_row / crossing_coeff = crossing_base / (cr × banks)
    
    if cr > 1e-10:
        tpr_div_cr_coeff = crossing_base / (cr * banks)
        print(f"  tiles_per_row / crossing_coeff = {tpr_div_cr_coeff:.6f}")
    
    # 比较两个值
    if nc > 1e-10 and cr > 1e-10:
        # tiles_per_row / non_crossing_coeff = A
        # tiles_per_row / crossing_coeff = B
        # 则: crossing_coeff / non_crossing_coeff = A / B
        A = rb_tiles / (nc * banks)
        B = crossing_base / (cr * banks)
        print(f"\n  验证: A/B = crossing_coeff/non_crossing_coeff")
        print(f"    A = {A:.6f}")
        print(f"    B = {B:.6f}")
        print(f"    A/B = {A/B:.6f}")
        
        # 如果知道 tiles_per_row，可以计算出 non_crossing_coeff 和 crossing_coeff
        # 假设 tiles_per_row 从代码中的平均值来
        
    print()
    return {
        'name': name,
        'rb_tiles': rb_tiles,
        'reuse': reuse,
        'nc': nc,
        'cr': cr,
        'total': seq_total
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
        r = analyze_workload(name, wl)
        results.append(r)
    
    # 汇总
    print("\n" + "=" * 80)
    print("汇总表格")
    print("=" * 80)
    print(f"{'Workload':<12} {'rb_tiles':>12} {'reuse':>10} {'nc':>12} {'cr':>12} {'total':>12}")
    print("-" * 72)
    for r in results:
        print(f"{r['name']:<12} {r['rb_tiles']:>12.4f} {r['reuse']:>10.1f} {r['nc']:>12.6f} {r['cr']:>12.6f} {r['total']:>12.6f}")


if __name__ == "__main__":
    main()
