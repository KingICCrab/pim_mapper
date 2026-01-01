#!/usr/bin/env python3
"""
检验 Sequential 模式的 crossing ratio 是使用平均值还是特定组合值

ILP 结果反推显示 crossing_coeff/non_crossing_coeff 比值和平均值计算有很大差异
需要确认是否 Sequential 模式用的是固定的平均值
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')

from pim_optimizer.arch.pim_arch import PIMArchitecture
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer import PIMOptimizer


def analyze_sequential_mode(name, wl_params):
    """分析 Sequential 模式用的参数"""
    
    print("=" * 80)
    print(f"Workload: {name}")
    print("=" * 80)
    
    workload = ConvWorkload(**wl_params)
    arch = PIMArchitecture()
    
    optimizer = PIMOptimizer(arch, verbose=False)
    result = optimizer.optimize([workload])
    model = optimizer.model
    
    banks = 16
    
    # 获取关键变量
    rb_tiles = model.getVarByName('RB_TILES_INPUT_(0)').X
    reuse = model.getVarByName('REUSE_INPUT_(0)').X
    nc = model.getVarByName('ROW_ACT_INPUT_NC_(0)').X
    cr = model.getVarByName('ROW_ACT_INPUT_CR_(0)').X
    crossing_base = model.getVarByName('CROSSING_INPUT_BASE_(0)').X
    
    print(f"\n【ILP Sequential 变量】")
    print(f"  rb_tiles = {rb_tiles:.6f}")
    print(f"  reuse = {reuse:.6f}")
    print(f"  crossing_base = rb_tiles × reuse = {rb_tiles * reuse:.6f} (ILP: {crossing_base:.6f})")
    print(f"  nc = {nc:.10f}")
    print(f"  cr = {cr:.10f}")
    
    # 从代码看 Sequential 模式公式:
    # nc = rb_tiles × unique_rows_factor × non_crossing_coeff / banks
    # cr = crossing_base × unique_rows_factor × crossing_coeff / banks
    #
    # 其中:
    # unique_rows_factor = 1 / tiles_per_row
    # non_crossing_coeff = (1 - cr_b) × (1 - cr_r)
    # crossing_coeff = 2 × (cr_b + cr_r)
    #
    # 所以:
    # nc × banks = rb_tiles / tiles_per_row × non_crossing_coeff
    # cr × banks = crossing_base / tiles_per_row × crossing_coeff
    
    # 反推 tiles_per_row 和 coefficients
    # 从两个方程:
    # nc × banks × tiles_per_row = rb_tiles × non_crossing_coeff
    # cr × banks × tiles_per_row = crossing_base × crossing_coeff
    #
    # 除以:
    # nc / cr = rb_tiles × non_crossing_coeff / (crossing_base × crossing_coeff)
    #         = rb_tiles × non_crossing_coeff / (rb_tiles × reuse × crossing_coeff)
    #         = non_crossing_coeff / (reuse × crossing_coeff)
    #
    # 所以:
    # nc / cr × reuse = non_crossing_coeff / crossing_coeff
    
    if cr > 1e-15:
        nc_over_cr = nc / cr
        coeff_ratio_from_ilp = nc_over_cr * reuse
        print(f"\n【从 ILP 反推】")
        print(f"  nc / cr = {nc_over_cr:.10f}")
        print(f"  nc / cr × reuse = non_crossing_coeff / crossing_coeff = {coeff_ratio_from_ilp:.10f}")
        print(f"  crossing_coeff / non_crossing_coeff = {1 / coeff_ratio_from_ilp:.6f}")
        
        # 反推 tiles_per_row
        # nc × banks = rb_tiles / tiles_per_row × non_crossing_coeff
        # tiles_per_row = rb_tiles × non_crossing_coeff / (nc × banks)
        
        # 设 non_crossing_coeff = x, crossing_coeff = 2(cr_b + cr_r)
        # x / (2(cr_b + cr_r)) = coeff_ratio_from_ilp
        # 
        # 设 cr_b + cr_r = s，则:
        # (1 - cr_b)(1 - cr_r) / (2s) = coeff_ratio_from_ilp
        # 
        # 如果 cr_b 和 cr_r 很小, (1-cr_b)(1-cr_r) ≈ 1
        # 则: 1 / (2s) ≈ coeff_ratio_from_ilp
        # s ≈ 1 / (2 × coeff_ratio_from_ilp)
        
        approx_cr_sum = 1 / (2 * coeff_ratio_from_ilp) if coeff_ratio_from_ilp > 0 else 0
        print(f"  近似 cr_b + cr_r ≈ {approx_cr_sum:.6f}")
        
        # 如果 cr_b + cr_r 很大 (接近或超过 1)
        # 则需要考虑 (1-cr_b)(1-cr_r) 的实际值
        # 对于我们看到的 ratio = 13.1，这意味着:
        # crossing_coeff / non_crossing_coeff = 13.1
        # 2(cr_b + cr_r) / [(1-cr_b)(1-cr_r)] = 13.1
        
        # 假设 cr_b = cr_r = c (对称情况):
        # 2 × 2c / (1-c)^2 = 13.1
        # 4c / (1-c)^2 = 13.1
        # 4c = 13.1 × (1 - 2c + c^2)
        # 4c = 13.1 - 26.2c + 13.1c^2
        # 13.1c^2 - 30.2c + 13.1 = 0
        
        import math
        A, B, C = 13.1, -30.2, 13.1
        discriminant = B**2 - 4*A*C
        if discriminant >= 0:
            c1 = (-B + math.sqrt(discriminant)) / (2*A)
            c2 = (-B - math.sqrt(discriminant)) / (2*A)
            print(f"\n  假设 cr_b = cr_r = c (对称):")
            print(f"    解方程 4c / (1-c)^2 = ratio")
            print(f"    c1 = {c1:.6f}, c2 = {c2:.6f}")
            # 选择在 [0, 1] 范围内的解
            for c in [c1, c2]:
                if 0 <= c <= 1:
                    print(f"    选择 c = {c:.6f}")
                    print(f"    验证: 4c / (1-c)^2 = {4*c / (1-c)**2:.6f}")
    
    # 获取 Row-Aligned 的 crossing ratio 变量
    print(f"\n【Row-Aligned 的 selected crossing ratio】")
    try:
        cr_h = model.getVarByName('INPUT_CROSSING_RATIO_H_(0)')
        cr_w = model.getVarByName('INPUT_CROSSING_RATIO_W_(0)')
        if cr_h and cr_w:
            print(f"  INPUT_CROSSING_RATIO_H = {cr_h.X:.6f}")
            print(f"  INPUT_CROSSING_RATIO_W = {cr_w.X:.6f}")
    except Exception as e:
        print(f"  无法获取: {e}")
    
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
        analyze_sequential_mode(name, wl)


if __name__ == "__main__":
    main()
