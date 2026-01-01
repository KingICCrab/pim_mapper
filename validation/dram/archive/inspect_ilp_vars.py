#!/usr/bin/env python3
"""
深入检查 ILP 模型中 Sequential Row Activation 的实际计算参数
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')

from pim_optimizer.arch.pim_arch import PIMArchitecture
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer import PIMOptimizer


def inspect_workload(name, wl_params):
    """检查 ILP 模型中的实际变量值"""
    
    print("=" * 80)
    print(f"Workload: {name}")
    print("=" * 80)
    
    workload = ConvWorkload(**wl_params)
    arch = PIMArchitecture()
    
    optimizer = PIMOptimizer(arch, verbose=False)
    result = optimizer.optimize([workload])
    model = optimizer.model
    
    print(f"\n【Workload 参数】")
    H_in = wl_params['Q'] + wl_params['S'] - 1
    W_in = wl_params['P'] + wl_params['R'] - 1
    print(f"  N={wl_params['N']}, K={wl_params['K']}, C={wl_params['C']}")
    print(f"  P={wl_params['P']}, Q={wl_params['Q']}, R={wl_params['R']}, S={wl_params['S']}")
    print(f"  Input size: H={H_in}, W={W_in}")
    
    # 列出所有包含 INPUT 和 CROSSING 的变量
    print(f"\n【Sequential 相关变量】")
    
    var_names_to_check = [
        'INPUT_ROW_ACT_SEQ_(0)',
        'ROW_ACT_INPUT_NC_(0)',
        'ROW_ACT_INPUT_CR_(0)',
        'RB_TILES_INPUT_(0)',
        'REUSE_INPUT_(0)',
        'LOG_RB_TILES_INPUT_(0)',
        'LOG_REUSE_INPUT_(0)',
        'LOG_ALL_IRRELEVANT_INPUT_(0)',
        'CROSSING_INPUT_BASE_(0)',
        'LOG_CROSSING_INPUT_(0)',
    ]
    
    for var_name in var_names_to_check:
        var = model.getVarByName(var_name)
        if var:
            print(f"  {var_name} = {var.X:.6f}")
    
    # 检查 log_mem_reads_input
    print(f"\n【Memory Reads 相关】")
    for suffix in ['', '_bounded']:
        var_name = f'LOG_MEM_READS_INPUT{suffix}_(0)'
        var = model.getVarByName(var_name)
        if var:
            print(f"  {var_name} = {var.X:.6f}")
    
    # 检查 exp 变量
    var = model.getVarByName('V_EXP_mem_reads_inst_(0,3,0)')
    if var:
        print(f"  mem_reads_inst (exp) = {var.X:.6f}")
    
    # 检查 row activation 相关的常量（从代码中计算）
    print(f"\n【计算的常量（从 ILP 代码中）】")
    
    # 从 ILP 结果反推
    rb_tiles_var = model.getVarByName('RB_TILES_INPUT_(0)')
    reuse_var = model.getVarByName('REUSE_INPUT_(0)')
    nc_var = model.getVarByName('ROW_ACT_INPUT_NC_(0)')
    cr_var = model.getVarByName('ROW_ACT_INPUT_CR_(0)')
    crossing_base_var = model.getVarByName('CROSSING_INPUT_BASE_(0)')
    
    if rb_tiles_var and nc_var:
        rb_tiles = rb_tiles_var.X
        nc = nc_var.X
        # nc = rb_tiles × unique_rows_factor × non_crossing_coeff / banks
        # unique_rows_factor = 1 / tiles_per_row
        # 所以: nc × banks = rb_tiles / tiles_per_row × non_crossing_coeff
        #       tiles_per_row × non_crossing_coeff = rb_tiles × banks / nc (if nc > 0)
        banks = 16
        if nc > 1e-6:
            product = rb_tiles / (nc * banks)
            print(f"  (tiles_per_row / non_crossing_coeff) = rb_tiles / (nc × banks)")
            print(f"    = {rb_tiles:.4f} / ({nc:.6f} × {banks}) = {product:.4f}")
    
    if crossing_base_var and cr_var:
        crossing_base = crossing_base_var.X
        cr = cr_var.X
        banks = 16
        # cr = crossing_base × unique_rows_factor × crossing_coeff / banks
        # = crossing_base / tiles_per_row × crossing_coeff / banks
        if cr > 1e-6:
            ratio = crossing_base / (cr * banks)
            print(f"  (tiles_per_row / crossing_coeff) = crossing_base / (cr × banks)")
            print(f"    = {crossing_base:.4f} / ({cr:.6f} × {banks}) = {ratio:.4f}")
    
    # 如果能找到 rb_tiles × reuse
    if rb_tiles_var and reuse_var and crossing_base_var:
        rb_tiles = rb_tiles_var.X
        reuse = reuse_var.X
        crossing_base = crossing_base_var.X
        print(f"  rb_tiles × reuse = {rb_tiles:.4f} × {reuse:.4f} = {rb_tiles * reuse:.4f}")
        print(f"  crossing_base = {crossing_base:.4f}")
        print(f"  比值 = {(rb_tiles * reuse) / crossing_base:.4f}" if crossing_base > 1e-6 else "  比值 = N/A")
    
    print()


def main():
    test_workloads = [
        {"name": "tiny", "N": 1, "K": 8, "C": 8, "P": 4, "Q": 4, "R": 3, "S": 3},
        {"name": "small", "N": 1, "K": 32, "C": 32, "P": 14, "Q": 14, "R": 3, "S": 3},
        {"name": "medium_1x1", "N": 1, "K": 64, "C": 64, "P": 14, "Q": 14, "R": 1, "S": 1},
        {"name": "medium_3x3", "N": 1, "K": 64, "C": 64, "P": 14, "Q": 14, "R": 3, "S": 3},
        {"name": "large", "N": 1, "K": 128, "C": 128, "P": 7, "Q": 7, "R": 3, "S": 3},
    ]
    
    for wl in test_workloads:
        name = wl.pop('name')
        wl['name'] = name
        inspect_workload(name, wl)


if __name__ == "__main__":
    main()
