#!/usr/bin/env python3
"""
提取每个 workload 的详细 mapping 和 layout 信息来验证 row activation 计算
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')

from pim_optimizer.arch.pim_arch import PIMArchitecture
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer import PIMOptimizer


def extract_mapping_details(name, wl_params):
    """提取详细的 mapping 信息"""
    
    print("=" * 80)
    print(f"Workload: {name}")
    print("=" * 80)
    
    workload = ConvWorkload(**wl_params)
    arch = PIMArchitecture()
    
    optimizer = PIMOptimizer(arch, verbose=False)
    result = optimizer.optimize([workload])
    model = optimizer.model
    
    # Workload 参数
    print(f"\n【Workload 参数】")
    print(f"  N={wl_params['N']}, K={wl_params['K']}, C={wl_params['C']}")
    print(f"  P={wl_params['P']}, Q={wl_params['Q']}, R={wl_params['R']}, S={wl_params['S']}")
    print(f"  MACs = {workload.macs}")
    print(f"  Input size: H={wl_params['Q']+wl_params['S']-1}, W={wl_params['P']+wl_params['R']-1}")
    
    # Layout choice
    print(f"\n【Layout Choice】")
    layout_seq = model.getVarByName('X_LAYOUT_(0,0,sequential)')
    layout_ra = model.getVarByName('X_LAYOUT_(0,0,row_aligned)')
    
    if layout_seq and layout_ra:
        print(f"  Input Sequential: {layout_seq.X:.0f}")
        print(f"  Input Row-aligned: {layout_ra.X:.0f}")
        layout = "Sequential" if layout_seq.X > 0.5 else "Row-aligned"
        print(f"  Selected: {layout}")
    
    # Row Activation 结果
    print(f"\n【Row Activation 结果】")
    seq_val = model.getVarByName('INPUT_ROW_ACT_SEQ_(0)')
    ra_val = model.getVarByName('INPUT_ROW_ACT_RA_(0)')
    total_val = model.getVarByName('INPUT_TOTAL_ROW_ACT_(0)')
    
    print(f"  Sequential mode: {seq_val.X:.4f}" if seq_val else "  Sequential mode: N/A")
    print(f"  Row-aligned mode: {ra_val.X:.4f}" if ra_val else "  Row-aligned mode: N/A")
    print(f"  Final selected: {total_val.X:.4f}" if total_val else "  Final selected: N/A")
    
    # 关键中间变量
    print(f"\n【关键中间变量】")
    
    # rb_tiles
    rb_tiles = model.getVarByName('RB_TILES_INPUT_(0)')
    print(f"  rb_tiles (unique tiles): {rb_tiles.X:.4f}" if rb_tiles else "  rb_tiles: N/A")
    
    # reuse
    reuse = model.getVarByName('REUSE_INPUT_(0)')
    print(f"  reuse factor: {reuse.X:.4f}" if reuse else "  reuse: N/A")
    
    # Non-crossing and crossing
    nc = model.getVarByName('ROW_ACT_INPUT_NC_(0)')
    cr = model.getVarByName('ROW_ACT_INPUT_CR_(0)')
    print(f"  non-crossing part: {nc.X:.4f}" if nc else "  non-crossing: N/A")
    print(f"  crossing part: {cr.X:.4f}" if cr else "  crossing: N/A")
    
    # Tile 相关变量
    print(f"\n【Tile 信息 (DRAM level)】")
    
    # 尝试获取各维度的 bound
    dims = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
    for j, dim_name in enumerate(dims):
        # 找到选择的 divisor
        for i in range(20):  # 假设最多 20 个 divisors
            xb_var = model.getVarByName(f'xb_(0,3,1,{j},{i})')  # level 3 (DRAM), temporal
            if xb_var and xb_var.X > 0.5:
                # 需要从 workload 获取实际的 divisor 值
                if j < len(workload.divisors) and i < len(workload.divisors[j]):
                    div_val = workload.divisors[j][i]
                    print(f"  {dim_name}_dram = {div_val}")
                break
    
    # Input tile H/W
    print(f"\n【Input Tile 选择】")
    
    # 尝试找 xh, xw 变量
    for i in range(20):
        xh = model.getVarByName(f'X_INPUT_YH_(0,{i})')
        if xh and xh.X > 0.5:
            print(f"  xh[{i}] = 1 (selected)")
            
    for i in range(20):
        xw = model.getVarByName(f'X_INPUT_YW_(0,{i})')
        if xw and xw.X > 0.5:
            print(f"  xw[{i}] = 1 (selected)")
    
    # Block size
    print(f"\n【Block Size】")
    for i in range(20):
        bh = model.getVarByName(f'X_ROWBUF_INPUT_BLOCK_H_(0,{i})')
        if bh and bh.X > 0.5:
            print(f"  block_h[{i}] = 1 (selected)")
            
    for i in range(20):
        bw = model.getVarByName(f'X_ROWBUF_INPUT_BLOCK_W_(0,{i})')
        if bw and bw.X > 0.5:
            print(f"  block_w[{i}] = 1 (selected)")
    
    # Memory reads
    print(f"\n【Memory Reads (DRAM level)】")
    for t, name in [(0, 'Input'), (1, 'Weight'), (2, 'Output')]:
        mr = model.getVarByName(f'V_EXP_mem_reads_(0,3,{t})')
        mri = model.getVarByName(f'V_EXP_mem_reads_inst_(0,3,{t})')
        if mr:
            print(f"  {name} mem_reads: {mr.X:.2f}")
        if mri:
            print(f"  {name} mem_reads_inst: {mri.X:.2f}")
    
    # DRAM cycles
    print(f"\n【DRAM Cycles】")
    dram_cyc = model.getVarByName('V_mem_cycles_(0,3)')
    print(f"  DRAM cycles: {dram_cyc.X:.2f}" if dram_cyc else "  DRAM cycles: N/A")
    
    print()
    return {
        'name': name,
        'seq': seq_val.X if seq_val else 0,
        'ra': ra_val.X if ra_val else 0,
        'selected': total_val.X if total_val else 0,
        'layout': layout if layout_seq else 'Unknown'
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
        r = extract_mapping_details(name, wl)
        results.append(r)
    
    # 汇总表格
    print("\n" + "=" * 80)
    print("汇总表格")
    print("=" * 80)
    print(f"{'Workload':<15} {'Sequential':>12} {'Row-aligned':>12} {'Selected':>12} {'Layout':<12}")
    print("-" * 65)
    for r in results:
        print(f"{r['name']:<15} {r['seq']:>12.2f} {r['ra']:>12.2f} {r['selected']:>12.2f} {r['layout']:<12}")


if __name__ == "__main__":
    main()
