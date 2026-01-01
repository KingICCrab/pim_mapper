#!/usr/bin/env python3
"""
从 ILP 结果中提取并打印完整的 Mapping 信息
输出到 mapping_results.txt 文件
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')

from pim_optimizer.arch.pim_arch import PIMArchitecture
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer import PIMOptimizer

# 输出文件路径
OUTPUT_FILE = '/Users/haochenzhao/Projects/pim_optimizer/validation/dram/mapping_results.txt'


def extract_mapping(name, wl_params, out_file):
    """提取并打印 ILP 选择的 mapping"""
    
    out_file.write("=" * 80 + "\n")
    out_file.write(f"Workload: {name}\n")
    out_file.write("=" * 80 + "\n")
    
    workload = ConvWorkload(**wl_params)
    # 使用 from_yaml 加载完整配置（包括 PE array、memory hierarchy）
    arch = PIMArchitecture.from_yaml('/Users/haochenzhao/Projects/pim_optimizer/examples/configs/arch.yaml')
    
    optimizer = PIMOptimizer(arch, verbose=False)
    result = optimizer.optimize([workload])
    model = optimizer.model
    vars = optimizer.vars
    
    # 检查求解状态
    import gurobipy as gp
    if model.Status != gp.GRB.OPTIMAL and model.Status != gp.GRB.SUBOPTIMAL:
        out_file.write(f"\n【求解失败】Status = {model.Status}\n")
        out_file.write(f"  Status codes: OPTIMAL=2, INFEASIBLE=3, UNBOUNDED=5, INF_OR_UNBD=4\n")
        return
    
    out_file.write(f"\n【Workload 参数】\n")
    out_file.write(f"  Dimensions: R={workload.bounds[0]}, S={workload.bounds[1]}, P={workload.bounds[2]}, Q={workload.bounds[3]}, C={workload.bounds[4]}, K={workload.bounds[5]}, N={workload.bounds[6]}\n")
    out_file.write(f"  Stride: {workload.stride}\n")
    out_file.write(f"  Dilation: {workload.dilation}\n")
    out_file.write(f"  Total MACs: {workload.macs}\n")
    
    # 维度名称
    dim_names = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
    
    # 内存层级名称
    mem_names = {0: 'PE', 1: 'GlobalBuffer', 2: 'RowBuffer', 3: 'DRAM'}
    
    out_file.write(f"\n【Tile Sizes (xb 变量)】\n")
    out_file.write("  xb[w, m, s, j, i] = 1 表示维度 j 在 memory level m 的 tile factor\n")
    
    # PE level (m=0): s=0,1,2 for spatial H,W,internal, s=3 for temporal
    # Other levels (m>0): s=0 for spatial (dummy), s=1 for temporal
    
    for m in range(4):
        out_file.write(f"\n  {mem_names[m]} (level {m}):\n")
        if m == 0:  # PE
            spatial_s = [0, 1, 2]  # H, W, internal
            temporal_s = 3
            spatial_names = {0: 'H-spatial', 1: 'W-spatial', 2: 'internal'}
        else:
            spatial_s = [0]  # dummy
            temporal_s = 1
            spatial_names = {0: 'spatial(dummy)'}
        
        # Temporal
        tiles = []
        for j in range(7):
            divs = workload.divisors[j]
            for i, div in enumerate(divs):
                xb_var = vars.xb.get((0, m, temporal_s, j, i))
                if xb_var is not None and xb_var.X > 0.5:
                    tiles.append(f"{dim_names[j]}={div}")
        if tiles:
            out_file.write(f"    Temporal: {', '.join(tiles)}\n")
        
        # Spatial (for PE)
        if m == 0:
            for s in spatial_s:
                tiles = []
                for j in range(7):
                    divs = workload.divisors[j]
                    for i, div in enumerate(divs):
                        xb_var = vars.xb.get((0, m, s, j, i))
                        if xb_var is not None and xb_var.X > 0.5:
                            tiles.append(f"{dim_names[j]}={div}")
                if tiles:
                    out_file.write(f"    {spatial_names[s]}: {', '.join(tiles)}\n")
    
    # Loop permutation (xp)
    out_file.write(f"\n【Loop Permutation (xp 变量)】\n")
    out_file.write("  xp[w, m, p, j] = 1 表示 permutation level p 选择了维度 j\n")
    
    for m in range(4):
        perm = []
        for p in range(7):  # 最多 7 个 permutation levels
            for j in range(7):
                xp_var = vars.xp.get((0, m, p, j))
                if xp_var is not None and xp_var.X > 0.5:
                    perm.append(dim_names[j])
        if perm:
            out_file.write(f"  {mem_names[m]}: {' -> '.join(perm)} (outermost -> innermost)\n")
    
    out_file.write(f"\n【Data Layout (Input)】\n")
    # Block sizes for Input
    h_divisors = getattr(workload, 'hw_divisors', {}).get('H', [1])
    w_divisors = getattr(workload, 'hw_divisors', {}).get('W', [1])
    
    out_file.write(f"  H divisors: {h_divisors}\n")
    out_file.write(f"  W divisors: {w_divisors}\n")
    
    # 查找选择的 block_h, block_w
    selected_bh, selected_bw = None, None
    for i, bh in enumerate(h_divisors):
        var = vars.rowbuf_input_block_h.get((0, i))
        if var is not None and var.X > 0.5:
            selected_bh = bh
            out_file.write(f"  Selected block_h: {bh}\n")
    
    for i, bw in enumerate(w_divisors):
        var = vars.rowbuf_input_block_w.get((0, i))
        if var is not None and var.X > 0.5:
            selected_bw = bw
            out_file.write(f"  Selected block_w: {bw}\n")
    
    # Layout choice
    selected_layout = None
    for layout in ["sequential", "row_aligned"]:
        var = vars.layout_choice.get((0, 0, layout))  # Input = datatype 0
        if var is not None and var.X > 0.5:
            selected_layout = layout
            out_file.write(f"  Selected layout: {layout}\n")
    
    out_file.write(f"\n【Row Activation 关键变量】\n")
    
    # 获取 macs_scale_factor 用于反归一化
    # macs_scale_factor = MAX_BOUND / (1.02 * macs)
    # 所以 实际值 = 归一化值 / macs_scale_factor = 归一化值 * (1.02 * macs) / MAX_BOUND
    MAX_BOUND = 1e4
    macs_scale_factor = MAX_BOUND / (1.02 * workload.macs)
    activation_latency = 25.0  # default
    out_file.write(f"  (macs = {workload.macs}, macs_scale_factor = {macs_scale_factor:.6f})\n")
    out_file.write(f"  (activation_latency = {activation_latency} cycles)\n")
    out_file.write("\n")
    
    # 当前的变量名称
    # Row Activation 变量 (每个 datatype: 0=input, 1=weight, 2=output)
    tensor_names = {0: 'input', 1: 'weight', 2: 'output'}
    
    for t_id, t_name in tensor_names.items():
        out_file.write(f"  【{t_name.upper()} (t_id={t_id})】\n")
        
        # log_reuse_penalty
        var = model.getVarByName(f'log_reuse_penalty_({0},{t_id})')
        if var is not None:
            out_file.write(f"    log_reuse_penalty = {var.X:.6f}\n")
        
        # reuse_penalty
        var = model.getVarByName(f'reuse_penalty_({0},{t_id})')
        if var is not None:
            out_file.write(f"    reuse_penalty = {var.X:.6f}\n")
        
        # row_acts_row_aligned
        var = model.getVarByName(f'row_acts_row_aligned_({0},{t_id})')
        if var is not None:
            out_file.write(f"    row_acts_row_aligned = {var.X:.6f}\n")
        
        # row_acts (sequential)
        var = model.getVarByName(f'row_acts_({0},{t_id})')
        if var is not None:
            out_file.write(f"    row_acts_seq = {var.X:.6f}\n")
        
        # Input Block Crossing (only for input)
        if t_id == 0:
            var = model.getVarByName(f'input_block_crossing_acts_({0})')
            if var is not None:
                out_file.write(f"    input_block_crossing_acts = {var.X:.6f}\n")
            
            var = model.getVarByName(f'selected_ibc_count_({0})')
            if var is not None:
                out_file.write(f"    selected_ibc_count = {var.X:.6f}\n")
        
        # total_row_acts
        var = model.getVarByName(f'total_row_acts_({0},{t_id})')
        if var is not None:
            out_file.write(f"    total_row_acts = {var.X:.6f}\n")
        
        # Layout conditional parts
        var = model.getVarByName(f'dram_crossing_seq_({0},{t_id})')
        if var is not None:
            out_file.write(f"    dram_crossing_seq (layout cond) = {var.X:.6f}\n")
        
        var = model.getVarByName(f'row_aligned_acts_({0},{t_id})')
        if var is not None:
            out_file.write(f"    row_aligned_acts (layout cond) = {var.X:.6f}\n")
        
        # Scaled row activations
        var = model.getVarByName(f'row_acts_scaled_({0},{t_id})')
        if var is not None:
            scaled_val = var.X
            actual_acts = scaled_val / macs_scale_factor
            out_file.write(f"    row_acts_scaled = {scaled_val:.6f} (actual: {actual_acts:.2f} acts)\n")
        
        # Row activation cycles
        var = model.getVarByName(f'row_acts_cycles_({0},{t_id})')
        if var is not None:
            scaled_cycles = var.X
            actual_cycles = scaled_cycles / macs_scale_factor
            out_file.write(f"    row_acts_cycles = {scaled_cycles:.6f} (actual: {actual_cycles:.2f} cycles)\n")
        
        out_file.write("\n")
    
    # DRAM latency per datatype (from objective.py)
    out_file.write(f"  【DRAM Latency Per Datatype】\n")
    for t_id, t_name in tensor_names.items():
        var = model.getVarByName(f'V_dram_latency_{t_name}_({0})')
        if var is not None:
            scaled_val = var.X
            actual_val = scaled_val / macs_scale_factor
            out_file.write(f"    V_dram_latency_{t_name} = {scaled_val:.6f} (actual: {actual_val:.2f} cycles)\n")
    
    out_file.write(f"\n  【DRAM Total Latency (max of three)】\n")
    var = model.getVarByName(f'V_dram_latency_({0})')
    if var is not None:
        scaled_val = var.X
        actual_val = scaled_val / macs_scale_factor
        out_file.write(f"    V_dram_latency = {scaled_val:.6f} (actual: {actual_val:.2f} cycles)\n")
    
    # Memory reads at RowBuffer level (for data transfer)
    out_file.write(f"\n  【Memory Reads at RowBuffer】\n")
    for t_id, t_name in tensor_names.items():
        var = model.getVarByName(f'MEM_READS_0_2_{t_id}')
        if var is not None:
            scaled_val = var.X
            actual_val = scaled_val / macs_scale_factor
            out_file.write(f"    MEM_READS_0_2_{t_id} ({t_name}) = {scaled_val:.6f} (actual: {actual_val:.2f} reads)\n")
    
    # Final latency
    out_file.write(f"\n  【Final Latency】\n")
    var = model.getVarByName(f'LATENCY_0')
    if var is not None:
        scaled_val = var.X
        actual_val = scaled_val / macs_scale_factor
        out_file.write(f"    LATENCY_0 = {scaled_val:.6f} (actual: {actual_val:.2f} cycles)\n")
    
    var = model.getVarByName(f'COMPUTE_CYCLES_0')
    if var is not None:
        scaled_val = var.X
        actual_val = scaled_val / macs_scale_factor
        out_file.write(f"    COMPUTE_CYCLES_0 = {scaled_val:.6f} (actual: {actual_val:.2f} cycles)\n")
        out_file.write(f"    COMPUTE_CYCLES_0 = {scaled_val:.6f} (actual: {actual_val:.2f} cycles)\n")
    
    out_file.write("\n")


def main():
    test_workloads = [
        {"name": "tiny", "N": 1, "K": 8, "C": 8, "P": 4, "Q": 4, "R": 3, "S": 3},
        {"name": "small", "N": 1, "K": 32, "C": 32, "P": 14, "Q": 14, "R": 3, "S": 3},
        {"name": "medium_1x1", "N": 1, "K": 64, "C": 64, "P": 14, "Q": 14, "R": 1, "S": 1},
    ]
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_file:
        for wl in test_workloads:
            name = wl.pop('name')
            wl['name'] = name
            extract_mapping(name, wl, out_file)
    
    print(f"Results written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
