#!/usr/bin/env python3
"""
完整的 DRAM Row Activation 验证脚本

对比 ILP 预测值与 Ramulator2 模拟值
"""

import sys
import os
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')

import numpy as np
from tqdm import tqdm
from pim_optimizer.arch.pim_arch import PIMArchitecture
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer import PIMOptimizer

from ramulator_runner import RamulatorRunner, RamulatorResult
from trace_generator import TraceGenerator, DRAMConfig


def count_row_activations_from_trace(trace_path: str, dram_config: DRAMConfig = None) -> dict:
    """
    从 trace 文件中计算 row activation 次数。
    
    逻辑：
    1. 解析每个地址
    2. 根据 DRAM 地址映射计算 (channel, rank, bank, row)
    3. 每个 bank 维护当前打开的 row
    4. 如果新访问的 row 不同于当前打开的 row，计一次 row activation
    
    地址映射 (简化的 RoBaRaCoCh)：
    - 使用 row_buffer_bytes 作为行大小
    - bank 由基地址决定 (不同 tensor 在不同 bank)
    - row = addr // row_buffer_bytes
    
    Returns:
        dict: {
            'total_row_acts': int,
            'per_bank_acts': dict[(ch, rank, bank)] -> int,
            'total_accesses': int,
        }
    """
    cfg = dram_config or DRAMConfig()
    
    # DRAM 参数
    row_buffer_bytes = cfg.row_buffer_bytes  # 每 row 的大小（字节）
    num_banks = cfg.num_banks
    
    # 每个 bank 当前打开的 row (-1 表示无)
    open_rows = {}  # bank_id -> row_id
    
    # 统计
    total_row_acts = 0
    per_bank_acts = {}
    total_accesses = 0
    
    with open(trace_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 解析格式: "LD 0x12345678" 或 "ST 0x12345678"
            parts = line.split()
            if len(parts) < 2:
                continue
            
            try:
                addr = int(parts[1], 16)
            except ValueError:
                continue
            
            total_accesses += 1
            
            # 地址解码：bank | row | col 映射
            #
            # 地址布局: | bank | row | column offset |
            #           | 2b   | 14b | 10 bits      |
            #
            # Base addresses (from trace_generator.py):
            #   Input:  0x00000000 (Bank 0)
            #   Weight: 0x01000000 (Bank 1)
            #   Output: 0x02000000 (Bank 2)
            #
            # bank_size = row_buffer_bytes * num_rows = 1024 * 16384 = 16MB
            
            col_bits = int(np.log2(row_buffer_bytes))  # 10 for 1024
            num_rows = 16384  # 从 DRAMConfig
            row_bits = int(np.log2(num_rows))  # 14 for 16384
            
            # 从地址提取 bank, row, col
            col = addr & ((1 << col_bits) - 1)
            row = (addr >> col_bits) & ((1 << row_bits) - 1)
            bank = addr >> (col_bits + row_bits)
            
            # 检查是否需要 row activation
            if bank not in open_rows:
                # 第一次访问这个 bank，需要 activation
                open_rows[bank] = row
                total_row_acts += 1
                per_bank_acts[bank] = per_bank_acts.get(bank, 0) + 1
            elif open_rows[bank] != row:
                # 访问不同的 row，需要 activation
                open_rows[bank] = row
                total_row_acts += 1
                per_bank_acts[bank] = per_bank_acts.get(bank, 0) + 1
            # else: row hit，不需要 activation
    
    return {
        'total_row_acts': total_row_acts,
        'per_bank_acts': per_bank_acts,
        'total_accesses': total_accesses,
    }


def generate_trace_for_mapping(optimizer, workload, output_path):
    """
    从 optimizer 结果生成 trace。
    
    使用 TraceGenerator 类，根据 ILP 提取的 mapping 生成真实的内存访问 trace。
    """
    # 从 optimizer 获取 mapping
    # 注意：需要先运行 optimize()，result 已经在外部获取
    model = optimizer.model
    
    # 手动提取 mapping 信息（因为我们在这里没有 result 对象）
    from pim_optimizer.mapping import Mapping
    from pim_optimizer.model.variables import SpatialDim
    
    mapping = Mapping()
    mapping.workload_name = workload.name
    mapping.workload_bounds = list(workload.bounds)
    
    arch = optimizer.arch
    num_mems = arch.num_mems
    vars = optimizer.vars
    w = 0  # 单 workload 场景
    
    # Extract loop bounds
    for m in range(num_mems):
        if m == 0:
            mapping.loop_bounds[m] = {"H": {}, "W": {}, "Internal": {}, "temporal": {}}
            s_names = {SpatialDim.H: "H", SpatialDim.W: "W", SpatialDim.INTERNAL: "Internal", SpatialDim.TEMPORAL: "temporal"}
            s_range = SpatialDim.num_dims_pe()
        else:
            mapping.loop_bounds[m] = {"spatial": {}, "temporal": {}}
            s_names = {0: "spatial", 1: "temporal"}
            s_range = SpatialDim.num_dims_other()
        
        for j, divs in enumerate(workload.divisors):
            for s in range(s_range):
                s_name = s_names[s]
                for i, div in enumerate(divs):
                    if vars.xb[w, m, s, j, i].X > 0.5:
                        mapping.loop_bounds[m][s_name][j] = div
    
    # Extract permutation
    for m in range(num_mems):
        mapping.permutation[m] = {}
        for p in range(len(workload.bounds)):
            for j in range(len(workload.bounds)):
                if vars.xp[w, m, p, j].X > 0.5:
                    mapping.permutation[m][p] = j
    
    # Extract layout
    for t in range(3):
        if (w, t, "row_aligned") in vars.layout_choice:
            if vars.layout_choice[w, t, "row_aligned"].X > 0.5:
                mapping.layout[t] = "row_aligned"
            else:
                mapping.layout[t] = "sequential"
    
    # Extract input block_h and block_w
    h_divisors = getattr(workload, 'hw_divisors', {}).get('H', [1])
    w_divisors = getattr(workload, 'hw_divisors', {}).get('W', [1])
    
    block_h = 1
    for i, h_div in enumerate(h_divisors):
        if (w, i) in vars.rowbuf_input_block_h:
            if vars.rowbuf_input_block_h[w, i].X > 0.5:
                block_h = h_div
                break
    
    block_w = 1
    for j, w_div in enumerate(w_divisors):
        if (w, j) in vars.rowbuf_input_block_w:
            if vars.rowbuf_input_block_w[w, j].X > 0.5:
                block_w = w_div
                break
    
    mapping.tile_info = {'block_h': block_h, 'block_w': block_w}
    
    # 打印提取的 mapping 信息（调试用）
    print(f"  Extracted mapping:")
    print(f"    block_h={block_h}, block_w={block_w}")
    print(f"    layout: input={mapping.layout.get(0, 'seq')}, weight={mapping.layout.get(1, 'seq')}, output={mapping.layout.get(2, 'seq')}")
    
    # 使用 TraceGenerator 生成 trace
    dram_cfg = DRAMConfig()
    trace_gen = TraceGenerator(dram_cfg)
    
    try:
        traces = trace_gen.generate_trace(mapping, workload)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(traces))
        
        return len(traces)
    except Exception as e:
        print(f"  TraceGenerator error: {e}")
        # Fallback: 简单 trace 生成
        return _generate_simple_trace(model, workload, output_path)


def _generate_simple_trace(model, workload, output_path):
    """简单的 trace 生成（备用）"""
    # 获取 macs_scale_factor 用于反归一化
    MAX_BOUND = 1e4
    macs_scale_factor = MAX_BOUND / (1.02 * workload.macs)
    
    # 获取 RowBuffer 层 (m=2) 的 memory reads（需要反归一化）
    input_reads = 0
    var = model.getVarByName('MEM_READS_0_2_0')
    if var:
        input_reads = int(var.X / macs_scale_factor)
    
    weight_reads = 0
    var = model.getVarByName('MEM_READS_0_2_1')
    if var:
        weight_reads = int(var.X / macs_scale_factor)
    
    output_reads = 0
    var = model.getVarByName('MEM_READS_0_2_2')
    if var:
        output_reads = int(var.X / macs_scale_factor)
    
    print(f"  Fallback trace: Input={input_reads}B, Weight={weight_reads}B, Output={output_reads}B")
    
    traces = []
    row_size = 1024  # Row buffer size in bytes
    
    # 生成跨 row 的访问模式
    base_addr = 0x0
    for i in range(max(1, input_reads // 64)):
        # 每隔 row_size 强制跨 row
        addr = base_addr + (i % 16) * 64 + (i // 16) * row_size
        traces.append(f"LD 0x{addr:08x}")
    
    base_addr = 0x10000000
    for i in range(max(1, weight_reads // 64)):
        addr = base_addr + (i % 16) * 64 + (i // 16) * row_size
        traces.append(f"LD 0x{addr:08x}")
    
    base_addr = 0x20000000
    for i in range(max(1, output_reads // 64)):
        addr = base_addr + (i % 16) * 64 + (i // 16) * row_size
        traces.append(f"LD 0x{addr:08x}")
        traces.append(f"ST 0x{addr:08x}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(traces))
    
    return len(traces)


def run_validation():
    """运行完整验证"""
    
    print("=" * 70)
    print("DRAM Row Activation 验证")
    print("=" * 70)
    
    # 检查 Ramulator2
    runner = RamulatorRunner()
    if not runner.is_available():
        print(f"Warning: Ramulator2 not found at {runner.ramulator_bin}")
        print("Will only show ILP predictions")
        use_ramulator = False
    else:
        print(f"Using Ramulator2: {runner.ramulator_bin}")
        use_ramulator = True
    
    # 测试用例
    test_workloads = [
        {"name": "tiny", "N": 1, "K": 8, "C": 8, "P": 4, "Q": 4, "R": 3, "S": 3},
        {"name": "small", "N": 1, "K": 32, "C": 32, "P": 14, "Q": 14, "R": 3, "S": 3},
        {"name": "medium_1x1", "N": 1, "K": 64, "C": 64, "P": 14, "Q": 14, "R": 1, "S": 1},
        {"name": "medium_3x3", "N": 1, "K": 64, "C": 64, "P": 14, "Q": 14, "R": 3, "S": 3},
        {"name": "large", "N": 1, "K": 128, "C": 128, "P": 7, "Q": 7, "R": 3, "S": 3},
    ]
    
    arch = PIMArchitecture.from_yaml('examples/configs/arch.yaml')
    
    results = []
    
    print()
    print("-" * 120)
    header = f"{'Workload':<15} {'MACs':>12} {'ILP Row Acts':>14} {'ILP DRAM Cyc':>14}"
    header += f" {'Trace Row Acts':>14} {'RA Error%':>10}"
    if use_ramulator:
        header += f" {'Sim Cycles':>12}"
    print(header)
    print("-" * 120)
    
    output_dir = "validation_output"
    os.makedirs(output_dir, exist_ok=True)
    
    for wl_params in test_workloads:
        name = wl_params.pop('name')
        workload = ConvWorkload(name=name, **wl_params)
        
        # 运行 ILP optimizer
        optimizer = PIMOptimizer(arch, verbose=False)
        result = optimizer.optimize([workload])
        model = optimizer.model
        
        # 获取预测值
        macs = workload.macs
        
        # 获取 macs_scale_factor 用于反归一化
        MAX_BOUND = 1e4
        macs_scale_factor = MAX_BOUND / (1.02 * macs)
        
        # Row activations (当前变量名: total_row_acts_(w,t_id))
        total_row_act = 0
        for t_id, t_name in [(0, 'input'), (1, 'weight'), (2, 'output')]:
            var = model.getVarByName(f'total_row_acts_(0,{t_id})')
            if var:
                total_row_act += var.X
        
        # DRAM latency (当前变量名: V_dram_latency_(w))
        var_cycles = model.getVarByName('V_dram_latency_(0)')
        dram_cycles_scaled = var_cycles.X if var_cycles else 0
        dram_cycles = dram_cycles_scaled / macs_scale_factor  # 反归一化
        
        row = f"{name:<15} {macs:>12} {total_row_act:>14.2f} {dram_cycles:>14.2f}"
        
        # 生成 trace 并用 Python 计算 row activation
        trace_path = os.path.join(output_dir, f"{name}_trace.txt")
        print(f"\nGenerating trace for {name}...")
        num_traces = generate_trace_for_mapping(optimizer, workload, trace_path)
        print(f"  Generated {num_traces} trace lines")
        
        # Python 计算 row activation
        trace_stats = count_row_activations_from_trace(trace_path, DRAMConfig())
        trace_row_acts = trace_stats['total_row_acts']
        print(f"  Trace row activations: {trace_row_acts}")
        
        # 计算误差
        if trace_row_acts > 0:
            ra_error = abs(total_row_act - trace_row_acts) / trace_row_acts * 100
        else:
            ra_error = 0 if total_row_act == 0 else 100
        
        row += f" {trace_row_acts:>14} {ra_error:>9.2f}%"
        
        if use_ramulator:
            # 运行 Ramulator2 (只看 cycles)
            try:
                sim_result = runner.run(trace_path)
                sim_cycles = sim_result.cycles
                row += f" {sim_cycles:>12}"
                
                results.append({
                    'name': name,
                    'macs': macs,
                    'ilp_row_acts': total_row_act,
                    'ilp_cycles': dram_cycles,
                    'trace_row_acts': trace_row_acts,
                    'sim_cycles': sim_cycles,
                    'ra_error': ra_error
                })
            except Exception as e:
                row += f" {'ERROR':>12}"
                print(f"  Ramulator error: {e}")
        
        print(row)
        
        # 恢复 name
        wl_params['name'] = name
    
    print("-" * 120)
    
    # 统计
    if results:
        ra_errors = [r['ra_error'] for r in results if 'ra_error' in r]
        if ra_errors:
            print(f"\nRow Activation Error:")
            print(f"  Mean: {np.mean(ra_errors):.2f}%")
            print(f"  Max:  {np.max(ra_errors):.2f}%")
            print(f"  Min:  {np.min(ra_errors):.2f}%")
    
    print("\n" + "=" * 70)
    print("详细 ILP 预测值分解")
    print("=" * 70)
    
    for wl_params in test_workloads:
        name = wl_params['name']
        workload = ConvWorkload(**wl_params)
        
        optimizer = PIMOptimizer(arch, verbose=False)
        result = optimizer.optimize([workload])
        model = optimizer.model
        
        # 获取 macs_scale_factor 用于反归一化
        MAX_BOUND = 1e4
        macs_scale_factor = MAX_BOUND / (1.02 * workload.macs)
        
        print(f"\n{name}:")
        
        # 每个 datatype 的 row activations 和 DRAM latency
        tensor_names = {0: 'input', 1: 'weight', 2: 'output'}
        for t_id, t_name in tensor_names.items():
            # total_row_acts
            var = model.getVarByName(f'total_row_acts_(0,{t_id})')
            row_acts = var.X if var else 0
            
            # DRAM latency per datatype
            var = model.getVarByName(f'V_dram_latency_{t_name}_(0)')
            dram_lat_scaled = var.X if var else 0
            dram_lat = dram_lat_scaled / macs_scale_factor
            
            print(f"  {t_name.capitalize():8s}: row_acts={row_acts:8.2f}, dram_latency={dram_lat:10.2f} cycles")
        
        # Layout choice
        layout_seq = model.getVarByName('layout_choice_(0,0,sequential)')
        layout = "Sequential" if layout_seq and layout_seq.X > 0.5 else "Row-aligned"
        print(f"  Selected layout: {layout}")
        
        # Total DRAM latency
        var = model.getVarByName('V_dram_latency_(0)')
        total_dram = (var.X / macs_scale_factor) if var else 0
        print(f"  Total DRAM latency: {total_dram:.2f} cycles (max of three)")


if __name__ == "__main__":
    run_validation()
