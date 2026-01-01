#!/usr/bin/env python3
"""
验证修正后的公式与 trace 的相关性

修正后的公式:
row_acts = C_factor × (
    reuse_penalty × block_nums +
    2 × K_factor × (h_crossing × w_non + h_non × w_crossing) +
    4 × K_factor × h_crossing × w_crossing
)

其中:
- block_nums = ceil(H/block_h) × ceil(W/block_w)  -- 整个 input 的 block 数量
- reuse_penalty = K_factor (K外层) 或 1 (K内层)
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

import math
from validation.dram.trace_generator import TraceGenerator, DRAMConfig
from validation.dram_v2.experiments.mapping_sweep import SimpleMapping
from validation.dram_v2.core.mapping_space import MappingConfig, DIM_P, DIM_Q, DIM_C, DIM_K
from src.pim_optimizer.workload.conv import ConvWorkload


def compute_crossing_counts(block_size, tile_size, step, num_tiles):
    """计算 crossing 和 non-crossing tile 数量"""
    if block_size <= 0 or tile_size <= 0 or step <= 0 or num_tiles <= 0:
        return 0, 0
    
    if tile_size > block_size:
        return 0, num_tiles  # 全部 crossing
    
    g = math.gcd(step, block_size)
    period = block_size // g
    
    crossing_in_period = 0
    for k in range(period):
        pos = (k * step) % block_size
        if pos + tile_size > block_size:
            crossing_in_period += 1
    
    non_crossing_in_period = period - crossing_in_period
    
    num_complete_periods = num_tiles // period
    remainder = num_tiles % period
    
    crossing_in_remainder = 0
    for k in range(remainder):
        pos = (k * step) % block_size
        if pos + tile_size > block_size:
            crossing_in_remainder += 1
    
    non_crossing_in_remainder = remainder - crossing_in_remainder
    
    h_non = num_complete_periods * non_crossing_in_period + non_crossing_in_remainder
    h_crossing = num_complete_periods * crossing_in_period + crossing_in_remainder
    
    return h_non, h_crossing


def compute_corrected_formula(workload, config: MappingConfig, k_innermost=False):
    """
    计算修正后的公式
    
    row_acts = C_factor × (
        reuse_penalty × block_nums +
        2 × K_factor × (h_crossing × w_non + h_non × w_crossing) +
        4 × K_factor × h_crossing × w_crossing
    )
    """
    H = workload.input_size['H']
    W = workload.input_size['W']
    R, S = workload.R, workload.S
    stride_h, stride_w = 1, 1
    dilation_h, dilation_w = 1, 1
    
    P_l3, Q_l3, C_l3, K_l3 = config.P_l3, config.Q_l3, config.C_l3, config.K_l3
    block_h, block_w = config.block_h, config.block_w
    
    # Buffer tile sizes
    P_tile = workload.P // P_l3
    Q_tile = workload.Q // Q_l3
    
    # Input tile size (sliding window)
    tile_h = stride_h * (P_tile - 1) + dilation_h * (R - 1) + 1
    tile_w = stride_w * (Q_tile - 1) + dilation_w * (S - 1) + 1
    
    # Step between tiles
    step_h = P_tile * stride_h
    step_w = Q_tile * stride_w
    
    # 计算 H/W 方向的 non/crossing tiles
    h_non, h_crossing = compute_crossing_counts(block_h, tile_h, step_h, P_l3)
    w_non, w_crossing = compute_crossing_counts(block_w, tile_w, step_w, Q_l3)
    
    # block_nums = ceil(H/block_h) × ceil(W/block_w)
    block_nums = math.ceil(H / block_h) * math.ceil(W / block_w)
    
    # 因子
    C_factor = C_l3
    K_factor = K_l3
    reuse_penalty = 1 if k_innermost else K_factor
    
    # 修正后的公式
    row_acts = C_factor * (
        reuse_penalty * block_nums +
        2 * K_factor * (h_crossing * w_non + h_non * w_crossing) +
        4 * K_factor * h_crossing * w_crossing
    )
    
    return {
        'row_acts': row_acts,
        'block_nums': block_nums,
        'h_non': h_non,
        'h_crossing': h_crossing,
        'w_non': w_non,
        'w_crossing': w_crossing,
        'tile_h': tile_h,
        'tile_w': tile_w,
        'C_factor': C_factor,
        'K_factor': K_factor,
        'reuse_penalty': reuse_penalty,
    }


def analyze_trace(workload, config, dram_config):
    """获取 trace 的实际 row switches"""
    mapping = SimpleMapping(config, workload)
    generator = TraceGenerator(dram_config)
    trace = generator.generate_trace(mapping, workload)
    
    row_size = dram_config.row_buffer_bytes
    bank_size = row_size * dram_config.num_rows
    
    input_rows = []
    for line in trace:
        parts = line.split()
        if len(parts) >= 2:
            addr = int(parts[1], 16)
            bank = addr // bank_size
            if bank == 0:  # Input bank
                row = (addr % bank_size) // row_size
                input_rows.append(row)
    
    row_switches = sum(1 for i in range(1, len(input_rows)) if input_rows[i] != input_rows[i-1])
    # row_activations = row_switches + 1 (第一次访问)
    row_activations = row_switches + 1 if input_rows else 0
    return row_switches, row_activations


def main():
    print("=" * 110)
    print("修正后公式验证：block_nums = ceil(H/block_h) × ceil(W/block_w)")
    print("=" * 110)
    
    workload = ConvWorkload(name='test', R=3, S=3, P=4, Q=4, C=4, K=4, N=1)
    H, W = workload.input_size['H'], workload.input_size['W']
    dram_config = DRAMConfig(row_buffer_bytes=1024, num_banks=4, num_rows=16384, element_size=1)
    
    print(f"\nWorkload: P={workload.P}, Q={workload.Q}, C={workload.C}, K={workload.K}, R={workload.R}, S={workload.S}")
    print(f"Input: H={H}, W={W}")
    print()
    
    # 测试配置
    configs = [
        # (P_l3, Q_l3, C_l3, K_l3, block_h, block_w, k_innermost)
        (2, 2, 2, 2, 6, 6, False),
        (2, 2, 2, 2, 6, 6, True),
        (4, 4, 4, 4, 6, 6, False),
        (1, 1, 4, 4, 6, 6, False),
        (2, 2, 2, 2, 3, 3, False),
        (2, 2, 2, 2, 3, 3, True),
        (2, 2, 4, 4, 6, 6, False),
        (4, 4, 2, 2, 6, 6, False),
        (2, 2, 2, 2, 2, 2, False),
        (4, 4, 4, 4, 3, 3, False),
        (1, 1, 2, 2, 6, 6, False),
        (2, 2, 1, 1, 6, 6, False),
    ]
    
    print(f"{'Config':<28} {'Trace':>8} {'Formula':>8} {'Err%':>8} | {'blks':>5} {'h_n':>4} {'h_x':>4} {'w_n':>4} {'w_x':>4}")
    print("-" * 95)
    
    results = []
    for (P_l3, Q_l3, C_l3, K_l3, block_h, block_w, k_inner) in configs:
        config = MappingConfig(
            P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3,
            permutation=(DIM_K, DIM_C, DIM_P, DIM_Q) if k_inner else (DIM_P, DIM_Q, DIM_C, DIM_K),
            input_layout='row_aligned',
            weight_layout='sequential',
            output_layout='sequential',
            block_h=block_h, block_w=block_w
        )
        
        trace_switches, trace_acts = analyze_trace(workload, config, dram_config)
        formula = compute_corrected_formula(workload, config, k_innermost=k_inner)
        
        # 比较 row_activations (= row_switches + 1)
        error = (formula['row_acts'] - trace_acts) / trace_acts * 100 if trace_acts > 0 else 0
        
        k_str = "K_in" if k_inner else "K_out"
        config_str = f"P{P_l3}Q{Q_l3}C{C_l3}K{K_l3}_b{block_h}x{block_w}_{k_str}"
        
        print(f"{config_str:<28} {trace_acts:>8} {formula['row_acts']:>8} {error:>8.1f} | "
              f"{formula['block_nums']:>5} {formula['h_non']:>4} {formula['h_crossing']:>4} "
              f"{formula['w_non']:>4} {formula['w_crossing']:>4}")
        
        results.append({
            'trace': trace_acts, 
            'formula': formula['row_acts'], 
            'error': error,
            'config': config_str
        })
    
    # 相关性分析
    print()
    print("=" * 110)
    print("相关性分析")
    print("=" * 110)
    
    trace_vals = [r['trace'] for r in results]
    formula_vals = [r['formula'] for r in results]
    
    from scipy.stats import spearmanr, pearsonr
    spearman_corr, sp_pval = spearmanr(trace_vals, formula_vals)
    pearson_corr, pe_pval = pearsonr(trace_vals, formula_vals)
    
    mae = sum(abs(r['trace'] - r['formula']) for r in results) / len(results)
    mape = sum(abs(r['error']) for r in results) / len(results)
    
    print(f"\nSpearman Correlation: {spearman_corr:.4f} (p={sp_pval:.6f})")
    print(f"Pearson Correlation:  {pearson_corr:.4f} (p={pe_pval:.6f})")
    print(f"Mean Absolute Error:  {mae:.1f}")
    print(f"Mean Absolute % Error: {mape:.1f}%")
    
    # 排名对比
    print(f"\n排名对比:")
    trace_order = sorted(range(len(trace_vals)), key=lambda i: trace_vals[i])
    formula_order = sorted(range(len(formula_vals)), key=lambda i: formula_vals[i])
    
    print(f"  Trace 排名:   {trace_order}")
    print(f"  Formula 排名: {formula_order}")
    
    # 匹配数
    matches = sum(1 for i in range(len(trace_order)) if trace_order[i] == formula_order[i])
    print(f"  排名完全匹配: {matches}/{len(trace_order)}")


if __name__ == "__main__":
    main()
