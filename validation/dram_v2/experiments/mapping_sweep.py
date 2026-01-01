#!/usr/bin/env python3
"""
Mapping Sweep: 批量运行多个 mapping，收集 row switch 统计

使用新的 mapping 模块进行枚举和约束验证
"""

import sys
import csv
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from src.pim_optimizer.workload.conv import ConvWorkload
from validation.dram.trace_generator import TraceGenerator, DRAMConfig

# 使用新的 mapping 模块
from validation.dram_v2.core.mapping import (
    MappingConfig, MappingEnumerator, TilingMode,
    WorkloadConfig, ArchConfig,
    to_trace_generator_mapping,
    DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N,
    DIM_NAMES
)

from validation.dram_v2.formula.row_activation_formula import (
    FormulaConfig, compute_total_row_switches_formula
)


def compute_formula_result(config: MappingConfig, workload: ConvWorkload, 
                           dram_config: DRAMConfig) -> Dict:
    """使用公式计算 row switches"""
    formula_config = FormulaConfig(
        R=workload.R, S=workload.S,
        P=workload.P, Q=workload.Q,
        C=workload.C, K=workload.K, N=workload.N,
        H=workload.input_size['H'], W=workload.input_size['W'],
        stride=workload.stride if hasattr(workload, 'stride') else (1, 1),
        dilation=workload.dilation if hasattr(workload, 'dilation') else (1, 1),
        row_buffer_bytes=dram_config.row_buffer_bytes,
        element_size=dram_config.element_size,
        P_l3=config.P_l3, Q_l3=config.Q_l3,
        C_l3=config.C_l3, K_l3=config.K_l3,
        block_h=config.block_h, block_w=config.block_w,
        input_layout=config.input_layout
    )
    return compute_total_row_switches_formula(formula_config)


def count_row_switches(rows: List[int]) -> int:
    """统计 row switches"""
    if len(rows) < 2:
        return 0
    switches = 0
    prev = rows[0]
    for r in rows[1:]:
        if r != prev:
            switches += 1
        prev = r
    return switches


def create_workload_config(workload: ConvWorkload) -> WorkloadConfig:
    """从 ConvWorkload 创建 WorkloadConfig"""
    return WorkloadConfig(
        P=workload.P, Q=workload.Q,
        C=workload.C, K=workload.K,
        R=workload.R, S=workload.S,
        H=workload.input_size['H'],
        W=workload.input_size['W'],
        N=workload.N
    )


def run_single_mapping(config: MappingConfig, workload: ConvWorkload, 
                       dram_config: DRAMConfig) -> Dict:
    """运行单个 mapping，返回统计结果"""
    
    # 使用新的适配器转换为 trace_generator 格式
    mapping = to_trace_generator_mapping(config)
    
    # 生成 trace
    generator = TraceGenerator(dram_config)
    trace = generator.generate_trace(mapping, workload)
    
    # 分析 trace
    row_size = dram_config.row_buffer_bytes
    bank_size = row_size * dram_config.num_rows
    
    bank_rows = {0: [], 1: [], 2: []}  # Input, Weight, Output
    
    for line in trace:
        parts = line.split()
        if len(parts) >= 2:
            addr = int(parts[1], 16)
            bank = addr // bank_size
            if bank in bank_rows:
                bank_addr = addr % bank_size
                row = bank_addr // row_size
                bank_rows[bank].append(row)
    
    # Trace 统计
    trace_input_rs = count_row_switches(bank_rows[0])
    trace_weight_rs = count_row_switches(bank_rows[1])
    trace_output_rs = count_row_switches(bank_rows[2])
    trace_total_rs = trace_input_rs + trace_weight_rs + trace_output_rs
    
    # Formula 计算
    formula_result = compute_formula_result(config, workload, dram_config)
    formula_input_rs = formula_result['input']
    formula_weight_rs = formula_result['weight']
    formula_output_rs = formula_result['output']
    formula_total_rs = formula_result['total']
    
    # 获取 permutation 字符串
    perm_str = '-'.join(DIM_NAMES[d] for d in config.permutation_l3)
    
    result = {
        'config': {
            'P_l3': config.P_l3, 'Q_l3': config.Q_l3,
            'C_l3': config.C_l3, 'K_l3': config.K_l3,
            'P_l2': config.P_l2, 'Q_l2': config.Q_l2,
            'C_l2': config.C_l2, 'K_l2': config.K_l2,
            'permutation': perm_str,
            'block_h': config.block_h, 'block_w': config.block_w,
            'input_layout': config.input_layout,
            'weight_layout': config.weight_layout,
            'output_layout': config.output_layout,
        },
        # Trace 结果
        'input_accesses': len(bank_rows[0]),
        'weight_accesses': len(bank_rows[1]),
        'output_accesses': len(bank_rows[2]),
        'input_unique_rows': len(set(bank_rows[0])),
        'weight_unique_rows': len(set(bank_rows[1])),
        'output_unique_rows': len(set(bank_rows[2])),
        'trace_input_rs': trace_input_rs,
        'trace_weight_rs': trace_weight_rs,
        'trace_output_rs': trace_output_rs,
        'trace_total_rs': trace_total_rs,
        # Formula 结果
        'formula_input_rs': formula_input_rs,
        'formula_weight_rs': formula_weight_rs,
        'formula_output_rs': formula_output_rs,
        'formula_total_rs': formula_total_rs,
        # 误差分析
        'input_error': abs(trace_input_rs - formula_input_rs) / max(1, trace_input_rs),
        'weight_error': abs(trace_weight_rs - formula_weight_rs) / max(1, trace_weight_rs),
        'output_error': abs(trace_output_rs - formula_output_rs) / max(1, trace_output_rs),
        'total_error': abs(trace_total_rs - formula_total_rs) / max(1, trace_total_rs),
    }
    # 兼容旧字段名
    result['input_row_switches'] = trace_input_rs
    result['weight_row_switches'] = trace_weight_rs
    result['output_row_switches'] = trace_output_rs
    result['total_row_switches'] = trace_total_rs
    
    return result


def run_sweep_sampled(workload: ConvWorkload,
                      sample_size: int = 100,
                      output_dir: Path = None) -> List[Dict]:
    """
    从完整空间采样指定数量的 mapping
    包含: Tiling + Permutation + Block Size + Layout
    
    使用新的 MappingEnumerator 进行枚举和约束验证
    """
    print("=" * 70)
    print(f"Full Space Sampling ({sample_size} mappings)")
    print("=" * 70)
    print(f"Workload: {workload.name}")
    print()
    
    # 创建 WorkloadConfig 和 ArchConfig
    wl_config = create_workload_config(workload)
    arch_config = ArchConfig(
        row_buffer_bytes=1024,
        global_buffer_bytes=256*1024
    )
    
    # 创建 MappingEnumerator (启用约束验证)
    enumerator = MappingEnumerator(
        wl_config, arch_config, 
        validate_constraints=True
    )
    enumerator.summary()
    print()
    
    # DRAM config
    dram_config = DRAMConfig(row_buffer_bytes=1024, num_banks=4,
                             num_rows=16384, element_size=1)
    
    # 使用 MappingEnumerator 枚举并采样
    print("Generating valid mappings (satisfying constraints)...")
    
    configs = list(enumerator.enumerate(
        mode=TilingMode.L3_L2,
        vary_permutation=True,
        vary_layout=True,
        vary_block_size=True,
        sample_total=sample_size,
        filter_invalid=True
    ))
    
    print(f"Sampled: {len(configs)} configurations")
    print()
    
    results = []
    for i, config in enumerate(configs):
        start = time.time()
        result = run_single_mapping(config, workload, dram_config)
        elapsed = time.time() - start
        
        results.append(result)
        
        if (i + 1) % 20 == 0 or i == 0:
            cfg = result['config']
            print(f"  [{i+1:3d}/{len(configs)}] "
                  f"P{cfg['P_l3']}_Q{cfg['Q_l3']}_C{cfg['C_l3']}_K{cfg['K_l3']} "
                  f"blk({cfg['block_h']}x{cfg['block_w']}) | "
                  f"Total: {result['total_row_switches']:6d} | {elapsed:.2f}s")
    
    # 按 total_row_switches 排序
    results.sort(key=lambda x: x['total_row_switches'])
    
    print()
    print("=" * 70)
    print("Top 10 (lowest row switches):")
    print("=" * 70)
    for i, r in enumerate(results[:10]):
        cfg = r['config']
        print(f"  {i+1:2d}. P{cfg['P_l3']}_Q{cfg['Q_l3']}_C{cfg['C_l3']}_K{cfg['K_l3']} "
              f"{cfg['permutation']:12s} blk({cfg['block_h']}x{cfg['block_w']}) "
              f"-> Total: {r['total_row_switches']}")
    
    print()
    print("Bottom 5 (highest row switches):")
    print("-" * 70)
    for i, r in enumerate(results[-5:]):
        cfg = r['config']
        print(f"  {i+1}. P{cfg['P_l3']}_Q{cfg['Q_l3']}_C{cfg['C_l3']}_K{cfg['K_l3']} "
              f"{cfg['permutation']:12s} blk({cfg['block_h']}x{cfg['block_w']}) "
              f"-> Total: {r['total_row_switches']}")
    
    # 保存结果
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{workload.name}_full_sweep.csv"
        
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['P_l3', 'Q_l3', 'C_l3', 'K_l3', 'P_l2', 'Q_l2', 'C_l2', 'K_l2',
                          'permutation', 'block_h', 'block_w', 
                          'input_layout', 'weight_layout', 'output_layout',
                          'trace_input_rs', 'trace_weight_rs', 'trace_output_rs', 'trace_total_rs',
                          'formula_input_rs', 'formula_weight_rs', 'formula_output_rs', 'formula_total_rs',
                          'input_error', 'weight_error', 'output_error', 'total_error']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for r in results:
                cfg = r['config']
                row = {
                    'P_l3': cfg['P_l3'], 'Q_l3': cfg['Q_l3'],
                    'C_l3': cfg['C_l3'], 'K_l3': cfg['K_l3'],
                    'P_l2': cfg['P_l2'], 'Q_l2': cfg['Q_l2'],
                    'C_l2': cfg['C_l2'], 'K_l2': cfg['K_l2'],
                    'permutation': cfg['permutation'],
                    'block_h': cfg['block_h'],
                    'block_w': cfg['block_w'],
                    'input_layout': cfg['input_layout'],
                    'weight_layout': cfg['weight_layout'],
                    'output_layout': cfg['output_layout'],
                    'trace_input_rs': r['trace_input_rs'],
                    'trace_weight_rs': r['trace_weight_rs'],
                    'trace_output_rs': r['trace_output_rs'],
                    'trace_total_rs': r['trace_total_rs'],
                    'formula_input_rs': r['formula_input_rs'],
                    'formula_weight_rs': r['formula_weight_rs'],
                    'formula_output_rs': r['formula_output_rs'],
                    'formula_total_rs': r['formula_total_rs'],
                    'input_error': f"{r['input_error']:.4f}",
                    'weight_error': f"{r['weight_error']:.4f}",
                    'output_error': f"{r['output_error']:.4f}",
                    'total_error': f"{r['total_error']:.4f}",
                }
                writer.writerow(row)
        
        print(f"\nResults saved to: {csv_path}")
    
    return results


def compute_correlation_metrics(results: List[Dict]) -> Dict:
    """
    计算 Trace vs Formula 的相关性指标
    
    对于 ILP，Spearman 相关系数（排名相关）比 Pearson（绝对相关）更重要。
    """
    from scipy import stats
    import numpy as np
    
    # 提取数据
    trace_input = [r['trace_input_rs'] for r in results]
    trace_weight = [r['trace_weight_rs'] for r in results]
    trace_output = [r['trace_output_rs'] for r in results]
    trace_total = [r['trace_total_rs'] for r in results]
    
    formula_input = [r['formula_input_rs'] for r in results]
    formula_weight = [r['formula_weight_rs'] for r in results]
    formula_output = [r['formula_output_rs'] for r in results]
    formula_total = [r['formula_total_rs'] for r in results]
    
    metrics = {}
    
    # Spearman 相关系数（排名）
    metrics['spearman_input'] = stats.spearmanr(trace_input, formula_input).statistic
    metrics['spearman_weight'] = stats.spearmanr(trace_weight, formula_weight).statistic
    metrics['spearman_output'] = stats.spearmanr(trace_output, formula_output).statistic
    metrics['spearman_total'] = stats.spearmanr(trace_total, formula_total).statistic
    
    # Pearson 相关系数（线性）
    metrics['pearson_input'] = stats.pearsonr(trace_input, formula_input).statistic
    metrics['pearson_weight'] = stats.pearsonr(trace_weight, formula_weight).statistic
    metrics['pearson_output'] = stats.pearsonr(trace_output, formula_output).statistic
    metrics['pearson_total'] = stats.pearsonr(trace_total, formula_total).statistic
    
    # Top-K 一致性
    for k in [5, 10, 20]:
        trace_sorted = sorted(range(len(trace_total)), key=lambda i: trace_total[i])
        formula_sorted = sorted(range(len(formula_total)), key=lambda i: formula_total[i])
        
        trace_top_k = set(trace_sorted[:k])
        formula_top_k = set(formula_sorted[:k])
        overlap = len(trace_top_k & formula_top_k)
        metrics[f'top_{k}_overlap'] = overlap / k
    
    # 平均绝对误差百分比
    metrics['mape_input'] = np.mean([abs(t - f) / max(1, t) for t, f in zip(trace_input, formula_input)])
    metrics['mape_weight'] = np.mean([abs(t - f) / max(1, t) for t, f in zip(trace_weight, formula_weight)])
    metrics['mape_output'] = np.mean([abs(t - f) / max(1, t) for t, f in zip(trace_output, formula_output)])
    metrics['mape_total'] = np.mean([abs(t - f) / max(1, t) for t, f in zip(trace_total, formula_total)])
    
    return metrics


def print_correlation_report(results: List[Dict]):
    """打印相关性分析报告"""
    metrics = compute_correlation_metrics(results)
    
    print()
    print("=" * 70)
    print("Formula vs Trace Correlation Report")
    print("=" * 70)
    
    print("\n[1] Spearman Rank Correlation (重要 for ILP ranking):")
    print(f"    Input:  {metrics['spearman_input']:.4f}")
    print(f"    Weight: {metrics['spearman_weight']:.4f}")
    print(f"    Output: {metrics['spearman_output']:.4f}")
    print(f"    Total:  {metrics['spearman_total']:.4f}")
    
    print("\n[2] Pearson Linear Correlation:")
    print(f"    Input:  {metrics['pearson_input']:.4f}")
    print(f"    Weight: {metrics['pearson_weight']:.4f}")
    print(f"    Output: {metrics['pearson_output']:.4f}")
    print(f"    Total:  {metrics['pearson_total']:.4f}")
    
    print("\n[3] Top-K Overlap (formula 的 top-K 中有多少在 trace 的 top-K 中):")
    print(f"    Top-5:  {metrics['top_5_overlap']:.0%}")
    print(f"    Top-10: {metrics['top_10_overlap']:.0%}")
    print(f"    Top-20: {metrics['top_20_overlap']:.0%}")
    
    print("\n[4] Mean Absolute Percentage Error (MAPE):")
    print(f"    Input:  {metrics['mape_input']:.2%}")
    print(f"    Weight: {metrics['mape_weight']:.2%}")
    print(f"    Output: {metrics['mape_output']:.2%}")
    print(f"    Total:  {metrics['mape_total']:.2%}")
    
    # 评估
    print("\n" + "=" * 70)
    print("Assessment for ILP:")
    print("=" * 70)
    
    spearman_ok = metrics['spearman_total'] > 0.85
    top10_ok = metrics['top_10_overlap'] > 0.7
    
    if spearman_ok and top10_ok:
        print("✅ Formula is GOOD for ILP optimization!")
        print("   - Spearman correlation > 0.85: ranking is reliable")
        print("   - Top-10 overlap > 70%: formula finds similar best mappings")
    elif spearman_ok:
        print("⚠️ Formula is ACCEPTABLE for ILP (good ranking, but top-K needs work)")
    else:
        print("❌ Formula needs improvement for ILP")
        print("   - Spearman correlation should be > 0.85")
        print("   - Top-10 overlap should be > 70%")


def main():
    """主入口"""
    # 创建 workload: ResNet-L1
    workload = ConvWorkload(
        name="ResNet-L1",
        R=7, S=7,
        P=56, Q=56,
        C=3, K=64,
        N=1
    )
    
    output_dir = Path('/Users/haochenzhao/Projects/pim_optimizer/validation/dram_v2/results')
    
    # 从完整空间采样 100 个 mapping
    results = run_sweep_sampled(
        workload,
        sample_size=100,
        output_dir=output_dir
    )
    
    # 打印相关性报告
    print_correlation_report(results)


if __name__ == "__main__":
    main()
