#!/usr/bin/env python3
"""
Row Activation 验证脚本（精简版）

只验证 ILP 预测的 row activation 与 trace 计算的 row activation
不运行 Ramulator2（cycle 模拟太慢）
"""

import sys
import os
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')

import numpy as np
from tqdm import tqdm
from pim_optimizer.arch.pim_arch import PIMArchitecture
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer import PIMOptimizer

from trace_generator import TraceGenerator, DRAMConfig
from full_validation import count_row_activations_from_trace, generate_trace_for_mapping


def run_row_activation_validation():
    """只验证 row activation"""
    
    print("=" * 80)
    print("Row Activation 验证 (ILP vs Trace)")
    print("=" * 80)
    
    # 测试用例 - 从小到大
    test_workloads = [
        {"name": "tiny", "N": 1, "K": 8, "C": 8, "P": 4, "Q": 4, "R": 3, "S": 3},
        {"name": "small", "N": 1, "K": 16, "C": 16, "P": 8, "Q": 8, "R": 3, "S": 3},
        {"name": "medium", "N": 1, "K": 32, "C": 32, "P": 7, "Q": 7, "R": 3, "S": 3},
    ]
    
    arch = PIMArchitecture.from_yaml('examples/configs/arch.yaml')
    
    results = []
    
    print()
    print("-" * 100)
    print(f"{'Workload':<12} {'MACs':>10} {'ILP Input':>10} {'ILP Wgt':>10} {'ILP Out':>10} "
          f"{'ILP Total':>10} {'Trace':>10} {'Error%':>10}")
    print("-" * 100)
    
    output_dir = "validation_output"
    os.makedirs(output_dir, exist_ok=True)
    
    for wl_params in test_workloads:
        name = wl_params.pop('name')
        workload = ConvWorkload(name=name, **wl_params)
        
        print(f"\n[{name}] Running ILP optimizer...")
        
        # 运行 ILP optimizer
        optimizer = PIMOptimizer(arch, verbose=False)
        result = optimizer.optimize([workload])
        model = optimizer.model
        
        macs = workload.macs
        
        # 获取每个 datatype 的 row activation
        ilp_row_acts = {}
        total_ilp = 0
        for t_id, t_name in [(0, 'input'), (1, 'weight'), (2, 'output')]:
            var = model.getVarByName(f'total_row_acts_(0,{t_id})')
            if var:
                ilp_row_acts[t_name] = var.X
                total_ilp += var.X
            else:
                ilp_row_acts[t_name] = 0
        
        # 生成 trace
        trace_path = os.path.join(output_dir, f"{name}_trace.txt")
        print(f"[{name}] Generating trace...")
        num_traces = generate_trace_for_mapping(optimizer, workload, trace_path)
        print(f"[{name}] Generated {num_traces} traces")
        
        # Python 计算 row activation
        print(f"[{name}] Counting row activations...")
        trace_stats = count_row_activations_from_trace(trace_path, DRAMConfig())
        trace_row_acts = trace_stats['total_row_acts']
        
        # 计算误差
        if trace_row_acts > 0:
            error = abs(total_ilp - trace_row_acts) / trace_row_acts * 100
        else:
            error = 0 if total_ilp == 0 else 100
        
        # 打印结果
        print(f"{name:<12} {macs:>10} {ilp_row_acts['input']:>10.1f} {ilp_row_acts['weight']:>10.1f} "
              f"{ilp_row_acts['output']:>10.1f} {total_ilp:>10.1f} {trace_row_acts:>10} {error:>9.1f}%")
        
        results.append({
            'name': name,
            'macs': macs,
            'ilp_input': ilp_row_acts['input'],
            'ilp_weight': ilp_row_acts['weight'],
            'ilp_output': ilp_row_acts['output'],
            'ilp_total': total_ilp,
            'trace_total': trace_row_acts,
            'error': error,
            'per_bank': trace_stats['per_bank_acts'],
        })
        
        # 恢复 name
        wl_params['name'] = name
    
    print("-" * 100)
    
    # 统计
    if results:
        errors = [r['error'] for r in results]
        print(f"\n统计:")
        print(f"  平均误差: {np.mean(errors):.2f}%")
        print(f"  最大误差: {np.max(errors):.2f}%")
        print(f"  最小误差: {np.min(errors):.2f}%")
        
        print(f"\n详细 per-bank 分布:")
        for r in results:
            print(f"  {r['name']}: {dict(r['per_bank'])}")
    
    print("\n" + "=" * 80)
    return results


if __name__ == "__main__":
    run_row_activation_validation()
