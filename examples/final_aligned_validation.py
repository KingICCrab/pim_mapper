#!/usr/bin/env python3
"""
完整的架构对齐验证

使用 pim-optimizer-aligned-v3.yaml 配置:
- 16 channels × 8 PUs × 16 MACs = 2048 MACs
- 等于 pim_optimizer 的 256 PEs × 8 MACs = 2048 MACs
"""

import subprocess
import os

UNINDP_DIR = '/Users/haochenzhao/Projects/UniNDP'
CONFIG_FILE = 'config/pim-optimizer-aligned-v3.yaml'

print("="*70, flush=True)
print("对齐配置验证: pim_optimizer vs UniNDP", flush=True)
print("="*70, flush=True)

# 确保使用对齐配置
os.system(f'cd {UNINDP_DIR} && cp {CONFIG_FILE} config/hbm-pim.yaml')

print(f"\n配置: {CONFIG_FILE}", flush=True)
print("  16 channels × 8 PUs × 16 MACs = 2048 MACs", flush=True)
print("  = pim_optimizer: 256 PEs × 8 MACs", flush=True)

# 测试用例
test_cases = [
    (64, 64, 64, 1),
    (128, 128, 128, 1),
    (256, 256, 256, 1),
    (32, 512, 32, 1),    # 窄矩阵
    (512, 32, 512, 1),   # 宽矩阵
]

print("\n" + "="*70, flush=True)
print("运行测试", flush=True)
print("="*70, flush=True)

results = []
total_macs = 2048

for M, K, N, B in test_cases:
    name = f"aligned_{M}x{K}x{N}"
    cmd = f"cd {UNINDP_DIR} && python compile.py -A hbm-pim -W mm -S {M} {K} {N} {B} -N {name} -WS test_workspace -O test 2>/dev/null"
    
    print(f"\n测试: GEMM {M}×{K}×{N}...", flush=True)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # 读取 CSV 结果
    csv_file = f"{UNINDP_DIR}/test_workspace/test/csv/_{name}.csv"
    try:
        with open(csv_file, 'r') as f:
            line = f.readline().strip()
            parts = line.split(',')
            # CSV: name, workload, best_result, inst_num, pu_dram_num, host_dram_num, row_change_num, baseline, ...
            best_cycles = int(parts[2])  # best_result
            baseline_cycles = int(parts[7])  # baseline_sim_result
            
            total_ops = M * K * N * B
            theory_min = total_ops / total_macs
            utilization = (total_ops / (best_cycles * total_macs)) * 100
            speedup = baseline_cycles / best_cycles
            
            print(f"  总操作数: {total_ops:,}", flush=True)
            print(f"  理论最小: {theory_min:,.0f} cycles", flush=True)
            print(f"  最优 cycles: {best_cycles:,}", flush=True)
            print(f"  基线 cycles: {baseline_cycles:,}", flush=True)
            print(f"  优化加速比: {speedup:.2f}x", flush=True)
            print(f"  硬件利用率: {utilization:.1f}%", flush=True)
            
            results.append({
                'workload': f'{M}×{K}×{N}',
                'total_ops': total_ops,
                'theory_min': theory_min,
                'best_cycles': best_cycles,
                'baseline_cycles': baseline_cycles,
                'speedup': speedup,
                'utilization': utilization
            })
    except Exception as e:
        print(f"  错误: {e}", flush=True)

# 总结
print("\n" + "="*70, flush=True)
print("总结", flush=True)
print("="*70, flush=True)

print(f"\n{'Workload':<15} {'Ops':>12} {'Theory':>8} {'Best':>8} {'Base':>8} {'Speedup':>8} {'Util%':>8}", flush=True)
print("-" * 75, flush=True)
for r in results:
    print(f"{r['workload']:<15} {r['total_ops']:>12,} {r['theory_min']:>8,.0f} {r['best_cycles']:>8,} {r['baseline_cycles']:>8,} {r['speedup']:>8.2f}x {r['utilization']:>7.1f}%", flush=True)

if results:
    avg_util = sum(r['utilization'] for r in results) / len(results)
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    print("-" * 75, flush=True)
    print(f"{'平均':<15} {'':<12} {'':<8} {'':<8} {'':<8} {avg_speedup:>8.2f}x {avg_util:>7.1f}%", flush=True)

print("\n" + "="*70, flush=True)
print("架构对齐验证完成!", flush=True)
print("="*70, flush=True)
