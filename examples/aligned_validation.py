#!/usr/bin/env python3
"""
完整的架构对齐验证

使用 pim-optimizer-aligned-v3.yaml 配置:
- 16 channels × 8 PUs × 16 MACs = 2048 MACs
- 等于 pim_optimizer 的 256 PEs × 8 MACs = 2048 MACs
"""

import subprocess
import os
import re
import sys

def log(msg):
    print(msg, flush=True)

UNINDP_DIR = '/Users/haochenzhao/Projects/UniNDP'
CONFIG_FILE = 'config/pim-optimizer-aligned-v3.yaml'

log("="*70)
log("对齐配置验证: pim_optimizer vs UniNDP")
log("="*70)

# 确保使用对齐配置
os.system(f'cd {UNINDP_DIR} && cp {CONFIG_FILE} config/hbm-pim.yaml')

print(f"\n配置: {CONFIG_FILE}")
print("  16 channels × 8 PUs × 16 MACs = 2048 MACs")
print("  等于 pim_optimizer: 256 PEs × 8 MACs = 2048 MACs")

# 测试用例
test_cases = [
    (64, 64, 64, 1),
    (128, 128, 128, 1),
    (256, 256, 256, 1),
    (512, 512, 512, 1),
    (32, 512, 32, 1),    # 不同形状
    (512, 32, 512, 1),   # 不同形状
]

print("\n" + "="*70)
print("运行测试")
print("="*70)

results = []
total_macs = 2048

for M, K, N, B in test_cases:
    name = f"test_{M}x{K}x{N}"
    cmd = f"cd {UNINDP_DIR} && python compile.py -A hbm-pim -W mm -S {M} {K} {N} {B} -N {name} -WS test_workspace -O test 2>&1"
    
    print(f"\n测试: GEMM {M}×{K}×{N}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # 读取 CSV 结果
    csv_file = f"{UNINDP_DIR}/test_workspace/test/csv/_{name}.csv"
    try:
        with open(csv_file, 'r') as f:
            line = f.readline().strip()
            parts = line.split(',')
            # CSV 格式: name,workload,?,?,?,?,?,cycles,...
            cycles = int(parts[7])  # 调整索引
            
            total_ops = M * K * N * B
            theory_min = total_ops / total_macs
            utilization = (total_ops / (cycles * total_macs)) * 100
            
            print(f"  总操作数: {total_ops}")
            print(f"  理论最小: {theory_min:.0f} cycles")
            print(f"  实际 cycles: {cycles}")
            print(f"  硬件利用率: {utilization:.1f}%")
            
            results.append({
                'workload': f'{M}×{K}×{N}',
                'total_ops': total_ops,
                'theory_min': theory_min,
                'cycles': cycles,
                'utilization': utilization
            })
    except Exception as e:
        print(f"  错误: {e}")

# 总结
print("\n" + "="*70)
print("总结")
print("="*70)

print(f"\n{'Workload':<15} {'Total Ops':<12} {'Theory':<10} {'Actual':<10} {'Util%':<8}")
print("-" * 55)
for r in results:
    print(f"{r['workload']:<15} {r['total_ops']:<12} {r['theory_min']:<10.0f} {r['cycles']:<10} {r['utilization']:<8.1f}")

avg_util = sum(r['utilization'] for r in results) / len(results) if results else 0
print("-" * 55)
print(f"平均利用率: {avg_util:.1f}%")

print("\n配置对齐完成！")
print(f"总 MACs: {total_macs} (= 256 PEs × 8 MACs = 16 ch × 8 PUs × 16 MACs)")
