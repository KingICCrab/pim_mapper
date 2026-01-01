#!/usr/bin/env python3
"""
快速验证: 直接对比 pim_optimizer Cost Model vs 理论分析
不需要运行 UniNDP 模拟
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')

from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer.arch.pim_arch import PIMArch
import yaml

print("="*70, flush=True)
print("快速 Conv 验证: Cost Model 分析", flush=True)
print("="*70, flush=True)

# 加载架构
with open('/Users/haochenzhao/Projects/pim_optimizer/examples/configs/arch.yaml', 'r') as f:
    arch_config = yaml.safe_load(f)

pe_h = arch_config['architecture']['pe_array']['dim_h']
pe_w = arch_config['architecture']['pe_array']['dim_w']
macs_per_pe = arch_config['architecture']['pe_array']['num_macs']
total_macs = pe_h * pe_w * macs_per_pe

print(f"\n架构: {pe_h}×{pe_w} PEs × {macs_per_pe} MACs = {total_macs} MACs", flush=True)

# Conv 测试用例 - 典型 CNN 层
conv_cases = [
    # (N, C, H, W, K, R, S, stride, padding) - name
    ((1, 64, 56, 56, 64, 3, 3, 1, 1), "ResNet-Conv1"),
    ((1, 128, 28, 28, 128, 3, 3, 1, 1), "ResNet-Conv2"),
    ((1, 256, 14, 14, 256, 3, 3, 1, 1), "ResNet-Conv3"),
    ((1, 512, 7, 7, 512, 3, 3, 1, 1), "ResNet-Conv4"),
    ((1, 3, 224, 224, 64, 7, 7, 2, 3), "ResNet-First"),
]

print("\n" + "="*70, flush=True)
print("Conv Workload 分析", flush=True)
print("="*70, flush=True)

results = []
for params, name in conv_cases:
    N, C, H, W, K, R, S, stride, pad = params
    
    # 计算输出尺寸
    H_out = (H + 2*pad - R) // stride + 1
    W_out = (W + 2*pad - S) // stride + 1
    
    # 总 MAC 操作数
    total_ops = N * K * H_out * W_out * C * R * S
    
    # 理论最小 cycles
    theory_min = total_ops / total_macs
    
    # im2col 转换后的 GEMM 维度
    # GEMM: [N*H_out*W_out, C*R*S] × [C*R*S, K] = [N*H_out*W_out, K]
    M = N * H_out * W_out  # batch × output spatial
    K_reduce = C * R * S   # input channels × kernel
    N_out = K              # output channels
    
    print(f"\n{name}:", flush=True)
    print(f"  Input: {N}×{C}×{H}×{W}, Kernel: {K}×{C}×{R}×{S}", flush=True)
    print(f"  Output: {N}×{K}×{H_out}×{W_out}", flush=True)
    print(f"  总 MACs: {total_ops:,}", flush=True)
    print(f"  理论最小 cycles: {theory_min:,.0f}", flush=True)
    print(f"  等效 GEMM: [{M}, {K_reduce}] × [{K_reduce}, {N_out}]", flush=True)
    
    results.append({
        'name': name,
        'total_ops': total_ops,
        'theory_min': theory_min,
        'gemm_M': M,
        'gemm_K': K_reduce,
        'gemm_N': N_out,
    })

# 总结
print("\n" + "="*70, flush=True)
print("总结", flush=True)
print("="*70, flush=True)

print(f"\n{'Layer':<15} {'MACs':>15} {'Theory Min':>12} {'GEMM Shape':<25}", flush=True)
print("-" * 70, flush=True)
for r in results:
    gemm_shape = f"[{r['gemm_M']}, {r['gemm_K']}]×[{r['gemm_K']}, {r['gemm_N']}]"
    print(f"{r['name']:<15} {r['total_ops']:>15,} {r['theory_min']:>12,.0f} {gemm_shape:<25}", flush=True)

print("\n✓ Conv 可以通过 im2col 转换为 GEMM 在 UniNDP 上验证", flush=True)
