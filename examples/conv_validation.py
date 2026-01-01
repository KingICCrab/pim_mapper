#!/usr/bin/env python3
"""
Conv 到 GEMM 转换验证

Conv 通过 im2col 转换为 GEMM:
- Conv: [N, C, H, W] * [K, C, R, S] -> [N, K, P, Q]
- GEMM (im2col): [N*P*Q, C*R*S] × [C*R*S, K] -> [N*P*Q, K]

即:
- M = N * P * Q (batch × output spatial)
- K_gemm = C * R * S (input channels × kernel size)
- N_gemm = K (output channels)
"""

import subprocess
import os

UNINDP_DIR = '/Users/haochenzhao/Projects/UniNDP'
CONFIG_FILE = 'config/pim-optimizer-aligned-v3.yaml'

print("="*70, flush=True)
print("Conv → GEMM 转换验证 (im2col)", flush=True)
print("="*70, flush=True)

# 使用对齐配置
os.system(f'cd {UNINDP_DIR} && cp {CONFIG_FILE} config/hbm-pim.yaml')

print(f"\n配置: {CONFIG_FILE}", flush=True)
print("  16 channels × 8 PUs × 16 MACs = 2048 MACs", flush=True)

# Conv 测试用例 (常见 CNN layers)
# (name, N, C, H, W, K, R, S, stride, padding)
conv_layers = [
    # ResNet-50 layers
    ("ResNet50_conv1", 1, 3, 224, 224, 64, 7, 7, 2),     # 第一层 7×7 conv
    ("ResNet50_conv2", 1, 64, 56, 56, 64, 3, 3, 1),      # 3×3 conv
    ("ResNet50_conv3", 1, 64, 56, 56, 256, 1, 1, 1),     # 1×1 conv
    ("ResNet50_conv4", 1, 256, 14, 14, 512, 3, 3, 1),    # 3×3 conv
    ("ResNet50_conv5", 1, 512, 7, 7, 2048, 1, 1, 1),     # 1×1 conv
    
    # VGG-like layers
    ("VGG_conv", 1, 64, 112, 112, 128, 3, 3, 1),
    
    # MobileNet-like depthwise separable
    ("MobileNet_pw", 1, 32, 112, 112, 64, 1, 1, 1),      # pointwise 1×1
]

print("\n" + "="*70, flush=True)
print("Conv → GEMM 转换", flush=True)
print("="*70, flush=True)

def conv_to_gemm(N, C, H, W, K, R, S, stride):
    """将 Conv 参数转换为等效 GEMM 参数"""
    # 输出尺寸 (假设 same padding)
    P = (H + stride - 1) // stride  # output height
    Q = (W + stride - 1) // stride  # output width
    
    # GEMM 维度 (im2col)
    M = N * P * Q       # batch × output spatial
    K_gemm = C * R * S  # input channels × kernel
    N_gemm = K          # output channels
    
    return M, K_gemm, N_gemm, P, Q

results = []
total_macs = 2048

for layer in conv_layers:
    name, N, C, H, W, K, R, S, stride = layer
    
    # 转换为 GEMM
    M, K_gemm, N_gemm, P, Q = conv_to_gemm(N, C, H, W, K, R, S, stride)
    
    # Conv MACs = N × K × P × Q × C × R × S
    conv_macs = N * K * P * Q * C * R * S
    # GEMM MACs = M × K_gemm × N_gemm  
    gemm_macs = M * K_gemm * N_gemm
    
    print(f"\n{name}:", flush=True)
    print(f"  Conv: N={N}, C={C}, H={H}, W={W}, K={K}, R={R}, S={S}, stride={stride}", flush=True)
    print(f"  Output: P={P}, Q={Q}", flush=True)
    print(f"  → GEMM: M={M}, K={K_gemm}, N={N_gemm}", flush=True)
    print(f"  Conv MACs: {conv_macs:,} = GEMM MACs: {gemm_macs:,}", flush=True)
    
    # 运行 UniNDP 模拟
    test_name = f"conv_{name}"
    cmd = f"cd {UNINDP_DIR} && python compile.py -A hbm-pim -W mm -S {M} {K_gemm} {N_gemm} 1 -N {test_name} -WS test_workspace -O test 2>/dev/null"
    
    subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # 读取结果
    csv_file = f"{UNINDP_DIR}/test_workspace/test/csv/_{test_name}.csv"
    try:
        with open(csv_file, 'r') as f:
            line = f.readline().strip()
            parts = line.split(',')
            best_cycles = int(parts[2])
            baseline_cycles = int(parts[7])
            
            theory_min = gemm_macs / total_macs
            utilization = (gemm_macs / (best_cycles * total_macs)) * 100
            
            print(f"  理论最小: {theory_min:,.0f} cycles", flush=True)
            print(f"  实际最优: {best_cycles:,} cycles", flush=True)
            print(f"  利用率: {utilization:.1f}%", flush=True)
            
            results.append({
                'name': name,
                'conv_shape': f"{N}×{C}×{H}×{W}→{K}×{R}×{S}",
                'gemm_shape': f"{M}×{K_gemm}×{N_gemm}",
                'macs': gemm_macs,
                'theory': theory_min,
                'cycles': best_cycles,
                'baseline': baseline_cycles,
                'util': utilization
            })
    except Exception as e:
        print(f"  错误: {e}", flush=True)

# 总结
print("\n" + "="*70, flush=True)
print("Conv 验证总结", flush=True)
print("="*70, flush=True)

print(f"\n{'Layer':<20} {'GEMM Shape':<20} {'MACs':>12} {'Cycles':>10} {'Util%':>8}", flush=True)
print("-" * 75, flush=True)
for r in results:
    print(f"{r['name']:<20} {r['gemm_shape']:<20} {r['macs']:>12,} {r['cycles']:>10,} {r['util']:>7.1f}%", flush=True)

if results:
    avg_util = sum(r['util'] for r in results) / len(results)
    print("-" * 75, flush=True)
    print(f"平均利用率: {avg_util:.1f}%", flush=True)

# 恢复原配置
os.system(f'cd {UNINDP_DIR} && cp config/hbm-pim.yaml.bak config/hbm-pim.yaml 2>/dev/null')

print("\n" + "="*70, flush=True)
print("Conv 验证完成!", flush=True)
print("="*70, flush=True)
