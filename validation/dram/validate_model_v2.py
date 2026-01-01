#!/usr/bin/env python3
"""
验证简化 Row Activation 模型的有效性

关键问题：简化模型能否正确排序不同 mapping 的好坏？

方法：
1. 用 optimizer 生成多个不同的 mapping 
2. 用 Trace 计算真实的 row_switches
3. 用简化模型计算预测值
4. 比较排序的一致性
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/validation/dram')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
from trace_generator import TraceGenerator, DRAMConfig
import math

def count_row_activations_by_tensor(trace, bank_size, row_size):
    """Count row activations for input tensor (bank 0)."""
    input_base = 0
    last_row = None
    activations = 0
    rows_seen = set()
    
    for line in trace:
        if 'LD' in line or 'ST' in line:
            parts = line.split()
            addr = int(parts[1], 16)
            # Check if in input bank (bank 0)
            bank = addr // bank_size
            if bank == 0:
                row = (addr % bank_size) // row_size
                rows_seen.add(row)
                if last_row is not None and row != last_row:
                    activations += 1
                last_row = row
    
    return activations, len(rows_seen)


def model_plan_a(P_l3, Q_l3, C_l3, R_l2, K_l3, BC=0):
    """方案 A: (P × Q × C × R + BC) × K"""
    return (P_l3 * Q_l3 * C_l3 * R_l2 + BC * R_l2) * K_l3


def model_plan_b(H_in, W_in, block_h, block_w, P_l3, Q_l3, R_l2, S_l2, C_l3, K_l3):
    """方案 B: unique_rows × penalty × K"""
    num_h_blocks = math.ceil(H_in / block_h)
    num_w_blocks = math.ceil(W_in / block_w)
    unique_rows = num_h_blocks * num_w_blocks * C_l3
    
    # 简单的 penalty: 每个 block 被访问的次数
    accesses_h = P_l3 * R_l2
    accesses_w = Q_l3 * S_l2
    penalty = (accesses_h / num_h_blocks) * (accesses_w / num_w_blocks)
    
    return unique_rows * penalty * K_l3


def model_current_ilp(P_l3, Q_l3, C_l3, K_l3, BC=0):
    """当前 ILP: (P × Q × C + BC) × K (没有 R)"""
    return (P_l3 * Q_l3 * C_l3 + BC) * K_l3


def extract_loop_params(mapping):
    """从 mapping 中提取循环参数"""
    loop_bounds = mapping.loop_bounds
    tile_info = mapping.tile_info
    
    # Level 3 (DRAM) 的 temporal
    l3_temporal = loop_bounds.get(3, {}).get('temporal', {})
    P_l3 = l3_temporal.get(2, 1)  # P at level 3
    Q_l3 = l3_temporal.get(3, 1)  # Q at level 3
    C_l3 = l3_temporal.get(4, 1)  # C at level 3
    K_l3 = l3_temporal.get(5, 1)  # K at level 3
    
    # Level 2 (RowBuffer) 的 temporal
    l2_temporal = loop_bounds.get(2, {}).get('temporal', {})
    R_l2 = l2_temporal.get(0, 1)  # R at level 2
    S_l2 = l2_temporal.get(1, 1)  # S at level 2
    
    # Block size
    block_h = tile_info.get('block_h', 31)
    block_w = tile_info.get('block_w', 31)
    
    return {
        'P_l3': P_l3, 'Q_l3': Q_l3, 'C_l3': C_l3, 'K_l3': K_l3,
        'R_l2': R_l2, 'S_l2': S_l2,
        'block_h': block_h, 'block_w': block_w,
    }


print("=" * 70)
print("实验: 使用 Optimizer 生成 mapping 并验证模型")
print("=" * 70)

# ResNet Layer 1 workload
workload = ConvWorkload(name='resnet_l1', N=1, K=64, C=3, P=56, Q=56, R=7, S=7)

print(f"\nWorkload: {workload.name}")
print(f"  R={workload.R}, S={workload.S}, P={workload.P}, Q={workload.Q}")
print(f"  C={workload.C}, K={workload.K}")
print(f"  H_in={workload.input_size['H']}, W_in={workload.input_size['W']}")

# Optimize
print("\n正在优化 mapping...")
optimizer = PIMOptimizer()
result = optimizer.optimize([workload], objective='latency')
mapping = result.mappings[0]

# Print mapping info
print("\n=== Mapping 信息 ===")
print(f"Loop bounds: {mapping.loop_bounds}")
print(f"Tile info: {mapping.tile_info}")

params = extract_loop_params(mapping)
print(f"\n提取的参数:")
for k, v in params.items():
    print(f"  {k} = {v}")

# Generate trace
print("\n正在生成 Trace...")
gen = TraceGenerator(DRAMConfig())
trace = gen.generate_trace(mapping, workload)

# Count row activations
bank_size = 1024 * 16384  # 16MB per bank
row_size = 1024

trace_switches, trace_unique = count_row_activations_by_tensor(trace, bank_size, row_size)
print(f"\nTrace 结果:")
print(f"  row_switches: {trace_switches}")
print(f"  unique_rows: {trace_unique}")

# Calculate model predictions
H_in, W_in = workload.input_size['H'], workload.input_size['W']
block_h = params['block_h']
block_w = params['block_w']

current_ilp = model_current_ilp(params['P_l3'], params['Q_l3'], params['C_l3'], params['K_l3'])
plan_a = model_plan_a(params['P_l3'], params['Q_l3'], params['C_l3'], params['R_l2'], params['K_l3'])
plan_b = model_plan_b(H_in, W_in, block_h, block_w, 
                       params['P_l3'], params['Q_l3'], params['R_l2'], params['S_l2'], 
                       params['C_l3'], params['K_l3'])

print(f"\n模型预测:")
print(f"  Current ILP: {current_ilp}")
print(f"  Plan A: {plan_a}")
print(f"  Plan B: {plan_b:.0f}")

# Get ILP's prediction
if hasattr(mapping, 'metrics'):
    ilp_prediction = mapping.metrics.get('row_activations_input', 0)
    print(f"  ILP (from mapping): {ilp_prediction:.0f}")

print("\n" + "=" * 70)
print("对比分析")
print("=" * 70)

print(f"""
| 方法         | 预测值  | 与 Trace 比率 |
|-------------|--------|--------------|
| Trace       | {trace_switches:6d} | 1.00x        |
| Current ILP | {current_ilp:6d} | {current_ilp/trace_switches if trace_switches > 0 else 0:.2f}x        |
| Plan A      | {plan_a:6d} | {plan_a/trace_switches if trace_switches > 0 else 0:.2f}x        |
| Plan B      | {plan_b:6.0f} | {plan_b/trace_switches if trace_switches > 0 else 0:.2f}x        |
""")

print("""
关键发现:

1. 如果 Plan A/B 的比率接近 1.0 或成比例关系，说明模型有效
2. 重要的是排序准确性，不是绝对数值
3. 需要测试多个 mapping 来验证排序一致性
""")
