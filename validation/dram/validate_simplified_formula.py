"""
验证简化公式：

total = base_switches_per_C × C × K

其中 base_switches_per_C 只与 (P, Q, R, S) 相关
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload

DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6


def extract_params(mapping, workload):
    dram_factors = {d: 1 for d in range(7)}
    if 3 in mapping.loop_bounds:
        for key in ['spatial', 'temporal']:
            if key in mapping.loop_bounds[3]:
                for d, bound in mapping.loop_bounds[3][key].items():
                    dram_factors[d] *= bound
    
    level2_factors = {d: 1 for d in range(7)}
    if 2 in mapping.loop_bounds:
        for key in ['spatial', 'temporal']:
            if key in mapping.loop_bounds[2]:
                for d, bound in mapping.loop_bounds[2][key].items():
                    level2_factors[d] *= bound
    
    buffer_tile = {d: 1 for d in range(7)}
    for level in [0, 1]:
        if level not in mapping.loop_bounds:
            continue
        level_bounds = mapping.loop_bounds[level]
        if level == 0:
            for key in ['H', 'W', 'Internal', 'temporal']:
                if key in level_bounds:
                    for d, bound in level_bounds[key].items():
                        buffer_tile[d] *= bound
        else:
            for key in ['spatial', 'temporal']:
                if key in level_bounds:
                    for d, bound in level_bounds[key].items():
                        buffer_tile[d] *= bound
    
    block_h = mapping.tile_info.get('block_h', 31)
    block_w = mapping.tile_info.get('block_w', 31)
    
    stride_h = workload.stride[0] if hasattr(workload, 'stride') else 1
    stride_w = workload.stride[1] if hasattr(workload, 'stride') else 1
    dilation_h = workload.dilation[0] if hasattr(workload, 'dilation') else 1
    dilation_w = workload.dilation[1] if hasattr(workload, 'dilation') else 1
    
    return dram_factors, level2_factors, buffer_tile, block_h, block_w, stride_h, stride_w, dilation_h, dilation_w


def compute_base_switches_per_C(dram, l2, buf, block_h, block_w, stride_h, stride_w, dilation_h, dilation_w):
    """只遍历 P, Q, R, S，计算单个 channel 内的 switches."""
    
    P_l3 = dram[DIM_P]
    Q_l3 = dram[DIM_Q]
    R_l2 = l2[DIM_R]
    S_l2 = l2[DIM_S]
    
    P_tile = buf[DIM_P]
    Q_tile = buf[DIM_Q]
    R_tile = buf[DIM_R]
    S_tile = buf[DIM_S]
    
    H_per_tile = (P_tile - 1) * stride_h + R_tile * dilation_h
    W_per_tile = (Q_tile - 1) * stride_w + S_tile * dilation_w
    
    # 单个 channel 内的 row 编号 = h_block * num_w_blocks + w_block
    # num_w_blocks 需要估算
    num_w_blocks = 2  # 假设，实际应该计算
    
    prev_row = -1
    switches = 0
    
    for p in range(P_l3):
        for q in range(Q_l3):
            for r in range(R_l2):
                for s in range(S_l2):
                    h_start = p * P_tile * stride_h + r * R_tile * dilation_h
                    w_start = q * Q_tile * stride_w + s * S_tile * dilation_w
                    
                    h_end = h_start + H_per_tile - 1
                    w_end = w_start + W_per_tile - 1
                    
                    h_blocks = list(range(h_start // block_h, (h_end // block_h) + 1))
                    w_blocks = list(range(w_start // block_w, (w_end // block_w) + 1))
                    
                    rows = set()
                    for hb in h_blocks:
                        for wb in w_blocks:
                            row = hb * num_w_blocks + wb  # 单 channel 内
                            rows.add(row)
                    
                    for row in sorted(rows):
                        if row != prev_row:
                            switches += 1
                            prev_row = row
    
    return switches


def validate():
    test_workloads = [
        ("ResNet-L1", ConvWorkload(name="ResNet-L1", R=7, S=7, P=56, Q=56, C=3, K=64, N=1)),
        ("ResNet-L2", ConvWorkload(name="ResNet-L2", R=3, S=3, P=56, Q=56, C=64, K=64, N=1)),
        ("ResNet-L3", ConvWorkload(name="ResNet-L3", R=1, S=1, P=56, Q=56, C=64, K=256, N=1)),
        ("VGG-L1", ConvWorkload(name="VGG-L1", R=3, S=3, P=224, Q=224, C=3, K=64, N=1)),
    ]
    
    optimizer = PIMOptimizer()
    
    print("=" * 100)
    print("验证简化公式: total = base_per_C × C × K")
    print("=" * 100)
    
    for name, workload in test_workloads:
        result = optimizer.optimize([workload], objective="latency")
        mapping = result.mappings[0]
        
        dram, l2, buf, block_h, block_w, stride_h, stride_w, dilation_h, dilation_w = extract_params(mapping, workload)
        
        C_l3 = dram[DIM_C]
        K_l3 = dram[DIM_K]
        
        # 计算完整 base_switches (包含 C)
        # 从之前的结果: base_switches = 1344 for ResNet-L1
        # 1344 / C_l3 = 1344 / 3 = 448
        
        base_per_C = compute_base_switches_per_C(dram, l2, buf, block_h, block_w, 
                                                  stride_h, stride_w, dilation_h, dilation_w)
        
        simplified = base_per_C * C_l3 * K_l3
        
        # 与之前的 full formula 对比
        # (从 validate_new_formula.py 复制 full 计算)
        full_result = compute_full_switches(dram, l2, buf, block_h, block_w,
                                            stride_h, stride_w, dilation_h, dilation_w)
        
        print(f"\n{name}:")
        print(f"  C={C_l3}, K={K_l3}")
        print(f"  base_per_C = {base_per_C}")
        print(f"  简化公式: {base_per_C} × {C_l3} × {K_l3} = {simplified}")
        print(f"  完整公式 (prev_row tracking): {full_result}")
        print(f"  匹配: {'✓' if simplified == full_result else '✗'}")


def compute_full_switches(dram, l2, buf, block_h, block_w, stride_h, stride_w, dilation_h, dilation_w):
    """完整的 prev_row tracking 计算 (包含 C 和 K)."""
    
    K_l3 = dram[DIM_K]
    C_l3 = dram[DIM_C]
    P_l3 = dram[DIM_P]
    Q_l3 = dram[DIM_Q]
    R_l2 = l2[DIM_R]
    S_l2 = l2[DIM_S]
    
    P_tile = buf[DIM_P]
    Q_tile = buf[DIM_Q]
    R_tile = buf[DIM_R]
    S_tile = buf[DIM_S]
    
    H_per_tile = (P_tile - 1) * stride_h + R_tile * dilation_h
    W_per_tile = (Q_tile - 1) * stride_w + S_tile * dilation_w
    
    num_h_blocks = 2  # 简化假设
    num_w_blocks = 2
    
    prev_row = -1
    switches = 0
    
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r in range(R_l2):
                        for s in range(S_l2):
                            h_start = p * P_tile * stride_h + r * R_tile * dilation_h
                            w_start = q * Q_tile * stride_w + s * S_tile * dilation_w
                            
                            h_end = h_start + H_per_tile - 1
                            w_end = w_start + W_per_tile - 1
                            
                            h_blocks = list(range(h_start // block_h, (h_end // block_h) + 1))
                            w_blocks = list(range(w_start // block_w, (w_end // block_w) + 1))
                            
                            rows = set()
                            for hb in h_blocks:
                                for wb in w_blocks:
                                    row = (c * num_h_blocks + hb) * num_w_blocks + wb
                                    rows.add(row)
                            
                            for row in sorted(rows):
                                if row != prev_row:
                                    switches += 1
                                    prev_row = row
    
    return switches


if __name__ == "__main__":
    validate()
