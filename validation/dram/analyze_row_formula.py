"""
验证公式: base_per_C = H_switches × (Q_l3 × S_l2)

其中 H_switches 是 H 方向的 unique block switches

观察:
- ResNet-L3: H_sw × Q×S = 8 × 56 = 448 ✓
- VGG-L1:    H_sw × Q×S = 252 × 224 = 56448 ✓
- ResNet-L1: H_sw × Q×S = 8 × 7 = 56 ✗ (actual=448)
- ResNet-L2: H_sw × Q×S = 1 × 7 = 7 ✗ (actual=812)

问题: 有些 case 不匹配

让我分析为什么 ResNet-L1 不匹配
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
import math

DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6


def analyze_row_formula(name, workload, dram, l2, buf, block_h, block_w, stride_h, stride_w, dilation_h, dilation_w):
    """分析 row = h_block × num_w_blocks + w_block 的公式."""
    
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
    
    H_step = P_tile * stride_h
    W_step = Q_tile * stride_w
    R_step = R_tile * dilation_h
    S_step = S_tile * dilation_w
    
    H_in = workload.input_size['H']
    W_in = workload.input_size['W']
    
    num_h_blocks = (H_in + block_h - 1) // block_h
    num_w_blocks = (W_in + block_w - 1) // block_w
    
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"num_h_blocks={num_h_blocks}, num_w_blocks={num_w_blocks}")
    print(f"Loops: P={P_l3}, Q={Q_l3}, R={R_l2}, S={S_l2}")
    print(f"Step: H={H_step}, W={W_step}, R={R_step}, S={S_step}")
    print(f"Tile: H={H_per_tile}, W={W_per_tile}")
    print(f"Block: {block_h} × {block_w}")
    
    # 完整计算
    prev_row = -1
    full_switches = 0
    row_sequence = []
    
    for p in range(P_l3):
        for q in range(Q_l3):
            for r in range(R_l2):
                for s in range(S_l2):
                    h_start = p * H_step + r * R_step
                    w_start = q * W_step + s * S_step
                    
                    h_end = h_start + H_per_tile - 1
                    w_end = w_start + W_per_tile - 1
                    
                    h_blocks = list(range(h_start // block_h, h_end // block_h + 1))
                    w_blocks = list(range(w_start // block_w, w_end // block_w + 1))
                    
                    rows = set()
                    for hb in h_blocks:
                        for wb in w_blocks:
                            row = hb * num_w_blocks + wb
                            rows.add(row)
                    
                    for row in sorted(rows):
                        if row != prev_row:
                            full_switches += 1
                            row_sequence.append(row)
                            prev_row = row
    
    # 分析 row_sequence
    unique_rows = len(set(row_sequence))
    total_switches = len(row_sequence)
    
    print(f"\nRow sequence analysis:")
    print(f"  Total switches: {total_switches}")
    print(f"  Unique rows visited: {unique_rows}")
    print(f"  Total possible rows: {num_h_blocks * num_w_blocks}")
    print(f"  Switches / unique: {total_switches / unique_rows:.1f}")
    
    # 分析: 为什么有些 row 被访问多次?
    # 因为循环顺序是 P -> Q -> R -> S
    # 每次 Q 变化时,可能访问相同的 row
    
    # 尝试分解公式:
    # row = h_block × num_w_blocks + w_block
    # 
    # 当 Q 循环时, w_block 变化
    # 当 P 循环时, h_block 变化
    # 
    # 每次 h_block 变化,所有 w_block 都会被访问一遍
    
    # 统计每个 h_block 被访问多少次
    h_block_visits = {}
    for p in range(P_l3):
        for r in range(R_l2):
            h_start = p * H_step + r * R_step
            h_end = h_start + H_per_tile - 1
            h_blocks = list(range(h_start // block_h, h_end // block_h + 1))
            for hb in h_blocks:
                h_block_visits[hb] = h_block_visits.get(hb, 0) + 1
    
    print(f"\nH block visits: {h_block_visits}")
    print(f"Total H block visits: {sum(h_block_visits.values())}")
    
    # 统计每个 w_block 被访问多少次  
    w_block_visits = {}
    for q in range(Q_l3):
        for s in range(S_l2):
            w_start = q * W_step + s * S_step
            w_end = w_start + W_per_tile - 1
            w_blocks = list(range(w_start // block_w, w_end // block_w + 1))
            for wb in w_blocks:
                w_block_visits[wb] = w_block_visits.get(wb, 0) + 1
    
    print(f"W block visits: {w_block_visits}")
    print(f"Total W block visits: {sum(w_block_visits.values())}")
    
    # 公式: switches = H_visits × W_visits / 某种因子?
    h_total = sum(h_block_visits.values())
    w_total = sum(w_block_visits.values())
    
    print(f"\n公式验证:")
    print(f"  H_visits × W_visits = {h_total} × {w_total} = {h_total * w_total}")
    print(f"  H_visits × Q×S = {h_total} × {Q_l3 * S_l2} = {h_total * Q_l3 * S_l2}")
    print(f"  W_visits × P×R = {w_total} × {P_l3 * R_l2} = {w_total * P_l3 * R_l2}")
    print(f"  Actual switches: {total_switches}")


def main():
    test_workloads = [
        ("ResNet-L1", ConvWorkload(name="ResNet-L1", R=7, S=7, P=56, Q=56, C=3, K=64, N=1)),
        ("ResNet-L2", ConvWorkload(name="ResNet-L2", R=3, S=3, P=56, Q=56, C=64, K=64, N=1)),
        ("ResNet-L3", ConvWorkload(name="ResNet-L3", R=1, S=1, P=56, Q=56, C=64, K=256, N=1)),
        ("VGG-L1", ConvWorkload(name="VGG-L1", R=3, S=3, P=224, Q=224, C=3, K=64, N=1)),
    ]
    
    optimizer = PIMOptimizer()
    
    for name, workload in test_workloads:
        result = optimizer.optimize([workload], objective="latency")
        mapping = result.mappings[0]
        
        dram = {d: 1 for d in range(7)}
        if 3 in mapping.loop_bounds:
            for key in ['spatial', 'temporal']:
                if key in mapping.loop_bounds[3]:
                    for d, bound in mapping.loop_bounds[3][key].items():
                        dram[d] *= bound
        
        l2 = {d: 1 for d in range(7)}
        if 2 in mapping.loop_bounds:
            for key in ['spatial', 'temporal']:
                if key in mapping.loop_bounds[2]:
                    for d, bound in mapping.loop_bounds[2][key].items():
                        l2[d] *= bound
        
        buf = {d: 1 for d in range(7)}
        for level in [0, 1]:
            if level not in mapping.loop_bounds:
                continue
            level_bounds = mapping.loop_bounds[level]
            if level == 0:
                for key in ['H', 'W', 'Internal', 'temporal']:
                    if key in level_bounds:
                        for d, bound in level_bounds[key].items():
                            buf[d] *= bound
            else:
                for key in ['spatial', 'temporal']:
                    if key in level_bounds:
                        for d, bound in level_bounds[key].items():
                            buf[d] *= bound
        
        block_h = mapping.tile_info.get('block_h', 31)
        block_w = mapping.tile_info.get('block_w', 31)
        stride_h = workload.stride[0] if hasattr(workload, 'stride') else 1
        stride_w = workload.stride[1] if hasattr(workload, 'stride') else 1
        dilation_h = workload.dilation[0] if hasattr(workload, 'dilation') else 1
        dilation_w = workload.dilation[1] if hasattr(workload, 'dilation') else 1
        
        analyze_row_formula(name, workload, dram, l2, buf, block_h, block_w,
                           stride_h, stride_w, dilation_h, dilation_w)


if __name__ == "__main__":
    main()
