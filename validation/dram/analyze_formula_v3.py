"""
发现的模式:
- VGG-L1: 4032 = 18 × 224 = H_tile_blocks × Q_l3
- ResNet-L3: 224 = 4 × 56 = H_tile_blocks × Q_l3

让我验证这个公式: switches_per_p = H_tile_span × Q_l3 × S_l2
其中 H_tile_span = ceil((H_tile + H_step-1) / block_h) 或类似的东西
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
import math

DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6


def compute_formula(name, workload, dram, l2, buf, block_h, block_w, stride_h, stride_w, dilation_h, dilation_w):
    """验证 switches_per_p 的公式."""
    
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
    print(f"Loops: P={P_l3}, Q={Q_l3}, R={R_l2}, S={S_l2}")
    print(f"Step: H={H_step}, W={W_step}, R={R_step}, S={S_step}")
    print(f"Tile span: H={H_per_tile}, W={W_per_tile}")
    print(f"Blocks: {num_h_blocks} × {num_w_blocks}, size={block_h}×{block_w}")
    
    # 计算实际 switches
    total_switches = 0
    prev_row = -1
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
                    
                    for hb in h_blocks:
                        for wb in w_blocks:
                            row = hb * num_w_blocks + wb
                            if row != prev_row:
                                total_switches += 1
                                prev_row = row
    
    print(f"\nActual total switches: {total_switches}")
    
    # 验证公式 1: 基于 unique rows
    # 计算访问的 unique rows
    unique_rows = set()
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
                    
                    for hb in h_blocks:
                        for wb in w_blocks:
                            row = hb * num_w_blocks + wb
                            unique_rows.add(row)
    
    num_unique_rows = len(unique_rows)
    print(f"Unique rows accessed: {num_unique_rows}")
    
    # 计算每个 iteration 访问的 blocks 数
    # tile 可能跨越的 H blocks 数
    h_tile_blocks = (H_per_tile + block_h - 1) // block_h
    w_tile_blocks = (W_per_tile + block_w - 1) // block_w
    
    print(f"\nTile may span: H={h_tile_blocks}, W={w_tile_blocks} blocks")
    
    # 计算 H 方向: 一个 P 内, R 循环会访问的 total H range
    h_range_per_p = H_per_tile + (R_l2 - 1) * R_step
    h_blocks_per_p = (h_range_per_p + block_h - 1) // block_h
    
    # 计算 W 方向: 一个 Q 内, S 循环会访问的 total W range  
    w_range_per_q = W_per_tile + (S_l2 - 1) * S_step
    w_blocks_per_q = (w_range_per_q + block_w - 1) // block_w
    
    print(f"Range per P (with R): H_range={h_range_per_p}, blocks={h_blocks_per_p}")
    print(f"Range per Q (with S): W_range={w_range_per_q}, blocks={w_blocks_per_q}")
    
    # 计算 H 方向 unique switches (考虑 block 边界对齐)
    # 每个 p 开始的 h_start = p * H_step
    # 在这个 p 内, r 从 0 到 R_l2-1
    # h 值范围: [p*H_step, p*H_step + h_range_per_p - 1]
    # 这会跨越几个 h_block?
    
    # 更准确: 统计整个 P 循环中 unique h_block configurations
    h_unique_sequences = []
    for p in range(P_l3):
        p_h_blocks = set()
        for r in range(R_l2):
            h_start = p * H_step + r * R_step
            h_end = h_start + H_per_tile - 1
            for hb in range(h_start // block_h, h_end // block_h + 1):
                p_h_blocks.add(hb)
        h_unique_sequences.append(len(p_h_blocks))
    
    h_total_visits = sum(h_unique_sequences)
    print(f"H blocks per p: {h_unique_sequences[:10]}..., sum={h_total_visits}")
    
    w_unique_sequences = []
    for q in range(Q_l3):
        q_w_blocks = set()
        for s in range(S_l2):
            w_start = q * W_step + s * S_step
            w_end = w_start + W_per_tile - 1
            for wb in range(w_start // block_w, w_end // block_w + 1):
                q_w_blocks.add(wb)
        w_unique_sequences.append(len(q_w_blocks))
    
    w_total_visits = sum(w_unique_sequences)
    print(f"W blocks per q: {w_unique_sequences[:10]}..., sum={w_total_visits}")
    
    # 尝试公式: total = H_total_visits × W_total_visits
    predict_1 = h_total_visits * w_total_visits
    
    # 尝试公式: total = sum(H_blocks[p]) × Q × S
    predict_2 = h_total_visits * Q_l3 * S_l2
    
    # 尝试公式: total = P × R × sum(W_blocks[q])
    predict_3 = P_l3 * R_l2 * w_total_visits
    
    print(f"\n公式验证:")
    print(f"  Actual: {total_switches}")
    print(f"  H_total × W_total = {h_total_visits} × {w_total_visits} = {predict_1}")
    print(f"  H_total × Q × S = {h_total_visits} × {Q_l3} × {S_l2} = {predict_2}")  
    print(f"  P × R × W_total = {P_l3} × {R_l2} × {w_total_visits} = {predict_3}")
    
    # 最有希望的公式: unique_rows × average_visits_per_row
    if num_unique_rows > 0:
        avg_visits = total_switches / num_unique_rows
        print(f"  Unique rows × avg visits = {num_unique_rows} × {avg_visits:.1f} = {total_switches}")
    
    return total_switches


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
        
        compute_formula(name, workload, dram, l2, buf, block_h, block_w,
                       stride_h, stride_w, dilation_h, dilation_w)


if __name__ == "__main__":
    main()
