"""
完善 base_per_C 解析公式

发现:
- ResNet-L3: tiles × blocks_per_tile = 112 × 4 = 448 ✓
- VGG-L1:    tiles × blocks_per_tile = 3136 × 18 = 56448 ✓
- ResNet-L1: 估计 30, 实际 448 (差很多)
- ResNet-L2: 估计 980, 实际 812

问题出在:
1. blocks_per_tile 计算不准确 (应该考虑 crossing)
2. tiles_per_block 的共享模型不对

让我重新分析
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
import math

DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6


def compute_h_direction_switches(P_l3, R_l2, H_step, R_step, H_tile, block_h):
    """
    计算 H 方向的 switches (row block 变化次数)
    
    访问 h 位置: h_start = p * H_step + r * R_step
    访问 h_block: h_start // block_h
    
    当 h_block 变化时,就有一次 switch
    """
    prev_h_block = -1
    switches = 0
    
    for p in range(P_l3):
        for r in range(R_l2):
            h_start = p * H_step + r * R_step
            h_block = h_start // block_h
            
            # 检查是否 crossing
            h_end = h_start + H_tile - 1
            h_block_end = h_end // block_h
            num_h_blocks = h_block_end - h_block + 1
            
            # 每个访问的 h_block 都可能是一次 switch
            for hb in range(h_block, h_block_end + 1):
                if hb != prev_h_block:
                    switches += 1
                    prev_h_block = hb
    
    return switches


def compute_w_direction_switches(Q_l3, S_l2, W_step, S_step, W_tile, block_w):
    """计算 W 方向的 switches."""
    prev_w_block = -1
    switches = 0
    
    for q in range(Q_l3):
        for s in range(S_l2):
            w_start = q * W_step + s * S_step
            w_block = w_start // block_w
            
            w_end = w_start + W_tile - 1
            w_block_end = w_end // block_w
            
            for wb in range(w_block, w_block_end + 1):
                if wb != prev_w_block:
                    switches += 1
                    prev_w_block = wb
    
    return switches


def analyze_decomposed(name, workload, dram, l2, buf, block_h, block_w, stride_h, stride_w, dilation_h, dilation_w):
    """分解成 H 和 W 方向分别分析."""
    
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
    
    # 单独计算 H 和 W 方向
    h_switches = compute_h_direction_switches(P_l3, R_l2, H_step, R_step, H_per_tile, block_h)
    w_switches = compute_w_direction_switches(Q_l3, S_l2, W_step, S_step, W_per_tile, block_w)
    
    # 完整 2D 计算
    prev_row = -1
    full_switches = 0
    
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
                            prev_row = row
    
    total_tiles = P_l3 * Q_l3 * R_l2 * S_l2
    
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"Loops: P={P_l3}, Q={Q_l3}, R={R_l2}, S={S_l2}, Total tiles={total_tiles}")
    print(f"Tile: H={H_per_tile}, W={W_per_tile}")
    print(f"Block: {block_h} × {block_w}")
    print(f"Step: H={H_step}, W={W_step}, R={R_step}, S={S_step}")
    print(f"\n方向分解:")
    print(f"  H direction switches: {h_switches}")
    print(f"  W direction switches: {w_switches}")
    print(f"  H × W (product): {h_switches * w_switches}")
    print(f"  Full 2D switches: {full_switches}")
    print(f"  Ratio H×W / Full: {h_switches * w_switches / full_switches:.2f}")
    
    # 公式假设: switches = H_switches × W_switches / num_w_blocks
    # 因为 H 每变化一次，会遍历所有 W
    
    formula_1 = h_switches * w_switches
    formula_2 = h_switches * Q_l3 * S_l2  # H switches × W iterations
    
    print(f"\n公式尝试:")
    print(f"  公式1: H_sw × W_sw = {formula_1}")
    print(f"  公式2: H_sw × (Q×S) = {h_switches} × {Q_l3 * S_l2} = {formula_2}")
    print(f"  Actual: {full_switches}")
    
    return full_switches, h_switches, w_switches


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
        
        analyze_decomposed(name, workload, dram, l2, buf, block_h, block_w,
                          stride_h, stride_w, dilation_h, dilation_w)


if __name__ == "__main__":
    main()
