"""
深入分析 switches 的真正来源

关键观察:
- ResNet-L1: Switches/tiles = 0.33 (比 1 小!)
- VGG-L1: Switches/tiles = 18 (很大)

这说明:
- 当 tile > block 时，每个 tile 访问多个 blocks → switches > tiles
- 当 tile < block 时，多个 tiles 可以共享同一个 block → switches < tiles

让我们分析 H_tile/block_h 和 W_tile/block_w 的比例
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
import math

DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6


def analyze_pattern(name, workload, dram, l2, buf, block_h, block_w, stride_h, stride_w, dilation_h, dilation_w):
    """分析访问模式."""
    
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
    
    # Step size in input space
    H_step = P_tile * stride_h  # 每次 p++ 移动的距离
    W_step = Q_tile * stride_w  # 每次 q++ 移动的距离
    R_step = R_tile * dilation_h  # 每次 r++ 移动的距离
    S_step = S_tile * dilation_w  # 每次 s++ 移动的距离
    
    H_in = workload.input_size['H']
    W_in = workload.input_size['W']
    
    num_h_blocks = (H_in + block_h - 1) // block_h
    num_w_blocks = (W_in + block_w - 1) // block_w
    
    # prev_row tracking
    prev_row = -1
    switches = 0
    
    for p in range(P_l3):
        for q in range(Q_l3):
            for r in range(R_l2):
                for s in range(S_l2):
                    h_start = p * H_step + r * R_step
                    w_start = q * W_step + s * S_step
                    
                    h_end = h_start + H_per_tile - 1
                    w_end = w_start + W_per_tile - 1
                    
                    h_block_start = h_start // block_h
                    h_block_end = h_end // block_h
                    w_block_start = w_start // block_w
                    w_block_end = w_end // block_w
                    
                    h_blocks = list(range(h_block_start, h_block_end + 1))
                    w_blocks = list(range(w_block_start, w_block_end + 1))
                    
                    rows = set()
                    for hb in h_blocks:
                        for wb in w_blocks:
                            row = hb * num_w_blocks + wb
                            rows.add(row)
                    
                    for row in sorted(rows):
                        if row != prev_row:
                            switches += 1
                            prev_row = row
    
    total_tiles = P_l3 * Q_l3 * R_l2 * S_l2
    
    # 分析比例
    h_blocks_per_tile = math.ceil(H_per_tile / block_h)  # 每个 tile 覆盖多少 H blocks
    w_blocks_per_tile = math.ceil(W_per_tile / block_w)  # 每个 tile 覆盖多少 W blocks
    
    tiles_per_h_block = max(1, block_h // H_step) if H_step > 0 else 1  # 多少 tiles 共享一个 H block
    tiles_per_w_block = max(1, block_w // W_step) if W_step > 0 else 1  # 多少 tiles 共享一个 W block
    
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"Tile: H={H_per_tile}, W={W_per_tile}")
    print(f"Block: {block_h} × {block_w}")
    print(f"Step: H_step={H_step}, W_step={W_step}, R_step={R_step}, S_step={S_step}")
    print(f"\n比例分析:")
    print(f"  H_tile/block_h = {H_per_tile}/{block_h} = {H_per_tile/block_h:.2f}")
    print(f"  W_tile/block_w = {W_per_tile}/{block_w} = {W_per_tile/block_w:.2f}")
    print(f"  blocks_per_tile ≈ {h_blocks_per_tile} × {w_blocks_per_tile} = {h_blocks_per_tile * w_blocks_per_tile}")
    print(f"  tiles_per_block ≈ {tiles_per_h_block} × {tiles_per_w_block} = {tiles_per_h_block * tiles_per_w_block}")
    
    # 尝试公式
    # 当 tile < block: switches ≈ tiles / tiles_per_block
    # 当 tile > block: switches ≈ tiles × blocks_per_tile
    
    if H_per_tile <= block_h and W_per_tile <= block_w:
        # 小 tile: 多 tiles 共享 blocks
        formula = "tiles / tiles_sharing"
        sharing = tiles_per_h_block * tiles_per_w_block
        estimate = total_tiles / sharing
    else:
        # 大 tile: 每 tile 访问多 blocks
        formula = "tiles × blocks_per_tile"
        estimate = total_tiles * h_blocks_per_tile * w_blocks_per_tile
    
    print(f"\n估计公式: {formula}")
    print(f"  Estimated: {estimate:.0f}")
    print(f"  Actual:    {switches}")
    print(f"  Ratio:     {switches/estimate:.2f}")
    
    return {
        'name': name,
        'tiles': total_tiles,
        'switches': switches,
        'H_tile': H_per_tile,
        'W_tile': W_per_tile,
        'block_h': block_h,
        'block_w': block_w,
        'H_step': H_step,
        'W_step': W_step,
    }


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
        
        analyze_pattern(name, workload, dram, l2, buf, block_h, block_w,
                       stride_h, stride_w, dilation_h, dilation_w)


if __name__ == "__main__":
    main()
