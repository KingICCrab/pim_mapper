"""
修正分析: 正确计算 num_h_blocks 和 num_w_blocks

问题: 之前假设 num_w_blocks = 2 是错误的
实际应该根据 Input 尺寸和 block 大小计算
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload

DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6


def analyze_corrected(name, workload, dram, l2, buf, block_h, block_w, stride_h, stride_w, dilation_h, dilation_w):
    """正确分析 base_per_C."""
    
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
    
    # 正确计算 Input 尺寸
    H_in = workload.input_size['H']
    W_in = workload.input_size['W']
    
    # 正确计算 block 数量
    num_h_blocks = (H_in + block_h - 1) // block_h
    num_w_blocks = (W_in + block_w - 1) // block_w
    
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Input: H={H_in}, W={W_in}")
    print(f"Block: {block_h} × {block_w}")
    print(f"Num blocks: {num_h_blocks} × {num_w_blocks} = {num_h_blocks * num_w_blocks}")
    print(f"\nLoop factors:")
    print(f"  P_l3={P_l3}, Q_l3={Q_l3}, R_l2={R_l2}, S_l2={S_l2}")
    print(f"  P_tile={P_tile}, Q_tile={Q_tile}")
    print(f"  H_tile={H_per_tile}, W_tile={W_per_tile}")
    
    total_tiles = P_l3 * Q_l3 * R_l2 * S_l2
    
    # 统计 crossing
    no_cross = 0
    cross_h = 0
    cross_w = 0
    cross_both = 0
    
    # prev_row tracking
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
                    
                    h_block_start = h_start // block_h
                    h_block_end = h_end // block_h
                    w_block_start = w_start // block_w
                    w_block_end = w_end // block_w
                    
                    crosses_h = h_block_end > h_block_start
                    crosses_w = w_block_end > w_block_start
                    
                    if crosses_h and crosses_w:
                        cross_both += 1
                    elif crosses_h:
                        cross_h += 1
                    elif crosses_w:
                        cross_w += 1
                    else:
                        no_cross += 1
                    
                    # rows in this tile
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
    
    total_blocks = no_cross * 1 + cross_h * 2 + cross_w * 2 + cross_both * 4
    
    print(f"\nTile 分类 (total={total_tiles}):")
    print(f"  不跨: {no_cross}, 跨H: {cross_h}, 跨W: {cross_w}, 跨HW: {cross_both}")
    print(f"  Total blocks accessed: {total_blocks}")
    print(f"  Actual switches (prev_row): {switches}")
    
    # 分析: switches 和 blocks 的关系
    print(f"\n关键指标:")
    print(f"  Unique rows in data: {num_h_blocks * num_w_blocks}")
    print(f"  Total tiles: {total_tiles}")
    print(f"  Switches: {switches}")
    print(f"  Switches / tiles: {switches / total_tiles:.2f}")
    print(f"  Switches / unique_rows: {switches / (num_h_blocks * num_w_blocks):.2f}")
    
    return switches, total_tiles, num_h_blocks * num_w_blocks


def main():
    test_workloads = [
        ("ResNet-L1", ConvWorkload(name="ResNet-L1", R=7, S=7, P=56, Q=56, C=3, K=64, N=1)),
        ("ResNet-L2", ConvWorkload(name="ResNet-L2", R=3, S=3, P=56, Q=56, C=64, K=64, N=1)),
        ("ResNet-L3", ConvWorkload(name="ResNet-L3", R=1, S=1, P=56, Q=56, C=64, K=256, N=1)),
        ("VGG-L1", ConvWorkload(name="VGG-L1", R=3, S=3, P=224, Q=224, C=3, K=64, N=1)),
    ]
    
    optimizer = PIMOptimizer()
    results = []
    
    for name, workload in test_workloads:
        result = optimizer.optimize([workload], objective="latency")
        mapping = result.mappings[0]
        
        # Extract params
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
        
        sw, tiles, unique = analyze_corrected(name, workload, dram, l2, buf, block_h, block_w,
                                               stride_h, stride_w, dilation_h, dilation_w)
        results.append((name, sw, tiles, unique))
    
    # 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"\n{'Workload':<12} {'Tiles':<8} {'Unique Rows':<12} {'Switches':<10} {'Sw/Tiles':<10}")
    print("-" * 60)
    for name, sw, tiles, unique in results:
        print(f"{name:<12} {tiles:<8} {unique:<12} {sw:<10} {sw/tiles:.2f}")
    
    print("""
    
发现:
  Switches ≈ Tiles × (某个系数)
  这个系数与 crossing 情况和访问模式有关
  
可能的简化公式:
  base_per_C ≈ P_l3 × Q_l3 × R_l2 × S_l2 × crossing_factor
  
其中 crossing_factor 取决于 tile 和 block 的关系
    """)


if __name__ == "__main__":
    main()
