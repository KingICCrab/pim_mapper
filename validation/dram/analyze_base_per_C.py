"""
分析 base_per_C 的结构，寻找解析公式的可能性

base_per_C 由两部分组成：
1. 不跨 block 的访问次数
2. 跨 block 时的额外 switch

让我们分解分析
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload

DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6


def analyze_base_per_C(name, dram, l2, buf, block_h, block_w, stride_h, stride_w, dilation_h, dilation_w):
    """详细分析 base_per_C 的组成."""
    
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
    
    num_w_blocks = 2  # 假设
    
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"P_l3={P_l3}, Q_l3={Q_l3}, R_l2={R_l2}, S_l2={S_l2}")
    print(f"H_tile={H_per_tile}, W_tile={W_per_tile}")
    print(f"block_h={block_h}, block_w={block_w}")
    
    total_tiles = P_l3 * Q_l3 * R_l2 * S_l2
    print(f"Total tiles: {total_tiles}")
    
    # 分类统计
    no_cross = 0      # 不跨 block
    cross_h = 0       # 只跨 H
    cross_w = 0       # 只跨 W
    cross_both = 0    # 跨 H 和 W
    
    # prev_row tracking 计算
    prev_row = -1
    switches_tracking = 0
    
    # 简单计数 (每个 tile 的 block 数)
    blocks_accessed = 0
    
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
                        num_blocks = 4
                    elif crosses_h:
                        cross_h += 1
                        num_blocks = 2
                    elif crosses_w:
                        cross_w += 1
                        num_blocks = 2
                    else:
                        no_cross += 1
                        num_blocks = 1
                    
                    blocks_accessed += num_blocks
                    
                    # prev_row tracking
                    h_blocks = list(range(h_block_start, h_block_end + 1))
                    w_blocks = list(range(w_block_start, w_block_end + 1))
                    
                    rows = set()
                    for hb in h_blocks:
                        for wb in w_blocks:
                            row = hb * num_w_blocks + wb
                            rows.add(row)
                    
                    for row in sorted(rows):
                        if row != prev_row:
                            switches_tracking += 1
                            prev_row = row
    
    print(f"\n分类统计:")
    print(f"  不跨 block:   {no_cross} tiles × 1 = {no_cross}")
    print(f"  跨 H:         {cross_h} tiles × 2 = {cross_h * 2}")
    print(f"  跨 W:         {cross_w} tiles × 2 = {cross_w * 2}")
    print(f"  跨 H+W:       {cross_both} tiles × 4 = {cross_both * 4}")
    print(f"  Total blocks: {blocks_accessed}")
    
    print(f"\n计数方式对比:")
    print(f"  简单 blocks:        {blocks_accessed}")
    print(f"  prev_row tracking:  {switches_tracking}")
    print(f"  差异:               {blocks_accessed - switches_tracking}")
    
    # 分析差异来源
    # prev_row tracking 会减少计数，因为相邻 tiles 可能访问相同 row
    
    # 尝试解析公式
    print(f"\n尝试解析公式:")
    
    # 假设 1: unique rows 访问次数
    # 如果没有 crossing，unique rows = P_l3 × Q_l3 × R_l2 × S_l2 / tiles_per_row
    
    # 假设 2: 基于 crossing 的计算
    # switches = no_cross + cross_h + cross_w + cross_both + extra_for_crossing
    
    simple_formula = no_cross + cross_h * 2 + cross_w * 2 + cross_both * 4
    print(f"  简单公式 (blocks):  {simple_formula}")
    print(f"  实际 switches:      {switches_tracking}")
    
    # 分析: prev_row tracking 减少的量
    reduction = simple_formula - switches_tracking
    print(f"  减少量:             {reduction}")
    print(f"  减少比例:           {reduction/simple_formula*100:.1f}%")
    
    return {
        'total_tiles': total_tiles,
        'no_cross': no_cross,
        'cross_h': cross_h,
        'cross_w': cross_w,
        'cross_both': cross_both,
        'blocks_accessed': blocks_accessed,
        'switches_tracking': switches_tracking,
        'reduction': reduction,
    }


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
        
        r = analyze_base_per_C(name, dram, l2, buf, block_h, block_w,
                               stride_h, stride_w, dilation_h, dilation_w)
        results.append((name, r))
    
    # 总结
    print("\n" + "=" * 60)
    print("总结: 解析公式可行性分析")
    print("=" * 60)
    
    print(f"\n{'Workload':<12} {'Tiles':<8} {'Blocks':<8} {'Switches':<10} {'Reduction%':<10}")
    print("-" * 50)
    for name, r in results:
        print(f"{name:<12} {r['total_tiles']:<8} {r['blocks_accessed']:<8} "
              f"{r['switches_tracking']:<10} {r['reduction']/r['blocks_accessed']*100:.1f}%")
    
    print("""
    
分析结论:
  - blocks_accessed = 简单计数 (每个 tile 的 blocks 数之和)
  - switches_tracking = prev_row tracking 的实际值
  - reduction = 由于相邻 tiles 复用 row 而减少的计数
  
关键问题: 
  reduction 的量取决于循环顺序和访问模式
  难以用简单解析公式表示
    """)


if __name__ == "__main__":
    main()
