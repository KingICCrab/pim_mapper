"""
基于 ILP 框架推导正确的 Input Row Activation 公式

ILP 当前公式:
    total = row_acts_aligned + block_crossing_acts
    row_acts_aligned = Π_{j ∈ all_dims} DRAM_factor[j]  (只有 L3)
    
问题: 缺少 L2 的 R/S 循环的影响

正确公式应该是:
    total = base_per_C × C × K
    
其中 base_per_C 需要考虑:
1. L3 的 P × Q 循环
2. L2 的 R × S 循环 (被 ILP 忽略了!)

循环嵌套顺序: P → Q → R → S (外到内)
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
import math

DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6


def derive_formula(name, workload, dram, l2, buf, block_h, block_w, stride_h, stride_w, dilation_h, dilation_w):
    """推导 base_per_C 的解析公式."""
    
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
    print(f"L3 Loops: P={P_l3}, Q={Q_l3}")
    print(f"L2 Loops: R={R_l2}, S={S_l2}")
    print(f"Tile: P={P_tile}, Q={Q_tile}, R={R_tile}, S={S_tile}")
    print(f"Step: H={H_step}, W={W_step}, R={R_step}, S={S_step}")
    print(f"Tile span: H={H_per_tile}, W={W_per_tile}")
    print(f"Blocks: {num_h_blocks} × {num_w_blocks}, size={block_h}×{block_w}")
    
    # =================================================================
    # 方法 1: 模拟计算 (Ground Truth)
    # =================================================================
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
    
    print(f"\n实际 switches: {total_switches}")
    
    # =================================================================
    # 方法 2: ILP 当前公式 (只有 L3)
    # =================================================================
    ilp_row_acts_aligned = P_l3 * Q_l3  # 只计算 relevant dims (P, Q)
    print(f"\nILP row_acts_aligned (只有 L3): P×Q = {P_l3}×{Q_l3} = {ilp_row_acts_aligned}")
    
    # =================================================================
    # 方法 3: 考虑 L2 的简单扩展
    # =================================================================
    # 如果简单地把 L2 也乘进去: P × Q × R × S
    simple_extension = P_l3 * Q_l3 * R_l2 * S_l2
    print(f"简单扩展 (L3+L2): P×Q×R×S = {P_l3}×{Q_l3}×{R_l2}×{S_l2} = {simple_extension}")
    
    # =================================================================
    # 方法 4: 分析 H 和 W 方向的 unique block visits
    # =================================================================
    
    # H 方向: 统计整个 P×R 循环中访问的 (h_block, visit_count)
    h_block_visits = {}
    for p in range(P_l3):
        for r in range(R_l2):
            h_start = p * H_step + r * R_step
            h_end = h_start + H_per_tile - 1
            for hb in range(h_start // block_h, h_end // block_h + 1):
                h_block_visits[hb] = h_block_visits.get(hb, 0) + 1
    
    H_total_visits = sum(h_block_visits.values())
    H_unique_blocks = len(h_block_visits)
    
    # W 方向: 统计整个 Q×S 循环中访问的 (w_block, visit_count)
    w_block_visits = {}
    for q in range(Q_l3):
        for s in range(S_l2):
            w_start = q * W_step + s * S_step
            w_end = w_start + W_per_tile - 1
            for wb in range(w_start // block_w, w_end // block_w + 1):
                w_block_visits[wb] = w_block_visits.get(wb, 0) + 1
    
    W_total_visits = sum(w_block_visits.values())
    W_unique_blocks = len(w_block_visits)
    
    print(f"\nH direction: unique={H_unique_blocks}, total_visits={H_total_visits}")
    print(f"W direction: unique={W_unique_blocks}, total_visits={W_total_visits}")
    
    # 公式尝试: H_total × W_total (有时候太大)
    formula_1 = H_total_visits * W_total_visits
    print(f"\n公式1: H_visits × W_visits = {H_total_visits} × {W_total_visits} = {formula_1}")
    
    # =================================================================
    # 方法 5: 分析循环结构 - 计算 unique row switches
    # =================================================================
    # 关键观察: 循环顺序是 P -> Q -> R -> S
    # 
    # 对于每个 P 值:
    #   - Q 从头遍历
    #   - 每个 Q 内, R×S 遍历
    #
    # Row = h_block × num_w_blocks + w_block
    # 
    # 当 Q 变化时: w_block 可能变化 → row 变化
    # 当 R 变化时: h_block 可能变化 → row 变化
    # 当 S 变化时: w_block 可能变化 → row 变化
    # 当 P 变化时: h_block 可能变化, 且 Q/R/S 重新开始
    
    # 计算每个 P 内的 row switches
    switches_per_p = []
    prev_row_global = -1
    
    for p in range(P_l3):
        count = 0
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
                            if row != prev_row_global:
                                count += 1
                                prev_row_global = row
        
        switches_per_p.append(count)
    
    print(f"\nSwitches per P: {switches_per_p[:10]}{'...' if len(switches_per_p) > 10 else ''}")
    print(f"Sum = {sum(switches_per_p)}")
    
    # =================================================================
    # 方法 6: 基于周期性的解析公式
    # =================================================================
    # 关键: 找到 H 方向和 W 方向的 block switch pattern
    
    # H 方向: 每次 P 或 R 变化时, h_start 变化
    # h_start = p × H_step + r × R_step
    # h_block = h_start // block_h
    # 
    # 问题: h_block 什么时候变化?
    # 当 h_start 跨越 block_h 边界时
    
    # 计算 H 方向的 unique (h_start // block_h) patterns in P×R loop
    h_patterns = []
    for p in range(P_l3):
        for r in range(R_l2):
            h_start = p * H_step + r * R_step
            h_block = h_start // block_h
            h_patterns.append(h_block)
    
    # 统计 H pattern 的 unique switches (连续相同的不算)
    h_switches = 1  # 第一个总是算
    for i in range(1, len(h_patterns)):
        if h_patterns[i] != h_patterns[i-1]:
            h_switches += 1
    
    # W 方向同理
    w_patterns = []
    for q in range(Q_l3):
        for s in range(S_l2):
            w_start = q * W_step + s * S_step
            w_block = w_start // block_w
            w_patterns.append(w_block)
    
    w_switches = 1
    for i in range(1, len(w_patterns)):
        if w_patterns[i] != w_patterns[i-1]:
            w_switches += 1
    
    print(f"\nH pattern switches (in P×R): {h_switches}")
    print(f"W pattern switches (in Q×S): {w_switches}")
    
    # 公式尝试: h_switches × w_switches?
    formula_2 = h_switches * w_switches
    print(f"公式2: h_switches × w_switches = {h_switches} × {w_switches} = {formula_2}")
    
    # =================================================================
    # 方法 7: 考虑 tile 跨越多个 blocks 的情况
    # =================================================================
    # 一个 tile 可能跨越多个 h_blocks 和 w_blocks
    # tile span: [h_start, h_start + H_per_tile - 1]
    # 跨越的 blocks: range(h_start // block_h, (h_start + H_per_tile - 1) // block_h + 1)
    
    # 计算 H 方向: 每个 (p, r) 访问多少个 h_blocks?
    h_blocks_per_pr = []
    for p in range(P_l3):
        for r in range(R_l2):
            h_start = p * H_step + r * R_step
            h_end = h_start + H_per_tile - 1
            n_blocks = h_end // block_h - h_start // block_h + 1
            h_blocks_per_pr.append(n_blocks)
    
    avg_h_blocks = sum(h_blocks_per_pr) / len(h_blocks_per_pr) if h_blocks_per_pr else 1
    
    # W 方向同理
    w_blocks_per_qs = []
    for q in range(Q_l3):
        for s in range(S_l2):
            w_start = q * W_step + s * S_step
            w_end = w_start + W_per_tile - 1
            n_blocks = w_end // block_w - w_start // block_w + 1
            w_blocks_per_qs.append(n_blocks)
    
    avg_w_blocks = sum(w_blocks_per_qs) / len(w_blocks_per_qs) if w_blocks_per_qs else 1
    
    print(f"\nAvg H blocks per (p,r): {avg_h_blocks:.2f}")
    print(f"Avg W blocks per (q,s): {avg_w_blocks:.2f}")
    
    # =================================================================
    # 方法 8: 正确的公式推导
    # =================================================================
    # 核心观察: row = h_block × num_w_blocks + w_block
    # 
    # row 变化 ⟺ h_block 变化 OR w_block 变化
    # 
    # 但由于循环顺序是 P → Q → R → S:
    # - 内层 S 循环可能改变 w_block
    # - 内层 R 循环可能改变 h_block
    # - 中层 Q 循环可能改变 w_block
    # - 外层 P 循环可能改变 h_block
    #
    # 关键: 每次 row 变化都需要一次 activation
    
    # 统计: 对于每个 (p, q), 内部 R×S 循环产生多少 unique rows?
    unique_rows_per_pq = []
    for p in range(P_l3):
        for q in range(Q_l3):
            rows_in_pq = set()
            for r in range(R_l2):
                for s in range(S_l2):
                    h_start = p * H_step + r * R_step
                    w_start = q * W_step + s * S_step
                    
                    h_end = h_start + H_per_tile - 1
                    w_end = w_start + W_per_tile - 1
                    
                    for hb in range(h_start // block_h, h_end // block_h + 1):
                        for wb in range(w_start // block_w, w_end // block_w + 1):
                            row = hb * num_w_blocks + wb
                            rows_in_pq.add(row)
            
            unique_rows_per_pq.append(len(rows_in_pq))
    
    print(f"\nUnique rows per (p,q): {unique_rows_per_pq[:20]}{'...' if len(unique_rows_per_pq) > 20 else ''}")
    print(f"Sum of unique rows: {sum(unique_rows_per_pq)}")
    print(f"Max unique rows per (p,q): {max(unique_rows_per_pq)}")
    
    # 但是! 不同 (p,q) 可能访问相同的 row
    # 当 Q 增加导致 w_block 变化, 然后回到相同的 row 时, 仍然需要 switch
    
    # =================================================================
    # 总结
    # =================================================================
    print(f"\n{'='*70}")
    print("总结")
    print(f"{'='*70}")
    print(f"实际 total switches: {total_switches}")
    print(f"ILP 当前 (L3 only): {ilp_row_acts_aligned}")
    print(f"简单 L3+L2: {simple_extension}")
    print(f"H_visits × W_visits: {formula_1}")
    print(f"h_switches × w_switches: {formula_2}")
    
    # 找到正确的公式
    if total_switches == formula_1:
        print(f"✓ 公式: H_visits × W_visits = {formula_1}")
    elif total_switches == formula_2:
        print(f"✓ 公式: h_switches × w_switches = {formula_2}")
    elif total_switches == h_switches * W_total_visits:
        print(f"✓ 公式: h_switches × W_visits = {h_switches} × {W_total_visits} = {h_switches * W_total_visits}")
    elif total_switches == H_total_visits * w_switches:
        print(f"✓ 公式: H_visits × w_switches = {H_total_visits} × {w_switches} = {H_total_visits * w_switches}")
    else:
        print(f"✗ 需要找到新公式")
        # 尝试更多组合
        print(f"\n尝试其他公式:")
        print(f"  P×Q×R×S / factor: {simple_extension} / ? = {total_switches} → factor = {simple_extension / total_switches if total_switches else 'N/A'}")
    
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
        
        derive_formula(name, workload, dram, l2, buf, block_h, block_w,
                      stride_h, stride_w, dilation_h, dilation_w)


if __name__ == "__main__":
    main()
