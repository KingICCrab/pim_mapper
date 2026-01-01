"""
基于 block crossing 的正确公式推导

关键发现:
- 当 tile 跨越多个 blocks 时, 每次 R/S 循环都会在这些 blocks 之间切换
- 这是额外的 row switches 来源

正确公式应该分解为:
1. 基础 switches (不考虑 crossing)
2. Block crossing 导致的额外 switches
"""

import math


def analyze_with_crossing(name, P_l3, Q_l3, R_l2, S_l2, P_tile, Q_tile, R_tile, S_tile,
                          block_h, block_w, H_in, W_in, stride_h=1, stride_w=1, dilation_h=1, dilation_w=1):
    """分析 block crossing 的影响."""
    
    H_per_tile = (P_tile - 1) * stride_h + R_tile * dilation_h
    W_per_tile = (Q_tile - 1) * stride_w + S_tile * dilation_w
    
    H_step = P_tile * stride_h
    W_step = Q_tile * stride_w
    R_step = R_tile * dilation_h
    S_step = S_tile * dilation_w
    
    num_h_blocks = (H_in + block_h - 1) // block_h
    num_w_blocks = (W_in + block_w - 1) // block_w
    
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"L3 Loops: P={P_l3}, Q={Q_l3}")
    print(f"L2 Loops: R={R_l2}, S={S_l2}")
    print(f"Tile span: H={H_per_tile}, W={W_per_tile}")
    print(f"Block size: {block_h} × {block_w}")
    print(f"Num blocks: {num_h_blocks} × {num_w_blocks}")
    
    # =================================================================
    # Ground Truth
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
    
    print(f"\n实际 total switches: {total_switches}")
    
    # =================================================================
    # 分析每个 (p, q) 的 crossing 情况
    # =================================================================
    
    # 对于每个 (p, q), 统计:
    # 1. h_blocks: R 循环会访问的 h_block 集合
    # 2. w_blocks: S 循环会访问的 w_block 集合
    # 3. 如果 len(h_blocks) > 1 or len(w_blocks) > 1, 会有 crossing
    
    no_crossing_count = 0
    h_crossing_only_count = 0
    w_crossing_only_count = 0
    both_crossing_count = 0
    
    # 也统计每种情况的 switches
    no_crossing_switches = 0
    h_crossing_switches = 0
    w_crossing_switches = 0
    both_crossing_switches = 0
    
    prev_row_local = -1
    
    for p in range(P_l3):
        for q in range(Q_l3):
            # 这个 (p, q) 内的所有 h_blocks 和 w_blocks
            all_h_blocks = set()
            all_w_blocks = set()
            
            for r in range(R_l2):
                for s in range(S_l2):
                    h_start = p * H_step + r * R_step
                    w_start = q * W_step + s * S_step
                    
                    h_end = h_start + H_per_tile - 1
                    w_end = w_start + W_per_tile - 1
                    
                    for hb in range(h_start // block_h, h_end // block_h + 1):
                        all_h_blocks.add(hb)
                    for wb in range(w_start // block_w, w_end // block_w + 1):
                        all_w_blocks.add(wb)
            
            h_crossing = len(all_h_blocks) > 1
            w_crossing = len(all_w_blocks) > 1
            
            # 计算这个 (p, q) 内的 switches
            pq_switches = 0
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
                            if row != prev_row_local:
                                pq_switches += 1
                                prev_row_local = row
            
            if not h_crossing and not w_crossing:
                no_crossing_count += 1
                no_crossing_switches += pq_switches
            elif h_crossing and not w_crossing:
                h_crossing_only_count += 1
                h_crossing_switches += pq_switches
            elif not h_crossing and w_crossing:
                w_crossing_only_count += 1
                w_crossing_switches += pq_switches
            else:
                both_crossing_count += 1
                both_crossing_switches += pq_switches
    
    print(f"\n(p, q) 分类统计:")
    print(f"  No crossing: {no_crossing_count} 个, {no_crossing_switches} switches")
    print(f"  H crossing only: {h_crossing_only_count} 个, {h_crossing_switches} switches")
    print(f"  W crossing only: {w_crossing_only_count} 个, {w_crossing_switches} switches")
    print(f"  Both crossing: {both_crossing_count} 个, {both_crossing_switches} switches")
    print(f"  Total: {no_crossing_switches + h_crossing_switches + w_crossing_switches + both_crossing_switches}")
    
    # =================================================================
    # 基于分类的公式推导
    # =================================================================
    
    # 对于 no crossing 的 (p, q):
    # - 只访问 1 个 row
    # - switches = 1 (或 0 如果和上一个 (p,q) 相同)
    
    # 对于 H crossing only 的 (p, q):
    # - 访问多个 h_blocks, 但只有 1 个 w_block
    # - R 循环会在不同 h_blocks 之间切换
    # - switches ≈ len(all_h_blocks) × R_l2? (取决于具体模式)
    
    # 对于 W crossing only 的 (p, q):
    # - 访问 1 个 h_block, 但多个 w_blocks
    # - 每次迭代都会遍历多个 w_blocks
    # - switches ≈ len(all_w_blocks) × R_l2 × S_l2
    
    # 让我计算平均 switches per crossing type
    print(f"\n平均 switches per (p, q):")
    if no_crossing_count > 0:
        print(f"  No crossing: {no_crossing_switches / no_crossing_count:.2f}")
    if h_crossing_only_count > 0:
        print(f"  H crossing: {h_crossing_switches / h_crossing_only_count:.2f}")
    if w_crossing_only_count > 0:
        print(f"  W crossing: {w_crossing_switches / w_crossing_only_count:.2f}")
    if both_crossing_count > 0:
        print(f"  Both crossing: {both_crossing_switches / both_crossing_count:.2f}")
    
    # =================================================================
    # 尝试解析公式
    # =================================================================
    
    # 核心观察: 
    # - No crossing (p,q): 最多 1 switch (从上一个 row 切换过来)
    # - W crossing (p,q): 每次 (r, s) 迭代访问 w_blocks 数个 rows
    #   如果 tile 跨 2 个 w_blocks, 每次 R 都会访问 row0, row1, row0, row1...
    #   switches = 2 × R_l2 - 1 (第一个不算)
    
    # 让我验证这个公式
    print(f"\n验证 W crossing 公式:")
    
    # 找一个 W crossing only 的 (p, q)
    for p in range(P_l3):
        for q in range(Q_l3):
            all_h_blocks = set()
            all_w_blocks = set()
            
            for r in range(R_l2):
                for s in range(S_l2):
                    h_start = p * H_step + r * R_step
                    w_start = q * W_step + s * S_step
                    
                    h_end = h_start + H_per_tile - 1
                    w_end = w_start + W_per_tile - 1
                    
                    for hb in range(h_start // block_h, h_end // block_h + 1):
                        all_h_blocks.add(hb)
                    for wb in range(w_start // block_w, w_end // block_w + 1):
                        all_w_blocks.add(wb)
            
            if len(all_h_blocks) == 1 and len(all_w_blocks) > 1:
                print(f"  (p={p}, q={q}): h_blocks={all_h_blocks}, w_blocks={all_w_blocks}")
                
                # 预测 switches: len(w_blocks) × R_l2 - (len(w_blocks) - 1)
                # = len(w_blocks) × (R_l2 - 1) + 1
                num_w = len(all_w_blocks)
                predicted = num_w * R_l2 - (num_w - 1)
                predicted_v2 = num_w * R_l2 * S_l2 - (num_w - 1)
                
                # 实际计算
                prev = -1
                actual = 0
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
                                if row != prev:
                                    actual += 1
                                    prev = row
                
                print(f"    Actual: {actual}, Predicted1: {predicted}, Predicted2: {predicted_v2}")
                break
        else:
            continue
        break
    
    # =================================================================
    # 新公式推导
    # =================================================================
    print(f"\n{'='*70}")
    print("公式推导")
    print(f"{'='*70}")
    
    # 基础部分: 假设没有 crossing
    # 每个 (p, q) 访问 1 个 row
    # P_l3 × Q_l3 个 (p, q), 但很多访问相同的 row
    
    # 统计 unique rows
    unique_rows = set()
    for p in range(P_l3):
        for q in range(Q_l3):
            for r in range(R_l2):
                for s in range(S_l2):
                    h_start = p * H_step + r * R_step
                    w_start = q * W_step + s * S_step
                    
                    hb = h_start // block_h
                    wb = w_start // block_w
                    row = hb * num_w_blocks + wb
                    unique_rows.add(row)
    
    print(f"Unique rows (不考虑 tile span): {len(unique_rows)}")
    
    # 实际 unique rows (考虑 tile span)
    unique_rows_full = set()
    for p in range(P_l3):
        for q in range(Q_l3):
            for r in range(R_l2):
                for s in range(S_l2):
                    h_start = p * H_step + r * R_step
                    w_start = q * W_step + s * S_step
                    
                    h_end = h_start + H_per_tile - 1
                    w_end = w_start + W_per_tile - 1
                    
                    for hb in range(h_start // block_h, h_end // block_h + 1):
                        for wb in range(w_start // block_w, w_end // block_w + 1):
                            row = hb * num_w_blocks + wb
                            unique_rows_full.add(row)
    
    print(f"Unique rows (考虑 tile span): {len(unique_rows_full)}")
    print(f"Total possible rows: {num_h_blocks * num_w_blocks}")
    
    return total_switches


def main():
    # ResNet-L1
    analyze_with_crossing(
        name="ResNet-L1",
        P_l3=28, Q_l3=7, R_l2=7, S_l2=1,
        P_tile=2, Q_tile=8, R_tile=1, S_tile=7,
        block_h=31, block_w=31,
        H_in=62, W_in=62
    )
    
    # ResNet-L2
    analyze_with_crossing(
        name="ResNet-L2",
        P_l3=28, Q_l3=7, R_l2=1, S_l2=1,
        P_tile=2, Q_tile=8, R_tile=1, S_tile=3,
        block_h=58, block_w=2,
        H_in=58, W_in=58
    )
    
    # ResNet-L3 (R_l2=1, S_l2=1)
    analyze_with_crossing(
        name="ResNet-L3",
        P_l3=2, Q_l3=56, R_l2=1, S_l2=1,
        P_tile=4, Q_tile=1, R_tile=1, S_tile=1,
        block_h=1, block_w=28,
        H_in=56, W_in=56
    )
    
    # VGG-L1 (R_l2=1, S_l2=1)
    analyze_with_crossing(
        name="VGG-L1",
        P_l3=14, Q_l3=224, R_l2=1, S_l2=1,
        P_tile=16, Q_tile=1, R_tile=3, S_tile=1,
        block_h=1, block_w=226,
        H_in=226, W_in=226
    )


if __name__ == "__main__":
    main()
