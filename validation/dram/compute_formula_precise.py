"""
正确公式推导 - 基于 block crossing 分析

核心发现:
1. 当 tile 跨越多个 blocks 时, 每次循环迭代会在这些 blocks 之间切换
2. 平均 switches = 跨越的 blocks 数

公式:
  total_switches = Σ (num_blocks_covered × visits_count) for each (p, q)

简化公式:
  - H 方向跨越 blocks: ceil((H_per_tile) / block_h)  
  - W 方向跨越 blocks: ceil((W_per_tile) / block_w)
  - 但实际跨越取决于起始位置对齐
"""

import math


def compute_precise_formula(name, P_l3, Q_l3, R_l2, S_l2, P_tile, Q_tile, R_tile, S_tile,
                            block_h, block_w, H_in, W_in, stride_h=1, stride_w=1, dilation_h=1, dilation_w=1):
    """计算精确公式."""
    
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
    
    # Ground Truth
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
    
    print(f"Ground Truth: {total_switches}")
    
    # =================================================================
    # 公式 A: H_visits × W_visits (只对 R_l2=1, S_l2=1 有效)
    # =================================================================
    
    # H_visits: P×R 循环中访问的 total h_block count
    h_block_visits = {}
    for p in range(P_l3):
        for r in range(R_l2):
            h_start = p * H_step + r * R_step
            h_end = h_start + H_per_tile - 1
            for hb in range(h_start // block_h, h_end // block_h + 1):
                h_block_visits[hb] = h_block_visits.get(hb, 0) + 1
    
    H_visits = sum(h_block_visits.values())
    
    # W_visits: Q×S 循环中访问的 total w_block count
    w_block_visits = {}
    for q in range(Q_l3):
        for s in range(S_l2):
            w_start = q * W_step + s * S_step
            w_end = w_start + W_per_tile - 1
            for wb in range(w_start // block_w, w_end // block_w + 1):
                w_block_visits[wb] = w_block_visits.get(wb, 0) + 1
    
    W_visits = sum(w_block_visits.values())
    
    formula_A = H_visits * W_visits
    print(f"公式 A (H_visits × W_visits): {H_visits} × {W_visits} = {formula_A}")
    
    # =================================================================
    # 公式 B: 基于 (p, q) 分类的精确计算
    # =================================================================
    
    # 对于每个 (p, q), 计算它覆盖的 unique rows
    # 然后考虑循环顺序
    
    total_formula_b = 0
    prev_row_b = -1
    
    # 统计每个 P 内的 switches
    switches_per_p = []
    
    for p in range(P_l3):
        p_switches = 0
        # 对于这个 P, 遍历 Q, R, S
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
                            if row != prev_row_b:
                                p_switches += 1
                                prev_row_b = row
        
        switches_per_p.append(p_switches)
        total_formula_b += p_switches
    
    print(f"公式 B (Σ switches_per_p): {total_formula_b}")
    print(f"  switches_per_p: {switches_per_p[:10]}{'...' if len(switches_per_p) > 10 else ''}")
    
    # =================================================================
    # 公式 C: 分析 switches_per_p 的模式
    # =================================================================
    
    # 观察: switches_per_p 可能有规律
    unique_sp = sorted(set(switches_per_p))
    print(f"  Unique values: {unique_sp}")
    
    # 对于每个 unique value, 统计出现次数
    for val in unique_sp:
        count = sum(1 for x in switches_per_p if x == val)
        print(f"    {val}: {count} 次")
    
    # =================================================================
    # 公式 D: 尝试找到 switches_per_p 的解析表达式
    # =================================================================
    
    # 对于每个 P:
    # 1. H 方向: 这个 P 的 R 循环会访问哪些 h_blocks?
    # 2. W 方向: Q×S 循环会访问哪些 w_blocks? (这对所有 P 相同)
    
    # W 方向分析 (对所有 P 相同)
    w_sequence = []  # Q×S 循环中访问的 w_block 序列
    for q in range(Q_l3):
        for s in range(S_l2):
            w_start = q * W_step + s * S_step
            w_end = w_start + W_per_tile - 1
            w_blocks = tuple(range(w_start // block_w, w_end // block_w + 1))
            w_sequence.append(w_blocks)
    
    # W 方向的 switches (不考虑 H)
    w_switches_per_q = []
    prev_w = None
    for w_config in w_sequence:
        if w_config != prev_w:
            w_switches_per_q.append(1)
            prev_w = w_config
        else:
            w_switches_per_q.append(0)
    
    total_w_switches = sum(w_switches_per_q)
    
    print(f"\nW 方向分析:")
    print(f"  W sequence length: {len(w_sequence)}")
    print(f"  Unique W configs: {len(set(w_sequence))}")
    print(f"  Total W switches: {total_w_switches}")
    
    # H 方向分析 (每个 P 不同)
    h_sequences = []  # 每个 P 的 R 循环访问的 h_block 序列
    for p in range(P_l3):
        h_seq = []
        for r in range(R_l2):
            h_start = p * H_step + r * R_step
            h_end = h_start + H_per_tile - 1
            h_blocks = tuple(range(h_start // block_h, h_end // block_h + 1))
            h_seq.append(h_blocks)
        h_sequences.append(h_seq)
    
    # 分析每个 P 的 H sequence
    h_info_per_p = []
    for p, h_seq in enumerate(h_sequences):
        unique_h = len(set(h_seq))
        total_h_blocks = sum(len(cfg) for cfg in h_seq)
        h_info_per_p.append((unique_h, total_h_blocks))
    
    print(f"\nH 方向分析:")
    print(f"  H info per P (unique_configs, total_blocks): {h_info_per_p[:10]}...")
    
    # =================================================================
    # 公式 E: 基于周期性的公式
    # =================================================================
    
    # 观察: 循环顺序是 P → Q → R → S
    # 在一个 P 内:
    #   - Q 遍历: 改变 w_start
    #   - R 遍历: 改变 h_start  
    #   - S 遍历: 改变 w_start (但通常 S_l2=1)
    
    # 关键: 每当 h_block 或 w_block 变化, 就有一次 switch
    
    # 对于一个 P, 假设 H 方向访问 n_h 个 unique h_blocks
    # 假设 W 方向访问 n_w 个 unique w_blocks
    # 
    # 如果完全独立: switches = n_h × n_w
    # 但由于循环顺序, 实际 switches 取决于:
    # - Q 变化时 w_block 是否变化
    # - R 变化时 h_block 是否变化
    
    # 简化假设: R_l2 = 1 时, 公式 A 成立
    # 当 R_l2 > 1 时, 需要考虑 R 循环中的 h_block 变化
    
    if R_l2 == 1 and S_l2 == 1:
        print(f"\n特殊情况: R_l2=1, S_l2=1")
        print(f"  公式 A 应该成立: {formula_A}")
    else:
        print(f"\n一般情况: R_l2={R_l2}, S_l2={S_l2}")
        # 需要考虑 R×S 循环的影响
    
    # =================================================================
    # 验证结果
    # =================================================================
    print(f"\n{'='*70}")
    print("验证")
    print(f"{'='*70}")
    print(f"Ground Truth: {total_switches}")
    print(f"公式 A: {formula_A} {'✓' if formula_A == total_switches else '✗'}")
    
    return total_switches, formula_A


def main():
    results = []
    
    # ResNet-L1
    gt, fa = compute_precise_formula(
        name="ResNet-L1",
        P_l3=28, Q_l3=7, R_l2=7, S_l2=1,
        P_tile=2, Q_tile=8, R_tile=1, S_tile=7,
        block_h=31, block_w=31,
        H_in=62, W_in=62
    )
    results.append(("ResNet-L1", gt, fa))
    
    # ResNet-L2
    gt, fa = compute_precise_formula(
        name="ResNet-L2",
        P_l3=28, Q_l3=7, R_l2=1, S_l2=1,
        P_tile=2, Q_tile=8, R_tile=1, S_tile=3,
        block_h=58, block_w=2,
        H_in=58, W_in=58
    )
    results.append(("ResNet-L2", gt, fa))
    
    # ResNet-L3
    gt, fa = compute_precise_formula(
        name="ResNet-L3",
        P_l3=2, Q_l3=56, R_l2=1, S_l2=1,
        P_tile=4, Q_tile=1, R_tile=1, S_tile=1,
        block_h=1, block_w=28,
        H_in=56, W_in=56
    )
    results.append(("ResNet-L3", gt, fa))
    
    # VGG-L1
    gt, fa = compute_precise_formula(
        name="VGG-L1",
        P_l3=14, Q_l3=224, R_l2=1, S_l2=1,
        P_tile=16, Q_tile=1, R_tile=3, S_tile=1,
        block_h=1, block_w=226,
        H_in=226, W_in=226
    )
    results.append(("VGG-L1", gt, fa))
    
    # 总结
    print(f"\n{'='*70}")
    print("总结")
    print(f"{'='*70}")
    for name, gt, fa in results:
        match = "✓" if gt == fa else "✗"
        print(f"{name}: Ground Truth={gt}, Formula A={fa} {match}")


if __name__ == "__main__":
    main()
