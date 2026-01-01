"""
基于 ILP 框架推导正确的 Input Row Activation 公式
使用手动设定的参数，避免 Gurobi license 问题
"""

import math

def derive_formula(name, P_l3, Q_l3, R_l2, S_l2, P_tile, Q_tile, R_tile, S_tile,
                   block_h, block_w, H_in, W_in, stride_h=1, stride_w=1, dilation_h=1, dilation_w=1):
    """推导 base_per_C 的解析公式."""
    
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
    print(f"Tile: P={P_tile}, Q={Q_tile}, R={R_tile}, S={S_tile}")
    print(f"Step: H={H_step}, W={W_step}, R={R_step}, S={S_step}")
    print(f"Tile span: H={H_per_tile}, W={W_per_tile}")
    print(f"Blocks: {num_h_blocks} × {num_w_blocks}, size={block_h}×{block_w}")
    print(f"Input: {H_in} × {W_in}")
    
    # =================================================================
    # Ground Truth: 模拟计算
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
    # 方法 1: ILP 当前公式 (只有 L3)
    # =================================================================
    ilp_current = P_l3 * Q_l3
    print(f"\nILP 当前 (L3 only): P×Q = {ilp_current}")
    
    # =================================================================
    # 方法 2: 统计 H 和 W 方向的 total visits
    # =================================================================
    
    # H 方向: 统计 P×R 循环
    h_block_visits = {}
    for p in range(P_l3):
        for r in range(R_l2):
            h_start = p * H_step + r * R_step
            h_end = h_start + H_per_tile - 1
            for hb in range(h_start // block_h, h_end // block_h + 1):
                h_block_visits[hb] = h_block_visits.get(hb, 0) + 1
    
    H_total_visits = sum(h_block_visits.values())
    
    # W 方向: 统计 Q×S 循环
    w_block_visits = {}
    for q in range(Q_l3):
        for s in range(S_l2):
            w_start = q * W_step + s * S_step
            w_end = w_start + W_per_tile - 1
            for wb in range(w_start // block_w, w_end // block_w + 1):
                w_block_visits[wb] = w_block_visits.get(wb, 0) + 1
    
    W_total_visits = sum(w_block_visits.values())
    
    print(f"H_total_visits: {H_total_visits}")
    print(f"W_total_visits: {W_total_visits}")
    
    formula_1 = H_total_visits * W_total_visits
    print(f"公式1: H_visits × W_visits = {formula_1}")
    
    # =================================================================
    # 方法 3: 计算 H 和 W 方向的 switches (连续相同不算)
    # =================================================================
    
    # H 方向 switches
    h_patterns = []
    for p in range(P_l3):
        for r in range(R_l2):
            h_start = p * H_step + r * R_step
            h_end = h_start + H_per_tile - 1
            h_blocks = tuple(range(h_start // block_h, h_end // block_h + 1))
            h_patterns.append(h_blocks)
    
    h_switches = 1
    for i in range(1, len(h_patterns)):
        if h_patterns[i] != h_patterns[i-1]:
            h_switches += 1
    
    # W 方向 switches
    w_patterns = []
    for q in range(Q_l3):
        for s in range(S_l2):
            w_start = q * W_step + s * S_step
            w_end = w_start + W_per_tile - 1
            w_blocks = tuple(range(w_start // block_w, w_end // block_w + 1))
            w_patterns.append(w_blocks)
    
    w_switches = 1
    for i in range(1, len(w_patterns)):
        if w_patterns[i] != w_patterns[i-1]:
            w_switches += 1
    
    print(f"H_switches: {h_switches}")
    print(f"W_switches: {w_switches}")
    
    formula_2 = h_switches * w_switches
    print(f"公式2: h_switches × w_switches = {formula_2}")
    
    # =================================================================
    # 方法 4: H_switches × W_total_visits
    # =================================================================
    formula_3 = h_switches * W_total_visits
    print(f"公式3: h_switches × W_visits = {formula_3}")
    
    # =================================================================
    # 方法 5: H_total_visits × w_switches
    # =================================================================
    formula_4 = H_total_visits * w_switches
    print(f"公式4: H_visits × w_switches = {formula_4}")
    
    # =================================================================
    # 检查哪个公式正确
    # =================================================================
    print(f"\n比较:")
    print(f"  实际: {total_switches}")
    for i, f in enumerate([formula_1, formula_2, formula_3, formula_4], 1):
        match = "✓" if f == total_switches else "✗"
        print(f"  公式{i}: {f} {match}")
    
    return total_switches, H_total_visits, W_total_visits, h_switches, w_switches


def main():
    # 基于之前的 optimizer 结果手动设置参数
    
    # ResNet-L1: R=7, S=7, P=56, Q=56
    # 从之前输出: L3 Loops P=28, Q=7; L2 Loops R=7, S=1
    # Tile: P=2, Q=8, R=1, S=7
    # Block: 31×31, Input: 62×62
    derive_formula(
        name="ResNet-L1",
        P_l3=28, Q_l3=7, R_l2=7, S_l2=1,
        P_tile=2, Q_tile=8, R_tile=1, S_tile=7,
        block_h=31, block_w=31,
        H_in=62, W_in=62
    )
    
    # ResNet-L2: R=3, S=3, P=56, Q=56
    # 从之前输出: L3 Loops P=28, Q=7; L2 Loops R=1, S=1
    # Tile: P=2, Q=8, R=1, S=3
    # Block: 58×2, Input: 58×58
    derive_formula(
        name="ResNet-L2",
        P_l3=28, Q_l3=7, R_l2=1, S_l2=1,
        P_tile=2, Q_tile=8, R_tile=1, S_tile=3,
        block_h=58, block_w=2,
        H_in=58, W_in=58
    )
    
    # ResNet-L3: R=1, S=1, P=56, Q=56
    # 从之前输出: L3 Loops P=2, Q=56; L2 Loops R=1, S=1
    # Tile: P=4, Q=1, R=1, S=1
    # Block: 1×28, Input: 56×56
    derive_formula(
        name="ResNet-L3",
        P_l3=2, Q_l3=56, R_l2=1, S_l2=1,
        P_tile=4, Q_tile=1, R_tile=1, S_tile=1,
        block_h=1, block_w=28,
        H_in=56, W_in=56
    )
    
    # VGG-L1: R=3, S=3, P=224, Q=224
    # 从之前输出: L3 Loops P=14, Q=224; L2 Loops R=1, S=1
    # Tile: P=16, Q=1, R=3, S=1
    # Block: 1×226, Input: 226×226
    derive_formula(
        name="VGG-L1",
        P_l3=14, Q_l3=224, R_l2=1, S_l2=1,
        P_tile=16, Q_tile=1, R_tile=3, S_tile=1,
        block_h=1, block_w=226,
        H_in=226, W_in=226
    )


if __name__ == "__main__":
    main()
