"""
最终公式推导

关键观察:
- ResNet-L1: switches_per_p 有两个值 [14, 28], 但大多数是 14
- ResNet-L2: switches_per_p 都是 29, 28 × 29 = 812 ✓
- ResNet-L3: switches_per_p 都是 224, 2 × 224 = 448 ✓
- VGG-L1: switches_per_p 都是 4032, 14 × 4032 = 56448 ✓

规律: total = P_l3 × (平均 switches_per_p)

关键问题: 如何计算 switches_per_p?

对于一个 P:
  - Q×R×S 循环
  - 每次迭代访问一组 rows
  - switches_per_p = unique_rows_in_P × 某个因子

让我分析每个 case
"""

import math


def analyze_switches_per_p(name, P_l3, Q_l3, R_l2, S_l2, P_tile, Q_tile, R_tile, S_tile,
                           block_h, block_w, H_in, W_in, stride_h=1, stride_w=1, dilation_h=1, dilation_w=1):
    """分析 switches_per_p 的规律."""
    
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
    print(f"L3: P={P_l3}, Q={Q_l3}")
    print(f"L2: R={R_l2}, S={S_l2}")
    print(f"Tile: H={H_per_tile}, W={W_per_tile}")
    print(f"Block: {block_h} × {block_w}")
    print(f"Num blocks: {num_h_blocks} × {num_w_blocks}")
    
    # 计算每个 P 的 switches
    switches_per_p = []
    prev_row = -1
    
    for p in range(P_l3):
        p_switches = 0
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
                                p_switches += 1
                                prev_row = row
        
        switches_per_p.append(p_switches)
    
    total = sum(switches_per_p)
    print(f"\nTotal: {total}")
    print(f"switches_per_p: {switches_per_p[:10]}...")
    
    # 分析每个 P 内的 unique rows
    unique_rows_per_p = []
    for p in range(P_l3):
        rows = set()
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
                            rows.add(row)
        
        unique_rows_per_p.append(len(rows))
    
    print(f"unique_rows_per_p: {unique_rows_per_p[:10]}...")
    
    # 计算比率: switches_per_p / unique_rows_per_p
    ratios = [s / u if u > 0 else 0 for s, u in zip(switches_per_p, unique_rows_per_p)]
    print(f"ratios: {[f'{r:.2f}' for r in ratios[:10]]}...")
    
    # 分析: 每个 P 内, W 方向的 unique row switches
    w_unique_rows_per_p = []
    for p in range(P_l3):
        # 固定 H, 看 W 方向有多少 unique w_blocks
        w_blocks_all = set()
        for q in range(Q_l3):
            for s in range(S_l2):
                w_start = q * W_step + s * S_step
                w_end = w_start + W_per_tile - 1
                for wb in range(w_start // block_w, w_end // block_w + 1):
                    w_blocks_all.add(wb)
        w_unique_rows_per_p.append(len(w_blocks_all))
    
    print(f"W unique blocks (same for all P): {w_unique_rows_per_p[0]}")
    
    # 每个 P 内的 H unique blocks
    h_unique_per_p = []
    for p in range(P_l3):
        h_blocks_all = set()
        for r in range(R_l2):
            h_start = p * H_step + r * R_step
            h_end = h_start + H_per_tile - 1
            for hb in range(h_start // block_h, h_end // block_h + 1):
                h_blocks_all.add(hb)
        h_unique_per_p.append(len(h_blocks_all))
    
    print(f"H unique blocks per P: {h_unique_per_p[:10]}...")
    
    # 尝试公式: switches_per_p[i] = H_unique[i] × W_unique × 某个因子
    predicted = [h * w_unique_rows_per_p[0] for h, w in zip(h_unique_per_p, w_unique_rows_per_p)]
    print(f"H_unique × W_unique: {predicted[:10]}...")
    
    # 比较
    print(f"\n比较 switches_per_p vs H×W:")
    for i in range(min(5, P_l3)):
        print(f"  P={i}: actual={switches_per_p[i]}, H×W={predicted[i]}, ratio={switches_per_p[i]/predicted[i] if predicted[i] else 0:.2f}")
    
    # =================================================================
    # 新思路: 考虑循环顺序
    # =================================================================
    print(f"\n循环顺序分析:")
    
    # 对于一个 P:
    # 循环: Q → R → S
    # 每当 Q 变化, W 方向的访问模式可能变化
    # 每当 R 变化, H 方向的访问模式可能变化 (但在同一个 Q 内)
    
    # 关键: 每个 Q 内的 R×S 循环访问的 rows
    print(f"  每个 (P=0, Q) 内的访问:")
    for q in range(min(5, Q_l3)):
        rows_in_q = []
        for r in range(R_l2):
            for s in range(S_l2):
                h_start = 0 * H_step + r * R_step
                w_start = q * W_step + s * S_step
                
                h_end = h_start + H_per_tile - 1
                w_end = w_start + W_per_tile - 1
                
                h_blocks = list(range(h_start // block_h, h_end // block_h + 1))
                w_blocks = list(range(w_start // block_w, w_end // block_w + 1))
                
                for hb in h_blocks:
                    for wb in w_blocks:
                        rows_in_q.append(hb * num_w_blocks + wb)
        
        unique_in_q = len(set(rows_in_q))
        # 计算这个 Q 内的 switches
        switches_in_q = 1
        for i in range(1, len(rows_in_q)):
            if rows_in_q[i] != rows_in_q[i-1]:
                switches_in_q += 1
        
        print(f"    Q={q}: rows={rows_in_q[:20]}{'...' if len(rows_in_q) > 20 else ''}, unique={unique_in_q}, switches={switches_in_q}")
    
    # =================================================================
    # 公式推导
    # =================================================================
    print(f"\n公式推导:")
    
    # 观察: W_unique × R_l2 可能接近 switches_per_q
    # 因为每个 R 循环都会重新遍历 W 方向的所有 blocks
    
    w_unique = w_unique_rows_per_p[0]
    expected_per_q = w_unique * R_l2
    print(f"  W_unique × R_l2 = {w_unique} × {R_l2} = {expected_per_q}")
    print(f"  Expected total per P ≈ {expected_per_q} × Q_l3 = {expected_per_q * Q_l3}")
    print(f"  Actual per P: {switches_per_p[0]}")


def main():
    # ResNet-L1
    analyze_switches_per_p(
        name="ResNet-L1",
        P_l3=28, Q_l3=7, R_l2=7, S_l2=1,
        P_tile=2, Q_tile=8, R_tile=1, S_tile=7,
        block_h=31, block_w=31,
        H_in=62, W_in=62
    )
    
    # ResNet-L2
    analyze_switches_per_p(
        name="ResNet-L2",
        P_l3=28, Q_l3=7, R_l2=1, S_l2=1,
        P_tile=2, Q_tile=8, R_tile=1, S_tile=3,
        block_h=58, block_w=2,
        H_in=58, W_in=58
    )
    
    # ResNet-L3
    analyze_switches_per_p(
        name="ResNet-L3",
        P_l3=2, Q_l3=56, R_l2=1, S_l2=1,
        P_tile=4, Q_tile=1, R_tile=1, S_tile=1,
        block_h=1, block_w=28,
        H_in=56, W_in=56
    )


if __name__ == "__main__":
    main()
