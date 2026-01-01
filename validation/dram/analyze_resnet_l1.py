"""
深入分析 ResNet-L1 的 row 访问模式

核心问题: H_visits × W_visits = 1600, 但 actual = 448

差距因子: 1600 / 448 = 3.57

观察:
- Switches / unique rows = 112 
- 每个 row 平均被访问 112 次
- 但 H_visits × W_visits / 4 = 400, 仍然不对

让我追踪具体的访问模式
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

def trace_resnet_l1():
    """ResNet-L1 的参数"""
    P_l3, Q_l3, R_l2, S_l2 = 28, 7, 7, 1
    H_step, W_step = 2, 8
    R_step, S_step = 1, 7
    H_tile, W_tile = 2, 14
    block_h, block_w = 31, 31
    num_h_blocks, num_w_blocks = 2, 2
    
    print("ResNet-L1 Row Access Trace")
    print("=" * 70)
    print(f"Loops: P={P_l3}, Q={Q_l3}, R={R_l2}, S={S_l2}")
    print(f"Total iterations: P×Q×R×S = {P_l3 * Q_l3 * R_l2 * S_l2}")
    
    # 追踪 row 变化
    prev_row = -1
    switches = 0
    row_to_count = {}
    
    # 展示前几个迭代
    print(f"\n前20个迭代的 row 访问:")
    print(f"{'p':>3} {'q':>3} {'r':>3} | h_start | w_start | h_blk | w_blk | row | switch")
    print("-" * 70)
    
    count = 0
    for p in range(P_l3):
        for q in range(Q_l3):
            for r in range(R_l2):
                for s in range(S_l2):
                    h_start = p * H_step + r * R_step
                    w_start = q * W_step + s * S_step
                    
                    h_end = h_start + H_tile - 1
                    w_end = w_start + W_tile - 1
                    
                    h_blocks = list(range(h_start // block_h, h_end // block_h + 1))
                    w_blocks = list(range(w_start // block_w, w_end // block_w + 1))
                    
                    rows_this_iter = []
                    for hb in h_blocks:
                        for wb in w_blocks:
                            row = hb * num_w_blocks + wb
                            rows_this_iter.append(row)
                    
                    for row in sorted(set(rows_this_iter)):
                        row_to_count[row] = row_to_count.get(row, 0) + 1
                        if row != prev_row:
                            switches += 1
                            if count < 20:
                                print(f"{p:>3} {q:>3} {r:>3} | {h_start:>7} | {w_start:>7} | {h_blocks} | {w_blocks} | {row:>3} | *")
                            prev_row = row
                            count += 1
                        elif count < 20:
                            # print(f"{p:>3} {q:>3} {r:>3} | {h_start:>7} | {w_start:>7} | {h_blocks} | {w_blocks} | {row:>3} |")
                            pass  # 不打印重复的
    
    print(f"\nTotal switches: {switches}")
    print(f"Row visit counts: {row_to_count}")
    
    # 分析关键观察: 循环是 P -> Q -> R -> S
    # P 循环: h_start = 0, 2, 4, ...
    # Q 循环: w_start = 0, 8, 16, ...
    # R 循环: h_offset = 0, 1, 2, ..., 6
    # S 循环: 只有 S=1, 不变
    
    print("\n" + "=" * 70)
    print("关键分析:")
    print("=" * 70)
    
    # R 循环在最内层, 所以每个 (p, q) 组合会遍历 R=7 次
    # 但 r 只改变 h_offset, 不改变 h_block (因为 block_h=31 >> r_range=7)
    # 所以一个 (p,q) 内的 R 循环不会导致 row switch
    
    # 检查: 对于固定的 p, 不同的 q 会访问哪些 row?
    print("\n对于 p=0, 不同 q 访问的 rows:")
    for q in range(Q_l3):
        w_start = q * W_step
        w_end = w_start + W_tile - 1
        w_blocks = list(range(w_start // block_w, w_end // block_w + 1))
        print(f"  q={q}: w_start={w_start}, w_end={w_end}, w_blocks={w_blocks}")
    
    # 检查: 对于固定的 q, 不同的 p 会访问哪些 row?
    print("\n对于 q=0, 不同 p 访问的 rows (H 方向):")
    h_block_changes = []
    prev_hb = None
    for p in range(P_l3):
        for r in range(R_l2):
            h_start = p * H_step + r * R_step
            h_end = h_start + H_tile - 1
            h_blocks = list(range(h_start // block_h, h_end // block_h + 1))
            if h_blocks != prev_hb:
                print(f"  p={p}, r={r}: h_start={h_start}, h_end={h_end}, h_blocks={h_blocks}")
                prev_hb = h_blocks
                h_block_changes.append((p, r, h_blocks))
    
    print(f"\nH block changes: {len(h_block_changes)} times")
    
    # 现在计算真正的公式
    # 每次 p 循环的开始, Q 会从头遍历
    # 所以 switches 应该是 P 循环次数 × 每次 Q 遍历的 W 方向 switches
    
    # 但是! 关键: R 循环在 Q 循环内部!
    # 所以结构是: P -> Q -> R -> S
    # 对于每个 (p, q), 遍历 (r, s)
    
    # R 循环不会改变 w_blocks (因为 r 只影响 h)
    # 所以 R 循环内的 row 变化只来自 h_block 变化
    
    # 让我重新计算:
    # 1. 对于每个 p, 确定 H 方向的 block 变化次数
    # 2. 对于每个 q, 确定 W 方向的 block 变化次数
    
    print("\n" + "=" * 70)
    print("公式推导:")
    print("=" * 70)
    
    # 统计 P×R 循环中 H block 的 unique 变化
    h_unique_configs = []
    for p in range(P_l3):
        for r in range(R_l2):
            h_start = p * H_step + r * R_step
            h_end = h_start + H_tile - 1
            h_blocks = tuple(range(h_start // block_h, h_end // block_h + 1))
            if not h_unique_configs or h_unique_configs[-1] != h_blocks:
                h_unique_configs.append(h_blocks)
    
    print(f"H unique configurations: {len(h_unique_configs)}")
    # print(f"  {h_unique_configs[:10]}...")  # 只显示前10个
    
    # 统计 Q×S 循环中 W block 的 unique 变化
    w_unique_configs = []
    for q in range(Q_l3):
        for s in range(S_l2):
            w_start = q * W_step + s * S_step
            w_end = w_start + W_tile - 1
            w_blocks = tuple(range(w_start // block_w, w_end // block_w + 1))
            if not w_unique_configs or w_unique_configs[-1] != w_blocks:
                w_unique_configs.append(w_blocks)
    
    print(f"W unique configurations: {len(w_unique_configs)}")
    
    # 计算交叉
    # 由于循环顺序是 P -> Q -> R -> S
    # 对于每个 P, 会完整遍历 Q×R×S
    # 在 Q×R×S 内部, 先遍历 Q, 再遍历 R
    
    # 关键insight: 
    # - 当 Q 变化时, W block 可能变化
    # - 当 Q 固定, R 变化时, H block 可能变化
    # - 所以一个 P 循环内, 会产生 W_switches × R_h_switches 的变化?
    
    # 不对, 让我更仔细地分析
    
    # 对于每个 (p, q), 内部 R 循环会产生多少 row switches?
    switches_per_pq = 0
    for p in range(1):  # 只看 p=0
        for q in range(1):  # 只看 q=0
            prev_row_local = -1
            for r in range(R_l2):
                for s in range(S_l2):
                    h_start = p * H_step + r * R_step
                    w_start = q * W_step + s * S_step
                    
                    h_end = h_start + H_tile - 1
                    w_end = w_start + W_tile - 1
                    
                    h_blocks = list(range(h_start // block_h, h_end // block_h + 1))
                    w_blocks = list(range(w_start // block_w, w_end // block_w + 1))
                    
                    for hb in h_blocks:
                        for wb in w_blocks:
                            row = hb * num_w_blocks + wb
                            if row != prev_row_local:
                                switches_per_pq += 1
                                prev_row_local = row
    
    print(f"\nSwitches in (p=0, q=0) R-S loop: {switches_per_pq}")
    
    # 这应该等于 H_unique_in_R_loop × W_unique_in_S_loop
    # 在 ResNet-L1, S=1, 所以 W 方向没有变化
    # H 方向: r from 0 to 6, 但 block_h=31, 所以 h_block 不变
    
    # 所以每个 (p,q) 内部, row switches = 1 或者是 tile 跨越的 blocks 数
    
    print("\n分析每个 (p,q) 内部的 switches:")
    
    # 计算每个 (p,q) 的 switches
    switches_matrix = []
    for p in range(P_l3):
        row_switches = []
        for q in range(Q_l3):
            prev_row_local = -1
            count_local = 0
            for r in range(R_l2):
                for s in range(S_l2):
                    h_start = p * H_step + r * R_step
                    w_start = q * W_step + s * S_step
                    
                    h_end = h_start + H_tile - 1
                    w_end = w_start + W_tile - 1
                    
                    h_blocks = list(range(h_start // block_h, h_end // block_h + 1))
                    w_blocks = list(range(w_start // block_w, w_end // block_w + 1))
                    
                    for hb in h_blocks:
                        for wb in w_blocks:
                            row = hb * num_w_blocks + wb
                            if row != prev_row_local:
                                count_local += 1
                                prev_row_local = row
            row_switches.append(count_local)
        switches_matrix.append(row_switches)
    
    # 显示前几行
    print("Switches per (p,q):")
    for p in range(min(5, P_l3)):
        print(f"  p={p}: {switches_matrix[p]}")
    
    total_from_matrix = sum(sum(row) for row in switches_matrix)
    print(f"\nTotal from matrix: {total_from_matrix}")
    
    # 但这忽略了 q 之间的跨越!
    # 当 q 变化时, 可能需要额外的 switch
    
    # 正确的计算: 把整个 P->Q->R->S 当作一个大循环
    # 关键是: 每次 Q 增加, 会导致 W block 变化
    # 但 W block 变化后, 需要重新从 row 开始

    # 让我计算每个 P 循环内的 total switches (包括 Q 之间的跳转)
    print("\nSwitches per P (包括 Q 跳转):")
    switches_per_p = []
    for p in range(P_l3):
        prev_row_local = -1
        count_local = 0
        for q in range(Q_l3):
            for r in range(R_l2):
                for s in range(S_l2):
                    h_start = p * H_step + r * R_step
                    w_start = q * W_step + s * S_step
                    
                    h_end = h_start + H_tile - 1
                    w_end = w_start + W_tile - 1
                    
                    h_blocks = list(range(h_start // block_h, h_end // block_h + 1))
                    w_blocks = list(range(w_start // block_w, w_end // block_w + 1))
                    
                    for hb in h_blocks:
                        for wb in w_blocks:
                            row = hb * num_w_blocks + wb
                            if row != prev_row_local:
                                count_local += 1
                                prev_row_local = row
        switches_per_p.append(count_local)
    
    print(f"Switches per p: {switches_per_p[:10]}...")
    print(f"Sum: {sum(switches_per_p)}")
    
    # 这还是不对, 因为 P 之间的跳转也会导致 switch
    # 需要考虑全局的 prev_row 追踪


def main():
    trace_resnet_l1()


if __name__ == "__main__":
    main()
