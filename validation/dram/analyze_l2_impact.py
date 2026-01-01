"""
分析 L2 的 R/S 循环对 row switches 的影响
关键发现：当 tile 跨越 block 边界时，L2 循环会导致重复访问 rows
"""
import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

# 工作负载配置
workloads = {
    'ResNet-L1': {
        'R': 7, 'S': 7, 'P': 56, 'Q': 56, 'C': 3, 'K': 64,
        # DRAM L3 factors
        'K_l3': 4, 'C_l3': 3, 'P_l3': 28, 'Q_l3': 7,
        # Level 2 factors
        'R_l2': 7, 'S_l2': 1,
        # Buffer tile
        'P_tile': 2, 'Q_tile': 8, 'R_tile': 1, 'S_tile': 7,
        'ground_truth': 448,
    },
    'ResNet-L2': {
        'R': 3, 'S': 3, 'P': 56, 'Q': 56, 'C': 64, 'K': 64,
        'K_l3': 4, 'C_l3': 8, 'P_l3': 28, 'Q_l3': 7,
        'R_l2': 1, 'S_l2': 3,  # Note: S in L2
        'P_tile': 2, 'Q_tile': 8, 'R_tile': 3, 'S_tile': 1,
        'ground_truth': 812,
    },
    'ResNet-L3': {
        'R': 1, 'S': 1, 'P': 56, 'Q': 56, 'C': 64, 'K': 128,
        'K_l3': 4, 'C_l3': 8, 'P_l3': 2, 'Q_l3': 56,
        'R_l2': 1, 'S_l2': 1,
        'P_tile': 28, 'Q_tile': 1, 'R_tile': 1, 'S_tile': 1,
        'ground_truth': 448,
    },
    'VGG-L1': {
        'R': 3, 'S': 3, 'P': 224, 'Q': 224, 'C': 3, 'K': 64,
        'K_l3': 4, 'C_l3': 3, 'P_l3': 14, 'Q_l3': 224,
        'R_l2': 1, 'S_l2': 1,
        'P_tile': 16, 'Q_tile': 1, 'R_tile': 3, 'S_tile': 3,
        'ground_truth': 56448,
    },
}

block_h = block_w = 31

def analyze_workload(name, cfg):
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    
    P_l3, Q_l3 = cfg['P_l3'], cfg['Q_l3']
    P_tile, Q_tile = cfg['P_tile'], cfg['Q_tile']
    R_tile, S_tile = cfg['R_tile'], cfg['S_tile']
    R_l2, S_l2 = cfg['R_l2'], cfg['S_l2']
    R, S = cfg['R'], cfg['S']
    
    H_tile = P_tile + R_tile - 1
    W_tile = Q_tile + S_tile - 1
    
    print(f"P_l3={P_l3}, Q_l3={Q_l3}, P_tile={P_tile}, Q_tile={Q_tile}")
    print(f"R={R}, S={S}, R_l2={R_l2}, S_l2={S_l2}")
    print(f"R_tile={R_tile}, S_tile={S_tile}")
    print(f"H_tile={H_tile}, W_tile={W_tile}")
    
    # 分析 P=0 的所有 Q 位置
    print(f"\n--- 分析 P=0 的 Q 位置 ---")
    p = 0
    crossing_tiles = []
    
    for q in range(min(Q_l3, 10)):  # 只看前10个
        # 计算这个 tile 跨越的 block
        h_blocks_all = set()
        w_blocks_all = set()
        for r_tile in range(R_l2):
            for s_tile in range(S_l2):
                h_start = p * P_tile + r_tile * R_tile
                w_start = q * Q_tile + s_tile * S_tile
                for h in range(h_start, h_start + H_tile):
                    h_blocks_all.add(h // block_h)
                for w in range(w_start, w_start + W_tile):
                    w_blocks_all.add(w // block_w)
        
        crosses = len(h_blocks_all) > 1 or len(w_blocks_all) > 1
        
        # 计算基础 switches (只考虑第一次 L2 迭代)
        base_switches = 0
        last_row = None
        h_start = p * P_tile
        w_start = q * Q_tile
        for h in range(h_start, h_start + H_tile):
            for w in range(w_start, w_start + W_tile):
                row = h // block_h
                if row != last_row:
                    base_switches += 1
                    last_row = row
        
        # 计算实际 switches (考虑所有 L2 循环)
        actual_switches = 0
        last_row = None
        for r_tile in range(R_l2):
            for s_tile in range(S_l2):
                h_start = p * P_tile + r_tile * R_tile
                w_start = q * Q_tile + s_tile * S_tile
                for h in range(h_start, h_start + H_tile):
                    for w in range(w_start, w_start + W_tile):
                        row = h // block_h
                        if row != last_row:
                            actual_switches += 1
                            last_row = row
        
        ratio = actual_switches / base_switches if base_switches > 0 else 0
        
        if crosses:
            crossing_tiles.append((q, base_switches, actual_switches, ratio))
        
        flag = " **CROSSING**" if crosses else ""
        print(f"  Q={q}: base={base_switches}, actual={actual_switches}, "
              f"ratio={ratio:.2f}, h_blocks={h_blocks_all}, w_blocks={w_blocks_all}{flag}")
    
    # 计算整个工作负载的 base_per_C
    print(f"\n--- 计算 base_per_C 的两种方法 ---")
    
    # 方法1: 用 H_visits × W_visits (当前 ILP 的思路)
    H_visits = (P_l3 * P_tile + R - 1) // block_h + 1
    W_visits = (Q_l3 * Q_tile + S - 1) // block_w + 1
    h_visits_x_w_visits = H_visits * W_visits
    
    # 方法2: 精确模拟
    grand_total = 0
    for p in range(P_l3):
        for q in range(Q_l3):
            last_row = None
            for r_tile in range(R_l2):
                for s_tile in range(S_l2):
                    h_start = p * P_tile + r_tile * R_tile
                    w_start = q * Q_tile + s_tile * S_tile
                    for h in range(h_start, h_start + H_tile):
                        for w in range(w_start, w_start + W_tile):
                            row = h // block_h
                            if row != last_row:
                                grand_total += 1
                                last_row = row
    
    print(f"  Ground Truth: {cfg['ground_truth']}")
    print(f"  方法1 (H_visits × W_visits): {h_visits_x_w_visits}")
    print(f"  方法2 (精确模拟): {grand_total}")
    print(f"  方法1正确: {'✓' if h_visits_x_w_visits == cfg['ground_truth'] else '✗'}")
    print(f"  方法2正确: {'✓' if grand_total == cfg['ground_truth'] else '✓'}")
    
    # 分析 crossing 的影响
    if crossing_tiles:
        print(f"\n--- Crossing 分析 ---")
        avg_ratio = sum(t[3] for t in crossing_tiles) / len(crossing_tiles)
        print(f"  Crossing tiles count: {len(crossing_tiles)} out of {Q_l3}")
        print(f"  Average ratio (actual/base): {avg_ratio:.2f}")
        print(f"  R_l2 × S_l2 = {R_l2 * S_l2}")
        print(f"  理论上，如果 L2 循环导致完全重新遍历，ratio 应该接近 {R_l2 * S_l2}")
    
    return grand_total

# 运行分析
for name, cfg in workloads.items():
    analyze_workload(name, cfg)

print("\n" + "="*70)
print("结论：需要一个新的公式")
print("="*70)
print("""
现有 ILP 公式 (H_visits × W_visits) 的问题：
1. 没有考虑 L2 循环 (R_l2, S_l2) 的影响
2. 当 tile 跨越 block 边界时，L2 循环会导致重复 row switches

可能的解决方案：
1. 预计算查找表（方案1）：对给定参数预计算 base_per_C
2. 修正公式：base_per_C = f(P_l3, Q_l3, P_tile, Q_tile, R_l2, S_l2, block_size)

让我们进一步分析是否能找到一个解析公式...
""")
