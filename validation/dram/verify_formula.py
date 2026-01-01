"""
验证公式：total = base_per_C × C_l3 × K_l3

关键发现：base_per_C = P_l3 × Q_l3 × switches_per_tile
"""
import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

# 工作负载配置
workloads = {
    'ResNet-L1': {
        'R': 7, 'S': 7, 'P': 56, 'Q': 56, 'C': 3, 'K': 64,
        'K_l3': 4, 'C_l3': 3, 'P_l3': 28, 'Q_l3': 7,
        'R_l2': 7, 'S_l2': 1,
        'P_tile': 2, 'Q_tile': 8, 'R_tile': 1, 'S_tile': 7,
        'ground_truth': 448,
    },
    'ResNet-L2': {
        'R': 3, 'S': 3, 'P': 56, 'Q': 56, 'C': 64, 'K': 64,
        'K_l3': 4, 'C_l3': 8, 'P_l3': 28, 'Q_l3': 7,
        'R_l2': 1, 'S_l2': 3,
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

def compute_base_per_C(cfg):
    """精确计算 base_per_C"""
    P_l3, Q_l3 = cfg['P_l3'], cfg['Q_l3']
    P_tile, Q_tile = cfg['P_tile'], cfg['Q_tile']
    R_tile, S_tile = cfg['R_tile'], cfg['S_tile']
    R_l2, S_l2 = cfg['R_l2'], cfg['S_l2']
    
    H_tile = P_tile + R_tile - 1
    W_tile = Q_tile + S_tile - 1
    
    base_per_C = 0
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
                                base_per_C += 1
                                last_row = row
    return base_per_C

print("="*70)
print("验证公式: total = base_per_C × C_l3 × K_l3")
print("="*70)

for name, cfg in workloads.items():
    base_per_C = compute_base_per_C(cfg)
    C_l3, K_l3 = cfg['C_l3'], cfg['K_l3']
    computed_total = base_per_C * C_l3 * K_l3
    ground_truth = cfg['ground_truth']
    
    match = '✓' if computed_total == ground_truth else '✗'
    print(f"\n{name}:")
    print(f"  base_per_C = {base_per_C}")
    print(f"  C_l3 × K_l3 = {C_l3} × {K_l3} = {C_l3 * K_l3}")
    print(f"  Computed: {base_per_C} × {C_l3 * K_l3} = {computed_total}")
    print(f"  Ground Truth: {ground_truth}")
    print(f"  Match: {match}")

print("\n" + "="*70)
print("现在尝试找到 base_per_C 的解析公式")
print("="*70)

# base_per_C = sum_{p,q} switches(p,q)
# 需要分析 switches(p,q) 的规律

for name, cfg in workloads.items():
    print(f"\n--- {name} ---")
    
    P_l3, Q_l3 = cfg['P_l3'], cfg['Q_l3']
    P_tile, Q_tile = cfg['P_tile'], cfg['Q_tile']
    R_tile, S_tile = cfg['R_tile'], cfg['S_tile']
    R_l2, S_l2 = cfg['R_l2'], cfg['S_l2']
    
    H_tile = P_tile + R_tile - 1
    W_tile = Q_tile + S_tile - 1
    
    # 收集每个 p 的 switches 总数
    switches_per_p = []
    unique_patterns = {}
    
    for p in range(P_l3):
        p_switches = 0
        for q in range(Q_l3):
            last_row = None
            q_switches = 0
            for r_tile in range(R_l2):
                for s_tile in range(S_l2):
                    h_start = p * P_tile + r_tile * R_tile
                    w_start = q * Q_tile + s_tile * S_tile
                    for h in range(h_start, h_start + H_tile):
                        for w in range(w_start, w_start + W_tile):
                            row = h // block_h
                            if row != last_row:
                                q_switches += 1
                                last_row = row
            p_switches += q_switches
        switches_per_p.append(p_switches)
    
    # 找出有多少个不同的 switches_per_p 值
    unique_values = set(switches_per_p)
    print(f"  P_l3 = {P_l3}, Q_l3 = {Q_l3}")
    print(f"  Unique switches_per_p values: {sorted(unique_values)}")
    print(f"  Count per value: {[(v, switches_per_p.count(v)) for v in sorted(unique_values)]}")
    print(f"  Sum = {sum(switches_per_p)}")
    
    # 尝试找规律
    # 如果 tile 不跨越 block 边界，每个 (p,q) 贡献 1
    # 如果跨越 H 边界，贡献更多
    
    # 计算 H 方向的 block 边界位置
    print(f"\n  分析 H 方向:")
    print(f"  H_tile = {H_tile}, P_tile = {P_tile}")
    print(f"  H 边界周期 = {block_h}")
    
    # 对于每个 p，计算它的 tile 跨越了多少个 H block
    h_blocks_per_p = []
    for p in range(min(P_l3, 10)):
        h_start = p * P_tile
        h_end = h_start + H_tile + (R_l2 - 1) * R_tile - 1  # 考虑 L2 循环
        h_blocks = set(range(h_start // block_h, h_end // block_h + 1))
        h_blocks_per_p.append(len(h_blocks))
        if p < 5:
            print(f"    p={p}: h_range=[{h_start}, {h_end}], h_blocks={h_blocks}")
    
    # 计算总的 H blocks 数
    h_total = P_l3 * P_tile + cfg['R'] - 1  # = H dimension
    total_h_blocks = (h_total - 1) // block_h + 1
    print(f"  Total H = {h_total}, H_blocks = {total_h_blocks}")
