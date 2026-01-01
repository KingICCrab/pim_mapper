"""
最终验证：理解 Ground Truth 的含义

Ground Truth 448 = base_per_C (单个 C channel 内的 row switches)
Total = base_per_C × C_l3 × K_l3 = 448 × 3 × 4 = 5376

现在需要找到 base_per_C 的解析公式
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
        'base_per_C': 448,  # 这是我们要找公式的目标
        'trace_total': 5376,  # = 448 × 3 × 4
    },
    'ResNet-L3': {
        'R': 1, 'S': 1, 'P': 56, 'Q': 56, 'C': 64, 'K': 128,
        'K_l3': 4, 'C_l3': 8, 'P_l3': 2, 'Q_l3': 56,
        'R_l2': 1, 'S_l2': 1,
        'P_tile': 28, 'Q_tile': 1, 'R_tile': 1, 'S_tile': 1,
        'base_per_C': 448,  # 单个 C 的 row switches
        'trace_total': 14336,  # = 448 × 8 × 4
    },
    'VGG-L1': {
        'R': 3, 'S': 3, 'P': 224, 'Q': 224, 'C': 3, 'K': 64,
        'K_l3': 4, 'C_l3': 3, 'P_l3': 14, 'Q_l3': 224,
        'R_l2': 1, 'S_l2': 1,
        'P_tile': 16, 'Q_tile': 1, 'R_tile': 3, 'S_tile': 3,
        'base_per_C': 56448,  # 需要验证
        'trace_total': None,  # = base × 3 × 4
    },
}

block_h = block_w = 31

def compute_base_per_C(cfg):
    """精确计算 base_per_C（单个 C 内的 row switches）"""
    P_l3, Q_l3 = cfg['P_l3'], cfg['Q_l3']
    P_tile, Q_tile = cfg['P_tile'], cfg['Q_tile']
    R_tile, S_tile = cfg['R_tile'], cfg['S_tile']
    R_l2, S_l2 = cfg['R_l2'], cfg['S_l2']
    
    H_tile = P_tile + R_tile - 1
    W_tile = Q_tile + S_tile - 1
    
    switches = 0
    prev_row = -1
    
    # 单个 C channel 内的循环: P → Q → R → S
    for p in range(P_l3):
        for q in range(Q_l3):
            for r_tile in range(R_l2):
                for s_tile in range(S_l2):
                    h_start = p * P_tile + r_tile * R_tile
                    w_start = q * Q_tile + s_tile * S_tile
                    
                    for h in range(h_start, h_start + H_tile):
                        for w in range(w_start, w_start + W_tile):
                            row = h // block_h  # row_aligned layout
                            if row != prev_row:
                                switches += 1
                                prev_row = row
    
    return switches

print("="*70)
print("验证 base_per_C")
print("="*70)

for name, cfg in workloads.items():
    computed = compute_base_per_C(cfg)
    expected = cfg['base_per_C']
    match = '✓' if computed == expected else '✗'
    
    C_l3, K_l3 = cfg['C_l3'], cfg['K_l3']
    total = computed * C_l3 * K_l3
    
    print(f"\n{name}:")
    print(f"  Computed base_per_C: {computed}")
    print(f"  Expected base_per_C: {expected}")
    print(f"  Match: {match}")
    print(f"  Total = {computed} × {C_l3} × {K_l3} = {total}")

print("\n" + "="*70)
print("分析 base_per_C 的规律")
print("="*70)

for name, cfg in workloads.items():
    print(f"\n--- {name} ---")
    
    P_l3, Q_l3 = cfg['P_l3'], cfg['Q_l3']
    P_tile, Q_tile = cfg['P_tile'], cfg['Q_tile']
    R_tile, S_tile = cfg['R_tile'], cfg['S_tile']
    R_l2, S_l2 = cfg['R_l2'], cfg['S_l2']
    
    H_tile = P_tile + R_tile - 1
    W_tile = Q_tile + S_tile - 1
    
    # 计算每个 p 的 switches
    switches_per_p = []
    prev_row = -1
    
    for p in range(P_l3):
        p_switches = 0
        for q in range(Q_l3):
            for r_tile in range(R_l2):
                for s_tile in range(S_l2):
                    h_start = p * P_tile + r_tile * R_tile
                    w_start = q * Q_tile + s_tile * S_tile
                    
                    for h in range(h_start, h_start + H_tile):
                        for w in range(w_start, w_start + W_tile):
                            row = h // block_h
                            if row != prev_row:
                                p_switches += 1
                                prev_row = row
        switches_per_p.append(p_switches)
    
    # 分析规律
    unique_values = sorted(set(switches_per_p))
    counts = {v: switches_per_p.count(v) for v in unique_values}
    
    print(f"  P_l3={P_l3}, Q_l3={Q_l3}, R_l2={R_l2}, S_l2={S_l2}")
    print(f"  H_tile={H_tile}, W_tile={W_tile}")
    print(f"  Unique switches_per_p: {unique_values}")
    print(f"  Counts: {counts}")
    print(f"  Sum: {sum(switches_per_p)}")
    
    # 尝试找规律
    # switches_per_p[0] 应该是稳态值
    # 边界处可能不同
    
    base = min(switches_per_p)
    extra = sum(s - base for s in switches_per_p)
    
    print(f"  Base per p: {base}")
    print(f"  Extra switches: {extra}")
    print(f"  P_l3 × base = {P_l3 * base}")

print("\n" + "="*70)
print("关键发现")
print("="*70)
print("""
base_per_C 的计算依赖于：
1. 循环结构：P_l3 × Q_l3 × R_l2 × S_l2
2. Tile 大小：H_tile × W_tile
3. Block 大小：block_h × block_w
4. 边界位置：哪些 p 值会跨越 block 边界

公式形式：
  base_per_C = P_l3 × base_per_p + boundary_crossings

其中 base_per_p 是不跨边界时每个 p 的 switches
boundary_crossings 是跨边界时额外的 switches
""")
