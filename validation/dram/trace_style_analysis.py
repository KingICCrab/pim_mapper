"""
精确分析 Trace 的 row activation 计算逻辑

关键发现：Trace 在连续 C 和 K 循环时会有 row buffer reuse
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
row_size = 1024  # elements per row

def analyze_workload_trace_style(name, cfg):
    """
    使用 Trace 的方式计算 row activations:
    1. 循环顺序: K_l3 → C_l3 → P_l3 → Q_l3 → R_l2 → S_l2 (外到内)
    2. Input 只和 C, P, Q, R, S 相关 (不和 K 相关)
    3. Row buffer state 追踪: 记住上次访问的 row, 不同 row 时 +1
    """
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    
    P_l3, Q_l3 = cfg['P_l3'], cfg['Q_l3']
    K_l3, C_l3 = cfg['K_l3'], cfg['C_l3']
    P_tile, Q_tile = cfg['P_tile'], cfg['Q_tile']
    R_tile, S_tile = cfg['R_tile'], cfg['S_tile']
    R_l2, S_l2 = cfg['R_l2'], cfg['S_l2']
    
    H_tile = P_tile + R_tile - 1
    W_tile = Q_tile + S_tile - 1
    
    print(f"Loop structure:")
    print(f"  K_l3={K_l3}, C_l3={C_l3}, P_l3={P_l3}, Q_l3={Q_l3}")
    print(f"  R_l2={R_l2}, S_l2={S_l2}")
    print(f"  H_tile={H_tile}, W_tile={W_tile}")
    
    # Trace 风格的计算: 追踪 row buffer state
    row_acts = 0
    last_row = None  # row_aligned layout: row = h // block_h
    
    # 循环顺序: K → C → P → Q → R → S
    # Input 在 K 变化时不变 (K 只影响 Weight 和 Output)
    # Input 在 C 变化时改变 (访问不同 channel)
    # Input 在 P, Q, R, S 变化时改变 (访问不同 H, W 位置)
    
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r_tile in range(R_l2):
                        for s_tile in range(S_l2):
                            # 计算这次访问的 Input 位置
                            h_start = p * P_tile + r_tile * R_tile
                            w_start = q * Q_tile + s_tile * S_tile
                            
                            # 遍历 tile 内的所有元素
                            for h in range(h_start, h_start + H_tile):
                                for w in range(w_start, w_start + W_tile):
                                    # row_aligned layout: row = h // block_h
                                    row = h // block_h
                                    if row != last_row:
                                        row_acts += 1
                                        last_row = row
    
    print(f"\n结果:")
    print(f"  Ground Truth: {cfg['ground_truth']}")
    print(f"  Trace-style计算: {row_acts}")
    print(f"  Match: {'✓' if row_acts == cfg['ground_truth'] else '✗'}")
    
    # 如果不匹配，分析原因
    if row_acts != cfg['ground_truth']:
        print(f"\n差异分析:")
        print(f"  row_acts = {row_acts}")
        print(f"  ratio = {row_acts / cfg['ground_truth']:.2f}")
        
        # K_l3 循环：Input 不应该在 K 变化时重新访问
        # 但上面的循环在每次 K 迭代时都从 h=0 开始，导致 last_row 被重置
        
        # 正确的逻辑：当 K 变化但 C, P, Q, R, S 不变时，Input 地址相同
        # 所以不应该产生新的 row activation
        
        print(f"\n检查：K 循环不应该对 Input 产生 row activations")
        print(f"  实际 row_acts / K_l3 = {row_acts / K_l3}")
        print(f"  应该接近 ground_truth 如果 K 是唯一的外层循环")

    return row_acts

# 运行分析
print("="*70)
print("分析1: 简单的 Trace 风格计算（有问题）")
print("="*70)

for name, cfg in workloads.items():
    analyze_workload_trace_style(name, cfg)

print("\n" + "="*70)
print("分析2: 正确考虑 Input 的 K-reuse")
print("="*70)

def analyze_workload_with_k_reuse(name, cfg):
    """
    考虑 K-reuse：Input 不依赖 K，所以 K 循环内访问相同 Input tile
    """
    print(f"\n--- {name} ---")
    
    P_l3, Q_l3 = cfg['P_l3'], cfg['Q_l3']
    K_l3, C_l3 = cfg['K_l3'], cfg['C_l3']
    P_tile, Q_tile = cfg['P_tile'], cfg['Q_tile']
    R_tile, S_tile = cfg['R_tile'], cfg['S_tile']
    R_l2, S_l2 = cfg['R_l2'], cfg['S_l2']
    
    H_tile = P_tile + R_tile - 1
    W_tile = Q_tile + S_tile - 1
    
    # 方法：K 循环外部追踪 row_buffer，K 循环内复用
    # 等价于：计算单次 K 迭代的 row_acts × 1 (因为 K 内完全复用)
    
    row_acts = 0
    last_row = None
    
    # 只迭代 Input 相关的循环: C → P → Q → R → S
    # K 维度不影响 Input (完全复用)
    
    for c in range(C_l3):
        for p in range(P_l3):
            for q in range(Q_l3):
                for r_tile in range(R_l2):
                    for s_tile in range(S_l2):
                        h_start = p * P_tile + r_tile * R_tile
                        w_start = q * Q_tile + s_tile * S_tile
                        
                        for h in range(h_start, h_start + H_tile):
                            for w in range(w_start, w_start + W_tile):
                                row = h // block_h
                                if row != last_row:
                                    row_acts += 1
                                    last_row = row
    
    # 但是！K 循环是最外层，所以每次 K 迭代开始时，C, P, Q, R, S 都从 0 开始
    # 如果 K 是最外层，那么每次 K 迭代都会产生相同的 row_acts
    # 总 row_acts = single_pass × K_l3 ???
    
    # 不对！因为 row buffer 在 K 循环之间会被其他操作（Weight, Output）使用
    # 需要看循环是否真的能保持 row buffer state
    
    print(f"  Single pass (C,P,Q,R,S): {row_acts}")
    print(f"  Ground Truth: {cfg['ground_truth']}")
    print(f"  single_pass == GT: {'✓' if row_acts == cfg['ground_truth'] else '✗'}")
    
    # 尝试 × K_l3
    print(f"  Single pass × K_l3 = {row_acts * K_l3}")
    print(f"  Ground Truth / K_l3 = {cfg['ground_truth'] / K_l3:.2f}")
    
    return row_acts

for name, cfg in workloads.items():
    analyze_workload_with_k_reuse(name, cfg)

print("\n" + "="*70)
print("关键发现")
print("="*70)
print("""
Row activation 计数关键点：
1. Input 不依赖 K 维度，所以 K 循环内可以复用 row buffer
2. 但实际 Trace 的循环结构是什么？
   - 如果循环顺序是 K → C → P → Q → R → S，
     那么每次 K 迭代都会重新访问相同的 Input rows
     (假设 Weight/Output 访问没有 flush Input 的 row buffer)
3. 需要检查实际的 Trace generator 循环结构

可能的正确公式：
- 如果 Input row buffer 在 K 循环间被保持: row_acts = single_pass
- 如果 Input row buffer 在 K 循环间被 flush: row_acts = single_pass × K_l3
""")
