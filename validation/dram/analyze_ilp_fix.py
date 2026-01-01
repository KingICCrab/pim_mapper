"""
分析 ILP 应该如何计算 row activations 以匹配 Trace

Trace 逻辑：
- 循环顺序: K → C → P → Q → R → S
- 每次地址计算: h = p * P_tile + r * R_tile, w = q * Q_tile + s * S_tile
- row = h // block_h
- 每次 row 变化时 +1

当前 ILP 逻辑：
- row_acts_aligned = Π bound_j^{xj} (只考虑 Level 3 factors)
- 没有考虑 Level 2 的 R, S 循环

需要修复：把 Level 2 的循环因子也算进去
"""
import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6

# 工作负载配置 - 从实际 mapping 提取
workloads = {
    'ResNet-L1': {
        'R': 7, 'S': 7, 'P': 56, 'Q': 56, 'C': 3, 'K': 64,
        # Level 3 (DRAM) factors
        'K_l3': 4, 'C_l3': 3, 'P_l3': 28, 'Q_l3': 7,
        # Level 2 factors
        'R_l2': 7, 'S_l2': 1,
        # Buffer tile (Level 0+1)
        'P_tile': 2, 'Q_tile': 8, 'R_tile': 1, 'S_tile': 7,
        # Ground truth from Trace
        'trace_total': 5376,
        # ILP prediction
        'ilp_current': 2392,  # = 2352 + 40
    },
}

block_h = block_w = 31

def analyze_workload(name, cfg):
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    
    K_l3, C_l3 = cfg['K_l3'], cfg['C_l3']
    P_l3, Q_l3 = cfg['P_l3'], cfg['Q_l3']
    R_l2, S_l2 = cfg['R_l2'], cfg['S_l2']
    P_tile, Q_tile = cfg['P_tile'], cfg['Q_tile']
    R_tile, S_tile = cfg['R_tile'], cfg['S_tile']
    
    H_tile = P_tile + R_tile - 1
    W_tile = Q_tile + S_tile - 1
    
    print(f"\n循环结构:")
    print(f"  Level 3 (DRAM): K={K_l3}, C={C_l3}, P={P_l3}, Q={Q_l3}")
    print(f"  Level 2:        R={R_l2}, S={S_l2}")
    print(f"  Buffer tile:    P={P_tile}, Q={Q_tile}, R={R_tile}, S={S_tile}")
    print(f"  H_tile={H_tile}, W_tile={W_tile}")
    
    # 当前 ILP 公式: K × C × P × Q (只有 Level 3)
    ilp_l3_only = K_l3 * C_l3 * P_l3 * Q_l3
    print(f"\n当前 ILP (Level 3 only):")
    print(f"  K × C × P × Q = {K_l3} × {C_l3} × {P_l3} × {Q_l3} = {ilp_l3_only}")
    
    # 包含 Level 2: K × C × P × Q × R × S
    ilp_with_l2 = K_l3 * C_l3 * P_l3 * Q_l3 * R_l2 * S_l2
    print(f"\n包含 Level 2:")
    print(f"  K × C × P × Q × R × S = {ilp_l3_only} × {R_l2} × {S_l2} = {ilp_with_l2}")
    
    # 但这还不对，因为不是每个循环迭代都产生 row switch
    # 需要计算实际的 row switches
    
    print(f"\n精确模拟 (prev_row tracking):")
    total_switches = 0
    prev_row = -1
    
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r_tile in range(R_l2):
                        for s_tile in range(S_l2):
                            h_start = p * P_tile + r_tile * R_tile
                            w_start = q * Q_tile + s_tile * S_tile
                            
                            # 遍历 tile 内元素
                            for h in range(h_start, h_start + H_tile):
                                for w in range(w_start, w_start + W_tile):
                                    row = h // block_h
                                    if row != prev_row:
                                        total_switches += 1
                                        prev_row = row
    
    print(f"  Total switches: {total_switches}")
    print(f"  Trace 结果:     {cfg['trace_total']}")
    print(f"  Match: {'✓' if total_switches == cfg['trace_total'] else '✗'}")
    
    # 分析规律
    print(f"\n规律分析:")
    
    # 方法1: 基于 Tile 的计算
    # 如果 Tile 小于 block，每个 Tile 只访问 1 个 row
    # 如果 Tile 跨越 block 边界，每个 Tile 访问多个 row
    
    # 计算每个 (p, q) tile 访问的 row 数
    rows_per_pq = {}
    for p in range(min(P_l3, 5)):  # 只看前 5 个
        for q in range(min(Q_l3, 5)):
            rows = set()
            for r_tile in range(R_l2):
                for s_tile in range(S_l2):
                    h_start = p * P_tile + r_tile * R_tile
                    w_start = q * Q_tile + s_tile * S_tile
                    for h in range(h_start, h_start + H_tile):
                        rows.add(h // block_h)
            rows_per_pq[(p, q)] = len(rows)
    
    print(f"  Rows per (p,q) tile (前 5×5):")
    for p in range(min(P_l3, 5)):
        row_str = " ".join(f"{rows_per_pq.get((p, q), '?'):2d}" for q in range(min(Q_l3, 5)))
        print(f"    p={p}: [{row_str}]")
    
    # 方法2: H 方向的 row 访问次数
    # 每个 unique h 值会被访问多少次？
    h_visits = {}
    for k in range(1):  # 只看 K=0
        for c in range(1):  # 只看 C=0
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r_tile in range(R_l2):
                        for s_tile in range(S_l2):
                            h_start = p * P_tile + r_tile * R_tile
                            for h in range(h_start, h_start + H_tile):
                                h_visits[h] = h_visits.get(h, 0) + 1
    
    print(f"\n  H 值访问次数 (K=0, C=0):")
    print(f"    Unique h values: {len(h_visits)}")
    print(f"    Total h visits: {sum(h_visits.values())}")
    
    # 每个 row 的访问次数
    row_visits = {}
    for h, count in h_visits.items():
        row = h // block_h
        row_visits[row] = row_visits.get(row, 0) + count
    
    print(f"    Unique rows: {len(row_visits)}")
    print(f"    Row visits: {row_visits}")
    
    return total_switches

# 运行分析
for name, cfg in workloads.items():
    analyze_workload(name, cfg)

print("\n" + "="*70)
print("结论")
print("="*70)
print("""
当前 ILP 问题：
1. 只计算 Level 3 factors (K × C × P × Q)
2. 没有考虑 Level 2 的 R, S 循环
3. 假设每个 DRAM tile 只产生 1 次 row activation

修复方向：
1. 把 Level 2 的 R, S 因子也加入 row_acts_aligned
2. 或者更精确地：计算每个 Tile 的实际 row switches

但实际上，Trace 的 row activation 是基于连续访问的 prev_row tracking，
这需要更复杂的公式来精确计算。

简化公式（保守估计）：
  row_acts = K × C × P × Q × R × S × rows_per_tile
  
其中 rows_per_tile = ceil(H_tile / block_h) × ceil(W_tile / block_w)
""")
