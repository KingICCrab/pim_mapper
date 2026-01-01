"""
修正版：包含 C 维度对 row 计算的影响

row = (c * num_h_blocks + h_block) * num_w_blocks + w_block

这意味着每个 C channel 有自己的 row 空间
"""
import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6

# 工作负载配置
cfg = {
    'R': 7, 'S': 7, 'P': 56, 'Q': 56, 'C': 3, 'K': 64,
    'K_l3': 4, 'C_l3': 3, 'P_l3': 28, 'Q_l3': 7,
    'R_l2': 7, 'S_l2': 1,
    'P_tile': 2, 'Q_tile': 8, 'R_tile': 1, 'S_tile': 7,
    'trace_total': 5376,
}

block_h = block_w = 31
num_h_blocks = 2  # ceil(H_total / block_h)
num_w_blocks = 2  # ceil(W_total / block_w)

K_l3, C_l3 = cfg['K_l3'], cfg['C_l3']
P_l3, Q_l3 = cfg['P_l3'], cfg['Q_l3']
R_l2, S_l2 = cfg['R_l2'], cfg['S_l2']
P_tile, Q_tile = cfg['P_tile'], cfg['Q_tile']
R_tile, S_tile = cfg['R_tile'], cfg['S_tile']

H_tile = P_tile + R_tile - 1
W_tile = Q_tile + S_tile - 1

print(f"Config: K_l3={K_l3}, C_l3={C_l3}, P_l3={P_l3}, Q_l3={Q_l3}")
print(f"        R_l2={R_l2}, S_l2={S_l2}")
print(f"        H_tile={H_tile}, W_tile={W_tile}")
print(f"        num_h_blocks={num_h_blocks}, num_w_blocks={num_w_blocks}")

# 方法 1: 使用 row = (c * num_h_blocks + h_block) * num_w_blocks + w_block
print("\n" + "="*70)
print("方法 1: row = (c * num_h_blocks + h_block) * num_w_blocks + w_block")
print("="*70)

total_switches_v1 = 0
prev_row = -1

for k in range(K_l3):
    for c in range(C_l3):
        for p in range(P_l3):
            for q in range(Q_l3):
                for r_tile in range(R_l2):
                    for s_tile in range(S_l2):
                        h_start = p * P_tile + r_tile * R_tile
                        w_start = q * Q_tile + s_tile * S_tile
                        
                        for h in range(h_start, h_start + H_tile):
                            for w in range(w_start, w_start + W_tile):
                                h_block = h // block_h
                                w_block = w // block_w
                                # C 参与 row 计算
                                row = (c * num_h_blocks + h_block) * num_w_blocks + w_block
                                if row != prev_row:
                                    total_switches_v1 += 1
                                    prev_row = row

print(f"Total switches: {total_switches_v1}")
print(f"Trace target:   {cfg['trace_total']}")
print(f"Match: {'✓' if total_switches_v1 == cfg['trace_total'] else '✗'}")

# 方法 2: 不包含 C (row = h // block_h)
print("\n" + "="*70)
print("方法 2: row = h // block_h (不含 C)")
print("="*70)

total_switches_v2 = 0
prev_row = -1

for k in range(K_l3):
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
                                if row != prev_row:
                                    total_switches_v2 += 1
                                    prev_row = row

print(f"Total switches: {total_switches_v2}")
print(f"Trace target:   {cfg['trace_total']}")
print(f"Match: {'✓' if total_switches_v2 == cfg['trace_total'] else '✗'}")

# 分析差异
print("\n" + "="*70)
print("分析")
print("="*70)

print(f"""
方法 1 (含 C): {total_switches_v1}
方法 2 (不含 C): {total_switches_v2}
Trace 目标: {cfg['trace_total']}

关键问题: Trace 使用的是哪种 row 计算方式?

如果使用方法 1 (含 C):
- row = (c * num_h_blocks + h_block) * num_w_blocks + w_block
- 每个 C channel 有独立的 row 空间
- row 数 = C * num_h_blocks * num_w_blocks

如果使用方法 2 (不含 C):
- row = h // block_h
- 所有 C channels 共享相同的 row 空间
- row 数 = num_h_blocks

让我们检查 Trace generator 使用的是哪种方式...
""")

# 计算理论值
print("\n理论分析:")
print(f"  如果每个 (K, C, P, Q, R, S) 产生 1 次 activation:")
print(f"    Total = K × C × P × Q × R × S = {K_l3 * C_l3 * P_l3 * Q_l3 * R_l2 * S_l2}")

print(f"\n  如果方法 1 的 row 数为 C × num_h_blocks × num_w_blocks:")
print(f"    Unique rows = {C_l3} × {num_h_blocks} × {num_w_blocks} = {C_l3 * num_h_blocks * num_w_blocks}")
print(f"    但 switches 数取决于访问模式...")

print(f"\n  比例分析:")
print(f"    方法 1 / 方法 2 = {total_switches_v1} / {total_switches_v2} = {total_switches_v1 / total_switches_v2:.2f}")
print(f"    K_l3 × C_l3 = {K_l3 * C_l3}")
