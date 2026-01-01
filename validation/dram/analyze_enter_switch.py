"""
分析 enter switch 来源
"""
import math

# 参数
workload = {'N': 1, 'H': 62, 'W': 62, 'C': 3, 'K': 64, 'R': 7, 'S': 7, 'P': 56, 'Q': 56, 'stride_h': 1, 'stride_w': 1}
mapping = {
    'P_dram': 28, 'Q_dram': 7, 'C_dram': 3, 'K_dram': 4,
    'R_dram': 7, 'S_dram': 1
}

H, W = workload['H'], workload['W']
P, Q, R, S = workload['P'], workload['Q'], workload['R'], workload['S']
C, K = workload['C'], workload['K']
stride_h, stride_w = workload['stride_h'], workload['stride_w']

P_dram, Q_dram = mapping['P_dram'], mapping['Q_dram']
R_dram = mapping['R_dram']
C_dram, K_dram = mapping['C_dram'], mapping['K_dram']

# 计算 tile 尺寸
P_tile = math.ceil(P / P_dram)  # 2
Q_tile = math.ceil(Q / Q_dram)  # 8
H_tile = P_tile + R - 1  # 8
W_tile = Q_tile + S - 1  # 14

# Block 尺寸 (HBM2)
block_h, block_w = 31, 31

# 循环顺序: K -> C -> P -> Q -> R (S_dram=1)
print('=' * 80)
print('参数:')
print(f'  H={H}, W={W}, R={R}, S={S}, P={P}, Q={Q}')
print(f'  P_dram={P_dram}, Q_dram={Q_dram}, R_dram={R_dram}')
print(f'  P_tile={P_tile}, Q_tile={Q_tile}, H_tile={H_tile}, W_tile={W_tile}')
print(f'  block_h={block_h}, block_w={block_w}')
print('=' * 80)
print()

# 生成 tiles 并分析 enter switch
tiles = []
last_blocks = None

for k in range(K_dram):
    for c in range(C_dram):
        for p in range(P_dram):
            for q in range(Q_dram):
                for r in range(R_dram):
                    # 计算 H/W 范围
                    h_start = p * P_tile * stride_h + r
                    h_end = h_start + H_tile
                    w_start = q * Q_tile * stride_w  # S=0
                    w_end = w_start + W_tile
                    
                    # 计算覆盖的 blocks
                    h_block_start = h_start // block_h
                    h_block_end = (h_end - 1) // block_h
                    w_block_start = w_start // block_w
                    w_block_end = (w_end - 1) // block_w
                    
                    h_cross = h_block_end > h_block_start
                    w_cross = w_block_end > w_block_start
                    
                    current_blocks = set()
                    for hb in range(h_block_start, h_block_end + 1):
                        for wb in range(w_block_start, w_block_end + 1):
                            current_blocks.add((hb, wb))
                    
                    # 确定 tile 类型
                    if h_cross and w_cross:
                        tile_type = 'both'
                    elif h_cross:
                        tile_type = 'h_only'
                    elif w_cross:
                        tile_type = 'w_only'
                    else:
                        tile_type = 'non'
                    
                    # 判断 enter switch
                    if last_blocks is None:
                        enter_switch = True  # 第一个 tile
                    else:
                        enter_switch = current_blocks.isdisjoint(last_blocks)
                    
                    tiles.append({
                        'k_idx': k, 'c_idx': c, 'p_idx': p, 'q_idx': q, 'r_idx': r,
                        'type': tile_type,
                        'enter_switch': enter_switch,
                        'h_start': h_start, 'h_end': h_end,
                        'w_start': w_start, 'w_end': w_end,
                        'blocks': current_blocks
                    })
                    
                    last_blocks = current_blocks

print(f'Total tiles: {len(tiles)}')

# 分析 enter switch 来源
print()
print('=' * 80)
print('分析 enter switch 来源')
print('=' * 80)

# 按 KC 分组
kc_groups = {}
for t in tiles:
    kc = (t['k_idx'], t['c_idx'])
    if kc not in kc_groups:
        kc_groups[kc] = []
    kc_groups[kc].append(t)

# 统计每种边界类型
total_first = 0
total_p_bound = 0
total_q_bound = 0
total_r_internal = 0

detail_by_type = {
    'non': {'first': 0, 'p': 0, 'q': 0, 'r': 0},
    'h_only': {'first': 0, 'p': 0, 'q': 0, 'r': 0},
    'w_only': {'first': 0, 'p': 0, 'q': 0, 'r': 0},
    'both': {'first': 0, 'p': 0, 'q': 0, 'r': 0}
}

for kc, kc_tiles_list in kc_groups.items():
    for i, t in enumerate(kc_tiles_list):
        if not t['enter_switch']:
            continue
            
        tile_type = t['type']
        p, q, r = t['p_idx'], t['q_idx'], t['r_idx']
        
        if i == 0:
            total_first += 1
            detail_by_type[tile_type]['first'] += 1
        else:
            prev = kc_tiles_list[i-1]
            prev_p, prev_q, prev_r = prev['p_idx'], prev['q_idx'], prev['r_idx']
            
            # 判断边界类型
            if prev_q == Q_dram-1 and q == 0 and prev_p + 1 == p:
                total_p_bound += 1
                detail_by_type[tile_type]['p'] += 1
            elif prev_r == R_dram-1 and r == 0:
                total_q_bound += 1
                detail_by_type[tile_type]['q'] += 1
            else:
                total_r_internal += 1
                detail_by_type[tile_type]['r'] += 1

print()
print('Enter Switch 来源分解:')
print(f'  First tile (KC 起始): {total_first}')
print(f'  P boundary (Q wrap, Q_max -> Q_0): {total_p_bound}')
print(f'  Q boundary (R wrap, R_max -> R_0): {total_q_bound}')
print(f'  R internal (R 循环内部): {total_r_internal}')
print(f'  Sum: {total_first + total_p_bound + total_q_bound + total_r_internal}')
print()

print('按类型分解:')
header = f"{'Type':8} | {'First':6} | {'P_bound':8} | {'Q_bound':8} | {'R_int':6} | {'Total':6}"
print(header)
print('-' * 60)
for typ, counts in detail_by_type.items():
    total = sum(counts.values())
    print(f'{typ:8} | {counts["first"]:6} | {counts["p"]:8} | {counts["q"]:8} | {counts["r"]:6} | {total:6}')

print()
print('=' * 80)
print('公式验证')
print('=' * 80)
print()

# 验证 R_internal 公式
print(f'R_internal total = {total_r_internal}')
print(f'  w_only R_int = {detail_by_type["w_only"]["r"]}')
print(f'  预测 w_only R_int: P_dram × (R_dram-1) × w_cross_Q × KC = 28 × 6 × 1 × 12 = {28*6*1*12}')
print()

# 验证 Q_bound 公式
print(f'Q_bound total = {total_q_bound}')

print()
print('=' * 80)
print('公式框架总结')
print('=' * 80)
print()

# 统计 internal switches
internal_total = 0
for t in tiles:
    if t['type'] == 'non':
        internal_total += 0
    elif t['type'] == 'h_only':
        internal_total += 1
    elif t['type'] == 'w_only':
        internal_total += 1
    elif t['type'] == 'both':
        internal_total += 3

# 统计 enter switches
enter_total = sum(1 for t in tiles if t['enter_switch'])

print(f'Internal switches (确定的): {internal_total}')
print(f'Enter switches (边界相关): {enter_total}')
print(f'Total row activations: {internal_total + enter_total}')
print()

# 按来源细分 enter
print('Enter 来源细分:')
print(f'  First (KC 起始): {total_first} = {K_dram * C_dram} KC')
print(f'  P_bound (Q wrap): {total_p_bound}')
print(f'  Q_bound (R wrap): {total_q_bound}')
print(f'  R_internal (R 循环): {total_r_internal}')
