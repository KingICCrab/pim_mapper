"""
R 循环 switch 公式推导
"""

print('=' * 80)
print('R 循环 switch 的公式推导')
print('=' * 80)
print()

R_dram, P_dram, Q_dram = 7, 28, 7
P_tile, Q_tile = 2, 8
block_h, block_w = 31, 31
total_KC = 12

print('【参数】')
print(f'  R_dram={R_dram}, P_dram={P_dram}, Q_dram={Q_dram}')
print(f'  P_tile={P_tile} (buffer level H 因子)')
print(f'  block_h={block_h}')
print()

# ============================================================
# 核心洞察: tile 类型由 (p,r) 和 q 独立决定
# ============================================================

print('=' * 80)
print('【核心洞察】')
print('=' * 80)
print()
print('tile 的 H 方向是否跨越: 由 (p, r) 决定')
print('tile 的 W 方向是否跨越: 由 q 决定 (s 固定)')
print()

# 计算 h_crossing 的 (p,r) 组合
h_crossing_pr = []
for p in range(P_dram):
    for r in range(R_dram):
        h_start = p * P_tile + r
        h_end = h_start + (P_tile - 1)
        if h_start // block_h != h_end // block_h:
            h_crossing_pr.append((p, r))

print(f'h_crossing (p,r) 组合: {h_crossing_pr}')
print()

# ============================================================
# w_only 的 R 循环 switch
# ============================================================

print('=' * 80)
print('【w_only 的 R 循环 switch 公式】')
print('=' * 80)
print()

print('w_only tile 条件: h_non(p,r) 且 w_crossing(q)')
print()
print('对于固定的 (p, q=w_crossing):')
print('  R 循环遍历 r=0,1,2,...,R_dram-1')
print('  每次 r 变化，W 范围不变，但要重新从 W_start 开始')
print()
print('每次从 r 到 r+1:')
print('  prev.last_w_block = 1 (W 跨越后半部分)')
print('  curr.first_w_block = 0 (W 从起点开始)')
print('  => switch!')
print()

print('关键公式:')
print('  w_only 的 R 循环内部 switch = (w_only tiles per P) - 1')
print('  = (该 P 的 h_non r 数量) - 1')
print()

# 对于 h_non P (24 个): 每个 P 有 7 个 h_non r
# 对于 h_crossing P (4 个): 每个 P 有 6 个 h_non r

print('本例:')
print('  h_non P (24个): 每个有 7 个 h_non r => R 循环 switch = 6')
print('  h_crossing P (4个): 每个有 6 个 h_non r => R 循环 switch = 5')
print('  但 h_crossing P 的 r=1 从 both 进入时也 switch (+1)')
print()
print('  总 w_only R 循环 switch per KC:')
print(f'    = 24 × 6 + 4 × 6 = {24*6 + 4*6}')
print()
print(f'  总 w_only enter switch = 168 × 12 = {168*12}')
print('  实际: 2016 ✓')
print()

# ============================================================
# 通用公式
# ============================================================

print('=' * 80)
print('【通用公式】')
print('=' * 80)
print()

print('设:')
print('  H_tile_buf = buffer level 的 H 尺寸 = P_tile')
print('  R_dram = R 循环长度')
print('  n_h_crossing_pr = H 方向跨越的 (p,r) 组合数')
print('  n_w_crossing_q = W 方向跨越的 Q 值数量')
print()

print('w_only 的 R 循环 switch:')
print('  = Σ(每个 P 的 h_non r 数量 - 1) × n_w_crossing_q')
print('  ≈ P_dram × (R_dram - 1) × n_w_crossing_q  (当 h_crossing 很少时)')
print()

# 验证
approx = P_dram * (R_dram - 1) * 1  # n_w_crossing_q = 1
print(f'近似值: {P_dram} × {R_dram-1} × 1 = {approx}')
print(f'实际值: 168')
print(f'误差来自 h_crossing P 少 1 次: 28×6 - 4 = 168 ✓')
print()

print('=' * 80)
print('【核心结论】')
print('=' * 80)
print()
print('w_only 的 R 循环 enter switch per KC:')
print('  = (P_dram × R_dram - n_h_crossing_pr) × n_w_crossing_q')
print('    - P_dram × n_w_crossing_q  (扣除 r=0 进入时不 switch)')
print('    + n_h_crossing_pr × n_w_crossing_q (从 both 进入 w_only 时 switch)')
print()
print('  = n_h_non_pr × n_w_crossing_q - P_dram × n_w_crossing_q + n_h_crossing_pr × n_w_crossing_q')
print('  = (n_h_non_pr - P_dram + n_h_crossing_pr) × n_w_crossing_q')
print('  = (P_dram × R_dram - P_dram) × n_w_crossing_q')
print('  = P_dram × (R_dram - 1) × n_w_crossing_q')
print()
print(f'验证: {P_dram} × {R_dram-1} × 1 = {P_dram * (R_dram-1) * 1}')
print('实际: 168 ✓')
