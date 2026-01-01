#!/usr/bin/env python3
"""
分析 row activation 差异来源
"""

# ResNet-L1 参数
block_h, block_w = 31, 31
P_buf, Q_buf = 8, 2
R_buf, S_buf = 7, 1
stride_h, stride_w = 1, 1
H_per_tile, W_per_tile = 14, 2

P_l3, Q_l3, C_l3, K_l3 = 7, 28, 3, 4
S_l2 = 7

# Row buffer = 1024 bytes, 每个 element = 1 byte
row_buffer_bytes = 1024

# 地址 strides (from debug output)
stride_p_l3 = 1024   # h_block stride
stride_q_l3 = 7168   # w_block stride  
stride_c_l3 = 200704 # channel stride

print("=" * 80)
print("Row Activation 差异分析")
print("=" * 80)

# 模拟 trace 的 row activation 计算
# 按照修改后的访问顺序：for h_block: for w_block: for h: for w

current_row = None
row_switches = 0
tile_count = 0

# DRAM 循环
for k in range(K_l3):
    for c in range(C_l3):
        for q in range(Q_l3):
            for p in range(P_l3):
                for s in range(S_l2):
                    tile_count += 1
                    
                    # 计算 tile 的 H, W 范围
                    p_start = p * P_buf
                    q_start = q * Q_buf + s
                    h_start = p_start * stride_h
                    h_end = h_start + H_per_tile
                    w_start = q_start * stride_w
                    w_end = w_start + W_per_tile
                    
                    # 计算 block 范围
                    h_block_start = h_start // block_h
                    h_block_end = (h_end - 1) // block_h
                    w_block_start = w_start // block_w
                    w_block_end = (w_end - 1) // block_w
                    
                    # 按 block 顺序访问
                    for h_blk in range(h_block_start, h_block_end + 1):
                        for w_blk in range(w_block_start, w_block_end + 1):
                            # 计算这个 block 的 row
                            block_base = (h_blk * stride_p_l3 + 
                                         w_blk * stride_q_l3 + 
                                         c * stride_c_l3)
                            row = block_base // row_buffer_bytes
                            
                            if current_row != row:
                                row_switches += 1
                                current_row = row

print(f"\n模拟结果 (按 block 分组访问):")
print(f"  Total tiles: {tile_count}")
print(f"  Row switches: {row_switches}")

# 分解 row switch 来源
print(f"\n" + "=" * 80)
print("Row Switch 来源分解")
print("=" * 80)

# 1. Tile 内部的 block crossing
crossing_within_tile = 0
for k in range(K_l3):
    for c in range(C_l3):
        for q in range(Q_l3):
            for p in range(P_l3):
                for s in range(S_l2):
                    p_start = p * P_buf
                    q_start = q * Q_buf + s
                    h_start = p_start * stride_h
                    h_end = h_start + H_per_tile
                    w_start = q_start * stride_w
                    w_end = w_start + W_per_tile
                    
                    h_block_start = h_start // block_h
                    h_block_end = (h_end - 1) // block_h
                    w_block_start = w_start // block_w
                    w_block_end = (w_end - 1) // block_w
                    
                    # 计算这个 tile 访问了多少个 block
                    num_h_blocks = h_block_end - h_block_start + 1
                    num_w_blocks = w_block_end - w_block_start + 1
                    num_blocks = num_h_blocks * num_w_blocks
                    
                    # tile 内部的 crossing = num_blocks - 1
                    crossing_within_tile += (num_blocks - 1)

print(f"\n1. Tile 内部 block crossing (每个 crossing tile 贡献 1+):")
print(f"   Total: {crossing_within_tile}")

# 2. Tile 之间的切换
tile_boundary_switches = 0
prev_last_row = None

for k in range(K_l3):
    for c in range(C_l3):
        for q in range(Q_l3):
            for p in range(P_l3):
                for s in range(S_l2):
                    p_start = p * P_buf
                    q_start = q * Q_buf + s
                    h_start = p_start * stride_h
                    h_end = h_start + H_per_tile
                    w_start = q_start * stride_w
                    w_end = w_start + W_per_tile
                    
                    h_block_start = h_start // block_h
                    h_block_end = (h_end - 1) // block_h
                    w_block_start = w_start // block_w
                    w_block_end = (w_end - 1) // block_w
                    
                    # 这个 tile 的第一个 block
                    first_block_base = (h_block_start * stride_p_l3 + 
                                       w_block_start * stride_q_l3 + 
                                       c * stride_c_l3)
                    first_row = first_block_base // row_buffer_bytes
                    
                    # 这个 tile 的最后一个 block
                    last_block_base = (h_block_end * stride_p_l3 + 
                                      w_block_end * stride_q_l3 + 
                                      c * stride_c_l3)
                    last_row = last_block_base // row_buffer_bytes
                    
                    # tile 边界切换：前一个 tile 的最后 row 到当前 tile 的第一 row
                    if prev_last_row is not None and prev_last_row != first_row:
                        tile_boundary_switches += 1
                    
                    prev_last_row = last_row

print(f"\n2. Tile 之间的切换 (相邻 tile 访问不同 row):")
print(f"   Total: {tile_boundary_switches}")

# 3. 第一个 tile 的第一次访问
first_access = 1
print(f"\n3. 第一个 tile 的首次访问:")
print(f"   Total: {first_access}")

print(f"\n" + "=" * 80)
print("总结")
print("=" * 80)
theoretical_total = crossing_within_tile + tile_boundary_switches + first_access
print(f"\n  Tile 内部 crossing: {crossing_within_tile}")
print(f"  Tile 边界切换:      {tile_boundary_switches}")
print(f"  首次访问:           {first_access}")
print(f"  ---------------------------------")
print(f"  理论总计:           {theoretical_total}")
print(f"  模拟结果:           {row_switches}")
print(f"  Trace 实际:         5880")
print(f"  ILP 预测:           2384")

# ======================================================================
# 更详细的分析
# ======================================================================
print(f"\n" + "=" * 80)
print("Tile 边界切换的详细分析")
print("=" * 80)

# 按循环层次分析切换
s_loop_switches = 0
p_loop_switches = 0
q_loop_switches = 0
c_loop_switches = 0
k_loop_switches = 0

prev_row_info = None

for k in range(K_l3):
    for c in range(C_l3):
        for q in range(Q_l3):
            for p in range(P_l3):
                for s in range(S_l2):
                    p_start = p * P_buf
                    q_start = q * Q_buf + s
                    h_start = p_start * stride_h
                    w_start = q_start * stride_w
                    
                    h_block = h_start // block_h
                    w_block = w_start // block_w
                    
                    block_base = (h_block * stride_p_l3 + 
                                 w_block * stride_q_l3 + 
                                 c * stride_c_l3)
                    first_row = block_base // row_buffer_bytes
                    
                    if prev_row_info is not None:
                        prev_k, prev_c, prev_q, prev_p, prev_s, prev_row = prev_row_info
                        
                        if prev_row != first_row:
                            # 确定是哪个循环的边界
                            if prev_s != s and prev_p == p and prev_q == q and prev_c == c and prev_k == k:
                                s_loop_switches += 1
                            elif prev_p != p and prev_q == q and prev_c == c and prev_k == k:
                                p_loop_switches += 1
                            elif prev_q != q and prev_c == c and prev_k == k:
                                q_loop_switches += 1
                            elif prev_c != c and prev_k == k:
                                c_loop_switches += 1
                            else:
                                k_loop_switches += 1
                    
                    prev_row_info = (k, c, q, p, s, first_row)

print(f"\n  S 循环边界切换: {s_loop_switches}")
print(f"  P 循环边界切换: {p_loop_switches}")
print(f"  Q 循环边界切换: {q_loop_switches}")
print(f"  C 循环边界切换: {c_loop_switches}")
print(f"  K 循环边界切换: {k_loop_switches}")
print(f"  ---------------------------------")
print(f"  合计: {s_loop_switches + p_loop_switches + q_loop_switches + c_loop_switches + k_loop_switches}")

# ======================================================================
# ILP 模型分析
# ======================================================================
print(f"\n" + "=" * 80)
print("ILP 模型的 Row Activation 计算方式")
print("=" * 80)

# ILP 可能只计算了 crossing 贡献，忽略了 tile 边界切换
print(f"\n  ILP 预测: 2384")
print(f"\n  如果 ILP 只计算 crossing:")
h_crossing = 2352
w_crossing = 336
both_crossing = 48
unique_crossing = h_crossing + w_crossing - both_crossing
print(f"    H crossing: {h_crossing}")
print(f"    W crossing: {w_crossing}")
print(f"    Both: {both_crossing}")
print(f"    Unique crossing tiles: {unique_crossing}")
print(f"    每个 crossing tile 贡献 1 次 row act")
print(f"    Crossing 贡献: {crossing_within_tile}")

# 计算基础 row activation (非 crossing 的首次访问)
print(f"\n  如果 ILP 还计算了基础访问:")
# 假设 ILP 认为 non-crossing tiles 可以复用 row
# 需要的 unique rows
unique_rows = set()
for k in range(K_l3):
    for c in range(C_l3):
        for q in range(Q_l3):
            for p in range(P_l3):
                for s in range(S_l2):
                    p_start = p * P_buf
                    q_start = q * Q_buf + s
                    h_start = p_start * stride_h
                    w_start = q_start * stride_w
                    
                    h_block = h_start // block_h
                    w_block = w_start // block_w
                    
                    block_base = (h_block * stride_p_l3 + 
                                 w_block * stride_q_l3 + 
                                 c * stride_c_l3)
                    row = block_base // row_buffer_bytes
                    unique_rows.add(row)

print(f"    Unique rows 访问: {len(unique_rows)}")
print(f"    Crossing 贡献:    {crossing_within_tile}")
print(f"    合计:             {len(unique_rows) + crossing_within_tile}")
