#!/usr/bin/env python3
"""
最终精确分析 5880 row activations

关键发现：
- 地址计算方式影响 row activation
- Trace 使用 block-wise layout
- 需要精确模拟 Trace 的地址计算
"""

print("=" * 70)
print("精确模拟 Trace 的 Row Activation")
print("=" * 70)

# 参数
block_h, block_w = 31, 31
row_buffer_bytes = 1024
element_bytes = 2

input_H, input_W, input_C = 62, 62, 3
num_h_blocks = (input_H + block_h - 1) // block_h  # = 2
num_w_blocks = (input_W + block_w - 1) // block_w  # = 2

P_buf, Q_buf = 8, 2
R_buf, S_buf = 7, 1
stride_h, stride_w = 1, 1

P_l3, Q_l3, C_l3, K_l3 = 7, 28, 3, 4
S_l2 = 7

H_per_tile = 14
W_per_tile = 2

print(f"Block layout: ({block_h}, {block_w})")
print(f"Num blocks per channel: ({num_h_blocks}, {num_w_blocks})")

def calc_block_addr(c, h_block, w_block, h_in_block, w_in_block):
    """
    计算 block layout 下的地址 (bytes)
    
    Layout: [C][H_block][W_block][h_in_block][w_in_block]
    """
    block_elements = block_h * block_w
    
    # Channel offset
    channel_stride = num_h_blocks * num_w_blocks * block_elements * element_bytes
    base_c = c * channel_stride
    
    # H block offset  
    h_block_stride = num_w_blocks * block_elements * element_bytes
    base_h = h_block * h_block_stride
    
    # W block offset
    w_block_stride = block_elements * element_bytes
    base_w = w_block * w_block_stride
    
    # Intra-block offset
    offset = (h_in_block * block_w + w_in_block) * element_bytes
    
    return base_c + base_h + base_w + offset

def calc_row(addr):
    return addr // row_buffer_bytes

# 打印地址空间布局
print("\n地址空间布局:")
for c in range(input_C):
    for h_blk in range(num_h_blocks):
        for w_blk in range(num_w_blocks):
            addr_start = calc_block_addr(c, h_blk, w_blk, 0, 0)
            addr_end = calc_block_addr(c, h_blk, w_blk, block_h-1, block_w-1)
            row_start = calc_row(addr_start)
            row_end = calc_row(addr_end)
            print(f"  C={c}, Block({h_blk},{w_blk}): addr=[{addr_start}, {addr_end}], rows=[{row_start}, {row_end}]")

# 现在精确模拟 Trace 的访问模式
print("\n" + "=" * 70)
print("精确模拟 Trace 访问模式")
print("=" * 70)

# Trace 的循环顺序 (for Input):
# K_l3 -> C_l3 -> Q_l3 -> P_l3 -> S_l2
# 然后对于每个 tile, 使用 block-wise 访问:
# h_block -> w_block -> h_in_block -> w_in_block

def simulate_trace_exact():
    """精确模拟 Trace 的访问和 row activation"""
    last_row = None
    total_row_acts = 0
    row_visits = {}
    
    for k in range(K_l3):
        for c in range(C_l3):
            for q in range(Q_l3):
                for p in range(P_l3):
                    for s in range(S_l2):
                        # 计算 tile 范围
                        p_start = p * P_buf
                        q_start = q * Q_buf + s
                        
                        h_start = p_start * stride_h
                        h_end = min(h_start + H_per_tile, input_H)
                        w_start = q_start * stride_w
                        w_end = min(w_start + W_per_tile, input_W)
                        
                        # Block 范围
                        h_blk_s = h_start // block_h
                        h_blk_e = (h_end - 1) // block_h
                        w_blk_s = w_start // block_w
                        w_blk_e = (w_end - 1) // block_w
                        
                        # Block-wise 访问
                        for h_blk in range(h_blk_s, h_blk_e + 1):
                            for w_blk in range(w_blk_s, w_blk_e + 1):
                                # 计算此 block 内需要访问的 h, w 范围
                                h_lo = max(h_start, h_blk * block_h)
                                h_hi = min(h_end, (h_blk + 1) * block_h)
                                w_lo = max(w_start, w_blk * block_w)
                                w_hi = min(w_end, (w_blk + 1) * block_w)
                                
                                # 遍历 block 内的元素
                                for h in range(h_lo, h_hi):
                                    for w in range(w_lo, w_hi):
                                        h_in_block = h % block_h
                                        w_in_block = w % block_w
                                        
                                        addr = calc_block_addr(c, h_blk, w_blk, h_in_block, w_in_block)
                                        row = calc_row(addr)
                                        
                                        if last_row is None or row != last_row:
                                            total_row_acts += 1
                                            row_visits[row] = row_visits.get(row, 0) + 1
                                        last_row = row
    
    return total_row_acts, row_visits

total, row_visits = simulate_trace_exact()

print(f"\n精确模拟结果: {total} row activations")
print(f"Trace 实际值: 5880")
print(f"差值: {5880 - total}")

print("\n每个 row 的访问次数:")
for row in sorted(row_visits.keys()):
    print(f"  Row {row}: {row_visits[row]} visits")

# 如果还有差值，可能是因为 Trace 的统计逻辑不同
# 让我检查 Trace 是否按 tile 级别统计

print("\n" + "=" * 70)
print("按 tile 级别统计 (每个 tile 访问一个新 row 时计数)")  
print("=" * 70)

def simulate_tile_level():
    """每个 tile 开始时检查 row 是否变化"""
    last_row = None
    total_row_acts = 0
    
    for k in range(K_l3):
        for c in range(C_l3):
            for q in range(Q_l3):
                for p in range(P_l3):
                    for s in range(S_l2):
                        # 计算第一个元素的 row
                        p_start = p * P_buf
                        q_start = q * Q_buf + s
                        
                        h_start = p_start * stride_h
                        w_start = q_start * stride_w
                        
                        h_blk = h_start // block_h
                        w_blk = w_start // block_w
                        h_in = h_start % block_h
                        w_in = w_start % block_w
                        
                        addr = calc_block_addr(c, h_blk, w_blk, h_in, w_in)
                        row = calc_row(addr)
                        
                        if last_row is None or row != last_row:
                            total_row_acts += 1
                        last_row = row
    
    return total_row_acts

tile_level = simulate_tile_level()
print(f"Tile 级别统计: {tile_level}")

# 检查是否有 block crossing 的情况会增加 row activation
print("\n" + "=" * 70)
print("分析 crossing tiles 的额外 row activations")
print("=" * 70)

def analyze_crossing_contribution():
    """分析 crossing tiles 如何贡献额外的 row activation"""
    last_row = None
    total = 0
    
    h_cross_extra = 0
    w_cross_extra = 0
    both_cross_extra = 0
    
    for k in range(K_l3):
        for c in range(C_l3):
            for q in range(Q_l3):
                for p in range(P_l3):
                    for s in range(S_l2):
                        p_start = p * P_buf
                        q_start = q * Q_buf + s
                        
                        h_start = p_start * stride_h
                        h_end = min(h_start + H_per_tile, input_H)
                        w_start = q_start * stride_w
                        w_end = min(w_start + W_per_tile, input_W)
                        
                        h_blk_s = h_start // block_h
                        h_blk_e = (h_end - 1) // block_h
                        w_blk_s = w_start // block_w
                        w_blk_e = (w_end - 1) // block_w
                        
                        h_cross = (h_blk_s != h_blk_e)
                        w_cross = (w_blk_s != w_blk_e)
                        
                        tile_row_acts = 0
                        tile_first_row = None
                        
                        for h_blk in range(h_blk_s, h_blk_e + 1):
                            for w_blk in range(w_blk_s, w_blk_e + 1):
                                h_lo = max(h_start, h_blk * block_h)
                                h_hi = min(h_end, (h_blk + 1) * block_h)
                                w_lo = max(w_start, w_blk * block_w)
                                w_hi = min(w_end, (w_blk + 1) * block_w)
                                
                                for h in range(h_lo, h_hi):
                                    for w in range(w_lo, w_hi):
                                        h_in = h % block_h
                                        w_in = w % block_w
                                        addr = calc_block_addr(c, h_blk, w_blk, h_in, w_in)
                                        row = calc_row(addr)
                                        
                                        if tile_first_row is None:
                                            tile_first_row = row
                                        
                                        if last_row is None or row != last_row:
                                            total += 1
                                            tile_row_acts += 1
                                        last_row = row
                        
                        # 记录额外的 row activations (超过 1 次的部分)
                        extra = tile_row_acts - 1 if tile_row_acts > 0 else 0
                        if h_cross and w_cross:
                            both_cross_extra += extra
                        elif h_cross:
                            h_cross_extra += extra
                        elif w_cross:
                            w_cross_extra += extra
    
    return total, h_cross_extra, w_cross_extra, both_cross_extra

total2, h_extra, w_extra, both_extra = analyze_crossing_contribution()
print(f"总 row activations: {total2}")
print(f"H crossing tiles 的额外贡献: {h_extra}")
print(f"W crossing tiles 的额外贡献: {w_extra}")
print(f"Both crossing tiles 的额外贡献: {both_extra}")
