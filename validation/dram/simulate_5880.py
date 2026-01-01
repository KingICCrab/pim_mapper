#!/usr/bin/env python3
"""
使用正确的 R_l2=7 参数精确模拟 Trace 的 5880 row activations
"""

block_h, block_w = 31, 31
P_buf, Q_buf = 8, 2
stride_h, stride_w = 1, 1

# DRAM 循环因子
P_l3, Q_l3, C_l3, K_l3 = 7, 28, 3, 4
R_l2, S_l2 = 7, 7  # R_l2=7 是关键!

input_H, input_W = 62, 62

print("=" * 70)
print("使用 R_l2=7 模拟 Trace 的 row activations")
print("=" * 70)
print(f"P_l3={P_l3}, Q_l3={Q_l3}, R_l2={R_l2}, S_l2={S_l2}, C_l3={C_l3}, K_l3={K_l3}")
print(f"P_buf={P_buf}, Q_buf={Q_buf}")

def simulate_with_r_l2():
    """考虑 R_l2=7 的模拟"""
    last_block = None
    total = 0
    block_visits = {}
    
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r in range(R_l2):
                        for s in range(S_l2):
                            h_start = p * P_buf + r
                            h_end = min(h_start + P_buf, input_H)
                            w_start = q * Q_buf + s
                            w_end = min(w_start + Q_buf, input_W)
                            
                            h_blk_s = h_start // block_h
                            h_blk_e = (h_end - 1) // block_h
                            w_blk_s = w_start // block_w
                            w_blk_e = (w_end - 1) // block_w
                            
                            for h_blk in range(h_blk_s, h_blk_e + 1):
                                for w_blk in range(w_blk_s, w_blk_e + 1):
                                    block = (c, h_blk, w_blk)
                                    if last_block is None or block != last_block:
                                        total += 1
                                        block_visits[block] = block_visits.get(block, 0) + 1
                                        last_block = block
    
    return total, block_visits

total, visits = simulate_with_r_l2()

print(f"\n模拟结果: {total}")
print(f"Trace 实际值: 5880")
print()

print("每个 block 的访问次数:")
for block in sorted(visits.keys()):
    c, h_blk, w_blk = block
    print(f"  Channel {c}, Block ({h_blk},{w_blk}): {visits[block]}")

h0_total = sum(visits.get((c, 0, w), 0) for c in range(C_l3) for w in range(2))
h1_total = sum(visits.get((c, 1, w), 0) for c in range(C_l3) for w in range(2))

print()
print(f"h_block=0 总计: {h0_total}")
print(f"h_block=1 总计: {h1_total}")

# 如果还不匹配，可能需要调整循环顺序
if total != 5880:
    print()
    print("=" * 70)
    print("尝试不同的循环顺序...")
    print("=" * 70)
    
    # Trace generator 的实际顺序可能是 K -> C -> P -> Q 在 Level 3, 然后 R -> S 在 Level 2
    # 但访问 Input 时，只有 C, P, Q, R, S 相关，K 是 irrelevant
    
    # 让我检查 P 和 Q 的顺序是否正确
    # verify_crossing.py 中是 Q -> P -> S
    # 我这里用的是 P -> Q -> R -> S
    
    def simulate_qprs():
        """Q -> P -> R -> S 顺序"""
        last_block = None
        total = 0
        block_visits = {}
        
        for k in range(K_l3):
            for c in range(C_l3):
                for q in range(Q_l3):  # Q 在 P 之前
                    for p in range(P_l3):
                        for r in range(R_l2):
                            for s in range(S_l2):
                                h_start = p * P_buf + r
                                h_end = min(h_start + P_buf, input_H)
                                w_start = q * Q_buf + s
                                w_end = min(w_start + Q_buf, input_W)
                                
                                h_blk_s = h_start // block_h
                                h_blk_e = (h_end - 1) // block_h
                                w_blk_s = w_start // block_w
                                w_blk_e = (w_end - 1) // block_w
                                
                                for h_blk in range(h_blk_s, h_blk_e + 1):
                                    for w_blk in range(w_blk_s, w_blk_e + 1):
                                        block = (c, h_blk, w_blk)
                                        if last_block is None or block != last_block:
                                            total += 1
                                            block_visits[block] = block_visits.get(block, 0) + 1
                                            last_block = block
        
        return total, block_visits
    
    total2, visits2 = simulate_qprs()
    print(f"Q->P->R->S 顺序: {total2}")
    
    # 也许 S 在 R 之前?
    def simulate_qpsr():
        """Q -> P -> S -> R 顺序"""
        last_block = None
        total = 0
        
        for k in range(K_l3):
            for c in range(C_l3):
                for q in range(Q_l3):
                    for p in range(P_l3):
                        for s in range(S_l2):
                            for r in range(R_l2):
                                h_start = p * P_buf + r
                                h_end = min(h_start + P_buf, input_H)
                                w_start = q * Q_buf + s
                                w_end = min(w_start + Q_buf, input_W)
                                
                                h_blk_s = h_start // block_h
                                h_blk_e = (h_end - 1) // block_h
                                w_blk_s = w_start // block_w
                                w_blk_e = (w_end - 1) // block_w
                                
                                for h_blk in range(h_blk_s, h_blk_e + 1):
                                    for w_blk in range(w_blk_s, w_blk_e + 1):
                                        block = (c, h_blk, w_blk)
                                        if last_block is None or block != last_block:
                                            total += 1
                                            last_block = block
        
        return total, {}
    
    total3, _ = simulate_qpsr()
    print(f"Q->P->S->R 顺序: {total3}")
