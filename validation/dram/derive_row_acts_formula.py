"""
方案 B：推导 Input Row Activation 的解析公式

目标：
1. 从第一性原理理解 row activation 的定义
2. 推导解析公式
3. 验证 trace 结果的正确性

关键问题：
- row_aligned 模式下，数据是如何布局的？
- 每个 tile 访问哪些 row？
- row switch 在什么时候发生？
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from collections import defaultdict
from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload

DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6


def derive_row_activation_formula():
    """从第一性原理推导 row activation 公式"""
    
    # ==========================================================================
    # Step 1: 获取 Mapping 参数
    # ==========================================================================
    print("=" * 80)
    print("STEP 1: 获取 Mapping 参数")
    print("=" * 80)
    
    workload = ConvWorkload(name="ResNet-L1", R=7, S=7, P=56, Q=56, C=3, K=64, N=1)
    optimizer = PIMOptimizer()
    result = optimizer.optimize([workload], objective="latency")
    mapping = result.mappings[0]
    
    # Workload 参数
    H_in = workload.input_size['H']  # 62
    W_in = workload.input_size['W']  # 62
    C_in = workload.C  # 3
    N_in = workload.N  # 1
    
    print(f"\nInput Tensor: N={N_in}, C={C_in}, H={H_in}, W={W_in}")
    print(f"Total elements: {N_in * C_in * H_in * W_in}")
    
    # ==========================================================================
    # Step 2: 理解 row_aligned 数据布局
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: 理解 row_aligned 数据布局")
    print("=" * 80)
    
    row_size = 1024  # bytes，一个 row 能存储的元素数
    block_h = mapping.tile_info.get('block_h', 31)
    block_w = mapping.tile_info.get('block_w', 31)
    
    print(f"\nrow_size = {row_size} bytes")
    print(f"block_h = {block_h}, block_w = {block_w}")
    
    # 计算 block 数量
    num_blocks_h = (H_in + block_h - 1) // block_h  # ceil(62/31) = 2
    num_blocks_w = (W_in + block_w - 1) // block_w  # ceil(62/31) = 2
    
    print(f"num_blocks_h = {num_blocks_h}, num_blocks_w = {num_blocks_w}")
    
    # row_aligned 布局的关键：每个 (c, h_block, w_block) 对应一个独立的存储区域
    # 每个存储区域 padding 到 row 边界
    
    # 计算 strides
    # 最内层：w_block 变化，stride = row_size (因为 row_aligned)
    # 然后：h_block 变化，stride = row_size * num_blocks_w
    # 最后：c 变化，stride = row_size * num_blocks_w * num_blocks_h
    
    stride_w_block = row_size  # 1024
    stride_h_block = row_size * num_blocks_w  # 1024 * 2 = 2048? 
    
    # 但是从 trace 结果看:
    # stride_q_l3 = 1024 (w_block stride)
    # stride_p_l3 = 7168 = 7 * 1024 (h_block stride) 
    # 为什么是 7 而不是 2？
    
    print(f"\n疑问：从 trace 看 stride_p_l3 = 7168 = 7 × 1024")
    print(f"      但 num_blocks_w = 2，为什么 h_block stride 不是 2048？")
    
    # 让我检查实际的 stride
    # 从 analysis.txt:
    # stride_q_l3 = 1024 (Q/w_block)
    # stride_p_l3 = 7168 (P/h_block)  
    # stride_c_l3 = 200704 (C)
    
    # 验证：200704 / 7168 = 28 = P_l3
    # 所以 stride 是按 P_l3 tile 数量计算的，不是 block 数量！
    
    print(f"\n重新理解 stride:")
    print(f"  stride_q_l3 = 1024 = row_size (每个 Q tile 一个 row)")
    print(f"  stride_p_l3 = 7168 = 1024 × Q_l3 = 1024 × 7")
    print(f"  stride_c_l3 = 200704 = 7168 × P_l3 = 7168 × 28")
    
    # ==========================================================================
    # Step 3: 理解 Tile 和 Block 的关系
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: 理解 Tile 和 Block 的关系")
    print("=" * 80)
    
    # Buffer tile (Level 0+1)
    buffer_tile = {d: 1 for d in range(7)}
    for level in [0, 1]:
        if level not in mapping.loop_bounds:
            continue
        level_bounds = mapping.loop_bounds[level]
        if level == 0:
            for key in ['H', 'W', 'Internal', 'temporal']:
                if key in level_bounds:
                    for d, bound in level_bounds[key].items():
                        buffer_tile[d] *= bound
        else:
            for key in ['spatial', 'temporal']:
                if key in level_bounds:
                    for d, bound in level_bounds[key].items():
                        buffer_tile[d] *= bound
    
    P_per_tile = buffer_tile[DIM_P]  # 2
    Q_per_tile = buffer_tile[DIM_Q]  # 8
    
    # Access tile 大小 (滑动窗口)
    H_per_tile = P_per_tile  # stride=1, R 在内层循环
    W_per_tile = Q_per_tile + 6  # 8 + (S-1) = 8 + 6 = 14
    
    print(f"\nBuffer tile: P_per_tile={P_per_tile}, Q_per_tile={Q_per_tile}")
    print(f"Access tile: H_per_tile={H_per_tile}, W_per_tile={W_per_tile}")
    
    # DRAM factors
    level3_factors = {d: 1 for d in range(7)}
    if 3 in mapping.loop_bounds:
        for key in ['spatial', 'temporal']:
            if key in mapping.loop_bounds[3]:
                for d, bound in mapping.loop_bounds[3][key].items():
                    level3_factors[d] *= bound
    
    P_l3 = level3_factors[DIM_P]  # 28
    Q_l3 = level3_factors[DIM_Q]  # 7
    C_l3 = level3_factors[DIM_C]  # 3
    K_l3 = level3_factors[DIM_K]  # 4
    
    level2_factors = {d: 1 for d in range(7)}
    if 2 in mapping.loop_bounds:
        for key in ['spatial', 'temporal']:
            if key in mapping.loop_bounds[2]:
                for d, bound in mapping.loop_bounds[2][key].items():
                    level2_factors[d] *= bound
    
    R_l2 = level2_factors[DIM_R]  # 7
    
    print(f"\nDRAM factors: K_l3={K_l3}, C_l3={C_l3}, P_l3={P_l3}, Q_l3={Q_l3}, R_l2={R_l2}")
    
    # ==========================================================================
    # Step 4: 分析每个 Tile 访问哪些 Row
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: 分析每个 Tile 访问哪些 Row")
    print("=" * 80)
    
    # row_aligned 模式下的地址计算:
    # addr = p_tile * stride_p + q_tile * stride_q + c_tile * stride_c + offset_in_tile
    # row = addr // row_size
    
    # 关键洞察：
    # 1. 每个 (p_tile, q_tile, c_tile) 组合对应一个 "DRAM tile"
    # 2. 由于 row_aligned，每个 DRAM tile 的 base address 对齐到 row 边界
    # 3. 所以每个 DRAM tile 的 base 就决定了它访问的 row
    
    stride_q = row_size  # 1024
    stride_p = stride_q * Q_l3  # 1024 * 7 = 7168
    stride_c = stride_p * P_l3  # 7168 * 28 = 200704
    
    print(f"\nStrides: stride_q={stride_q}, stride_p={stride_p}, stride_c={stride_c}")
    
    # 计算每个 (p_tile, q_tile, c_tile) 的 base row
    print(f"\n每个 tile 的 base row (p_tile, q_tile, c_tile) -> row:")
    
    tile_to_row = {}
    unique_rows = set()
    
    for c in range(C_l3):
        for p in range(min(3, P_l3)):  # 只打印前 3 个 p
            for q in range(Q_l3):
                base_addr = p * stride_p + q * stride_q + c * stride_c
                row = base_addr // row_size
                tile_to_row[(p, q, c)] = row
                unique_rows.add(row)
                if p < 2:  # 只打印前 2 个 p
                    print(f"  ({p}, {q}, {c}) -> addr={base_addr}, row={row}")
    
    print(f"\n总共 {P_l3 * Q_l3 * C_l3} = {P_l3}×{Q_l3}×{C_l3} 个 unique tiles")
    
    # ==========================================================================
    # Step 5: 分析 Row Switch 发生的条件
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: 分析 Row Switch 发生的条件")
    print("=" * 80)
    
    # 循环顺序 (outer to inner): K -> C -> P -> Q -> R
    # Input 相关维度: C, P, Q (K 不相关, R 在内层不产生 DRAM 访问)
    
    # Row switch 发生在:
    # 1. tile 之间: 当 (p, q, c) 变化导致 row 变化
    # 2. tile 内部: 当 tile 跨越 block 边界
    
    print("\n情况 1: Tile 之间的 row switch")
    print("  循环顺序 K -> C -> P -> Q 意味着 Q 变化最快")
    print("  Q 变化: row 从 row_n 变到 row_(n+1)")
    print("  P 变化: row 从 row_n 跳到 row_(n+Q_l3)")
    print("  C 变化: row 从 row_n 跳到 row_(n+P_l3*Q_l3)")
    
    # 分析 tile 内部是否会跨 block
    print("\n情况 2: Tile 内部跨 block")
    print(f"  Access tile: H_per_tile={H_per_tile}, W_per_tile={W_per_tile}")
    print(f"  Block size: block_h={block_h}, block_w={block_w}")
    
    # 检查哪些 tile 会跨 block
    tiles_crossing_h = 0
    tiles_crossing_w = 0
    tiles_crossing_both = 0
    
    for p in range(P_l3):
        h_start = p * P_per_tile
        h_end = h_start + H_per_tile
        h_block_start = h_start // block_h
        h_block_end = (h_end - 1) // block_h
        crosses_h = h_block_start != h_block_end
        
        for q in range(Q_l3):
            w_start = q * Q_per_tile
            w_end = w_start + W_per_tile
            w_block_start = w_start // block_w
            w_block_end = (w_end - 1) // block_w
            crosses_w = w_block_start != w_block_end
            
            if crosses_h and crosses_w:
                tiles_crossing_both += 1
            elif crosses_h:
                tiles_crossing_h += 1
            elif crosses_w:
                tiles_crossing_w += 1
    
    tiles_no_crossing = P_l3 * Q_l3 - tiles_crossing_h - tiles_crossing_w - tiles_crossing_both
    
    print(f"\n  P_l3={P_l3}, Q_l3={Q_l3} 总共 {P_l3 * Q_l3} 个 (p,q) 组合")
    print(f"  不跨 block: {tiles_no_crossing}")
    print(f"  仅跨 H block: {tiles_crossing_h}")
    print(f"  仅跨 W block: {tiles_crossing_w}")
    print(f"  跨 H 和 W block: {tiles_crossing_both}")
    
    # ==========================================================================
    # Step 6: 推导解析公式
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: 推导解析公式")
    print("=" * 80)
    
    # 关键洞察：
    # 对于 row_aligned 模式，每个 (p_tile, q_tile, c_tile) 占用独立的 row(s)
    # 
    # Row activation = 在循环中访问的 unique (row) 数量？
    # 不对！Row activation = row SWITCH 数量 = 每次访问不同 row 时 +1
    
    print("\nRow Activation 定义:")
    print("  Row activation 是 row SWITCH 次数，不是 unique row 数量")
    print("  每次访问的 row 与上一次不同时，row activation +1")
    
    # 让我们逐步计算
    print("\n计算方法:")
    print("  总 tile 数 = K_l3 × C_l3 × P_l3 × Q_l3 × R_l2")
    print(f"            = {K_l3} × {C_l3} × {P_l3} × {Q_l3} × {R_l2}")
    print(f"            = {K_l3 * C_l3 * P_l3 * Q_l3 * R_l2}")
    
    # 由于 K 对 Input 不相关，K 循环不产生新的 Input 访问
    # 但是！K 循环会导致 row switch，因为在 K 变化后回到 C=0, P=0, Q=0
    
    print("\n由于 K 对 Input 不相关:")
    print("  K 变化不产生新的 Input 访问")
    print("  但 K 从 3 变回 0 时，(C,P,Q) 回到 (0,0,0)，可能产生 row switch")
    
    # ==========================================================================
    # Step 7: 模拟完整的循环并计数
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: 精确模拟循环计数")
    print("=" * 80)
    
    # 方法 A: 只考虑 tile 之间的 row switch (不考虑 tile 内部跨 block)
    print("\n方法 A: 只考虑 tile 之间的 row switch")
    
    current_row_A = None
    row_acts_A = 0
    
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r in range(R_l2):
                        # 计算这个 tile 的 base row
                        base_addr = p * stride_p + q * stride_q + c * stride_c
                        row = base_addr // row_size
                        
                        if current_row_A != row:
                            row_acts_A += 1
                            current_row_A = row
    
    print(f"  Row activations (方法 A): {row_acts_A}")
    
    # 方法 B: 考虑 tile 内部跨 block (每个 block 访问可能在不同 row)
    print("\n方法 B: 考虑 tile 内部跨 block")
    
    current_row_B = None
    row_acts_B = 0
    
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r in range(R_l2):
                        # 计算 H, W 范围
                        h_start = p * P_per_tile
                        h_end = h_start + H_per_tile
                        w_start = q * Q_per_tile
                        w_end = w_start + W_per_tile
                        
                        # Block 范围
                        h_block_start = h_start // block_h
                        h_block_end = (h_end - 1) // block_h
                        w_block_start = w_start // block_w
                        w_block_end = (w_end - 1) // block_w
                        
                        # 遍历每个涉及的 block
                        for h_block in range(h_block_start, h_block_end + 1):
                            for w_block in range(w_block_start, w_block_end + 1):
                                # 计算这个 block 的 row
                                block_base = h_block * stride_p + w_block * stride_q + c * stride_c
                                row = block_base // row_size
                                
                                if current_row_B != row:
                                    row_acts_B += 1
                                    current_row_B = row
    
    print(f"  Row activations (方法 B): {row_acts_B}")
    
    # 方法 C: 更精细 - 遍历每个元素
    print("\n方法 C: 遍历每个元素 (与 trace 一致)")
    
    stride_p_l2 = W_per_tile  # 14
    stride_q_l2 = 1
    
    current_row_C = None
    row_acts_C = 0
    access_count = 0
    
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r in range(R_l2):
                        # 计算 H, W 范围
                        h_start = p * P_per_tile
                        h_end = min(h_start + H_per_tile, H_in)
                        w_start = q * Q_per_tile
                        w_end = min(w_start + W_per_tile, W_in)
                        
                        # Block 范围
                        h_block_start = h_start // block_h
                        h_block_end = (h_end - 1) // block_h
                        w_block_start = w_start // block_w
                        w_block_end = (w_end - 1) // block_w
                        
                        # 遍历每个涉及的 block
                        for h_block in range(h_block_start, h_block_end + 1):
                            for w_block in range(w_block_start, w_block_end + 1):
                                h_lo = max(h_start, h_block * block_h)
                                h_hi = min(h_end, (h_block + 1) * block_h)
                                w_lo = max(w_start, w_block * block_w)
                                w_hi = min(w_end, (w_block + 1) * block_w)
                                
                                for h in range(h_lo, h_hi):
                                    for w in range(w_lo, w_hi):
                                        h_in_block = h % block_h
                                        w_in_block = w % block_w
                                        
                                        block_base = h_block * stride_p + w_block * stride_q + c * stride_c
                                        offset = h_in_block * stride_p_l2 + w_in_block * stride_q_l2
                                        addr = block_base + offset
                                        row = addr // row_size
                                        
                                        if current_row_C != row:
                                            row_acts_C += 1
                                            current_row_C = row
                                        
                                        access_count += 1
    
    print(f"  Row activations (方法 C): {row_acts_C}")
    print(f"  Total element accesses: {access_count}")
    
    # ==========================================================================
    # Step 8: 对比结果
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP 8: 结果对比")
    print("=" * 80)
    
    print(f"\n方法 A (tile base row only):  {row_acts_A}")
    print(f"方法 B (block-level):          {row_acts_B}")
    print(f"方法 C (element-level):        {row_acts_C}")
    print(f"Trace 结果:                    5880")
    
    if row_acts_C == 5880:
        print("\n✓ 方法 C 与 Trace 结果一致！")
    else:
        print(f"\n✗ 方法 C 与 Trace 不一致，差异: {row_acts_C - 5880}")
    
    # ==========================================================================
    # Step 9: 推导解析公式
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP 9: 推导解析公式")
    print("=" * 80)
    
    # 从方法 B 的结果，我们可以推导公式
    # Row acts = (tile 间 switch) + (tile 内 block crossing switch)
    
    # Tile 间 switch: 
    # - 每个新的 (c, p, q) 组合访问不同的 row(s)
    # - 但连续的 q 访问相邻的 row，可能没有 switch
    
    # 让我分析 tile 间的 switch pattern
    print("\n分析 tile 间 switch pattern:")
    
    # 一次完整的 C × P × Q 迭代
    switches_in_one_cpq = 0
    prev_row = None
    for c in range(C_l3):
        for p in range(P_l3):
            for q in range(Q_l3):
                base_addr = p * stride_p + q * stride_q + c * stride_c
                row = base_addr // row_size
                if prev_row != row:
                    switches_in_one_cpq += 1
                    prev_row = row
    
    print(f"  一次 C×P×Q 迭代的 switch: {switches_in_one_cpq}")
    print(f"  K_l3 × R_l2 次迭代: {K_l3} × {R_l2} = {K_l3 * R_l2}")
    print(f"  预期 tile 间 switch: {switches_in_one_cpq * K_l3 * R_l2}")
    
    # 但这忽略了 K 和 R 循环回绕时的 switch
    # 当 K 从 k 变到 k+1，(C,P,Q,R) 从 (C_l3-1, P_l3-1, Q_l3-1, R_l2-1) 变到 (0,0,0,0)
    # 这会产生一次额外的 switch（如果 row 不同）


if __name__ == "__main__":
    derive_row_activation_formula()
