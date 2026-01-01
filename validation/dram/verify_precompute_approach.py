#!/usr/bin/env python3
"""
验证预计算方法能否正确计算 Input Row Activations
并评估空间复杂度是否可行
"""

import math
from typing import Dict, Tuple, List, Set

# ResNet-L1 配置
WORKLOAD = {
    'N': 1, 'K': 64, 'C': 3, 'P': 56, 'Q': 56, 'R': 7, 'S': 7,
    'stride_h': 1, 'stride_w': 1,
    'dilation_h': 1, 'dilation_w': 1,
}

# 计算 Input 维度
H_in = WORKLOAD['stride_h'] * (WORKLOAD['P'] - 1) + WORKLOAD['dilation_h'] * (WORKLOAD['R'] - 1) + 1
W_in = WORKLOAD['stride_w'] * (WORKLOAD['Q'] - 1) + WORKLOAD['dilation_w'] * (WORKLOAD['S'] - 1) + 1
print(f"Input 维度: H={H_in}, W={W_in}, C={WORKLOAD['C']}")

# Block 配置
block_h, block_w = 31, 31

# 当前 ILP 结果对应的配置
# DRAM (L3): K=4, C=3, P=28, Q=7, R=1, S=1
# Level 2:   R=7, S=7
CURRENT_CONFIG = {
    'K_l3': 4, 'C_l3': 3, 'P_l3': 28, 'Q_l3': 7, 'R_l3': 1, 'S_l3': 1,
    'R_l2': 7, 'S_l2': 7,
    'P_tile': 2, 'Q_tile': 8,  # Buffer tile
}


def get_divisors(n: int) -> List[int]:
    """获取 n 的所有因子"""
    divisors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted(divisors)


def compute_row_activations_for_config(
    P_factor: int, Q_factor: int, R_factor: int, S_factor: int,
    C_factor: int, K_factor: int,
    workload: dict, block_h: int, block_w: int,
) -> int:
    """
    计算给定配置下的 Input row activations
    
    考虑完整的循环结构:
    - DRAM loops: K_l3, C_l3, P_l3, Q_l3
    - Level 2 loops: R_l2, S_l2 (如果 R/S 没有完全在 DRAM 层)
    """
    P, Q, R, S, C, K = workload['P'], workload['Q'], workload['R'], workload['S'], workload['C'], workload['K']
    stride_h, stride_w = workload['stride_h'], workload['stride_w']
    dilation_h, dilation_w = workload['dilation_h'], workload['dilation_w']
    
    # Tile sizes
    P_tile = P // P_factor
    Q_tile = Q // Q_factor
    R_tile = R // R_factor
    S_tile = S // S_factor
    
    # Input tile size (in H, W space)
    H_tile = P_tile * stride_h + (R_tile - 1) * dilation_h
    W_tile = Q_tile * stride_w + (S_tile - 1) * dilation_w
    
    # Loop bounds
    K_l3 = K_factor
    C_l3 = C_factor
    P_l3 = P_factor
    Q_l3 = Q_factor
    R_l2 = R_factor  # Level 2 R loops (内层)
    S_l2 = S_factor  # Level 2 S loops (内层)
    
    # 计算 row switches
    # Key insight: K 对 Input 是 irrelevant,不影响 row
    # 但每次 K 迭代开始时,可能需要重新 activate row
    
    prev_row = None
    row_switches = 0
    
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r in range(R_l2):
                        for s in range(S_l2):
                            # 计算当前访问的 h, w 起始位置
                            h_start = p * P_tile * stride_h + r * R_tile * dilation_h
                            w_start = q * Q_tile * stride_w + s * S_tile * dilation_w
                            
                            # 计算所在的 block
                            h_block = h_start // block_h
                            w_block = w_start // block_w
                            
                            # Row = (c, h_block, w_block)
                            current_row = (c, h_block, w_block)
                            
                            if current_row != prev_row:
                                row_switches += 1
                                prev_row = current_row
    
    return row_switches


def compute_row_activations_all_blocks(
    P_factor: int, Q_factor: int, R_factor: int, S_factor: int,
    C_factor: int, K_factor: int,
    workload: dict, block_h: int, block_w: int,
) -> int:
    """
    计算考虑 block crossing 的 row activations
    每个 tile 可能访问多个 blocks
    """
    P, Q, R, S, C, K = workload['P'], workload['Q'], workload['R'], workload['S'], workload['C'], workload['K']
    stride_h, stride_w = workload['stride_h'], workload['stride_w']
    dilation_h, dilation_w = workload['dilation_h'], workload['dilation_w']
    
    P_tile = P // P_factor
    Q_tile = Q // Q_factor
    R_tile = R // R_factor
    S_tile = S // S_factor
    
    H_tile = P_tile * stride_h + (R_tile - 1) * dilation_h
    W_tile = Q_tile * stride_w + (S_tile - 1) * dilation_w
    
    K_l3, C_l3, P_l3, Q_l3 = K_factor, C_factor, P_factor, Q_factor
    R_l2, S_l2 = R_factor, S_factor
    
    prev_rows: Set[Tuple] = set()
    row_switches = 0
    
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r in range(R_l2):
                        for s in range(S_l2):
                            h_start = p * P_tile * stride_h + r * R_tile * dilation_h
                            h_end = h_start + H_tile - 1
                            w_start = q * Q_tile * stride_w + s * S_tile * dilation_w
                            w_end = w_start + W_tile - 1
                            
                            # 所有访问的 blocks
                            h_blocks = range(h_start // block_h, (h_end // block_h) + 1)
                            w_blocks = range(w_start // block_w, (w_end // block_w) + 1)
                            
                            for hb in h_blocks:
                                for wb in w_blocks:
                                    row = (c, hb, wb)
                                    if row not in prev_rows:
                                        row_switches += 1
                                        prev_rows.add(row)
                                    # 注意: 这里用的是 "首次访问才计数"
                                    # Trace 用的是 "row 切换时计数"
    
    return row_switches


def compute_row_activations_trace_style(
    P_factor: int, Q_factor: int, R_factor: int, S_factor: int,
    C_factor: int, K_factor: int,
    workload: dict, block_h: int, block_w: int,
) -> int:
    """
    模拟 Trace 的计数方式: 每次 row 切换时计数
    考虑 tile 内部的 element 访问顺序
    """
    P, Q, R, S, C, K = workload['P'], workload['Q'], workload['R'], workload['S'], workload['C'], workload['K']
    stride_h, stride_w = workload['stride_h'], workload['stride_w']
    dilation_h, dilation_w = workload['dilation_h'], workload['dilation_w']
    
    P_tile = P // P_factor
    Q_tile = Q // Q_factor
    R_tile = R // R_factor
    S_tile = S // S_factor
    
    H_tile = P_tile * stride_h + (R_tile - 1) * dilation_h
    W_tile = Q_tile * stride_w + (S_tile - 1) * dilation_w
    
    K_l3, C_l3, P_l3, Q_l3 = K_factor, C_factor, P_factor, Q_factor
    R_l2, S_l2 = R_factor, S_factor
    
    prev_row = None
    row_switches = 0
    
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r in range(R_l2):
                        for s in range(S_l2):
                            h_start = p * P_tile * stride_h + r * R_tile * dilation_h
                            h_end = h_start + H_tile - 1
                            w_start = q * Q_tile * stride_w + s * S_tile * dilation_w
                            w_end = w_start + W_tile - 1
                            
                            # Tile 内部按行优先访问
                            for h in range(h_start, h_end + 1):
                                for w in range(w_start, w_end + 1):
                                    hb = h // block_h
                                    wb = w // block_w
                                    row = (c, hb, wb)
                                    
                                    if row != prev_row:
                                        row_switches += 1
                                        prev_row = row
    
    return row_switches


def compute_row_activations_correct(
    P_factor: int, Q_factor: int, R_factor: int, S_factor: int,
    C_factor: int, K_factor: int,
    workload: dict, block_h: int, block_w: int,
) -> int:
    """
    正确的 row activation 计算方式
    
    关键发现:
    1. DRAM loops = K × C × P × Q × R (R 在 Level 2)
    2. S 在 buffer tile 内,不产生额外的 DRAM iteration
    3. 每次 DRAM iteration 访问一个 H_tile × W_tile 的区域
       - H_tile = P_tile (因为 R_tile = 1)
       - W_tile = Q_tile + S_tile - 1 (因为 S 在 buffer 内)
    4. K 对 Input 是 irrelevant,但 K 循环会导致重复遍历 C×P×Q×R
    """
    P, Q, R, S, C, K = workload['P'], workload['Q'], workload['R'], workload['S'], workload['C'], workload['K']
    stride_h, stride_w = workload['stride_h'], workload['stride_w']
    dilation_h, dilation_w = workload['dilation_h'], workload['dilation_w']
    
    P_tile = P // P_factor
    Q_tile = Q // Q_factor
    S_tile = S  # S 完全在 buffer tile 内
    
    # DRAM iteration 访问的 Input 区域大小
    H_tile = P_tile * stride_h  # R_tile = 1 (R 在 Level 2)
    W_tile = Q_tile * stride_w + (S_tile - 1) * dilation_w  # S 在 buffer 内
    
    K_l3, C_l3, P_l3, Q_l3 = K_factor, C_factor, P_factor, Q_factor
    R_l2 = R_factor  # R 在 Level 2
    
    prev_row = None
    row_switches = 0
    
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r in range(R_l2):
                        # 这是一次 DRAM iteration
                        h_start = p * P_tile * stride_h + r * dilation_h
                        h_end = h_start + H_tile
                        w_start = q * Q_tile * stride_w
                        w_end = w_start + W_tile
                        
                        # 访问这个 tile 覆盖的所有 blocks
                        h_blocks = range(h_start // block_h, (h_end - 1) // block_h + 1)
                        w_blocks = range(w_start // block_w, (w_end - 1) // block_w + 1)
                        
                        for hb in h_blocks:
                            for wb in w_blocks:
                                row = (c, hb, wb)
                                if row != prev_row:
                                    row_switches += 1
                                    prev_row = row
    
    return row_switches


def compute_row_activations_analytical(
    P_factor: int, Q_factor: int,
    C_factor: int,
    R_total: int, S_total: int,
    workload: dict, block_h: int, block_w: int,
) -> int:
    """
    解析计算: 不需要完整枚举
    
    思路:
    1. 对于每个 (C, P, Q) tile, 计算它在 R×S 循环中访问的唯一 block 数
    2. 对于连续访问的 blocks, 只有切换时计数
    """
    P, Q, R, S, C, K = workload['P'], workload['Q'], workload['R'], workload['S'], workload['C'], workload['K']
    stride_h, stride_w = workload['stride_h'], workload['stride_w']
    dilation_h, dilation_w = workload['dilation_h'], workload['dilation_w']
    
    P_tile = P // P_factor
    Q_tile = Q // Q_factor
    
    # H, W 方向的访问模式分析
    # 对于固定的 (p, q), r 从 0 到 R-1 变化
    # h_start = p * P_tile * stride + r * dilation
    
    # 计算 H 方向的 block switches
    # 关键: gcd(dilation, block_h) 决定周期性
    
    C_l3, P_l3, Q_l3 = C_factor, P_factor, Q_factor
    
    total_switches = 0
    
    for c in range(C_l3):
        for p in range(P_l3):
            for q in range(Q_l3):
                # 这个 (c, p, q) tile 在 R×S 循环中的 row switches
                prev_row = None
                tile_switches = 0
                
                for r in range(R_total):
                    for s in range(S_total):
                        h_start = p * P_tile * stride_h + r * dilation_h
                        w_start = q * Q_tile * stride_w + s * dilation_w
                        
                        # 主 block (简化: 只看起始点)
                        hb = h_start // block_h
                        wb = w_start // block_w
                        row = (c, hb, wb)
                        
                        if row != prev_row:
                            tile_switches += 1
                            prev_row = row
                
                total_switches += tile_switches
    
    return total_switches


def estimate_precompute_table_size():
    """估算预计算表的空间复杂度"""
    print("\n" + "=" * 60)
    print("预计算表空间复杂度分析")
    print("=" * 60)
    
    # 对于每个维度,可能的因子数量
    dims = {
        'P': WORKLOAD['P'],  # 56
        'Q': WORKLOAD['Q'],  # 56
        'R': WORKLOAD['R'],  # 7
        'S': WORKLOAD['S'],  # 7
        'C': WORKLOAD['C'],  # 3
        'K': WORKLOAD['K'],  # 64
    }
    
    for name, val in dims.items():
        divs = get_divisors(val)
        print(f"  {name}={val}: {len(divs)} 因子 {divs}")
    
    # Block size 选项 (H_rb 因子)
    H_in = 62
    block_h_options = get_divisors(H_in)
    print(f"  block_h (H_in={H_in}): {len(block_h_options)} 选项 {block_h_options}")
    
    # 总组合数
    num_P = len(get_divisors(56))  # 8
    num_Q = len(get_divisors(56))  # 8
    num_R = len(get_divisors(7))   # 2
    num_S = len(get_divisors(7))   # 2
    num_C = len(get_divisors(3))   # 2
    num_K = len(get_divisors(64))  # 7
    num_block_h = len(get_divisors(62))  # 4
    num_block_w = len(get_divisors(62))  # 4
    
    # 方案 1: 完整预计算 (所有组合)
    total_full = num_P * num_Q * num_R * num_S * num_C * num_K * num_block_h * num_block_w
    print(f"\n方案 1: 完整预计算表")
    print(f"  组合数: {num_P}×{num_Q}×{num_R}×{num_S}×{num_C}×{num_K}×{num_block_h}×{num_block_w}")
    print(f"  = {total_full} 条目")
    print(f"  内存: ~{total_full * 4 / 1024:.1f} KB (int32)")
    
    # 方案 2: 只预计算 Input 相关的维度 (去掉 K)
    # Input 只依赖 P, Q, R, S, C, block_h, block_w
    total_input = num_P * num_Q * num_R * num_S * num_C * num_block_h * num_block_w
    print(f"\n方案 2: 只预计算 Input 相关维度 (去掉 K)")
    print(f"  组合数: {num_P}×{num_Q}×{num_R}×{num_S}×{num_C}×{num_block_h}×{num_block_w}")
    print(f"  = {total_input} 条目")
    print(f"  内存: ~{total_input * 4 / 1024:.1f} KB")
    
    # 方案 3: 分离 H 和 W 方向
    # H 方向: P, R, block_h
    # W 方向: Q, S, block_w
    total_h = num_P * num_R * num_block_h
    total_w = num_Q * num_S * num_block_w
    print(f"\n方案 3: 分离 H/W 方向 (如 block_crossing)")
    print(f"  H 方向: {num_P}×{num_R}×{num_block_h} = {total_h} 条目")
    print(f"  W 方向: {num_Q}×{num_S}×{num_block_w} = {total_w} 条目")
    print(f"  总计: {total_h + total_w} 条目")
    print(f"  内存: ~{(total_h + total_w) * 4 / 1024:.2f} KB")
    
    return total_input


def main():
    print("=" * 60)
    print("验证预计算方法: Input Row Activations")
    print("=" * 60)
    
    cfg = CURRENT_CONFIG
    
    print(f"\n当前配置:")
    print(f"  DRAM (L3): K={cfg['K_l3']}, C={cfg['C_l3']}, P={cfg['P_l3']}, Q={cfg['Q_l3']}")
    print(f"  Level 2:   R={cfg['R_l2']}, S={cfg['S_l2']}")
    print(f"  Tile:      P_tile={cfg['P_tile']}, Q_tile={cfg['Q_tile']}")
    print(f"  Block:     {block_h} × {block_w}")
    
    # 方法 1: 简化版 (只看主 block)
    result1 = compute_row_activations_for_config(
        P_factor=cfg['P_l3'], Q_factor=cfg['Q_l3'],
        R_factor=cfg['R_l2'], S_factor=cfg['S_l2'],
        C_factor=cfg['C_l3'], K_factor=cfg['K_l3'],
        workload=WORKLOAD, block_h=block_h, block_w=block_w
    )
    print(f"\n方法 1 (只看主 block): {result1}")
    
    # 方法 2: 考虑 all blocks (首次访问计数)
    result2 = compute_row_activations_all_blocks(
        P_factor=cfg['P_l3'], Q_factor=cfg['Q_l3'],
        R_factor=cfg['R_l2'], S_factor=cfg['S_l2'],
        C_factor=cfg['C_l3'], K_factor=cfg['K_l3'],
        workload=WORKLOAD, block_h=block_h, block_w=block_w
    )
    print(f"方法 2 (all blocks, 首次访问): {result2}")
    
    # 方法 3: 正确的计算 (基于分析)
    print("\n方法 3 (正确模型) 计算中...")
    result3 = compute_row_activations_correct(
        P_factor=cfg['P_l3'], Q_factor=cfg['Q_l3'],
        R_factor=cfg['R_l2'], S_factor=cfg['S_l2'],
        C_factor=cfg['C_l3'], K_factor=cfg['K_l3'],
        workload=WORKLOAD, block_h=block_h, block_w=block_w
    )
    print(f"方法 3 (K irrelevant, R 滑动): {result3}")
    
    # 方法 4: 解析计算
    result4 = compute_row_activations_analytical(
        P_factor=cfg['P_l3'], Q_factor=cfg['Q_l3'],
        C_factor=cfg['C_l3'],
        R_total=WORKLOAD['R'], S_total=WORKLOAD['S'],
        workload=WORKLOAD, block_h=block_h, block_w=block_w
    )
    print(f"方法 4 (解析计算): {result4}")
    
    print(f"\n目标 (Trace 实际值): 5376")
    print(f"ILP 当前预测: 2392")
    
    # 详细分析
    print("\n" + "-" * 60)
    print("详细分析: 768 × 7 = 5376")
    print("-" * 60)
    
    # 计算 768 (不含 R 循环)
    P_tile = WORKLOAD['P'] // cfg['P_l3']  # 2
    Q_tile = WORKLOAD['Q'] // cfg['Q_l3']  # 8
    R_tile = 7  # 完整 R 在一个 DRAM tile 内
    S_tile = 7  # 完整 S 在一个 DRAM tile 内
    
    H_tile_full = P_tile * WORKLOAD['stride_h'] + (R_tile - 1) * WORKLOAD['dilation_h']
    W_tile_full = Q_tile * WORKLOAD['stride_w'] + (S_tile - 1) * WORKLOAD['dilation_w']
    
    print(f"  DRAM tile (含完整 R, S):")
    print(f"    H_tile = {P_tile}×stride + ({R_tile}-1)×dilation = {H_tile_full}")
    print(f"    W_tile = {Q_tile}×stride + ({S_tile}-1)×dilation = {W_tile_full}")
    
    # 计算每个 (C, P, Q) DRAM tile 访问的 block 数
    blocks_per_tile = []
    for p in range(cfg['P_l3']):
        for q in range(cfg['Q_l3']):
            h_start = p * P_tile * WORKLOAD['stride_h']
            h_end = h_start + H_tile_full - 1
            w_start = q * Q_tile * WORKLOAD['stride_w']
            w_end = w_start + W_tile_full - 1
            
            h_blocks = len(range(h_start // block_h, (h_end // block_h) + 1))
            w_blocks = len(range(w_start // block_w, (w_end // block_w) + 1))
            blocks_per_tile.append(h_blocks * w_blocks)
    
    total_blocks = sum(blocks_per_tile) * cfg['C_l3']
    print(f"\n  C × P × Q tiles 的总 block 访问: {total_blocks}")
    print(f"  768 × R_l2 = 768 × 7 = {768 * 7}")
    
    # 空间复杂度分析
    estimate_precompute_table_size()
    
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    print(f"""
关键发现:
1. Trace 的 5376 = 768 × 7 = (C×P×Q blocks) × R
2. 这意味着 Trace 在每次 R 迭代时都检查 row switch
3. ILP 的 row_acts_aligned = 2352 只考虑了 DRAM tile 数量
4. 缺失的是: R 循环内的 row switches

ILP 修复方向:
- 不能简单用 Π bound_j^{{xj}}
- 需要考虑 R/S 在 Level 2 时, 每次 R/S 迭代的 h_start/w_start 变化
- 可以用预计算表: precompute_input_row_acts[P_factor, Q_factor, R, S, block_h, block_w]
""")


if __name__ == "__main__":
    main()
