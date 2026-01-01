#!/usr/bin/env python3
"""
验证 Input Row Activation 新公式 (crossing-based)

公式:
    row_acts = C_factor × (
        reuse_penalty × h_non × w_non +
        2 × K_factor × (h_crossing × w_non + h_non × w_crossing) +
        4 × K_factor × h_crossing × w_crossing
    )

其中:
    - reuse_penalty = K_factor 当 K 在外层, = 1 当 K 在内层
    - h_non: H 方向不跨越 block 边界的 tile 数
    - h_crossing: H 方向跨越 block 边界的 tile 数
    - w_non: W 方向不跨越 block 边界的 tile 数
    - w_crossing: W 方向跨越 block 边界的 tile 数
"""

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Workload:
    """卷积工作负载参数"""
    R: int  # 卷积核高度
    S: int  # 卷积核宽度
    P: int  # 输出高度
    Q: int  # 输出宽度
    C: int  # 输入通道
    K: int  # 输出通道
    N: int  # batch size
    stride_h: int = 1
    stride_w: int = 1
    dilation_h: int = 1
    dilation_w: int = 1
    
    @property
    def H(self) -> int:
        """输入高度"""
        return (self.P - 1) * self.stride_h + (self.R - 1) * self.dilation_h + 1
    
    @property
    def W(self) -> int:
        """输入宽度"""
        return (self.Q - 1) * self.stride_w + (self.S - 1) * self.dilation_w + 1


@dataclass
class Mapping:
    """Mapping 配置"""
    # DRAM level factors
    P_dram: int
    Q_dram: int
    C_dram: int
    K_dram: int
    R_dram: int  # R 在 DRAM level 的因子 (如果R在buffer level, 这里=1)
    S_dram: int  # S 在 DRAM level 的因子
    
    # Buffer level factors
    P_buf: int
    Q_buf: int
    C_buf: int
    K_buf: int
    R_buf: int  # R 在 buffer level 的因子
    S_buf: int  # S 在 buffer level 的因子
    
    # K 是否在外层 (相对于 P,Q,R,S)
    K_outer: bool = True


def compute_crossing_counts(
    num_tiles: int,
    tile_size: int,
    step: int,
    block_size: int
) -> Tuple[int, int]:
    """
    计算跨越和非跨越 block 边界的 tile 数量
    
    使用 GCD 周期方法:
    - g = gcd(step, block_size)
    - period = block_size / g
    - 枚举一个周期内的位置，统计 crossing
    
    Args:
        num_tiles: tile 总数 (P_dram 或 Q_dram)
        tile_size: 每个 tile 的尺寸 (Input tile 在该维度的大小)
        step: tile 之间的步长 (= P_buf * stride 或 Q_buf * stride)
        block_size: block 大小
    
    Returns:
        (non_crossing_count, crossing_count)
    """
    if num_tiles == 0:
        return (0, 0)
    
    if num_tiles == 1:
        # 只有一个 tile，检查它是否跨越
        start = 0
        end = start + tile_size - 1
        start_block = start // block_size
        end_block = end // block_size
        if start_block == end_block:
            return (1, 0)
        else:
            return (0, 1)
    
    # 使用 GCD 周期方法
    g = math.gcd(step, block_size)
    period = block_size // g
    
    # 统计一个周期内的 crossing 数量
    crossing_in_period = 0
    for i in range(period):
        start = (i * step) % block_size
        end = start + tile_size - 1
        # 检查是否跨越 block 边界
        if end >= block_size:
            crossing_in_period += 1
    
    non_crossing_in_period = period - crossing_in_period
    
    # 分解为完整周期 + 余数
    full_periods = num_tiles // period
    remainder = num_tiles % period
    
    # 完整周期的贡献
    total_crossing = full_periods * crossing_in_period
    total_non_crossing = full_periods * non_crossing_in_period
    
    # 余数部分：枚举前 remainder 个位置
    for i in range(remainder):
        start = (i * step) % block_size
        end = start + tile_size - 1
        if end >= block_size:
            total_crossing += 1
        else:
            total_non_crossing += 1
    
    return (total_non_crossing, total_crossing)


def compute_row_activations_formula(
    workload: Workload,
    mapping: Mapping,
    block_h: int,
    block_w: int
) -> dict:
    """
    使用新公式计算 Input Row Activations
    
    关键洞察：模拟器的逻辑是：
    1. 遍历 K -> C -> P -> Q -> R 的所有组合
    2. 对于每个组合，遍历 tile 覆盖的所有 blocks
    3. 对于每个 block，遍历 block 内的所有元素
    4. 当 (c, h_block, w_block) 变化时计数
    
    但关键是：tile 内的元素访问是连续的，所以同一个 block 内只产生一次 switch
    实际产生 switch 的情况是：
    - C 变化时：row 一定变化
    - P/Q/R 变化时：如果 (h_block, w_block) 变化，则 row 变化
    - K 变化时：回到之前的 (C, h_block, w_block)，产生 switch
    
    正确的公式应该是：
    - 每个 (K, C) 组合内，统计 (h_block, w_block) 变化次数
    - 但 K 循环在最外层，每次 K 迭代开始时，都会从 C=0 开始
    - 所以每次 K 迭代都会重新访问所有 rows
    
    实际上，关键是分析 P->Q->R 遍历时访问的 unique (h_block, w_block) 组合
    以及它们的访问顺序
    """
    P_tile = workload.P // mapping.P_dram
    Q_tile = workload.Q // mapping.Q_dram
    
    # H_per_tile 和 W_per_tile (每个 tile 覆盖的 Input 范围)
    R_buf_actual = 1 if mapping.R_dram > 1 else workload.R
    S_buf_actual = 1 if mapping.S_dram > 1 else workload.S
    H_per_tile = (P_tile - 1) * workload.stride_h + (R_buf_actual - 1) * workload.dilation_h + 1
    W_per_tile = (Q_tile - 1) * workload.stride_w + (S_buf_actual - 1) * workload.dilation_w + 1
    
    H_in = workload.H
    W_in = workload.W
    
    print(f"\n=== Tile 分析 ===")
    print(f"P_tile={P_tile}, Q_tile={Q_tile}")
    print(f"R_buf_actual={R_buf_actual}, S_buf_actual={S_buf_actual}")
    print(f"H_per_tile={H_per_tile}, W_per_tile={W_per_tile}")
    print(f"block_h={block_h}, block_w={block_w}")
    
    # 分析 P->Q->R 遍历时的 block 访问序列
    # 使用与模拟器相同的逻辑来统计 row switches
    row_switches_per_KC = 0
    prev_hw_block = None
    
    for p in range(mapping.P_dram):
        for q in range(mapping.Q_dram):
            for r in range(mapping.R_dram):
                # Tile 起始坐标
                p_start = p * P_tile
                q_start = q * Q_tile
                r_start = r * R_buf_actual
                
                h_start = p_start * workload.stride_h + r_start * workload.dilation_h
                w_start = q_start * workload.stride_w
                
                h_end = min(h_start + H_per_tile, H_in)
                w_end = min(w_start + W_per_tile, W_in)
                
                # 遍历 tile 覆盖的 blocks
                h_block_start = h_start // block_h
                h_block_end = (h_end - 1) // block_h
                w_block_start = w_start // block_w
                w_block_end = (w_end - 1) // block_w
                
                for h_block in range(h_block_start, h_block_end + 1):
                    for w_block in range(w_block_start, w_block_end + 1):
                        # 只统计 (h_block, w_block) 变化
                        hw_block = (h_block, w_block)
                        if prev_hw_block is None or hw_block != prev_hw_block:
                            row_switches_per_KC += 1
                        prev_hw_block = hw_block
    
    print(f"\n=== P->Q->R 遍历分析 ===")
    print(f"P_dram={mapping.P_dram}, Q_dram={mapping.Q_dram}, R_dram={mapping.R_dram}")
    print(f"row_switches_per_KC = {row_switches_per_KC}")
    
    # 每个 (K, C) 组合会遍历一次 P->Q->R
    # C 变化时，row 一定变化（因为 C 是 row_id 的一部分）
    # K 变化时，回到 C=0，也会产生 switch
    
    # 但实际上，模拟器的逻辑是遍历每个元素...
    # 让我重新看模拟器的逻辑
    
    # 模拟器逻辑：
    # for k: for c: for p: for q: for r:
    #   for h_block in tile: for w_block in tile:
    #     for h in block: for w in block:
    #       if (c, h_block, w_block) != prev: count++
    #
    # 关键：tile 内的元素会被逐个遍历，但同一个 (c, h_block, w_block) 只计数一次
    # 所以实际上，row_switches = unique (c, h_block, w_block) 被访问的次数
    # 但由于 K 在外层，每个 K 迭代都会重新访问所有 (c, h_block, w_block)
    
    # 更准确地说：
    # 对于每个 K 迭代：
    #   对于每个 C：
    #     遍历 P->Q->R，访问一系列 (h_block, w_block)
    #     每次 (h_block, w_block) 变化时计数（但 c 是固定的）
    #   当 C 变化时，(c, h_block, w_block) 一定变化，所以计数
    #   
    # 所以 switches_per_K = C × (1 + switches_within_C_due_to_PQR)
    # 但第一个 C 不需要 +1（从无到有的第一次访问除外）
    
    # 分析：
    # - 每个 K 迭代开始时，从 C=0 开始，此时 (c, h_block, w_block) 与上一个 K 结束时不同（通常）
    # - 所以每个 K 迭代都会产生 switch
    # - 在每个 K 内部：
    #   - C 从 0 到 C_dram-1
    #   - 每次 C 变化时，(c, h_block, w_block) 变化
    #   - 在每个 C 内部，(h_block, w_block) 可能变化多次
    
    # 统计每个 C 内的 (h_block, w_block) switches
    # 由于 P->Q->R 遍历顺序固定，我们可以直接统计
    
    # 但问题是：每个 C 结束时的最后一个 (h_block, w_block) 和下一个 C 开始时的第一个 (h_block, w_block) 
    # 是否相同？答案是不同（因为 c 变了）
    
    # 所以实际的公式是：
    # switches_per_K = (C_dram - 1) + C_dram × (row_switches_per_KC - 1) + 1
    #               = C_dram × row_switches_per_KC
    # 
    # 等等，这不对。让我仔细想：
    # - 第一个 K 的第一个 C 的第一个 (p,q,r)：从 None 到 (0, h0, w0)，count = 1
    # - 同一个 C 内 (h_block, w_block) 变化：count += (变化次数)
    # - C 变化时：(c, h_block, w_block) 一定变化，count += 1
    # - K 变化时：回到 C=0，但 prev 是上一个 K 结束时的值
    #   - 上一个 K 结束时的值是 (C_dram-1, h_last, w_last)
    #   - 新 K 开始时的值是 (0, h0, w0)
    #   - 这两个不同，所以 count += 1
    
    # 综上：
    # switches = K_dram × C_dram × row_switches_per_KC
    # 这是因为每个 K 迭代重新遍历所有 C，每个 C 内遍历所有 (h_block, w_block) 变化
    
    total_switches = mapping.K_dram * mapping.C_dram * row_switches_per_KC
    
    print(f"\n=== 总计算 ===")
    print(f"K_dram={mapping.K_dram}, C_dram={mapping.C_dram}")
    print(f"total_switches = K × C × row_switches_per_KC")
    print(f"              = {mapping.K_dram} × {mapping.C_dram} × {row_switches_per_KC}")
    print(f"              = {total_switches}")
    
    # 计算 unique rows (用于对比)
    num_h_blocks = (H_in + block_h - 1) // block_h
    num_w_blocks = (W_in + block_w - 1) // block_w
    unique_rows = num_h_blocks * num_w_blocks * mapping.C_dram
    
    print(f"\n=== Unique Rows (对比) ===")
    print(f"num_h_blocks={num_h_blocks}, num_w_blocks={num_w_blocks}")
    print(f"unique_rows = {num_h_blocks} × {num_w_blocks} × {mapping.C_dram} = {unique_rows}")
    print(f"如果每个 unique row 被访问 K 次: {unique_rows * mapping.K_dram}")
    
    return {
        'row_acts': total_switches,
        'row_switches_per_KC': row_switches_per_KC,
        'unique_rows': unique_rows,
    }


def compute_h_crossing_exact(
    workload: Workload,
    mapping: Mapping,
    block_h: int
) -> Tuple[int, int]:
    """
    精确计算 H 方向的 crossing 统计
    
    枚举所有 (p_dram, r_dram) 组合，计算每个 tile 是否跨越 block 边界
    """
    R_total = mapping.R_dram * mapping.R_buf
    P_tile = workload.P // mapping.P_dram
    R_tile = workload.R // R_total
    
    H_tile = (P_tile - 1) * workload.stride_h + (R_tile - 1) * workload.dilation_h + 1
    
    crossing = 0
    non_crossing = 0
    
    for p_dram in range(mapping.P_dram):
        for r_dram in range(mapping.R_dram):
            # 计算这个 tile 的 H 起始位置
            p_start = p_dram * P_tile
            r_start = r_dram * mapping.R_buf  # R_buf 是 tile 内的 R 尺寸
            
            h_start = p_start * workload.stride_h + r_start * workload.dilation_h
            h_end = h_start + H_tile - 1
            
            start_block = h_start // block_h
            end_block = h_end // block_h
            
            if start_block == end_block:
                non_crossing += 1
            else:
                crossing += 1
    
    return (non_crossing, crossing)


def compute_w_crossing_exact(
    workload: Workload,
    mapping: Mapping,
    block_w: int
) -> Tuple[int, int]:
    """
    精确计算 W 方向的 crossing 统计
    """
    S_total = mapping.S_dram * mapping.S_buf
    Q_tile = workload.Q // mapping.Q_dram
    S_tile = workload.S // S_total
    
    W_tile = (Q_tile - 1) * workload.stride_w + (S_tile - 1) * workload.dilation_w + 1
    
    crossing = 0
    non_crossing = 0
    
    for q_dram in range(mapping.Q_dram):
        for s_dram in range(mapping.S_dram):
            q_start = q_dram * Q_tile
            s_start = s_dram * mapping.S_buf
            
            w_start = q_start * workload.stride_w + s_start * workload.dilation_w
            w_end = w_start + W_tile - 1
            
            start_block = w_start // block_w
            end_block = w_end // block_w
            
            if start_block == end_block:
                non_crossing += 1
            else:
                crossing += 1
    
    return (non_crossing, crossing)


def simulate_row_activations(
    workload: Workload,
    mapping: Mapping,
    block_h: int,
    block_w: int
) -> dict:
    """
    精确模拟 row activations，并统计不同类型的 row switch
    
    分类统计：
    - non_crossing: tile 不跨越任何 block 边界
    - h_only_crossing: tile 只跨越 H 方向 block 边界
    - w_only_crossing: tile 只跨越 W 方向 block 边界  
    - both_crossing: tile 同时跨越 H 和 W 方向 block 边界
    """
    P_tile = workload.P // mapping.P_dram
    Q_tile = workload.Q // mapping.Q_dram
    
    R_buf_actual = 1 if mapping.R_dram > 1 else workload.R
    S_buf_actual = 1 if mapping.S_dram > 1 else workload.S
    H_per_tile = (P_tile - 1) * workload.stride_h + (R_buf_actual - 1) * workload.dilation_h + 1
    W_per_tile = (Q_tile - 1) * workload.stride_w + (S_buf_actual - 1) * workload.dilation_w + 1
    
    H_in = workload.H
    W_in = workload.W
    
    prev_row_id = None
    total_switches = 0
    
    # 分类统计
    stats = {
        'non_crossing_switches': 0,      # 在 non-crossing tile 内发生的 switch
        'h_only_crossing_switches': 0,   # 在 h-only crossing tile 内发生的 switch
        'w_only_crossing_switches': 0,   # 在 w-only crossing tile 内发生的 switch
        'both_crossing_switches': 0,     # 在 both crossing tile 内发生的 switch
        
        'non_crossing_tiles': 0,
        'h_only_crossing_tiles': 0,
        'w_only_crossing_tiles': 0,
        'both_crossing_tiles': 0,
    }
    
    # 循环顺序: K -> C -> P -> Q -> R
    for k in range(mapping.K_dram):
        for c in range(mapping.C_dram):
            for p in range(mapping.P_dram):
                for q in range(mapping.Q_dram):
                    for r in range(mapping.R_dram):
                        # Tile 起始坐标
                        p_start = p * P_tile
                        q_start = q * Q_tile
                        r_start = r * R_buf_actual
                        
                        h_start = p_start * workload.stride_h + r_start * workload.dilation_h
                        w_start = q_start * workload.stride_w
                        
                        h_end = min(h_start + H_per_tile, H_in)
                        w_end = min(w_start + W_per_tile, W_in)
                        
                        # 判断 tile 类型
                        h_block_start = h_start // block_h
                        h_block_end = (h_end - 1) // block_h
                        w_block_start = w_start // block_w
                        w_block_end = (w_end - 1) // block_w
                        
                        h_crosses = (h_block_start != h_block_end)
                        w_crosses = (w_block_start != w_block_end)
                        
                        if h_crosses and w_crosses:
                            tile_type = 'both_crossing'
                        elif h_crosses:
                            tile_type = 'h_only_crossing'
                        elif w_crosses:
                            tile_type = 'w_only_crossing'
                        else:
                            tile_type = 'non_crossing'
                        
                        stats[f'{tile_type}_tiles'] += 1
                        
                        # 统计这个 tile 内的 row switches
                        tile_switches = 0
                        for h_block in range(h_block_start, h_block_end + 1):
                            for w_block in range(w_block_start, w_block_end + 1):
                                h_lo = max(h_start, h_block * block_h)
                                h_hi = min(h_end, (h_block + 1) * block_h)
                                w_lo = max(w_start, w_block * block_w)
                                w_hi = min(w_end, (w_block + 1) * block_w)
                                
                                for h in range(h_lo, h_hi):
                                    for w in range(w_lo, w_hi):
                                        row_id = (c, h_block, w_block)
                                        if prev_row_id is None or row_id != prev_row_id:
                                            total_switches += 1
                                            tile_switches += 1
                                        prev_row_id = row_id
                        
                        stats[f'{tile_type}_switches'] += tile_switches
    
    stats['total_switches'] = total_switches
    return stats


def main():
    print("=" * 80)
    print("Input Row Activation 新公式验证")
    print("=" * 80)
    
    # ResNet-L1 配置
    workload = Workload(
        R=7, S=7, P=56, Q=56, C=3, K=64, N=1,
        stride_h=1, stride_w=1, dilation_h=1, dilation_w=1
    )
    
    mapping = Mapping(
        P_dram=28, Q_dram=7, C_dram=3, K_dram=4, R_dram=7, S_dram=1,
        P_buf=2, Q_buf=8, C_buf=1, K_buf=16, R_buf=1, S_buf=7,
        K_outer=True
    )
    
    block_h = 31
    block_w = 31
    
    print(f"\n=== 工作负载 ===")
    print(f"R={workload.R}, S={workload.S}, P={workload.P}, Q={workload.Q}")
    print(f"C={workload.C}, K={workload.K}, N={workload.N}")
    print(f"Input size: H={workload.H}, W={workload.W}")
    
    print(f"\n=== Mapping ===")
    print(f"DRAM level: P={mapping.P_dram}, Q={mapping.Q_dram}, C={mapping.C_dram}, K={mapping.K_dram}, R={mapping.R_dram}, S={mapping.S_dram}")
    print(f"Block size: {block_h} × {block_w}")
    
    # 计算公式参数
    result = compute_row_activations_formula(workload, mapping, block_h, block_w)
    
    # 模拟并分类统计
    print(f"\n=== 模拟计算 (分类统计) ===")
    stats = simulate_row_activations(workload, mapping, block_h, block_w)
    
    print(f"\n--- Tile 分类统计 ---")
    print(f"non_crossing tiles:    {stats['non_crossing_tiles']}")
    print(f"h_only_crossing tiles: {stats['h_only_crossing_tiles']}")
    print(f"w_only_crossing tiles: {stats['w_only_crossing_tiles']}")
    print(f"both_crossing tiles:   {stats['both_crossing_tiles']}")
    total_tiles = stats['non_crossing_tiles'] + stats['h_only_crossing_tiles'] + stats['w_only_crossing_tiles'] + stats['both_crossing_tiles']
    print(f"Total tiles:           {total_tiles}")
    
    print(f"\n--- Row Switches 分类统计 ---")
    print(f"non_crossing switches:    {stats['non_crossing_switches']}")
    print(f"h_only_crossing switches: {stats['h_only_crossing_switches']}")
    print(f"w_only_crossing switches: {stats['w_only_crossing_switches']}")
    print(f"both_crossing switches:   {stats['both_crossing_switches']}")
    print(f"Total switches:           {stats['total_switches']}")
    
    print(f"\n--- 每种类型 Tile 的平均 Switches ---")
    if stats['non_crossing_tiles'] > 0:
        print(f"non_crossing:    {stats['non_crossing_switches'] / stats['non_crossing_tiles']:.2f} switches/tile")
    if stats['h_only_crossing_tiles'] > 0:
        print(f"h_only_crossing: {stats['h_only_crossing_switches'] / stats['h_only_crossing_tiles']:.2f} switches/tile")
    if stats['w_only_crossing_tiles'] > 0:
        print(f"w_only_crossing: {stats['w_only_crossing_switches'] / stats['w_only_crossing_tiles']:.2f} switches/tile")
    if stats['both_crossing_tiles'] > 0:
        print(f"both_crossing:   {stats['both_crossing_switches'] / stats['both_crossing_tiles']:.2f} switches/tile")
    
    # 公式预测 vs 实际
    print(f"\n=== 公式 vs 实际对比 ===")
    
    # 原公式参数
    h_non, h_crossing = compute_h_crossing_exact(workload, mapping, block_h)
    w_non, w_crossing = compute_w_crossing_exact(workload, mapping, block_w)
    
    print(f"\n公式参数:")
    print(f"  h_non={h_non}, h_crossing={h_crossing}")
    print(f"  w_non={w_non}, w_crossing={w_crossing}")
    
    # 公式预测的 tile 组合数
    print(f"\n公式预测的 tile 组合数 (per K×C iteration):")
    non_cross_combo = h_non * w_non
    h_only_combo = h_crossing * w_non
    w_only_combo = h_non * w_crossing
    both_combo = h_crossing * w_crossing
    print(f"  non_crossing:    h_non × w_non = {h_non} × {w_non} = {non_cross_combo}")
    print(f"  h_only_crossing: h_crossing × w_non = {h_crossing} × {w_non} = {h_only_combo}")
    print(f"  w_only_crossing: h_non × w_crossing = {h_non} × {w_crossing} = {w_only_combo}")
    print(f"  both_crossing:   h_crossing × w_crossing = {h_crossing} × {w_crossing} = {both_combo}")
    
    # 实际 per K×C 的 tiles
    KC = mapping.K_dram * mapping.C_dram
    print(f"\n实际 tile 数 (per K×C = {KC} iteration):")
    print(f"  non_crossing:    {stats['non_crossing_tiles'] // KC}")
    print(f"  h_only_crossing: {stats['h_only_crossing_tiles'] // KC}")
    print(f"  w_only_crossing: {stats['w_only_crossing_tiles'] // KC}")
    print(f"  both_crossing:   {stats['both_crossing_tiles'] // KC}")
    
    print(f"\n" + "=" * 80)
    print("结果比较")
    print("=" * 80)
    print(f"公式结果:     {result['row_acts']}")
    print(f"模拟结果:     {stats['total_switches']}")
    print(f"Trace 实际值: 5376")
    print(f"ILP 原始值:   2392")


if __name__ == "__main__":
    main()
