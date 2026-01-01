#!/usr/bin/env python3
"""
Row Activation Formula for Input Tensor

基于 sliding window 访问模式的 row activation 公式。

关键概念：
1. Input 的访问模式是 sliding window（滑动窗口）
2. tile 的起始位置按周期性模式变化
3. crossing 发生在 tile 跨越 row boundary 时

公式思路：
- 对于 row_aligned layout：使用 block crossing 分析
- 对于 sequential layout：使用 unique rows 近似
"""

import math
from typing import Dict, Tuple
from dataclasses import dataclass


# 维度常量
DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6


@dataclass
class FormulaConfig:
    """公式计算所需的配置"""
    # Workload 参数
    R: int
    S: int
    P: int
    Q: int
    C: int
    K: int
    N: int
    H: int  # Input height
    W: int  # Input width
    stride: Tuple[int, int] = (1, 1)
    dilation: Tuple[int, int] = (1, 1)
    
    # DRAM 配置
    row_buffer_bytes: int = 1024
    element_size: int = 1
    
    # Mapping 参数
    P_l3: int = 1
    Q_l3: int = 1
    C_l3: int = 1
    K_l3: int = 1
    
    # L2 (Row Buffer) Tiling Factors
    # 用于计算 L1 (On-Chip Buffer) Tile Size
    P_l2: int = 1
    Q_l2: int = 1
    C_l2: int = 1
    K_l2: int = 1
    
    block_h: int = 1
    block_w: int = 1
    input_layout: str = 'row_aligned'
    
    # Loop Order (Inner to Outer)
    # 'Q_inner': for p in P: for q in Q: ... (Row Major-ish)
    # 'P_inner': for q in Q: for p in P: ... (Col Major-ish)
    loop_order: str = 'Q_inner'
    
    @property
    def row_elements(self) -> int:
        """一个 row buffer 能容纳的元素数"""
        return self.row_buffer_bytes // self.element_size


def compute_input_row_switches_formula(config: FormulaConfig) -> int:
    """
    计算 Input tensor 的 row switches（公式版本）
    
    关键理解（基于 trace_generator 分析）：
    
    1. row_aligned layout:
       - 每个 Buffer Tile (L1) 在 DRAM 中存储时被 padding 到 row boundary
       - input_aligned_tile_size = ceil(tile_size / row_size) * row_size
       - 因此 Tile 内部没有 row crossing (除非 Tile > Row，但那是必然的连续访问)
       - Row Switch 主要发生在访问不同的 Buffer Tile 时
       
    2. sequential layout:
       - 地址连续分配，没有 padding
       - Row Switch 取决于 Buffer Tile 的访问模式是否跨越 Row Boundary
    """
    if config.input_layout == 'sequential':
        return _compute_input_sequential_row_switches(config)
    else:
        return _compute_input_row_aligned_row_switches(config)


def _compute_input_sequential_row_switches(config: FormulaConfig) -> int:
    """
    Sequential layout: 地址连续分配
    
    使用 ILP 模型中的精确公式：
    row_acts = non_crossing_acts + 2 * crossing_count * reuse_penalty
    
    其中：
    - crossing_count: 跨越 Row Boundary 的 Buffer Tile (L1) 数量
    - non_crossing_acts: 不跨越的 Tile 所需的 Act 次数 (可共享 Row)
    - reuse_penalty: K 维度的重用因子 (导致 Crossing Tile 反复切换 Row)
    """
    # 每次访问的 tile 大小 (L1 Buffer Tile Size)
    # P_l3 是 DRAM (L3) 层的循环次数，P_l2 是 Row Buffer (L2) 层的循环次数
    # 实际加载到 On-Chip Buffer (L1) 的 Tile 大小由两者共同决定
    P_tile = config.P // (config.P_l3 * config.P_l2)
    Q_tile = config.Q // (config.Q_l3 * config.Q_l2)
    C_tile = config.C // (config.C_l3 * config.C_l2)
    
    # Input tile 在 input space 的大小（考虑 sliding window）
    H_tile = config.stride[0] * (P_tile - 1) + config.dilation[0] * (config.R - 1) + 1
    W_tile = config.stride[1] * (Q_tile - 1) + config.dilation[1] * (config.S - 1) + 1
    input_tile_elements = config.N * C_tile * H_tile * W_tile
    
    # Calculate Effective Tile Span (considering Block Layout)
    if config.block_h > 1 and config.block_w > 1:
        # Block Layout: N, C, H, W where H, W are block-strided
        # Stride W = 1
        # Stride H = block_w
        # Stride C = block_h * block_w
        # Stride N = block_h * block_w * C_tile (assuming C_tile is the full C per block)
        
        # Span = (N-1)*SN + (C-1)*SC + (H-1)*SH + (W-1)*SW + 1
        # Note: We use H_tile and W_tile which are the accessed dimensions
        span_elements = (
            (config.N - 1) * (config.block_h * config.block_w * C_tile) +
            (C_tile - 1) * (config.block_h * config.block_w) +
            (H_tile - 1) * config.block_w +
            (W_tile - 1) * 1 + 
            1
        )
        tile_bytes = span_elements * config.element_size
    else:
        tile_bytes = input_tile_elements * config.element_size
    
    row_bytes = config.row_buffer_bytes
    
    # 总的 unique input tiles (L1 Tiles)
    num_tiles = (config.P_l3 * config.P_l2) * (config.Q_l3 * config.Q_l2) * (config.C_l3 * config.C_l2)
    
    # 1. 计算 Crossing Count
    crossing_count = compute_dram_row_crossing_count(tile_bytes, row_bytes, num_tiles)
    
    # 2. 计算 Non-Crossing Acts
    non_crossing_count = num_tiles - crossing_count
    
    if tile_bytes > row_bytes:
        # Tile spans multiple rows
        rows_per_tile = math.ceil(tile_bytes / row_bytes)
        non_crossing_acts = non_crossing_count * rows_per_tile
    else:
        # Multiple tiles per row
        tiles_per_row = max(1, int(row_bytes / tile_bytes))
        non_crossing_acts = math.ceil(non_crossing_count / tiles_per_row) if non_crossing_count > 0 else 0
    
    # Define steps for block crossing calculation
    H_step = P_tile * config.stride[0]
    W_step = Q_tile * config.stride[1]
    
    # 3. Reuse Penalty & K Factor
    # K_factor: Tile 被访问的总次数 (K 维度)
    K_factor = config.K_l3 * config.K_l2
    
    # Reuse Penalty: 针对 Non-Crossing Tile 的重用惩罚
    # 如果 K 是外层循环 (Conservative)，则 ReusePenalty = K_factor
    # 如果 K 是内层循环 (Optimistic)，则 ReusePenalty = 1
    reuse_penalty = K_factor  # Conservative assumption
    
    # 4. 总 Row Acts (DRAM Row Crossing 部分)
    if tile_bytes > row_bytes:
        # Large Tile Case:
        # Tile 比 Row Buffer 大，无法驻留。
        # 每次访问 (reuse) 都需要重新激活所有行。
        rows_per_tile = math.ceil(tile_bytes / row_bytes)
        dram_row_crossing_acts = num_tiles * rows_per_tile * K_factor
    else:
        # Small Tile Case:
        # Crossing Tiles: 每次访问需要 2 次 Act (Ping-Pong)，且无法复用 Row (必须乘 K_factor)
        # Non-Crossing Tiles: 取决于 ReusePenalty (K 是否为外层循环)
        # 公式: N_non * ReusePenalty + N_crossing * 2 * K_factor
        dram_row_crossing_acts = non_crossing_acts * reuse_penalty + 2 * crossing_count * K_factor
    
    # 5. Input Block Crossing (H & W directions)
    # ILP 模型中，Input Tensor 总是包含 Block Crossing 项 (无论 Layout)
    # 这是为了捕捉 Sliding Window 在逻辑 Block 边界上的开销
    
    # H 方向 Block Crossing
    h_crossing, _ = _compute_block_crossing_1d(
        block_size=config.block_h,
        tile_size=H_tile,
        step=H_step,
        num_tiles=config.P_l3 * config.P_l2
    )
    
    # W 方向 Block Crossing
    w_crossing, _ = _compute_block_crossing_1d(
        block_size=config.block_w,
        tile_size=W_tile,
        step=W_step,
        num_tiles=config.Q_l3 * config.Q_l2
    )
    
    # ILP 系数为 2 (保守估计/Upper Bound)
    block_crossing_acts = 2 * (h_crossing + w_crossing) * reuse_penalty
    
    return dram_row_crossing_acts + block_crossing_acts


def _compute_input_row_aligned_row_switches(config: FormulaConfig) -> int:
    """
    Row-aligned layout: 基于 tile 分类的 row activation 公式
    
    ========== 关键约束 (Key Constraints) ==========
    本公式假设 R/S 维度在 DRAM Cell Level (L3) 和 DRAM Row Buffer Level (L2) 的因子为 1：
      - R_l3 = 1, R_l2 = 1
      - S_l3 = 1, S_l2 = 1
    
    这意味着 R/S 循环只发生在 On-Chip Buffer Level (L1) 或更低层级（PE Array 内部）。
    因此，整个 Kernel 窗口 (R×S) 在从 DRAM 加载 Input Tile 时被视为原子操作单元。
    
    理由：
    1. R/S 通常较小（1, 3, 5, 7），不值得在 DRAM/RowBuffer 层级切分。
    2. 切分 R/S 会导致频繁的 Row Buffer Thrashing（因为需要反复加载同一行的不同部分）。
    3. 如果违反此约束，公式将低估 Row Activation 开销。
    ================================================
    
    核心公式（修正版）：
    
    row_acts = C_factor × (
        reuse_penalty × (num_blocks_h × num_blocks_w) × 1 +
        K_factor × (h_crossing × w_non) × 2 +
        K_factor × (h_non × w_crossing) × 2 +
        K_factor × (h_crossing × w_crossing) × 4
    )
    
    其中：
    - num_blocks_h/w: Input Tensor 覆盖的 Unique Block 数量
    - h_non / h_crossing：H 方向不跨越/跨越 block 边界的 tile 数
    - w_non / w_crossing：W 方向不跨越/跨越 block 边界的 tile 数
    - reuse_penalty：K 维度的 reuse 惩罚（简化为 K_factor）
    
    关键修正：
    - Non-crossing 部分使用 Unique Blocks 数量计算，因为同一个 Block 内的多个 Non-Crossing Tile 共享一次 Row Activation。
    - Crossing 部分仍然基于 Tile 数量计算，因为每次跨越边界都可能导致 Row Buffer Thrashing (Ping-Pong)。

    P/R 循环对 Crossing 次数的影响机制 (Loop Analysis):

    1. P 循环 (The Sliding Driver - 驱动滑动):
       - 循环行为：P 循环负责在 Input Map 上移动窗口。
       - 公式映射：
         - 循环次数 -> `num_tiles` (P_l3): 公式会对所有 P_l3 次迭代进行求和。
         - 循环步长 -> `step` (H_step): 公式利用 GCD 分析步长与 Block Size 的相位关系。
       - 能否反映特点？能。公式通过 `_compute_block_crossing_1d` 精确计算了 P 循环每一步的落脚点，捕捉了"步长对齐"导致的 Crossing 数量波动。

    2. R 循环 (The Window Expander - 扩张窗口):
       - 循环行为：R 循环在 Tile 内部累加数据，它"撑大"了当前需要的 Input 区域。
       - 公式映射：
         - 循环范围 -> `tile_size` (H_tile): R 循环的迭代范围直接决定了 H_tile 的大小。
       - 能否反映特点？能。公式将 R 循环视为一个原子操作单元(Atomic Unit)。
         只要 R 循环撑大的区域 (H_tile) 触碰了边界，公式就会判定为 Crossing。
         这正确反映了"R 循环越大，单次访问越容易跨行"的物理事实。
    """
    # === 基本维度计算 ===
    # P_tile / Q_tile: On-Chip Buffer (L1) 中的 Tile 大小
    # 这里假设 P_l3 / Q_l3 是 DRAM (L3) 层的循环次数，
    # P_l2 / Q_l2 是 Row Buffer (L2) 层的循环次数。
    # 每次迭代加载一个 Tile 到 On-Chip Buffer (L1)。
    # Row Buffer (L2) 被视为透明缓存。
    P_tile = config.P // (config.P_l3 * config.P_l2)
    Q_tile = config.Q // (config.Q_l3 * config.Q_l2)
    
    # Input tile 在 input space 的大小（考虑 sliding window）
    H_tile = config.stride[0] * (P_tile - 1) + config.dilation[0] * (config.R - 1) + 1
    W_tile = config.stride[1] * (Q_tile - 1) + config.dilation[1] * (config.S - 1) + 1
    
    # Tile 的起始位置步长
    H_step = config.stride[0] * P_tile
    W_step = config.stride[1] * Q_tile
    
    # === Block 相关计算 ===
    block_h = config.block_h
    block_w = config.block_w
    
    # === 使用 GCD 周期性分析计算 crossing tiles ===
    # H 方向
    h_crossing, h_total = _compute_block_crossing_1d(
        block_size=block_h,
        tile_size=H_tile,
        step=H_step,
        num_tiles=config.P_l3 * config.P_l2
    )
    h_non = h_total - h_crossing
    
    # W 方向
    w_crossing, w_total = _compute_block_crossing_1d(
        block_size=block_w,
        tile_size=W_tile,
        step=W_step,
        num_tiles=config.Q_l3 * config.Q_l2
    )
    w_non = w_total - w_crossing
    
    # === Reuse factors ===
    C_factor = config.C_l3 * config.C_l2
    K_factor = config.K_l3 * config.K_l2
    
    # Reuse penalty: 简化假设 K 在最外层
    reuse_penalty = K_factor
    
    # === 应用公式：基于 L1 Tile 粒度的分类计费 ===
    # 核心思想：Micro-operation Simulation (微观行为模拟)
    # 我们将 Workload 拆解为 P_l3*P_l2 个 L1 Tiles，并根据其物理位置分类：
    #
    # 1. Non-Crossing Tiles (安全块):
    #    - 定义：完全落在一个 Row/Block 内部的 Tile。
    #    - 假设：利用 Spatial Locality。连续的 Non-Crossing Tiles 共享同一个 Row Open。
    #    - 计费：按 Unique Blocks 数量计费 (num_blocks * 1)。
    #
    # 2. Crossing Tiles (危险块):
    #    - 定义：跨越 Row/Block 边界的 Tile。
    #    - 假设：触发 Row Buffer Thrashing (Ping-Pong)。
    #          因为单 Bank 无法同时保持两行开启，每次读取该 Tile 都会导致 2 次 Row Switch。
    #    - 计费：按 Tile 数量计费 (num_crossing_tiles * 2)。
    #          这捕捉了 L2 循环导致的反复跨行访问开销。
    
    # 计算 Input Tensor 覆盖的 Block 总数
    H_input = config.stride[0] * (config.P - 1) + config.dilation[0] * (config.R - 1) + 1
    W_input = config.stride[1] * (config.Q - 1) + config.dilation[1] * (config.S - 1) + 1
    
    num_blocks_h = math.ceil(H_input / block_h)
    num_blocks_w = math.ceil(W_input / block_w)
    
    # 1. Non-crossing acts = Unique Blocks * Reuse
    # 修正：考虑 Loop Order 导致的 Thrashing
    # 如果 Inner Loop 覆盖了多个 Block，那么 Outer Loop 的每次迭代都会导致 Block 切换 (Thrashing)。
    # 只有当 Inner Loop 完全在一个 Block 内时，我们才能享受 "Sequential Access" 的红利。
    
    # 计算 L2 Loop 覆盖的 Block 数量 (近似)
    # H_span_l2 = stride * (P_l2 - 1) * P_tile + H_tile
    # 但为了简化，我们看 "跨越了多少个 block boundary"
    # 实际上，只要 Inner Loop 导致了 Row Switch，Outer Loop 就会放大它。
    
    # 估算 P_l2 循环覆盖的 Block 数
    h_span_l2 = config.stride[0] * (config.P_l2 * P_tile) 
    blocks_h_l2 = math.ceil(h_span_l2 / block_h) if block_h > 0 else 1
    
    # 估算 Q_l2 循环覆盖的 Block 数
    w_span_l2 = config.stride[1] * (config.Q_l2 * Q_tile)
    blocks_w_l2 = math.ceil(w_span_l2 / block_w) if block_w > 0 else 1
    
    # 根据 Loop Order 判断 Thrashing
    thrashing_multiplier = 1.0
    
    if config.loop_order == 'Q_inner':
        # Inner Loop (Q) 跨越 Block -> 潜在的 Ping-Pong
        if blocks_w_l2 > 1:
            # Outer Loop (P) 的每一次迭代都会触发一次 A->B->... 的扫描
            # 如果 P 循环主要在同一个 Block Row 内移动 (blocks_h_l2 小)，
            # 那么我们会反复打开/关闭这组 Block (A, B)。
            # 
            # Thrashing Multiplier = (Outer Loop Iterations) / (Unique Block Rows Visited)
            # 例如：P_l2=10, blocks_h_l2=1 -> 意味着我们在同一行 Block 上重复了 10 次 A->B 模式。
            #      Multiplier = 10。
            # 例如：P_l2=10, blocks_h_l2=10 -> 意味着每次 P 移动都去了新行。A0->B0, A1->B1...
            #      虽然每次都有切换，但都是 Compulsory Miss (新 Block)，不是 Thrashing (旧 Block)。
            #      Multiplier = 1。
            
            thrashing_multiplier = config.P_l2 / max(1, blocks_h_l2)

    elif config.loop_order == 'P_inner':
        # Inner Loop (P) 跨越 Block
        if blocks_h_l2 > 1:
            # Outer Loop (Q) 导致 Thrashing
            thrashing_multiplier = config.Q_l2 / max(1, blocks_w_l2)
            
    # 应用 Multiplier
    # 注意：Multiplier >= 1.0
    non_crossing_acts = reuse_penalty * (num_blocks_h * num_blocks_w) * thrashing_multiplier
    
    # 2. Crossing acts = Crossing Tiles * 2 * Reuse
    # 假设：Crossing Tile 必定导致 Ping-Pong。
    # 这里的 h_only_tiles 等是具体的 L1 Tile 数量，直接反映了 L2 循环的次数。
    
    # H-only crossing tiles: h_crossing × w_non 个，每个访问 2 个 blocks
    # 注意：这里 w_non 仍然是 Tile 数量，这是为了计算有多少个 Crossing 事件
    h_only_tiles = h_crossing * w_non
    h_only_crossing_acts = K_factor * h_only_tiles * 2
    
    # W-only crossing tiles: h_non × w_crossing 个，每个访问 2 个 blocks
    w_only_tiles = h_non * w_crossing
    w_only_crossing_acts = K_factor * w_only_tiles * 2
    
    # Both crossing tiles: h_crossing × w_crossing 个，每个访问 4 个 blocks
    both_tiles = h_crossing * w_crossing
    both_crossing_acts = K_factor * both_tiles * 4
    
    # 总 row activations
    # 注意：这里不再乘以 C_factor，因为 num_blocks 已经是整个 Input 的 Block 数？
    # 不，Input 在 Channel 维度被切分 (C_l3)。
    # 每次 C 循环，我们需要访问新的 Input Channel Group。
    # 所以 C_factor 仍然需要。
    
    row_acts = C_factor * (non_crossing_acts + h_only_crossing_acts + 
                           w_only_crossing_acts + both_crossing_acts)
    
    return row_acts


def _compute_block_crossing_1d(
    block_size: int,
    tile_size: int,
    step: int,
    num_tiles: int
) -> Tuple[int, int]:
    """
    计算 1D 方向的 block crossing 数量
    
    使用 GCD 周期性分析：
    - tile 起始位置模 block_size 呈周期性
    - 周期 = block_size / gcd(step, block_size)
    - crossing 发生在 (start_pos % block_size) + tile_size > block_size
    
    Args:
        block_size: block 大小
        tile_size: tile 大小
        step: 相邻 tile 的间隔
        num_tiles: tile 数量
        
    Returns:
        (crossing_count, total_count)
    """
    if block_size <= 0 or tile_size <= 0 or step <= 0 or num_tiles <= 0:
        return (0, 0)
    
    # 如果 tile 比 block 大，每个 tile 都 crossing
    if tile_size > block_size:
        return (num_tiles, num_tiles)
    
    # 如果 tile 正好等于 block，没有 crossing
    if tile_size == block_size:
        return (0, num_tiles)
    
    # GCD 周期分析
    g = math.gcd(step, block_size)
    period = block_size // g
    
    # 计算一个周期内有多少 crossing
    crossing_in_period = 0
    for k in range(period):
        start_pos = (k * step) % block_size
        if start_pos + tile_size > block_size:
            crossing_in_period += 1
    
    # 分解为完整周期 + 余数
    num_complete_periods = num_tiles // period
    remainder = num_tiles % period
    
    # 余数部分的 crossing
    crossing_in_remainder = 0
    for k in range(remainder):
        start_pos = (k * step) % block_size
        if start_pos + tile_size > block_size:
            crossing_in_remainder += 1
    
    total_crossing = num_complete_periods * crossing_in_period + crossing_in_remainder
    
    return (total_crossing, num_tiles)


def compute_dram_row_crossing_count(tile_bytes: int, row_bytes: int, num_tiles: int) -> int:
    """
    计算 DRAM Row Crossing 的数量 (用于 Sequential Layout)
    
    Args:
        tile_bytes: Tile 大小 (bytes)
        row_bytes: Row Buffer 大小 (bytes)
        num_tiles: Tile 数量
        
    Returns:
        跨越 Row Boundary 的 Tile 数量
    """
    if tile_bytes <= 0 or num_tiles <= 0:
        return 0
    if tile_bytes > row_bytes:
        return num_tiles  # 所有 tile 都跨越
    if tile_bytes == row_bytes:
        return 0  # 刚好对齐，不跨越
    
    g = math.gcd(tile_bytes, row_bytes)
    period = row_bytes // g
    
    # 计算一个周期内的 crossing
    # Crossing 发生在: (offset % row_bytes) + tile_bytes > row_bytes
    threshold = row_bytes - tile_bytes + 1
    cross_count_per_period = period - math.ceil(threshold / g)
    cross_count_per_period = max(0, cross_count_per_period)
    
    # 周期分解
    num_complete_periods = num_tiles // period
    remainder_tiles = num_tiles % period
    
    # 余数部分的 crossing
    crossings_in_remainder = 0
    for i in range(remainder_tiles):
        start_offset = i * tile_bytes
        start_row = start_offset // row_bytes
        end_row = (start_offset + tile_bytes - 1) // row_bytes
        if end_row > start_row:
            crossings_in_remainder += 1
            
    return num_complete_periods * cross_count_per_period + crossings_in_remainder


def compute_weight_row_switches_formula(config: FormulaConfig) -> int:
    """
    计算 Weight tensor 的 row switches（简化公式）
    
    Weight 是静态数据，访问模式分析：
    - Weight 只依赖 R, S, C, K 维度（不依赖 P, Q, N）
    - 每个 DRAM tile 访问 weight_tile = R × S × C_tile × K_tile
    - 当 R, S, C, K 维度的 loop 变化时，访问新的 weight tile
    
    关键观察（从 trace 数据）：
    - 大多数配置的 weight row switches = 9，非常稳定
    - 这说明 weight 访问主要受限于 unique rows，而非 reuse 模式
    
    简化公式：
    - sequential layout: row_switches ≈ unique_rows - 1（连续访问）
    - row_aligned layout: 基于 block crossing
    """
    R, S = config.R, config.S
    C_tile = config.C // (config.C_l3 * config.C_l2)
    K_tile = config.K // (config.K_l3 * config.K_l2)
    
    weight_tile_size = R * S * C_tile * K_tile
    unique_rows = math.ceil(weight_tile_size / config.row_elements)
    
    # Weight 被 P_l3 × Q_l3 × C_l3 × K_l3 次 DRAM iteration 访问
    # 但 Weight 只在 C 和 K 维度变化时加载新数据
    # P, Q 的变化重用同一个 weight tile
    
    # 简化模型：
    # - C_l3 × K_l3 个不同的 weight tiles
    # - 每个 tile 被 P_l3 × Q_l3 次重用
    # - row_switches ≈ C_l3 × K_l3 × (unique_rows - 1) + C_l3 × K_l3 - 1
    
    num_weight_tiles = (config.C_l3 * config.C_l2) * (config.K_l3 * config.K_l2)
    
    # 每个 weight tile 内部的 row switches（连续访问，很少）
    internal_switches = max(0, unique_rows - 1)
    
    # tile 之间的切换
    tile_switches = max(0, num_weight_tiles - 1)
    
    # 但 P, Q 的 reuse 会导致反复访问同一 tile
    # 这里简化为：每次 C/K 变化时切换 row，但连续重用时不切换
    # row_switches ≈ num_weight_tiles × (1 + internal_switches)
    
    # 更保守的估计（基于 trace 观察：大部分是 9）
    # 使用 unique rows 作为上界
    row_switches = num_weight_tiles * (1 + internal_switches) // ((config.P_l3 * config.P_l2) * (config.Q_l3 * config.Q_l2))
    row_switches = max(row_switches, unique_rows - 1)
    
    return max(0, row_switches)


def compute_output_row_switches_formula(config: FormulaConfig) -> int:
    """
    计算 Output tensor 的 row switches（简化公式）
    
    Output 访问模式：
    - 每个 DRAM tile 访问 output_tile = P_tile × Q_tile × K_tile
    - row switches ≈ unique_rows × reuse_penalty
    """
    P_tile = config.P // (config.P_l3 * config.P_l2)
    Q_tile = config.Q // (config.Q_l3 * config.Q_l2)
    K_tile = config.K // (config.K_l3 * config.K_l2)
    
    output_tile_size = config.N * P_tile * Q_tile * K_tile
    unique_rows = math.ceil(output_tile_size / config.row_elements)
    
    # Reuse penalty: output 在 R, S, C 维度被重用
    reuse_factor = config.C_l3 * config.C_l2  # 只有 C 影响 output reuse
    
    # Row switches
    row_switches = unique_rows * (config.P_l3 * config.P_l2) * (config.Q_l3 * config.Q_l2) * (config.K_l3 * config.K_l2) - 1
    
    return max(0, row_switches)


def compute_total_row_switches_formula(config: FormulaConfig) -> Dict[str, int]:
    """
    计算所有 tensor 的 row switches
    
    Returns:
        Dict with keys: 'input', 'weight', 'output', 'total'
    """
    input_rs = compute_input_row_switches_formula(config)
    weight_rs = compute_weight_row_switches_formula(config)
    output_rs = compute_output_row_switches_formula(config)
    
    return {
        'input': input_rs,
        'weight': weight_rs,
        'output': output_rs,
        'total': input_rs + weight_rs + output_rs
    }


# === 测试 ===
if __name__ == "__main__":
    # ResNet-L1 配置
    config = FormulaConfig(
        R=7, S=7, P=56, Q=56, C=3, K=64, N=1,
        H=62, W=62,
        stride=(1, 1), dilation=(1, 1),
        row_buffer_bytes=1024, element_size=1,
        P_l3=28, Q_l3=7, C_l3=3, K_l3=4,
        block_h=31, block_w=31,
        input_layout='row_aligned'
    )
    
    result = compute_total_row_switches_formula(config)
    print("Formula Results:")
    print(f"  Input:  {result['input']}")
    print(f"  Weight: {result['weight']}")
    print(f"  Output: {result['output']}")
    print(f"  Total:  {result['total']}")
