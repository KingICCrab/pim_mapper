#!/usr/bin/env python3
"""
验证新的 Input row activation 公式。

这个脚本使用 ILP 得到的 mapping，然后用新公式计算 row activations，
与 Trace 的结果进行比较。

新公式的关键修改：
- R_outer = R_level1 × R_level2 × R_level3 (不只是 R_DRAM)
- S_outer = S_level1 × S_level2 × S_level3 (不只是 S_DRAM)
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload
from validation.dram.trace_generator import TraceGenerator, DRAMConfig

DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6
DIM_NAMES = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']


def get_factor_at_level(mapping, level: int, dim: int) -> int:
    """获取指定 level 和 dimension 的 factor。"""
    if level not in mapping.loop_bounds:
        return 1
    
    factor = 1
    bounds = mapping.loop_bounds[level]
    
    if level == 0:
        # Level 0 使用 H, W, Internal, temporal 格式
        for key in ['H', 'W', 'Internal', 'temporal']:
            if key in bounds and dim in bounds[key]:
                factor *= bounds[key][dim]
    else:
        # Level 1+ 使用 spatial, temporal 格式
        for key in ['spatial', 'temporal']:
            if key in bounds and dim in bounds[key]:
                factor *= bounds[key][dim]
    
    return factor


def get_dram_factor(mapping, dim: int) -> int:
    """获取 DRAM level (2+3) 的 factor。"""
    factor = 1
    for level in [2, 3]:
        factor *= get_factor_at_level(mapping, level, dim)
    return factor


def compute_input_row_acts_new_formula(mapping, workload):
    """
    使用新公式计算 Input row activations。
    
    关键：R_outer 和 S_outer 需要考虑所有非 buffer 层的 factors。
    """
    # 获取参数
    P = workload.bounds[DIM_P]
    Q = workload.bounds[DIM_Q]
    R = workload.bounds[DIM_R]
    S = workload.bounds[DIM_S]
    stride_h = workload.stride[0] if hasattr(workload, 'stride') else 1
    stride_w = workload.stride[1] if hasattr(workload, 'stride') else 1
    dilation_h = workload.dilation[0] if hasattr(workload, 'dilation') else 1
    dilation_w = workload.dilation[1] if hasattr(workload, 'dilation') else 1
    
    # 获取 block size
    block_h = mapping.tile_info.get('block_h', 1)
    block_w = mapping.tile_info.get('block_w', 1)
    
    # 获取各 level 的 factors
    print("\n=== Factor Analysis ===")
    for level in range(4):
        factors = {DIM_NAMES[d]: get_factor_at_level(mapping, level, d) for d in range(7)}
        non_one = {k: v for k, v in factors.items() if v > 1}
        print(f"  Level {level}: {non_one if non_one else '(all 1s)'}")
    
    # 新公式：R_outer = R_level0 × R_level1 × R_level2 × R_level3
    # 但 Level 0 的 R 是在 buffer 内复用的，不会产生 row activation
    # 实际上，产生 row activation 的是 Level 2 和 Level 3 的 R factor
    R_l0 = get_factor_at_level(mapping, 0, DIM_R)
    R_l1 = get_factor_at_level(mapping, 1, DIM_R)
    R_l2 = get_factor_at_level(mapping, 2, DIM_R)
    R_l3 = get_factor_at_level(mapping, 3, DIM_R)
    
    S_l0 = get_factor_at_level(mapping, 0, DIM_S)
    S_l1 = get_factor_at_level(mapping, 1, DIM_S)
    S_l2 = get_factor_at_level(mapping, 2, DIM_S)
    S_l3 = get_factor_at_level(mapping, 3, DIM_S)
    
    # DRAM factors (Level 2 + 3)
    P_dram = get_dram_factor(mapping, DIM_P)
    Q_dram = get_dram_factor(mapping, DIM_Q)
    C_dram = get_dram_factor(mapping, DIM_C)
    K_dram = get_dram_factor(mapping, DIM_K)
    
    # R_outer: 所有非 buffer level 的 R factor 乘积
    # 这里 buffer 是 Level 0+1，所以 R_outer = R_l2 × R_l3
    R_outer = R_l2 * R_l3
    S_outer = S_l2 * S_l3
    
    print(f"\n=== R/S Factors ===")
    print(f"  R: L0={R_l0}, L1={R_l1}, L2={R_l2}, L3={R_l3}")
    print(f"  S: L0={S_l0}, L1={S_l1}, L2={S_l2}, L3={S_l3}")
    print(f"  R_outer (L2×L3) = {R_outer}")
    print(f"  S_outer (L2×L3) = {S_outer}")
    
    # Buffer tile size (Level 0+1)
    # 注意：如果某个维度在 DRAM level (Level 2+3)，则它的 buffer tile size 是 1
    P_buf = get_factor_at_level(mapping, 0, DIM_P) * get_factor_at_level(mapping, 1, DIM_P)
    Q_buf = get_factor_at_level(mapping, 0, DIM_Q) * get_factor_at_level(mapping, 1, DIM_Q)
    # R_buf: 如果 R 在 DRAM level (R_outer > 1)，则 R_buf = total_R / R_outer
    # 否则 R_buf = total_R
    R_buf = workload.R // R_outer if R_outer > 1 else workload.R
    # S_buf: 同理
    S_buf = workload.S // S_outer if S_outer > 1 else workload.S
    
    print(f"\n=== Buffer Tile (L0×L1) ===")
    print(f"  P_buf={P_buf}, Q_buf={Q_buf}")
    print(f"  R_buf={R_buf} (R={workload.R}, R_outer={R_outer})")
    print(f"  S_buf={S_buf} (S={workload.S}, S_outer={S_outer})")
    
    # Input access tile size (sliding window)
    H_per_tile = (P_buf - 1) * stride_h + (R_buf - 1) * dilation_h + 1
    W_per_tile = (Q_buf - 1) * stride_w + (S_buf - 1) * dilation_w + 1
    
    print(f"\n=== Access Tile ===")
    print(f"  H_per_tile = {H_per_tile}")
    print(f"  W_per_tile = {W_per_tile}")
    print(f"  block_h = {block_h}, block_w = {block_w}")
    
    # 计算 H 方向的 crossing
    # P_factor = P_dram (DRAM level P tiles)
    # 但实际上，对于 Input，P 和 R 都会影响 H 方向的访问
    # H range = [0, (P-1)*stride_h + (R-1)*dilation_h + 1)
    #
    # 每个 DRAM tile 访问的 H range:
    # - P_dram tiles in P direction
    # - R_outer tiles in R direction (R sliding)
    # 
    # 关键洞察：R_outer 循环会导致 Input tile 在 H 方向滑动！
    # 每次 R 迭代，h_start = p*stride_h + r*dilation_h
    
    print(f"\n=== DRAM Level Factors ===")
    print(f"  P_dram={P_dram}, Q_dram={Q_dram}")
    print(f"  C_dram={C_dram}, K_dram={K_dram}")
    
    # 计算 row activations (简化的 row_aligned 模式)
    # 对于 row_aligned：每个 unique (h_block, w_block, c) 组合是一个 row
    # 
    # 方案 A：按 DRAM tile 迭代计算
    # 每个 DRAM tile 访问的 block 数 = ceil(H_per_tile / block_h) × ceil(W_per_tile / block_w)
    # 但如果 R_outer > 1，相邻 R tiles 的 H range 会重叠或相邻
    
    # 计算总的 H/W 方向 tile 迭代次数
    # P_dram × R_outer = 总的 H 方向 tile 迭代次数
    # Q_dram × S_outer = 总的 W 方向 tile 迭代次数
    
    h_tile_iters = P_dram * R_outer
    w_tile_iters = Q_dram * S_outer
    
    print(f"\n=== Tile Iterations ===")
    print(f"  H direction: P_dram × R_outer = {P_dram} × {R_outer} = {h_tile_iters}")
    print(f"  W direction: Q_dram × S_outer = {Q_dram} × {S_outer} = {w_tile_iters}")
    
    # 计算每个 (h_tile, w_tile) 访问多少个 blocks
    # 这需要考虑 block boundary crossing
    
    # 简化计算：假设每个 tile 访问 unique blocks
    # unique_h_blocks ≈ H_per_tile / block_h (ceiling)
    # unique_w_blocks ≈ W_per_tile / block_w (ceiling)
    
    blocks_per_tile_h = (H_per_tile + block_h - 1) // block_h
    blocks_per_tile_w = (W_per_tile + block_w - 1) // block_w
    
    print(f"\n=== Blocks per Tile ===")
    print(f"  blocks_per_tile_h = ceil({H_per_tile}/{block_h}) = {blocks_per_tile_h}")
    print(f"  blocks_per_tile_w = ceil({W_per_tile}/{block_w}) = {blocks_per_tile_w}")
    
    # Row activations (简化模型)
    # 假设 row_aligned：每个 DRAM tile iteration 访问的 unique rows
    # = blocks_per_tile_h × blocks_per_tile_w × C_factor (per DRAM tile)
    # × h_tile_iters × w_tile_iters (total iterations)
    # 但这会重复计算重叠部分...
    
    # 更准确的模型：
    # 总的 unique (h_block, w_block, c) combinations
    # H_in = P + R - 1 (for stride=1, dilation=1)
    # W_in = Q + S - 1
    # num_h_blocks = ceil(H_in / block_h)
    # num_w_blocks = ceil(W_in / block_w)
    # unique_rows = num_h_blocks × num_w_blocks × C
    
    H_in = P + R - 1
    W_in = Q + S - 1
    num_h_blocks = (H_in + block_h - 1) // block_h
    num_w_blocks = (W_in + block_w - 1) // block_w
    C = workload.C
    
    unique_rows = num_h_blocks * num_w_blocks * C
    
    print(f"\n=== Unique Rows ===")
    print(f"  H_in = {H_in}, W_in = {W_in}")
    print(f"  num_h_blocks = {num_h_blocks}, num_w_blocks = {num_w_blocks}")
    print(f"  unique_rows = {num_h_blocks} × {num_w_blocks} × {C} = {unique_rows}")
    
    # 但 Trace 统计的是 row switches，不是 unique rows
    # row switches = 每次 row_id 变化时 +1
    # 
    # 对于 row_aligned 布局，如果循环顺序合适，可以最小化 switches
    # 最坏情况：每个 DRAM tile 都要 switch row
    # 
    # 关键问题：K 是 irrelevant dim，每次 K 迭代会重访相同的 Input rows
    # 所以 row_switches = unique_switches_per_K × K_factor
    
    # 计算每个 K iteration 内的 row switches
    # 遍历顺序：(per K) C -> P -> Q -> R -> S (具体顺序取决于 permutation)
    # 如果 P->R 或 Q->S 是连续的，可以减少 switches
    
    # 简化模型：假设每个 (P_dram × R_outer) × (Q_dram × S_outer) × C_dram tile 访问
    # 不同的 blocks，产生 block switches
    
    # 更简单：row_switches ≈ h_tile_iters × w_tile_iters × C_dram × blocks_per_tile × K_dram
    # 但这忽略了 block 内的连续访问
    
    # 实际公式：参考 Trace 的计算方式
    # Trace 迭代 K -> C -> P -> Q -> R -> (S in buffer)
    # 每次 (c, p, q, r) 变化时，h_start 和 w_start 可能变化
    # 如果新的 (h_block, w_block) != 旧的，就 switch
    
    # 让我们用精确模拟来验证
    row_switches_sim = simulate_row_switches(
        P_dram, Q_dram, R_outer, S_outer, C_dram, K_dram,
        P_buf, Q_buf, R_buf, S_buf,
        stride_h, stride_w, dilation_h, dilation_w,
        block_h, block_w, H_in, W_in, C
    )
    
    return {
        'R_outer': R_outer,
        'S_outer': S_outer,
        'h_tile_iters': h_tile_iters,
        'w_tile_iters': w_tile_iters,
        'unique_rows': unique_rows,
        'row_switches_sim': row_switches_sim,
    }


def simulate_row_switches(
    P_dram, Q_dram, R_outer, S_outer, C_dram, K_dram,
    P_buf, Q_buf, R_buf, S_buf,
    stride_h, stride_w, dilation_h, dilation_w,
    block_h, block_w, H_in, W_in, C
):
    """
    精确模拟 row switches。
    
    关键点：
    1. 遍历每个 element，不是只看 tile 起始位置
    2. R_buf 和 S_buf 是 buffer tile size，如果 R/S 在 DRAM level，则为 1
    3. Row switch 发生在 (c, h_block, w_block) 变化时
    """
    prev_row_id = None
    row_switches = 0
    
    # 对于每个 K iteration (K is irrelevant for Input)
    for k in range(K_dram):
        for c in range(C_dram):
            for p in range(P_dram):
                for q in range(Q_dram):
                    for r in range(R_outer):
                        # Tile 起始坐标
                        p_start = p * P_buf
                        q_start = q * Q_buf
                        r_start = r * R_buf  # R_buf = 1 if R is in DRAM level
                        s_start = 0  # S in buffer
                        
                        h_start = p_start * stride_h + r_start * dilation_h
                        w_start = q_start * stride_w + s_start * dilation_w
                        
                        # Tile 范围
                        H_per_tile = (P_buf - 1) * stride_h + (R_buf - 1) * dilation_h + 1
                        W_per_tile = (Q_buf - 1) * stride_w + (S_buf - 1) * dilation_w + 1
                        h_end = min(h_start + H_per_tile, H_in)
                        w_end = min(w_start + W_per_tile, W_in)
                        
                        # 遍历 tile 覆盖的 blocks
                        h_block_start = h_start // block_h
                        h_block_end = (h_end - 1) // block_h
                        w_block_start = w_start // block_w
                        w_block_end = (w_end - 1) // block_w
                        
                        for h_block in range(h_block_start, h_block_end + 1):
                            for w_block in range(w_block_start, w_block_end + 1):
                                # Block 内的有效范围
                                h_lo = max(h_start, h_block * block_h)
                                h_hi = min(h_end, (h_block + 1) * block_h)
                                w_lo = max(w_start, w_block * block_w)
                                w_hi = min(w_end, (w_block + 1) * block_w)
                                
                                # 遍历 block 内的每个 element
                                for h in range(h_lo, h_hi):
                                    for w in range(w_lo, w_hi):
                                        row_id = (c, h_block, w_block)
                                        if prev_row_id is None or row_id != prev_row_id:
                                            row_switches += 1
                                        prev_row_id = row_id
    
    return row_switches


def main():
    # 定义 workload
    workload = ConvWorkload(name="ResNet-L1", R=7, S=7, P=56, Q=56, C=3, K=64, N=1)
    
    print("=" * 80)
    print(f"WORKLOAD: {workload.name}")
    print(f"R={workload.R}, S={workload.S}, P={workload.P}, Q={workload.Q}")
    print(f"C={workload.C}, K={workload.K}, N={workload.N}")
    print("=" * 80)
    
    # 运行 ILP (使用原始代码)
    print("\n>>> Running ILP Optimizer (original code)...")
    optimizer = PIMOptimizer()
    result = optimizer.optimize([workload], objective="latency")
    mapping = result.mappings[0]
    
    print(f"\n>>> ILP Result:")
    print(f"  row_acts_input (ILP):  {mapping.metrics.get('row_activations_input', 0):.4f}")
    print(f"  row_acts_weight (ILP): {mapping.metrics.get('row_activations_weight', 0):.4f}")
    print(f"  row_acts_output (ILP): {mapping.metrics.get('row_activations_output', 0):.4f}")
    
    # 打印 mapping 配置
    print("\n>>> Mapping Configuration:")
    for level in range(4):
        factors = {DIM_NAMES[d]: get_factor_at_level(mapping, level, d) for d in range(7)}
        non_one = {k: v for k, v in factors.items() if v > 1}
        print(f"  Level {level}: {non_one if non_one else '(all 1s)'}")
    
    # 使用新公式计算
    print("\n>>> Computing with NEW formula...")
    new_result = compute_input_row_acts_new_formula(mapping, workload)
    
    print(f"\n>>> New Formula Result:")
    print(f"  row_switches_sim = {new_result['row_switches_sim']}")
    
    # 运行 Trace
    print("\n>>> Running Trace Generator...")
    dram_config = DRAMConfig(row_buffer_bytes=1024, num_banks=4, num_rows=16384, element_size=1)
    gen = TraceGenerator(dram_config)
    trace = gen.generate_trace(mapping, workload)
    
    # 统计 Trace 的 row switches
    bank_size = dram_config.row_buffer_bytes * dram_config.num_rows
    row_size = dram_config.row_buffer_bytes
    
    input_accesses = []
    for line in trace:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        addr = int(parts[1], 16)
        bank = addr // bank_size
        if bank == 0:  # Input bank
            row = (addr % bank_size) // row_size
            input_accesses.append(row)
    
    prev_row = None
    trace_switches = 0
    for row in input_accesses:
        if prev_row is None or row != prev_row:
            trace_switches += 1
        prev_row = row
    
    print(f"\n>>> Trace Result:")
    print(f"  Input accesses: {len(input_accesses)}")
    print(f"  Input row switches: {trace_switches}")
    
    # 比较
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    ilp_input = mapping.metrics.get('row_activations_input', 0)
    print(f"  ILP (original):      {ilp_input:.0f}")
    print(f"  New Formula (sim):   {new_result['row_switches_sim']}")
    print(f"  Trace (actual):      {trace_switches}")
    if trace_switches > 0:
        print(f"\n  ILP error:           {abs(ilp_input - trace_switches) / trace_switches * 100:.1f}%")
        print(f"  New Formula error:   {abs(new_result['row_switches_sim'] - trace_switches) / trace_switches * 100:.1f}%")


if __name__ == "__main__":
    main()
