"""
精确验证 Trace 的 5376 是如何计算出来的

已知:
- Trace (修复后): 5376
- 我的分析: 4800

差异: 576 = 5376 - 4800

可能原因:
1. 我的 H_per_tile 计算错误?
2. Trace 有额外的 row switch 逻辑?
3. C 循环也会触发 row switch?
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload

# Dimension indices
DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6

def simulate_trace():
    """精确模拟 Trace 的 row switch 计算."""
    
    workload = ConvWorkload(name="ResNet-L1", R=7, S=7, P=56, Q=56, C=3, K=64, N=1)
    optimizer = PIMOptimizer()
    result = optimizer.optimize([workload], objective="latency")
    mapping = result.mappings[0]
    
    # Extract all factors
    dram_factors = {d: 1 for d in range(7)}
    if 3 in mapping.loop_bounds:
        for key in ['spatial', 'temporal']:
            if key in mapping.loop_bounds[3]:
                for d, bound in mapping.loop_bounds[3][key].items():
                    dram_factors[d] *= bound
    
    level2_factors = {d: 1 for d in range(7)}
    if 2 in mapping.loop_bounds:
        for key in ['spatial', 'temporal']:
            if key in mapping.loop_bounds[2]:
                for d, bound in mapping.loop_bounds[2][key].items():
                    level2_factors[d] *= bound
    
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
    
    print("=" * 80)
    print("精确模拟 Trace Row Switches")
    print("=" * 80)
    
    print(f"\n配置:")
    print(f"  DRAM L3: K={dram_factors[DIM_K]}, C={dram_factors[DIM_C]}, "
          f"P={dram_factors[DIM_P]}, Q={dram_factors[DIM_Q]}")
    print(f"  Level 2: R={level2_factors[DIM_R]}")
    print(f"  Buffer:  P={buffer_tile[DIM_P]}, Q={buffer_tile[DIM_Q]}, "
          f"R={buffer_tile[DIM_R]}, S={buffer_tile[DIM_S]}")
    
    K_l3 = dram_factors[DIM_K]
    C_l3 = dram_factors[DIM_C]
    P_l3 = dram_factors[DIM_P]
    Q_l3 = dram_factors[DIM_Q]
    R_l2 = level2_factors[DIM_R]
    
    P_per_tile = buffer_tile[DIM_P]  # 2
    Q_per_tile = buffer_tile[DIM_Q]  # 8
    R_per_tile = buffer_tile[DIM_R]  # 1
    S_per_tile = buffer_tile[DIM_S]  # 7
    
    stride_h, stride_w = 1, 1
    dilation_h, dilation_w = 1, 1
    
    # 关键: H_per_tile 的计算
    H_per_tile = (P_per_tile - 1) * stride_h + R_per_tile * dilation_h  # (2-1)*1 + 1*1 = 2
    W_per_tile = (Q_per_tile - 1) * stride_w + S_per_tile * dilation_w  # (8-1)*1 + 7*1 = 14
    
    print(f"\n  H_per_tile = (P-1)*stride + R*dilation = ({P_per_tile}-1)*{stride_h} + {R_per_tile}*{dilation_h} = {H_per_tile}")
    print(f"  W_per_tile = (Q-1)*stride + S*dilation = ({Q_per_tile}-1)*{stride_w} + {S_per_tile}*{dilation_w} = {W_per_tile}")
    
    block_h = mapping.tile_info.get('block_h', 31)
    block_w = mapping.tile_info.get('block_w', 31)
    
    print(f"  Block: {block_h} × {block_w}")
    
    # 模拟 Trace 的循环
    print(f"\n" + "-" * 60)
    print("方法 1: 模拟 Trace 循环 (跟踪 prev_row)")
    print("-" * 60)
    
    prev_row = -1
    total_switches = 0
    row_activations = {}  # row -> count
    
    # Trace 循环顺序: K -> C -> P -> Q -> R (最内层)
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r in range(R_l2):
                        # 计算 h_start, w_start
                        h_start = p * P_per_tile * stride_h + r * R_per_tile * dilation_h
                        w_start = q * Q_per_tile * stride_w
                        # 修复后的计算应该包含 r 的滑动
                        
                        h_end = h_start + H_per_tile - 1
                        w_end = w_start + W_per_tile - 1
                        
                        # 计算访问的 blocks
                        h_blocks = list(range(h_start // block_h, (h_end // block_h) + 1))
                        w_blocks = list(range(w_start // block_w, (w_end // block_w) + 1))
                        
                        # 计算所有涉及的 rows (in row_aligned mode)
                        # row = (c * num_h_blocks + h_block) * num_w_blocks + w_block
                        num_h_blocks = 2  # ceil(62/31)
                        num_w_blocks = 2  # ceil(62/31)
                        
                        rows = set()
                        for hb in h_blocks:
                            for wb in w_blocks:
                                row = (c * num_h_blocks + hb) * num_w_blocks + wb
                                rows.add(row)
                        
                        # Count switches
                        for row in sorted(rows):
                            if row != prev_row:
                                total_switches += 1
                                prev_row = row
                            row_activations[row] = row_activations.get(row, 0) + 1
    
    print(f"  Total switches (with prev_row tracking): {total_switches}")
    print(f"  Row activation distribution:")
    for row in sorted(row_activations.keys()):
        print(f"    Row {row}: {row_activations[row]} times")
    
    # 方法 2: 每个 tile 独立计算 rows accessed
    print(f"\n" + "-" * 60)
    print("方法 2: 每个 tile 独立计算 rows (不跟踪 prev)")
    print("-" * 60)
    
    total_rows_accessed = 0
    
    for k in range(K_l3):
        for c in range(C_l3):
            for p in range(P_l3):
                for q in range(Q_l3):
                    for r in range(R_l2):
                        h_start = p * P_per_tile * stride_h + r * R_per_tile * dilation_h
                        w_start = q * Q_per_tile * stride_w
                        
                        h_end = h_start + H_per_tile - 1
                        w_end = w_start + W_per_tile - 1
                        
                        h_blocks = list(range(h_start // block_h, (h_end // block_h) + 1))
                        w_blocks = list(range(w_start // block_w, (w_end // block_w) + 1))
                        
                        num_rows = len(h_blocks) * len(w_blocks)
                        total_rows_accessed += num_rows
    
    print(f"  Total rows accessed (per tile): {total_rows_accessed}")
    
    # 去除 K 的影响 (Input 对 K reuse)
    without_k = total_rows_accessed // K_l3
    print(f"  Without K factor: {without_k}")
    
    # 方法 3: 检查 C 之间的切换
    print(f"\n" + "-" * 60)
    print("方法 3: 分析 C 循环的影响")
    print("-" * 60)
    
    # C 在外层循环，每次 C 改变都是新的 row
    # 所以 C 之间的切换也要计入
    
    total_c_aware = 0
    prev_row = -1
    
    for c in range(C_l3):
        for p in range(P_l3):
            for q in range(Q_l3):
                for r in range(R_l2):
                    h_start = p * P_per_tile * stride_h + r * R_per_tile * dilation_h
                    w_start = q * Q_per_tile * stride_w
                    
                    h_end = h_start + H_per_tile - 1
                    w_end = w_start + W_per_tile - 1
                    
                    h_blocks = list(range(h_start // block_h, (h_end // block_h) + 1))
                    w_blocks = list(range(w_start // block_w, (w_end // block_w) + 1))
                    
                    num_h_blocks = 2
                    num_w_blocks = 2
                    
                    rows = set()
                    for hb in h_blocks:
                        for wb in w_blocks:
                            row = (c * num_h_blocks + hb) * num_w_blocks + wb
                            rows.add(row)
                    
                    for row in sorted(rows):
                        if row != prev_row:
                            total_c_aware += 1
                            prev_row = row
    
    print(f"  C-aware switches (no K): {total_c_aware}")
    
    # 方法 4: 简单计算，不跟踪 prev
    print(f"\n" + "-" * 60)
    print("方法 4: C × P × Q × R × blocks_per_tile")
    print("-" * 60)
    
    total_simple = 0
    
    for c in range(C_l3):
        for p in range(P_l3):
            for q in range(Q_l3):
                for r in range(R_l2):
                    h_start = p * P_per_tile * stride_h + r * R_per_tile * dilation_h
                    w_start = q * Q_per_tile * stride_w
                    
                    h_end = h_start + H_per_tile - 1
                    w_end = w_start + W_per_tile - 1
                    
                    h_blocks = list(range(h_start // block_h, (h_end // block_h) + 1))
                    w_blocks = list(range(w_start // block_w, (w_end // block_w) + 1))
                    
                    total_simple += len(h_blocks) * len(w_blocks)
    
    print(f"  Simple total (no K): {total_simple}")
    
    print(f"\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    print(f"""
    方法 1 (模拟 prev_row tracking): {total_switches}
    方法 2 (per tile, with K):       {total_rows_accessed}
    方法 2 (per tile, without K):    {without_k}
    方法 3 (C-aware, no K):          {total_c_aware}
    方法 4 (simple, no K):           {total_simple}
    
    Trace 实际结果: 5376
    """)
    
    # 检查每个 row 被访问的次数
    print(f"\n" + "-" * 60)
    print("验证: 每个 row 的访问次数应该加起来等于?")
    print("-" * 60)
    
    print(f"  Sum of row activations: {sum(row_activations.values())}")
    
    # 真正的问题: Trace 计算的是什么?
    # 看看 5376 = 12 rows × 448 = 5376!
    print(f"\n  12 unique rows × 448 = {12 * 448}")
    print(f"  Trace 可能计算的是 total visits, 不是 switches!")


if __name__ == "__main__":
    simulate_trace()
