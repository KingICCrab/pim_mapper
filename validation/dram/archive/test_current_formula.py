#!/usr/bin/env python3
"""
测试当前 ILP 模型中 Input Row Activation 公式

公式 (Sequential 模式 - 4种情况):
E[acts] = (1-cr_b)(1-cr_r) × 1 + 2 × reuse × (cr_b + cr_r)

row_acts = rb_tiles / tiles_per_row / banks × E[acts]
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')

import math
from pim_optimizer.model.row_activation import (\n    compute_input_block_crossing_ratio,\n    compute_dram_row_crossing_ratio\n)


def test_sequential_formula():
    """测试 Sequential 模式的四种情况公式"""
    
    print("=" * 70)
    print("测试 Sequential 模式 Row Activation 公式")
    print("公式: E[acts] = (1-cr_b)(1-cr_r) + 2 × reuse × (cr_b + cr_r)")
    print("=" * 70)
    
    # 参数设置
    row_buffer_bytes = 1024  # 1KB DRAM row
    element_bytes = 2  # INT16
    banks = 16
    
    # 测试用例
    test_cases = [
        # (block_h, block_w, tile_h, tile_w, step_h, step_w, reuse, description)
        (8, 8, 4, 4, 1, 1, 1, "小 tile, 无 reuse"),
        (8, 8, 4, 4, 1, 1, 8, "小 tile, reuse=8"),
        (8, 8, 16, 16, 2, 2, 1, "大 tile (crossing), 无 reuse"),
        (8, 8, 16, 16, 2, 2, 8, "大 tile (crossing), reuse=8"),
        (4, 4, 6, 6, 2, 2, 4, "tile > block (both crossing)"),
        (32, 32, 8, 8, 1, 1, 16, "tile << block, 高 reuse"),
    ]
    
    print("\n" + "-" * 70)
    print(f"{'Case':<35} {'cr_b':>8} {'cr_r':>8} {'E[acts]':>10} {'备注'}")
    print("-" * 70)
    
    for block_h, block_w, tile_h, tile_w, step_h, step_w, reuse, desc in test_cases:
        # Block crossing (H and W)
        cr_b_h = compute_input_block_crossing_ratio(
            block_h=block_h, tile_h=tile_h, step=step_h,
            tiler_s=3, total_S=3, dilation=1
        )
        cr_b_w = compute_input_block_crossing_ratio(
            block_h=block_w, tile_h=tile_w, step=step_w,
            tiler_s=3, total_S=3, dilation=1
        )
        # Combined block crossing
        cr_b = 1.0 - (1.0 - cr_b_h) * (1.0 - cr_b_w)
        
        # Row crossing
        tile_bytes = tile_h * tile_w * element_bytes
        cr_r = compute_dram_row_crossing_ratio(tile_bytes, row_buffer_bytes)
        
        # 四种情况公式
        # E[acts] = (1-cr_b)(1-cr_r) + 2 × reuse × (cr_b + cr_r)
        e_acts = (1.0 - cr_b) * (1.0 - cr_r) + 2.0 * reuse * (cr_b + cr_r)
        
        note = ""
        if cr_b > 0 and cr_r > 0:
            note = "Both crossing"
        elif cr_b > 0:
            note = "Block CR only"
        elif cr_r > 0:
            note = "Row CR only"
        else:
            note = "No crossing"
        
        print(f"{desc:<35} {cr_b:>8.4f} {cr_r:>8.4f} {e_acts:>10.4f} {note}")
    
    print("-" * 70)
    
    # 验证简化公式
    print("\n" + "=" * 70)
    print("验证四种情况的数学推导")
    print("=" * 70)
    
    cr_b = 0.3
    cr_r = 0.2
    reuse = 4
    
    # 原始四种情况展开
    case_00 = (1 - cr_b) * (1 - cr_r) * 1           # 无 crossing
    case_10 = cr_b * (1 - cr_r) * 2 * reuse        # 只有 block crossing
    case_01 = (1 - cr_b) * cr_r * 2 * reuse        # 只有 row crossing
    case_11 = cr_b * cr_r * 4 * reuse              # 两种 crossing
    
    exact = case_00 + case_10 + case_01 + case_11
    
    # 简化公式
    simplified = (1 - cr_b) * (1 - cr_r) + 2 * reuse * (cr_b + cr_r)
    
    print(f"cr_b = {cr_b}, cr_r = {cr_r}, reuse = {reuse}")
    print()
    print("四种情况展开:")
    print(f"  Case (0,0): (1-{cr_b})(1-{cr_r}) × 1 = {case_00:.4f}")
    print(f"  Case (1,0): {cr_b}(1-{cr_r}) × 2 × {reuse} = {case_10:.4f}")
    print(f"  Case (0,1): (1-{cr_b}){cr_r} × 2 × {reuse} = {case_01:.4f}")
    print(f"  Case (1,1): {cr_b} × {cr_r} × 4 × {reuse} = {case_11:.4f}")
    print(f"  Total = {exact:.4f}")
    print()
    print(f"简化公式: (1-cr_b)(1-cr_r) + 2×reuse×(cr_b+cr_r) = {simplified:.4f}")
    print()
    
    if abs(exact - simplified) < 0.0001:
        print("✓ 公式验证通过！")
    else:
        print(f"✗ 公式不匹配！差异 = {exact - simplified:.6f}")
        print()
        # 推导正确的简化公式
        print("重新推导...")
        # case_10 + case_01 + case_11 
        # = cr_b(1-cr_r)×2×reuse + (1-cr_b)cr_r×2×reuse + cr_b×cr_r×4×reuse
        # = 2×reuse × [cr_b - cr_b×cr_r + cr_r - cr_b×cr_r + 2×cr_b×cr_r]
        # = 2×reuse × [cr_b + cr_r]
        crossing_part = 2 * reuse * (cr_b + cr_r)
        recalc = case_00 + crossing_part
        print(f"Crossing 部分 = 2 × {reuse} × ({cr_b} + {cr_r}) = {crossing_part:.4f}")
        print(f"重新计算总和 = {case_00:.4f} + {crossing_part:.4f} = {recalc:.4f}")
        print(f"与原始展开比较: {exact:.4f} vs {recalc:.4f}")


def test_with_optimizer():
    """使用 optimizer 测试实际值"""
    print("\n" + "=" * 70)
    print("使用 PIMOptimizer 测试实际 Row Activation")
    print("=" * 70)
    
    from pim_optimizer.arch.pim_arch import PIMArchitecture
    from pim_optimizer.workload.conv import ConvWorkload
    from pim_optimizer import PIMOptimizer
    
    test_workloads = [
        {"name": "small", "N": 1, "K": 8, "C": 8, "P": 4, "Q": 4, "R": 3, "S": 3},
        {"name": "medium", "N": 1, "K": 32, "C": 32, "P": 14, "Q": 14, "R": 3, "S": 3},
        {"name": "large_1x1", "N": 1, "K": 64, "C": 64, "P": 14, "Q": 14, "R": 1, "S": 1},
        {"name": "large_3x3", "N": 1, "K": 64, "C": 64, "P": 14, "Q": 14, "R": 3, "S": 3},
    ]
    
    arch = PIMArchitecture()
    
    print()
    print(f"{'Workload':<15} {'Sequential':>12} {'Row-aligned':>12} {'Selected':>12} {'Layout':<12}")
    print("-" * 65)
    
    for wl_params in test_workloads:
        workload = ConvWorkload(**wl_params)
        optimizer = PIMOptimizer(arch, verbose=False)
        result = optimizer.optimize([workload])
        model = optimizer.model
        
        seq = model.getVarByName('INPUT_ROW_ACT_SEQ_(0)')
        ra = model.getVarByName('INPUT_ROW_ACT_RA_(0)')
        total = model.getVarByName('INPUT_TOTAL_ROW_ACT_(0)')
        
        layout_seq = model.getVarByName('X_LAYOUT_(0,0,sequential)')
        layout = "Sequential" if layout_seq and layout_seq.X > 0.5 else "Row-aligned"
        
        seq_val = seq.X if seq else 0
        ra_val = ra.X if ra else 0
        total_val = total.X if total else 0
        
        print(f"{wl_params['name']:<15} {seq_val:>12.2f} {ra_val:>12.2f} {total_val:>12.2f} {layout:<12}")
    
    print("-" * 65)


if __name__ == "__main__":
    test_sequential_formula()
    test_with_optimizer()
