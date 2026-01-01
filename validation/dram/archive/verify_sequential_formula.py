"""
验证 Input Sequential 模式 Row Activation 公式

新公式（简化后）：
  Sequential crossing factor = 1 + row_crossing
  
其中：
  - row_crossing: tile 数据跨越 DRAM row 边界的 crossing ratio
  
关键理解：
  - Sequential 布局下，数据是地址连续的
  - 访问一个 tile 时，主要的 row activation 来源是跨越 DRAM row 边界
  - block_crossing 在 Sequential 模式下不增加 row activation，因为块内数据是连续的
"""
import math
import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')

from pim_optimizer.model.row_activation import (
    compute_input_sequential_row_activation,
    compute_input_block_crossing_ratio,
    compute_dram_row_crossing_ratio
)

def verify_formula():
    print("=" * 70)
    print("验证 Input Sequential 模式 Row Activation 公式")
    print("=" * 70)
    
    # Test case 1: 小 tile，不跨 DRAM row
    print("\n【测试 1】小 tile: tile_bytes < row_bytes")
    print("-" * 70)
    
    row_bytes = 1024
    element_bytes = 1
    
    # tile = 6x6 = 36 bytes << 1024
    tile_h, tile_w = 6, 6
    tile_bytes = tile_h * tile_w * element_bytes
    row_crossing = compute_dram_row_crossing_ratio(tile_bytes, row_bytes)
    expected_factor = 1.0 + row_crossing
    
    func_factor = compute_input_sequential_row_activation(
        block_h=6, block_w=6,
        tile_h=tile_h, tile_w=tile_w,
        step_h=1, step_w=1,
        row_bytes=row_bytes, element_bytes=element_bytes,
        total_S=3, total_R=3, dilation_h=1, dilation_w=1
    )
    
    print(f"  tile_bytes = {tile_bytes} bytes")
    print(f"  row_bytes = {row_bytes} bytes")
    print(f"  row_crossing = {row_crossing:.4f}")
    print(f"  Expected: 1 + {row_crossing:.4f} = {expected_factor:.4f}")
    print(f"  函数返回: {func_factor:.4f}")
    print(f"  匹配: {'✓' if abs(expected_factor - func_factor) < 0.001 else '✗'}")
    
    # Test case 2: 大 tile，跨 DRAM row
    print("\n【测试 2】大 tile: tile_bytes > row_bytes")
    print("-" * 70)
    
    # tile = 64x64 = 4096 bytes > 1024
    tile_h, tile_w = 64, 64
    tile_bytes = tile_h * tile_w * element_bytes
    row_crossing = compute_dram_row_crossing_ratio(tile_bytes, row_bytes)
    expected_factor = 1.0 + row_crossing
    
    func_factor = compute_input_sequential_row_activation(
        block_h=32, block_w=32,
        tile_h=tile_h, tile_w=tile_w,
        step_h=1, step_w=1,
        row_bytes=row_bytes, element_bytes=element_bytes,
        total_S=3, total_R=3, dilation_h=1, dilation_w=1
    )
    
    print(f"  tile_bytes = {tile_bytes} bytes")
    print(f"  row_bytes = {row_bytes} bytes")
    print(f"  row_crossing = {row_crossing:.4f}")
    print(f"  Expected: 1 + {row_crossing:.4f} = {expected_factor:.4f}")
    print(f"  函数返回: {func_factor:.4f}")
    print(f"  匹配: {'✓' if abs(expected_factor - func_factor) < 0.001 else '✗'}")
    
    # Test case 3: 对比 Sequential vs Row_aligned
    print("\n【测试 3】对比 Sequential vs Row_aligned")
    print("-" * 70)
    print("  Sequential 因为数据连续，通常有更好的 row buffer locality")
    print()
    
    for tile_h, tile_w in [(6, 6), (16, 16), (32, 32)]:
        block_h, block_w = 6, 6
        
        # Row_aligned: block crossing 为主
        block_cr_h = compute_input_block_crossing_ratio(block_h, tile_h, 1, 3, 3, 1)
        block_cr_w = compute_input_block_crossing_ratio(block_w, tile_w, 1, 3, 3, 1)
        ra_factor = 1 + block_cr_h + block_cr_w
        
        # Sequential: row crossing 为主
        seq_factor = compute_input_sequential_row_activation(
            block_h=block_h, block_w=block_w,
            tile_h=tile_h, tile_w=tile_w,
            step_h=1, step_w=1,
            row_bytes=row_bytes, element_bytes=element_bytes,
            total_S=3, total_R=3, dilation_h=1, dilation_w=1
        )
        
        tile_bytes = tile_h * tile_w * element_bytes
        print(f"  tile={tile_h}×{tile_w} ({tile_bytes}B): Row_aligned={ra_factor:.3f}, Sequential={seq_factor:.3f}")
    
    # Test case 4: 验证 Sequential <= Row_aligned (应该成立，因为更好的 locality)
    print("\n【测试 4】验证 Sequential 通常 <= Row_aligned (更好的 locality)")
    print("-" * 70)
    
    fail_count = 0
    for block_h in [4, 8, 16, 32]:
        for tile_h in [2, 4, 8]:
            for step in [1, 2, 4]:
                block_cr_h = compute_input_block_crossing_ratio(block_h, tile_h, step, 3, 3, 1)
                ra_factor = 1 + block_cr_h * 2  # 假设 h 和 w 一样
                
                seq_factor = compute_input_sequential_row_activation(
                    block_h=block_h, block_w=block_h,
                    tile_h=tile_h, tile_w=tile_h,
                    step_h=step, step_w=step,
                    row_bytes=row_bytes, element_bytes=element_bytes,
                    total_S=3, total_R=3, dilation_h=1, dilation_w=1
                )
                
                if seq_factor > ra_factor + 0.001:
                    print(f"  Note: block={block_h}, tile={tile_h}, step={step}: seq={seq_factor:.3f} > ra={ra_factor:.3f}")
                    fail_count += 1
    
    if fail_count == 0:
        print("  ✓ 所有测试: Sequential <= Row_aligned")
    else:
        print(f"  有 {fail_count} 个例外情况（大 tile 可能 Sequential 稍大）")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    verify_formula()
