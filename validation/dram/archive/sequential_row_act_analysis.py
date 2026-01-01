"""
Sequential 模式下 Input Row Activation 推导

问题：Input 在 sequential 布局下，如何计算 row activation？
"""
import math

def row_activation_sequential(
    C_tile: int,      # Channel tile size
    TH: int,          # Input tile height (Q_tile + S - 1)
    TW: int,          # Input tile width (P_tile + R - 1)
    H_in: int,        # Total input height
    W_in: int,        # Total input width
    elems_per_row: int = 512,  # DRAM row size in elements
) -> tuple[int, str]:
    """
    Sequential 模式下一个 tile 访问触发的行激活数。
    
    Sequential 存储布局:
        addr(c, h, w) = c × H_in × W_in + h × W_in + w
    
    关键观察:
        - Channel 之间的地址跳跃 = H_in × W_in
        - 当 H_in × W_in >= elems_per_row 时，不同 channel 的数据在不同 DRAM row
        - 因此行激活数 ≈ C_tile × (一个 channel 内 tile 跨越的行数)
    
    Returns:
        (row_activations, formula_explanation)
    """
    channel_stride = H_in * W_in
    
    # 一个 channel 内，tile 访问的地址跨度
    # 从 (h=0, w=0) 到 (h=TH-1, w=TW-1)
    tile_span_in_channel = (TH - 1) * W_in + TW
    
    # 一个 channel 内，tile 跨越多少 DRAM row
    rows_per_channel = math.ceil(tile_span_in_channel / elems_per_row)
    
    if channel_stride >= elems_per_row:
        # 常见情况：不同 channel 的 tile 在不同的 DRAM row
        total_rows = C_tile * rows_per_channel
        formula = f"C_tile × rows_per_ch = {C_tile} × {rows_per_channel}"
    else:
        # 小 feature map，多个 channel 可能共享 DRAM row，需要精确计算
        accessed_rows = set()
        for c in range(C_tile):
            for h in range(TH):
                for w in range(TW):
                    addr = c * H_in * W_in + h * W_in + w
                    accessed_rows.add(addr // elems_per_row)
        total_rows = len(accessed_rows)
        formula = f"exact enumeration (small feature map)"
    
    return total_rows, formula


def row_activation_row_aligned(
    C_tile: int,
    TH: int,
    TW: int,
    block_H: int,     # Row-aligned block height
    block_W: int,     # Row-aligned block width
    step_H: int,      # Stride in H direction (Q_factor × stride)
    step_W: int,      # Stride in W direction (P_factor × stride)
    total_S: int = 3,
    total_R: int = 3,
    dilation: int = 1,
) -> tuple[float, str]:
    """
    Row-Aligned 模式下的行激活数（使用 crossing ratio）。
    
    Row-Aligned 存储布局:
        数据按 (block_H × W_in) 的块存储，每块占据连续的 DRAM row。
        当 tile 访问跨越 block 边界时，触发额外的行激活。
    
    公式:
        row_activations = mem_reads × (1 + crossing_ratio_H + crossing_ratio_W)
    """
    def compute_crossing_ratio(block, tile, step, total_kernel, dilation):
        if block <= 0 or tile <= 0:
            return 0.0
        if tile > block:
            return 1.0
        
        g = math.gcd(step, block)
        period = block // g
        crossing_count = 0
        for k in range(period):
            pos_mod = (k * step) % block
            if pos_mod + tile > block:
                crossing_count += 1
        return crossing_count / period
    
    cr_h = compute_crossing_ratio(block_H, TH, step_H, total_S, dilation)
    cr_w = compute_crossing_ratio(block_W, TW, step_W, total_R, dilation)
    
    # 简化的 mem_reads 估计（单位化）
    base = C_tile  # 每个 channel 至少 1 次读取
    total = base * (1 + cr_h + cr_w)
    
    formula = f"base × (1 + cr_H + cr_W) = {C_tile} × (1 + {cr_h:.2f} + {cr_w:.2f})"
    return total, formula


def main():
    print("=" * 75)
    print("Sequential vs Row-Aligned 模式下 Input Row Activation 对比")
    print("=" * 75)
    
    # =========================================================================
    # 场景 1: 我们的测试 workload (小)
    # =========================================================================
    print("\n【场景 1】小 workload: N=1, K=8, C=8, P=4, Q=4, R=3, S=3")
    print("          Input: (1, 8, 6, 6), 共 288 元素, DRAM row = 512 元素")
    print()
    
    H_in, W_in, C_total = 6, 6, 8
    C_tile, TH, TW = 8, 6, 6  # 整个 input 作为一个 tile
    
    seq_rows, seq_formula = row_activation_sequential(C_tile, TH, TW, H_in, W_in)
    ra_rows, ra_formula = row_activation_row_aligned(
        C_tile, TH, TW, 
        block_H=6, block_W=6,  # ILP 选择的 block = 6
        step_H=1, step_W=1
    )
    
    print(f"  Sequential:   {seq_rows:>6.1f} rows  ({seq_formula})")
    print(f"  Row-Aligned:  {ra_rows:>6.1f} rows  ({ra_formula})")
    print(f"  结论: 此场景下 Sequential 更优")
    
    # =========================================================================
    # 场景 2: 大 workload
    # =========================================================================
    print("\n" + "=" * 75)
    print("【场景 2】大 workload: H_in=32, W_in=32, C=64")
    print("          DRAM row = 512 元素, block_H = 32")
    print()
    print(f"{'C_tile':>6} {'TH':>4} | {'Sequential':>12} {'Row-Aligned':>12} | 结论")
    print("-" * 75)
    
    H_in, W_in = 32, 32
    
    for C_tile in [8, 16, 32, 64]:
        for TH in [4, 8, 16, 32]:
            TW = TH
            
            seq_rows, _ = row_activation_sequential(C_tile, TH, TW, H_in, W_in)
            ra_rows, _ = row_activation_row_aligned(
                C_tile, TH, TW,
                block_H=32, block_W=32,
                step_H=1, step_W=1
            )
            
            if seq_rows < ra_rows:
                conclusion = "Sequential 更好"
            elif ra_rows < seq_rows:
                conclusion = "Row-Aligned 更好"
            else:
                conclusion = "相同"
            
            print(f"{C_tile:>6} {TH:>4} | {seq_rows:>12.1f} {ra_rows:>12.1f} | {conclusion}")
    
    # =========================================================================
    # 总结
    # =========================================================================
    print("\n" + "=" * 75)
    print("【推导总结】")
    print("=" * 75)
    print("""
1. Sequential 模式 Row Activation 公式:
   ────────────────────────────────────
   当 H_in × W_in >= elems_per_row (常见情况):
   
       rows_per_tile = C_tile × ⌈((TH-1) × W_in + TW) / elems_per_row⌉
   
   简化（当 tile 在单个 channel 内 fit in 1 row）:
       rows_per_tile ≈ C_tile

2. Row-Aligned 模式 Row Activation 公式:
   ────────────────────────────────────
       rows_per_tile = base × (1 + crossing_ratio_H + crossing_ratio_W)
   
   其中 crossing_ratio 由 (block, tile, step) 关系决定。

3. 两种模式的权衡:
   ────────────────────────────────────
   Sequential 更好的情况:
     - C_tile 较小
     - block 选择不当导致高 crossing ratio
     - Input feature map 很小（整个 fit in 1-2 rows）
   
   Row-Aligned 更好的情况:
     - C_tile 较大
     - block 与 step 对齐良好（低 crossing ratio）
     - Input feature map 较大

4. ILP 模型的问题:
   ────────────────────────────────────
   当前模型只实现了 Row-Aligned 的 crossing ratio 计算，
   但 layout_choice 变量可能选择 Sequential，导致不一致。
   
   修复方案:
   (a) 为 Sequential 模式添加单独的 row activation 计算
   (b) 或者强制 Input 使用 Row-Aligned 布局
""")


if __name__ == "__main__":
    main()
