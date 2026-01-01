#!/usr/bin/env python3
"""
深入分析: R、S 在 L2 层时是否影响 Row Activation

用户问题:
  即使 R_l3=1, S_l3=1 (R、S 在 L2 层)，
  如果 block_h 和 block_w 比较小，R、S 的遍历顺序应该会影响 row activation

分析:
  - 一个 L3 Input tile 的大小是 C_buffer × (P_buffer + R - 1) × (Q_buffer + S - 1)
  - 这个 tile 需要多次 row activation 来加载
  - 在 L2 层内，R、S 的遍历顺序决定了访问 Input 的地址序列
  - 不同的地址序列可能有不同的 row activation 模式
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

import math
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class DRAMConfig:
    """DRAM 配置"""
    block_h: int = 4      # 每个 DRAM row 有多少个 "高度"
    block_w: int = 256    # 每个 DRAM row 有多少个元素
    row_size: int = 1024  # DRAM row 大小 (元素数)


@dataclass
class Workload:
    K: int = 64
    C: int = 64
    P: int = 14
    Q: int = 14
    R: int = 3
    S: int = 3
    
    @property
    def H_in(self):
        return self.P + self.R - 1
    
    @property
    def W_in(self):
        return self.Q + self.S - 1


def compute_address(c: int, h: int, w: int, workload: Workload) -> int:
    """
    计算 Input[c][h][w] 的 DRAM 地址
    假设 Input 按 [C][H_in][W_in] 存储 (C-major)
    """
    return c * workload.H_in * workload.W_in + h * workload.W_in + w


def get_dram_row(address: int, dram: DRAMConfig) -> int:
    """获取地址所在的 DRAM row"""
    return address // dram.row_size


def simulate_l2_access(workload: Workload, 
                        p_tile: int, q_tile: int, c_tile: int,
                        P_buffer: int, Q_buffer: int, C_buffer: int,
                        l2_order: str,  # 'RS' or 'SR' or 'RSC' etc.
                        dram: DRAMConfig) -> dict:
    """
    模拟 L2 层访问 Input 的 row activation
    
    在一个 L3 tile 内部，L2 层按照 l2_order 遍历 (R, S, ...)
    """
    # 这个 L3 tile 覆盖的 Input 范围
    c_start = c_tile * C_buffer
    h_start = p_tile * P_buffer  # 对应 P 的起始位置
    w_start = q_tile * Q_buffer  # 对应 Q 的起始位置
    
    # Input tile 实际覆盖 [h_start : h_start + P_buffer + R - 1]
    h_tile_size = P_buffer + workload.R - 1
    w_tile_size = Q_buffer + workload.S - 1
    
    # 根据 L2 order 生成访问序列
    accesses = []
    
    if l2_order == 'RSC':
        # 先 R，再 S，再 C (within buffer)
        for r in range(workload.R):
            for s in range(workload.S):
                for c in range(C_buffer):
                    for p in range(P_buffer):
                        for q in range(Q_buffer):
                            h = h_start + p + r
                            w = w_start + q + s
                            c_abs = c_start + c
                            accesses.append((c_abs, h, w))
    
    elif l2_order == 'SRC':
        # 先 S，再 R，再 C
        for s in range(workload.S):
            for r in range(workload.R):
                for c in range(C_buffer):
                    for p in range(P_buffer):
                        for q in range(Q_buffer):
                            h = h_start + p + r
                            w = w_start + q + s
                            c_abs = c_start + c
                            accesses.append((c_abs, h, w))
    
    elif l2_order == 'CRS':
        # 先 C，再 R，再 S
        for c in range(C_buffer):
            for r in range(workload.R):
                for s in range(workload.S):
                    for p in range(P_buffer):
                        for q in range(Q_buffer):
                            h = h_start + p + r
                            w = w_start + q + s
                            c_abs = c_start + c
                            accesses.append((c_abs, h, w))
    
    elif l2_order == 'tile_sequential':
        # 顺序访问整个 tile (最优)
        for c in range(C_buffer):
            for h in range(h_tile_size):
                for w in range(w_tile_size):
                    c_abs = c_start + c
                    h_abs = h_start + h
                    w_abs = w_start + w
                    accesses.append((c_abs, h_abs, w_abs))
    
    # 计算 row activation
    row_activations = 0
    last_row = None
    unique_rows = set()
    
    for c, h, w in accesses:
        addr = compute_address(c, h, w, workload)
        row = get_dram_row(addr, dram)
        unique_rows.add(row)
        
        if row != last_row:
            row_activations += 1
            last_row = row
    
    return {
        'total_accesses': len(accesses),
        'row_activations': row_activations,
        'unique_rows': len(unique_rows),
        'l2_order': l2_order,
    }


def main():
    print("="*70)
    print("L2 层 R、S 遍历顺序对 Row Activation 的影响")
    print("="*70)
    
    workload = Workload(K=64, C=16, P=14, Q=14, R=3, S=3)
    
    print(f"\nWorkload: C={workload.C}, P={workload.P}, Q={workload.Q}, R={workload.R}, S={workload.S}")
    print(f"H_in={workload.H_in}, W_in={workload.W_in}")
    
    # L3 tiling
    P_l3, Q_l3, C_l3 = 2, 2, 2
    P_buffer = workload.P // P_l3  # 7
    Q_buffer = workload.Q // Q_l3  # 7
    C_buffer = workload.C // C_l3  # 8
    
    print(f"\nL3 Tiling: P_l3={P_l3}, Q_l3={Q_l3}, C_l3={C_l3}")
    print(f"Buffer sizes: P_buffer={P_buffer}, Q_buffer={Q_buffer}, C_buffer={C_buffer}")
    
    # Input tile 大小
    h_tile = P_buffer + workload.R - 1
    w_tile = Q_buffer + workload.S - 1
    tile_size = C_buffer * h_tile * w_tile
    print(f"Input tile: {C_buffer} × {h_tile} × {w_tile} = {tile_size} elements")
    
    # 测试不同的 DRAM 配置
    dram_configs = [
        DRAMConfig(row_size=1024),
        DRAMConfig(row_size=512),
        DRAMConfig(row_size=256),
    ]
    
    l2_orders = ['RSC', 'SRC', 'CRS', 'tile_sequential']
    
    for dram in dram_configs:
        print(f"\n{'='*60}")
        print(f"DRAM row_size = {dram.row_size}")
        print(f"{'='*60}")
        
        print(f"\n{'L2 Order':<20} {'Row Acts':<12} {'Unique Rows':<12}")
        print("-"*50)
        
        for order in l2_orders:
            result = simulate_l2_access(
                workload, 
                p_tile=0, q_tile=0, c_tile=0,
                P_buffer=P_buffer, Q_buffer=Q_buffer, C_buffer=C_buffer,
                l2_order=order,
                dram=dram
            )
            print(f"{order:<20} {result['row_activations']:<12} {result['unique_rows']:<12}")
    
    # 更细致的分析
    print("\n" + "="*70)
    print("详细分析: 小 row_size 场景")
    print("="*70)
    
    dram = DRAMConfig(row_size=128)  # 很小的 row size
    
    print(f"\nDRAM row_size = {dram.row_size}")
    print(f"每个 DRAM row 可存储 {dram.row_size} 个元素")
    
    for order in l2_orders:
        result = simulate_l2_access(
            workload,
            p_tile=0, q_tile=0, c_tile=0,
            P_buffer=P_buffer, Q_buffer=Q_buffer, C_buffer=C_buffer,
            l2_order=order,
            dram=dram
        )
        
        print(f"\n{order}:")
        print(f"  总访问: {result['total_accesses']}")
        print(f"  唯一 rows: {result['unique_rows']}")
        print(f"  Row activations: {result['row_activations']}")
        print(f"  平均每 row 访问次数: {result['total_accesses'] / result['row_activations']:.1f}")
    
    # 结论
    print("\n" + "="*70)
    print("结论")
    print("="*70)
    print("""
关键发现:

1. R、S 的遍历顺序确实会影响 Row Activation!
   - 即使 R_l3=1, S_l3=1 (R、S 在 L2 层)
   - L2 层内部的访问顺序决定了 DRAM 访问模式

2. 影响程度取决于:
   - DRAM row size (block_w)
   - Input tile 的形状
   - 数据在 DRAM 中的布局

3. 最优访问模式:
   - 顺序访问整个 tile (tile_sequential)
   - 或者按照 DRAM 行的方向遍历

4. 之前的公式需要修正:
   - 不能简单假设 R、S 在 L2 层就不影响 row activation
   - 需要考虑 L2 层的遍历顺序
""")


if __name__ == '__main__':
    main()
