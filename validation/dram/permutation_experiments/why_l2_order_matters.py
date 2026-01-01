#!/usr/bin/env python3
"""
完整分析: L2 层遍历顺序对 Row Activation 的影响

关键发现:
1. RSC vs CRS 差异巨大 (96 vs 48 row activations)
2. tile_sequential 最优 (16 row activations = unique rows)
3. 问题在于 R、S 循环在外层时，每次迭代都会跳跃访问

原因分析:
- Input 布局: [C][H][W] (假设 C-major)
- 地址计算: addr = c * H * W + h * W + w
- 连续访问 W 方向时，地址连续
- 但 R 循环时 h 变化，导致地址跳跃 W 个位置
- 如果跳跃跨越 DRAM row，就产生额外的 row activation
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Workload:
    C: int = 16
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


def visualize_access_pattern(workload: Workload, 
                              c_buf: int, h_tile: int, w_tile: int,
                              l2_order: str,
                              row_size: int) -> dict:
    """可视化访问模式"""
    
    # 生成访问序列
    accesses = []
    
    if l2_order == 'RSC':
        for r in range(workload.R):
            for s in range(workload.S):
                for c in range(c_buf):
                    # 假设 P_buffer 和 Q_buffer 的循环在最内层
                    for p in range(h_tile - workload.R + 1):
                        for q in range(w_tile - workload.S + 1):
                            h = p + r
                            w = q + s
                            accesses.append((c, h, w))
    
    elif l2_order == 'CRS':
        for c in range(c_buf):
            for r in range(workload.R):
                for s in range(workload.S):
                    for p in range(h_tile - workload.R + 1):
                        for q in range(w_tile - workload.S + 1):
                            h = p + r
                            w = q + s
                            accesses.append((c, h, w))
    
    elif l2_order == 'sequential':
        # 顺序访问 tile
        for c in range(c_buf):
            for h in range(h_tile):
                for w in range(w_tile):
                    accesses.append((c, h, w))
    
    # 计算地址和 row
    H = workload.H_in
    W = workload.W_in
    
    rows_accessed = []
    for c, h, w in accesses:
        addr = c * H * W + h * W + w
        row = addr // row_size
        rows_accessed.append(row)
    
    # 统计 row switches
    switches = 0
    last_row = None
    for row in rows_accessed:
        if row != last_row:
            switches += 1
            last_row = row
    
    return {
        'accesses': len(accesses),
        'switches': switches,
        'unique_rows': len(set(rows_accessed)),
    }


def analyze_why_order_matters():
    """分析为什么顺序很重要"""
    
    print("="*70)
    print("为什么 L2 层的循环顺序影响 Row Activation?")
    print("="*70)
    
    # 简单例子
    print("\n简化例子:")
    print("  Input tile: 2 × 4 × 4 (C × H × W)")
    print("  DRAM row_size: 8 elements")
    print("  R=2, S=2")
    
    # 数据布局
    print("\n数据在 DRAM 中的布局 (地址 = c*H*W + h*W + w):")
    print("  C=0: [h=0,w=0-3][h=1,w=0-3][h=2,w=0-3][h=3,w=0-3]")
    print("        addr 0-3   addr 4-7   addr 8-11  addr 12-15")
    print("        row 0      row 0      row 1      row 1")
    print("")
    print("  C=1: [h=0,w=0-3][h=1,w=0-3][h=2,w=0-3][h=3,w=0-3]")
    print("        addr 16-19 addr 20-23 addr 24-27 addr 28-31")
    print("        row 2      row 2      row 3      row 3")
    
    print("\n场景 1: RSC (R→S→C 顺序)")
    print("  r=0,s=0: 访问 (c,h=p+0,w=q+0) for c,p,q")
    print("  r=0,s=1: 访问 (c,h=p+0,w=q+1) for c,p,q")
    print("  r=1,s=0: 访问 (c,h=p+1,w=q+0) for c,p,q  ← h 变化!")
    print("  r=1,s=1: 访问 (c,h=p+1,w=q+1) for c,p,q")
    print("")
    print("  当 r 从 0 变到 1，h 增加 1")
    print("  地址跳跃 W=4，可能跨越 DRAM row!")
    
    print("\n场景 2: CRS (C→R→S 顺序)")
    print("  c=0: 遍历完整个 c=0 的所有 (r,s) 组合")
    print("  c=1: 遍历完整个 c=1 的所有 (r,s) 组合")
    print("")
    print("  每个 c 的数据在 DRAM 中是连续的")
    print("  遍历完一个 c 后才切换到下一个 c")
    print("  Row switches 更少!")
    
    print("\n场景 3: sequential (顺序访问 tile)")
    print("  按照 c → h → w 顺序访问")
    print("  地址完全连续，Row switches = unique rows")


def test_different_data_layouts():
    """测试不同数据布局的影响"""
    
    print("\n" + "="*70)
    print("数据布局的影响")
    print("="*70)
    
    # C-major: addr = c * H * W + h * W + w
    # H-major: addr = h * W * C + w * C + c
    # W-major: addr = w * H * C + h * C + c
    
    layouts = {
        'CHW': lambda c, h, w, H, W, C: c * H * W + h * W + w,
        'HWC': lambda c, h, w, H, W, C: h * W * C + w * C + c,
        'WHC': lambda c, h, w, H, W, C: w * H * C + h * C + c,
    }
    
    # 简单 tile
    c_buf, h_tile, w_tile = 4, 6, 6
    R, S = 3, 3
    H, W, C = 16, 16, 16
    row_size = 128
    
    print(f"\nTile: C_buf={c_buf}, H_tile={h_tile}, W_tile={w_tile}")
    print(f"R={R}, S={S}")
    print(f"row_size={row_size}")
    
    for layout_name, addr_func in layouts.items():
        print(f"\n--- 数据布局: {layout_name} ---")
        
        # 测试不同的遍历顺序
        for order in ['RSC', 'CRS', 'SRC']:
            accesses = []
            
            if order == 'RSC':
                for r in range(R):
                    for s in range(S):
                        for c in range(c_buf):
                            for p in range(h_tile - R + 1):
                                for q in range(w_tile - S + 1):
                                    accesses.append((c, p+r, q+s))
            elif order == 'CRS':
                for c in range(c_buf):
                    for r in range(R):
                        for s in range(S):
                            for p in range(h_tile - R + 1):
                                for q in range(w_tile - S + 1):
                                    accesses.append((c, p+r, q+s))
            elif order == 'SRC':
                for s in range(S):
                    for r in range(R):
                        for c in range(c_buf):
                            for p in range(h_tile - R + 1):
                                for q in range(w_tile - S + 1):
                                    accesses.append((c, p+r, q+s))
            
            # 计算 row switches
            switches = 0
            last_row = None
            for c, h, w in accesses:
                addr = addr_func(c, h, w, H, W, C)
                row = addr // row_size
                if row != last_row:
                    switches += 1
                    last_row = row
            
            print(f"  {order}: {switches} row switches")


def main():
    analyze_why_order_matters()
    test_different_data_layouts()
    
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print("""
1. L2 层的 R、S 遍历顺序确实影响 Row Activation!

2. 原因:
   - 不同的遍历顺序产生不同的地址访问序列
   - 地址跳跃可能跨越 DRAM row，产生额外 row activation
   
3. 影响因素:
   a) 数据布局 (CHW vs HWC vs WHC)
   b) L2 循环顺序 (RSC vs CRS vs SRC)
   c) DRAM row_size (block_w)
   d) Input tile 形状

4. 最优策略:
   - 让访问顺序匹配数据布局的连续方向
   - 例如 CHW 布局时，C 在最外层循环最优

5. 对 ILP 模型的影响:
   - 需要将 L2 层的循环顺序也作为决策变量
   - 或者在代价模型中考虑 L2 层的影响
""")


if __name__ == '__main__':
    main()
