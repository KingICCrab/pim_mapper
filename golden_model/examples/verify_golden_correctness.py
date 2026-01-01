#!/usr/bin/env python3
"""
手工验证 Golden Model 公式的正确性。

通过小规模例子，手工推导并验证公式。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from golden_model.analytical.cost_formulas import (
    LoopBounds, TileFactors,
    compute_input_access_count,
    compute_weight_access_count,
    compute_output_access_count,
    compute_input_tile_size,
    compute_weight_tile_size,
    compute_output_tile_size,
)


def manual_verification_1():
    """
    手工验证 Case 1: 最简单的 1×1 卷积
    
    Workload: N=2, C=2, K=2, P=2, Q=2, R=1, S=1
    Tiling:   全部 factor=1 (即每个 tile 只处理 1 个元素)
    
    手工推导:
    - 总共有 2×2×2×2×2×1×1 = 32 次 MAC 运算
    - 每次 MAC 需要: 1 input, 1 weight, 1 output
    
    Input[N,C,H,W] 访问分析:
    - Input tile size = 1×1×1×1 = 1
    - Input 无关维度 = K
    - 对于每个 (n,c,p,q) 组合，K 方向遍历 2 次
    - Input access count = K/K_factor = 2/1 = 2
    - Input total reads = 1 × 2 × (其他维度的总 tile 数)
    
    等等，让我更仔细地分析...
    
    实际上，我们需要考虑的是:
    - 有多少个 input tile? N_tiles × C_tiles × H_tiles × W_tiles
    - 每个 input tile 被访问多少次? K_outer 次 (因为 K 是无关的)
    
    Tile counts:
    - N_tiles = N/N_factor = 2/1 = 2
    - C_tiles = C/C_factor = 2/1 = 2  
    - P_tiles = P/P_factor = 2/1 = 2
    - Q_tiles = Q/Q_factor = 2/1 = 2
    - K_tiles = K/K_factor = 2/1 = 2
    - R_tiles = R/R_factor = 1/1 = 1
    - S_tiles = S/S_factor = 1/1 = 1
    
    Input 分析:
    - 每个 input tile 大小 = N_factor × C_factor × H_in × W_in = 1×1×1×1 = 1
    - Input tile 总数 = N_tiles × C_tiles × P_tiles × Q_tiles = 2×2×2×2 = 16
    - 每个 input tile 访问次数 = K_tiles = 2 (K 是无关的)
    - Input total reads = 16 × 2 = 32
    
    或者用公式:
    - Input tile size = 1
    - Input access count = K/K_factor = 2
    - 但这只是每个 tile 的访问次数
    - 实际 total reads = tile_size × access_count × num_tiles
                       = 1 × 2 × 16 = 32
    
    等等，我们的公式是:
    total_reads = tile_size × access_count
    
    这里 access_count = K_outer = K/K_factor = 2
    
    但这样 total_reads = 1 × 2 = 2，这显然不对。
    
    让我重新理解公式...
    
    啊，我明白了。公式的含义是:
    - tile_size = 单个 tile 的大小
    - access_count = 这种 tile 从外部存储被读取的总次数
    
    对于 Input:
    - 有 N_tiles × C_tiles × P_tiles × Q_tiles = 16 种不同的 tile
    - 每种 tile 被读 K_outer = 2 次
    - 但公式简化为: (N×C×H_in×W_in) × (K_outer)
    - 因为 total = num_tiles × tile_size × access_per_tile
              = (N/Nf × C/Cf × P/Pf × Q/Qf) × (Nf × Cf × Pf × Qf) × K_outer
              = N × C × P × Q × K_outer
    
    所以我们的公式其实是:
    - tile_size = Nf × Cf × H_in × W_in (单个 tile 大小)
    - access_count = (N/Nf × C/Cf × P/Pf × Q/Qf) × K_outer
                   = ... 不对
    
    让我再仔细看代码...
    """
    print("=" * 70)
    print("手工验证 Case 1: 1×1 Conv, 全部 tile=1")
    print("=" * 70)
    
    bounds = LoopBounds(N=2, C=2, K=2, P=2, Q=2, R=1, S=1)
    factors = TileFactors(N=1, C=1, K=1, P=1, Q=1, R=1, S=1)
    
    # 计算
    input_tile = compute_input_tile_size(factors)  # = 1×1×1×1 = 1
    input_access = compute_input_access_count(bounds, factors)  # = K/K_factor = 2
    
    print(f"\nBounds: N={bounds.N}, C={bounds.C}, K={bounds.K}, P={bounds.P}, Q={bounds.Q}")
    print(f"Factors: all = 1")
    print(f"\nInput tile size: {input_tile}")
    print(f"Input access count: {input_access}")
    print(f"Input total (tile × access): {input_tile * input_access}")
    
    # 手工计算
    # 嵌套循环展开:
    # for n in range(2):     # N
    #   for c in range(2):   # C  
    #     for k in range(2): # K
    #       for p in range(2): # P
    #         for q in range(2): # Q
    #           # 读 input[n,c,p,q], weight[k,c,0,0], 写 output[n,k,p,q]
    #
    # Input[n,c,p,q] 被访问的次数 = K 维度的循环次数 = 2
    # 共有 2×2×2×2 = 16 个不同的 input 元素
    # 每个被访问 2 次
    # 总 input 读取 = 16 × 2 = 32
    
    manual_input_reads = 2 * 2 * 2 * 2 * 2  # N×C×P×Q×K
    print(f"\n手工计算 input reads: {manual_input_reads}")
    
    # 但是我们的公式给出 tile × access = 1 × 2 = 2
    # 这显然不对！
    #
    # 问题在于：access_count 只是 K_outer，不包括其他维度的迭代
    # 
    # 正确的理解是:
    # total_input_reads = (所有 tile 的总数据量) × (因 K 迭代重复读取的次数)
    #                   = (N × C × H_in × W_in) × K_outer
    #                   = 2 × 2 × 2 × 2 × 2 = 32
    #
    # 但我们的公式是 tile_size × access_count
    # 这里 tile_size = Nf × Cf × H_in × W_in = 1
    # access_count = K_outer = 2
    #
    # 缺少了 num_tiles = (N/Nf) × (C/Cf) × (P/Pf) × (Q/Qf)
    
    # 让我重新检查代码中的公式定义...
    # 看起来我们需要修正公式
    
    print("\n" + "=" * 70)
    print("发现问题: 公式可能需要调整")
    print("=" * 70)
    print("""
当前公式:
    total_reads = tile_size × access_count
    
其中:
    tile_size = Nf × Cf × H_in × W_in  (单个 tile 的大小)
    access_count = K_outer = K / Kf    (K 是无关维度)

问题:
    这个公式缺少了 "有多少个 tile" 这一项
    
正确的公式应该是:
    total_reads = tile_size × num_tiles × access_per_tile
    
    或者等价地:
    total_reads = total_data_size × access_multiplier
                = (N × C × H_in × W_in) × K_outer
    
让我验证 Interstellar 原始论文的公式...
    """)


def manual_verification_2():
    """
    验证 Case 2: 理解 Interstellar 的原始公式
    
    Interstellar 的公式是:
    - get_if_access() = FX * FY * OC  (对于 output stationary)
    
    其中:
    - FX, FY = kernel size (R, S)
    - OC = output channels (K)
    
    这些是 "irrelevant" 维度的 **总循环次数**，不是 outer factor
    
    等等，让我仔细看 Interstellar 的代码...
    """
    print("\n" + "=" * 70)
    print("验证 Interstellar 原始公式")
    print("=" * 70)
    
    # 从 Interstellar cost_model.py:
    # def get_if_access(self):
    #     return self.fx * self.fy * self.oc
    #
    # 这里 fx, fy, oc 是什么?
    # 
    # 看 schedule.py:
    # fx = para_count[0] * order_count[0]  # FX 维度的总迭代
    # 
    # 所以 Interstellar 的公式是:
    # Input accesses = (total FX iterations) × (total FY iterations) × (total OC iterations)
    #
    # 这实际上是: 每个 input 元素被访问的总次数
    # = R × S × K (对于 output stationary dataflow)
    #
    # 但这是整个工作负载的访问次数，不考虑 tiling
    
    print("""
Interstellar 的公式 (for output stationary):
    input_accesses = FX × FY × OC = R × S × K
    
这是 **每个 input 位置** 被访问的次数。

总 input 读取 = input_size × input_accesses
             = (N × C × H × W) × (R × S × K)  -- 这是错的！
             
不对，这样会重复计算。让我重新理解...

对于 output stationary:
- 每个 output[n,k,p,q] 累积 C×R×S 次
- 在这个过程中，需要读取 input[n,c,p+r,q+s] 和 weight[k,c,r,s]
- input[n,c,p+r,q+s] 被哪些 output 使用? 
  - 不同的 k (有 K 个)
  - 不同的 r 会用到不同的 p+r (有 R 种 r 对应同一个 input)
  - 不同的 s 会用到不同的 q+s (有 S 种 s)
  
所以每个 input 元素被读 K×R×S 次 (如果没有 reuse)

而 weight[k,c,r,s] 被用于:
  - 不同的 n (有 N 个)
  - 不同的 p (有 P 个)
  - 不同的 q (有 Q 个)
  
所以每个 weight 元素被读 N×P×Q 次

这就是 "irrelevant loops" 的含义!
    """)


def manual_verification_3():
    """
    验证 Case 3: 带 tiling 的情况
    
    Tiling 如何影响访问次数?
    """
    print("\n" + "=" * 70)
    print("带 Tiling 的访问次数分析")
    print("=" * 70)
    
    print("""
假设 workload: N=4, C=4, K=4, P=4, Q=4, R=1, S=1
Tiling: Nf=2, Cf=2, Kf=2, Pf=2, Qf=2

分析:
- N_outer = 4/2 = 2, C_outer = 2, K_outer = 2, P_outer = 2, Q_outer = 2
- Input tile = (Nf=2) × (Cf=2) × (Pf+Rf-1=2) × (Qf+Sf-1=2) = 32 元素
- Weight tile = (Kf=2) × (Cf=2) × (Rf=1) × (Sf=1) = 4 元素
- Output tile = (Nf=2) × (Kf=2) × (Pf=2) × (Qf=2) = 32 元素

考虑 nested loop 结构 (假设 K 在最外层):

for k_o in range(K_outer):      # 2 次
  for n_o in range(N_outer):    # 2 次
    for c_o in range(C_outer):  # 2 次
      for p_o in range(P_outer):# 2 次
        for q_o in range(Q_outer):# 2 次
          # 在这里处理一个 tile
          # 读取: input[n_o:n_o+Nf, c_o:c_o+Cf, ...]
          # 读取: weight[k_o:k_o+Kf, c_o:c_o+Cf, ...]
          # 读/写: output[n_o:n_o+Nf, k_o:k_o+Kf, ...]

Input tile 访问分析:
- Input tile 由 (n_o, c_o, p_o, q_o) 索引
- 对于固定的 (n_o, c_o, p_o, q_o)，K_outer 循环不改变 input tile
- 所以每个 input tile 被读取 K_outer = 2 次
- 总共有 N_outer × C_outer × P_outer × Q_outer = 16 个不同的 input tiles
- 总 input reads = 16 × 32 × 2 = 1024 元素

Weight tile 访问分析:
- Weight tile 由 (k_o, c_o) 索引
- N, P, Q 都是无关的
- 每个 weight tile 被读取 N_outer × P_outer × Q_outer = 8 次
- 总共有 K_outer × C_outer = 4 个不同的 weight tiles
- 总 weight reads = 4 × 4 × 8 = 128 元素

Output tile 访问分析:
- Output tile 由 (n_o, k_o, p_o, q_o) 索引
- C, R, S 是无关的 (partial sum accumulation)
- 每个 output tile 被读取 C_outer × R_outer × S_outer = 2 次
- 总共有 N_outer × K_outer × P_outer × Q_outer = 16 个不同的 output tiles
- 总 output reads = 16 × 32 × 2 = 1024 元素
    """)
    
    # 用代码验证
    bounds = LoopBounds(N=4, C=4, K=4, P=4, Q=4, R=1, S=1)
    factors = TileFactors(N=2, C=2, K=2, P=2, Q=2, R=1, S=1)
    
    # 手工计算
    N_outer = bounds.N // factors.N  # 2
    C_outer = bounds.C // factors.C  # 2
    K_outer = bounds.K // factors.K  # 2
    P_outer = bounds.P // factors.P  # 2
    Q_outer = bounds.Q // factors.Q  # 2
    R_outer = bounds.R // factors.R  # 1
    S_outer = bounds.S // factors.S  # 1
    
    input_tile_size = factors.N * factors.C * factors.P * factors.Q  # 简化: R=S=1
    num_input_tiles = N_outer * C_outer * P_outer * Q_outer
    input_access_per_tile = K_outer
    manual_input_reads = num_input_tiles * input_tile_size * input_access_per_tile
    
    print(f"\n手工计算:")
    print(f"  Input tile size = {input_tile_size}")
    print(f"  Num input tiles = {num_input_tiles}")
    print(f"  Access per tile = {input_access_per_tile}")
    print(f"  Total input reads = {manual_input_reads}")
    
    # 用我们的公式计算
    formula_tile = compute_input_tile_size(factors)
    formula_access = compute_input_access_count(bounds, factors)
    formula_total = formula_tile * formula_access
    
    print(f"\n公式计算:")
    print(f"  tile_size = {formula_tile}")
    print(f"  access_count = {formula_access}")
    print(f"  total = {formula_total}")
    
    # 检查公式是否正确
    # 我们的公式: tile_size × access_count
    # = (Nf × Cf × Pf × Qf) × (K_outer)
    # = 32 × 2 = 64
    #
    # 但手工计算是 1024
    #
    # 差距 = 1024 / 64 = 16 = num_input_tiles!
    #
    # 所以我们的 access_count 只是 K_outer
    # 而不是 (总tile数) × K_outer
    
    print(f"\n差异分析:")
    print(f"  手工/公式 = {manual_input_reads / formula_total}")
    print(f"  这正好等于 num_input_tiles = {num_input_tiles}")
    print("""
结论: 我们的公式计算的是 "单个 tile 被读取的总次数 × tile 大小"
而不是 "所有 tile 的总读取量"

需要修正公式为:
    total_reads = tile_size × num_tiles × access_per_tile
    
或者重新定义:
    total_reads = total_unique_data × reuse_factor
    
其中:
    total_unique_data = 整个 tensor 的大小
    reuse_factor = 因为无关循环导致的重复访问次数
    """)


def correct_formula_verification():
    """
    验证修正后的正确公式
    """
    print("\n" + "=" * 70)
    print("修正后的正确公式验证")
    print("=" * 70)
    
    bounds = LoopBounds(N=4, C=4, K=4, P=4, Q=4, R=1, S=1)
    factors = TileFactors(N=2, C=2, K=2, P=2, Q=2, R=1, S=1)
    
    # 正确公式:
    # total_input_reads = total_input_size × K_outer
    # 其中 total_input_size = N × C × H_in × W_in
    
    H_in = bounds.P  # 简化 R=1
    W_in = bounds.Q  # 简化 S=1
    total_input_size = bounds.N * bounds.C * H_in * W_in
    K_outer = bounds.K // factors.K
    
    correct_input_reads = total_input_size * K_outer
    
    print(f"Input:")
    print(f"  Total input size = {total_input_size}")
    print(f"  K_outer (reuse factor) = {K_outer}")
    print(f"  Total input reads = {correct_input_reads}")
    
    # Weight
    total_weight_size = bounds.K * bounds.C * bounds.R * bounds.S
    N_outer = bounds.N // factors.N
    P_outer = bounds.P // factors.P
    Q_outer = bounds.Q // factors.Q
    weight_reuse = N_outer * P_outer * Q_outer
    
    correct_weight_reads = total_weight_size * weight_reuse
    
    print(f"\nWeight:")
    print(f"  Total weight size = {total_weight_size}")
    print(f"  Reuse factor (N×P×Q outer) = {weight_reuse}")
    print(f"  Total weight reads = {correct_weight_reads}")
    
    # Output
    total_output_size = bounds.N * bounds.K * bounds.P * bounds.Q
    C_outer = bounds.C // factors.C
    R_outer = bounds.R // factors.R
    S_outer = bounds.S // factors.S
    output_reuse = C_outer * R_outer * S_outer
    
    correct_output_reads = total_output_size * output_reuse
    
    print(f"\nOutput:")
    print(f"  Total output size = {total_output_size}")
    print(f"  Reuse factor (C×R×S outer) = {output_reuse}")
    print(f"  Total output reads = {correct_output_reads}")
    
    print(f"\n总内存访问: {correct_input_reads + correct_weight_reads + correct_output_reads}")


if __name__ == '__main__':
    manual_verification_1()
    manual_verification_2()
    manual_verification_3()
    correct_formula_verification()
    
    print("\n" + "=" * 70)
    print("总结: Golden Model 公式需要修正!")
    print("=" * 70)
    print("""
发现的问题:
    当前公式: total_reads = tile_size × access_count
    其中 access_count = K_outer (对于 input)
    
    这只计算了单个 tile 的读取，没有乘以 tile 的数量!

正确的公式:
    total_reads = total_data_size × reuse_factor
    
    或等价地:
    total_reads = tile_size × num_tiles × access_per_tile

对于 Input:
    total_reads = (N × C × H_in × W_in) × (K / K_factor)
    
对于 Weight:
    total_reads = (K × C × R × S) × (N/Nf × P/Pf × Q/Qf)
    
对于 Output:
    total_reads = (N × K × P × Q) × (C/Cf × R/Rf × S/Sf)
    """)
