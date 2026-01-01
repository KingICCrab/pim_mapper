#!/usr/bin/env python3
"""解释 step_h 是如何计算的"""

print("=" * 70)
print("step_h 的计算方法")
print("=" * 70)

print("""
【公式】
  step_h = Q_factor × stride_h

其中：
  - Q_factor: RowBuffer level 及以下所有 Q 维度的 tile factor 乘积
  - stride_h: 卷积在 H 方向的 stride

【为什么是 Q_factor？】

Input 的 H 维度与 Output 的 Q 维度相关联：
  input_h = (output_q - 1) × stride + (kernel_r - 1) × dilation + 1

当在 Q 维度迭代时，相邻的 output tiles 对应的 input tiles 之间的间隔是：
  step = stride × Q_factor

这是因为每移动 Q_factor 个 Q 维度，input 在 H 方向移动 Q_factor × stride。
""")

print("=" * 70)
print("Tiny Workload 的 step_h 计算")
print("=" * 70)

print("""
【从 mapping_results.txt 读取 Tile Sizes】

PE (level 0):
  Temporal: Q=2
  H-spatial: Q=2
  W-spatial: Q=1
  
GlobalBuffer (level 1):
  Temporal: Q=1
  
RowBuffer (level 2):
  Temporal: Q=1
  
【Q_factor 计算】

Q_factor 是 RowBuffer level 及以下所有 Q tile factors 的乘积：
  Q_factor = Q_temporal_PE × Q_Hspatial × Q_Wspatial × Q_GlobalBuffer
           = 2 × 2 × 1 × 1
           = 4

⚠️ 注意！上面我之前写的 Q_factor = 2 是错的！

让我重新查看...
""")

# 根据 mapping 重新计算
print("\n【重新分析】")
print("""
实际上，step_h 的计算取决于"在哪个 level 计算 crossing"：

对于 Row-Aligned 模式：
- 在 RowBuffer level 计算 crossing ratio
- 需要考虑 RowBuffer 内部的访问模式

对于 Sequential 模式：
- 使用不同的公式，基于预计算的 average crossing ratio

让我查看 Row-Aligned 模式中实际的 step 计算...
""")

# 实际的 step 计算逻辑
print("\n【Row-Aligned 模式的 step 计算】")
print("""
从代码 row_activation.py 的 compute_input_crossing_ratio:

step = Q_factor × stride

其中 Q_factor 是所有在 RowBuffer 之下的 Q tile factors 乘积。

对于 tiny workload:
- PE level Q temporal = 2
- PE level Q H-spatial = 2  
- PE level Q W-spatial = 1

但 crossing ratio 计算中的 step 是指：
"在一次 RowBuffer 访问内，连续访问 input tiles 的间隔"

这取决于 inner loop 的结构...

实际上 step_h 通常等于 stride (最内层移动)，
除非有特殊的 spatial mapping 改变了访问顺序。
""")

# 从 Row-Aligned 变量反推
print("\n【从结果反推 step_h】")

import math

block_h = 6
tile_h = 6
cr_h = 0.666667  # 从 mapping_results.txt

print(f"已知: block_h={block_h}, tile_h={tile_h}, cr_h={cr_h:.6f}")
print()

# 尝试不同的 step 值
for step in [1, 2, 3, 4, 6]:
    g = math.gcd(step, block_h)
    period = block_h // g
    crossing_count = 0
    for k in range(period):
        pos_mod = (k * step) % block_h
        if pos_mod + tile_h > block_h:
            crossing_count += 1
    cr = crossing_count / period if period > 0 else 0
    match = "✓ MATCH" if abs(cr - cr_h) < 0.001 else ""
    print(f"step={step}: period={period}, crossing_count={crossing_count}, cr={cr:.6f} {match}")

print("""
\n【结论】
step_h = 2 能产生 cr_h = 0.666667

这可能来自：
- PE level Q temporal factor = 2
- 或者其他计算方式

具体取决于代码中如何传递 Q_factor 参数。
""")
