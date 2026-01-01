#!/usr/bin/env python3
"""详细解释 step_h 的来源"""

print("=" * 70)
print("step_h 的计算详解")
print("=" * 70)

print("""
【关键理解】

在 row_activation.py 中:
    q_factor = yh_q_list[j]    # 取自枚举列表
    step_h = q_factor * stride_h

yh_q_list 是在 expressions.py 中构建的，包含 RowBuffer level 的 Q factor 选项。

【yh_q_list 的构建方式】(expressions.py line 170-183)

for dq, q_ in enumerate(workload.divisors[3]):   # divisors[3] = Q 维度的因子
    for ds, s_ in enumerate(workload.divisors[1]):   # divisors[1] = S 维度的因子
        yh_q_list.append(q_)  # <-- 这里！q_ 是 Q 维度的累积因子

所以 yh_q_list 存储的是 Q 维度从 level 0 到 RowBuffer level 的累积 tile factor。
""")

print("=" * 70)
print("Tiny Workload 的具体分析")
print("=" * 70)

print("""
【Workload 参数】
  Q = 4 (output height)
  S = 3 (kernel height)
  stride_h = 1
  
【Q 维度的 divisors】
  Q = 4 的因子: [1, 2, 4]
  
【S 维度的 divisors】
  S = 3 的因子: [1, 3]
  
【在 RowBuffer level 构建 yh_q_list】

枚举所有 (Q因子, S因子) 组合:
  q=1, s=1 → yh_q_list[0] = 1
  q=1, s=3 → yh_q_list[1] = 1
  q=2, s=1 → yh_q_list[2] = 2
  q=2, s=3 → yh_q_list[3] = 2
  q=4, s=1 → yh_q_list[4] = 4
  q=4, s=3 → yh_q_list[5] = 4
  
【对应的 yh (input tile height)】

unique_h = stride*(Q-1) + dilation*(S-1) + 1

  q=1, s=1: unique_h = 1*(1-1) + 1*(1-1) + 1 = 1
  q=1, s=3: unique_h = 1*(1-1) + 1*(3-1) + 1 = 3
  q=2, s=1: unique_h = 1*(2-1) + 1*(1-1) + 1 = 2
  q=2, s=3: unique_h = 1*(2-1) + 1*(3-1) + 1 = 4
  q=4, s=1: unique_h = 1*(4-1) + 1*(1-1) + 1 = 4
  q=4, s=3: unique_h = 1*(4-1) + 1*(3-1) + 1 = 6  <-- 这是选中的！
""")

print("=" * 70)
print("ILP 选择了哪个？")
print("=" * 70)

print("""
从 mapping_results.txt:
  Selected tile_h = 6 (对应 yh[5])
  
所以选中的是:
  q_ = 4
  s_ = 3
  yh_q_list[5] = 4
  
因此:
  q_factor = yh_q_list[5] = 4
  step_h = q_factor * stride_h = 4 * 1 = 4
  
【但 crossing_ratio = 0.666667 对应 step = 2，不是 4！】
""")

print("=" * 70)
print("问题分析")
print("=" * 70)

import math

block_h = 6
tile_h = 6

# 验证不同 step 值
print("\n【验证不同 step 产生的 crossing ratio】")
for step in [1, 2, 3, 4, 5, 6]:
    g = math.gcd(step, block_h)
    period = block_h // g
    crossing_count = 0
    for k in range(period):
        pos_mod = (k * step) % block_h
        if pos_mod + tile_h > block_h:
            crossing_count += 1
    cr = crossing_count / period if period > 0 else 0
    match = "✓ MATCH 0.666667" if abs(cr - 0.666667) < 0.001 else ""
    print(f"step={step}: cr={cr:.6f} {match}")

print("""
\n【结论】

step = 2 或 4 都能产生 cr = 0.666667

如果 q_factor = 4 (从 yh_q_list):
  step_h = 4 * 1 = 4
  cr_h = 0.666667 ✓

如果 q_factor = 2:
  step_h = 2 * 1 = 2  
  cr_h = 0.666667 ✓

两种情况都能得到相同的 crossing ratio，
因为 step=2 和 step=4 在 mod 6 周期中产生相同的 crossing 模式。

【最终答案】

step_h = yh_q_list[selected_idx] × stride_h

对于 tiny workload:
- 选中的 tile_h = 6，对应 q=4, s=3
- yh_q_list 对应值 = 4 (Q 方向的累积因子)
- stride_h = 1
- step_h = 4 × 1 = 4

但由于数学巧合，step=4 和 step=2 在 block=6 时产生相同的 crossing ratio。
""")
