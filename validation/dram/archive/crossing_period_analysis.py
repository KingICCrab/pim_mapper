#!/usr/bin/env python3
"""分析如何在 ILP 中精确表示 crossing 次数"""

import math

print("=" * 70)
print("用周期分解来精确计算 crossing")
print("=" * 70)

print("""
【思路】

将 crossing 计算分解为：
1. 完整周期数 × 每周期 crossing 次数
2. 不足一个周期的部分的 crossing 次数

公式：
  num_tiles = 实际的 tile 数量
  period = block_h / gcd(step, block_h)
  crossing_per_period = 每个周期内的 crossing 次数
  
  num_complete_periods = floor((num_tiles - 1) / period)
  remainder = (num_tiles - 1) % period + 1  # 最后不完整周期的 tile 数
  
  total_crossing = num_complete_periods × crossing_per_period 
                 + crossing_in_remainder(remainder)
""")

print("=" * 70)
print("具体例子")
print("=" * 70)

def analyze_crossing(input_h, tile_h, step, block_h):
    """精确分析 crossing"""
    
    # 1. 计算实际 tile 数量
    num_tiles = 0
    pos = 0
    while pos < input_h:
        num_tiles += 1
        pos += step
        if pos >= input_h:
            break
    
    # 注意：最后一个 tile 如果 pos >= input_h 就停止了
    # 实际上 num_tiles = ceil(input_h / step)，但要考虑边界
    
    # 更准确的计算
    if tile_h >= input_h:
        num_tiles = 1
    else:
        # 最后一个有效的起始位置是 < input_h 的最大的 k*step
        num_tiles = (input_h - tile_h) // step + 1
    
    # 2. 计算周期参数
    g = math.gcd(step, block_h)
    period = block_h // g
    
    # 3. 计算每个周期内的 crossing 次数和位置
    crossing_positions = []  # 周期内哪些位置会 crossing
    for k in range(period):
        pos_mod = (k * step) % block_h
        if pos_mod + tile_h > block_h:
            crossing_positions.append(k)
    crossing_per_period = len(crossing_positions)
    
    # 4. 分解为完整周期 + 不完整周期
    # 注意：第一个 tile 是 k=0，第 num_tiles 个 tile 是 k=num_tiles-1
    # 完整周期数 = floor((num_tiles - 1) / period)  (不包括最后一个不完整周期)
    if num_tiles <= 0:
        return {
            'num_tiles': 0,
            'period': period,
            'crossing_per_period': crossing_per_period,
            'num_complete_periods': 0,
            'remainder_tiles': 0,
            'crossing_in_remainder': 0,
            'total_crossing': 0,
            'exact_crossing': 0
        }
    
    # 从 k=0 到 k=num_tiles-1
    # k=0 到 k=period-1 是第一个周期
    # k=period 到 k=2*period-1 是第二个周期
    # ...
    
    num_complete_periods = (num_tiles) // period
    remainder_tiles = (num_tiles) % period
    
    # 5. 计算不完整周期的 crossing
    # 不完整周期包含 k = num_complete_periods * period, ..., num_tiles - 1
    # 在周期内的位置是 0, 1, ..., remainder_tiles - 1
    crossing_in_remainder = 0
    for k in range(remainder_tiles):
        if k in crossing_positions:
            crossing_in_remainder += 1
    
    # 6. 计算理论总 crossing
    total_crossing_formula = num_complete_periods * crossing_per_period + crossing_in_remainder
    
    # 7. 精确计算（枚举每个 tile）
    exact_crossing = 0
    for i in range(num_tiles):
        pos = i * step
        if pos >= input_h:
            break
        actual_end = min(pos + tile_h, input_h)
        actual_size = actual_end - pos
        pos_in_block = pos % block_h
        if pos_in_block + actual_size > block_h:
            exact_crossing += 1
    
    return {
        'num_tiles': num_tiles,
        'period': period,
        'crossing_positions': crossing_positions,
        'crossing_per_period': crossing_per_period,
        'num_complete_periods': num_complete_periods,
        'remainder_tiles': remainder_tiles,
        'crossing_in_remainder': crossing_in_remainder,
        'total_crossing_formula': total_crossing_formula,
        'exact_crossing': exact_crossing
    }


# 测试用例
test_cases = [
    (6, 6, 4, 6, "tiny workload"),
    (14, 6, 4, 6, "medium workload"),
    (100, 6, 4, 6, "large workload"),
    (50, 6, 4, 6, "workload 50"),
]

for input_h, tile_h, step, block_h, desc in test_cases:
    print(f"\n【{desc}】input_h={input_h}, tile_h={tile_h}, step={step}, block_h={block_h}")
    result = analyze_crossing(input_h, tile_h, step, block_h)
    
    print(f"  num_tiles = {result['num_tiles']}")
    print(f"  period = {result['period']}")
    print(f"  crossing_positions in period = {result['crossing_positions']}")
    print(f"  crossing_per_period = {result['crossing_per_period']}")
    print(f"  num_complete_periods = {result['num_complete_periods']}")
    print(f"  remainder_tiles = {result['remainder_tiles']}")
    print(f"  crossing_in_remainder = {result['crossing_in_remainder']}")
    print(f"  公式计算 total_crossing = {result['total_crossing_formula']}")
    print(f"  精确枚举 exact_crossing = {result['exact_crossing']}")
    
    match = "✓" if result['total_crossing_formula'] == result['exact_crossing'] else "✗ 不匹配！"
    print(f"  匹配: {match}")

print("\n" + "=" * 70)
print("ILP 建模分析")
print("=" * 70)

print("""
【在 ILP 中表示这个公式】

已知常数（对于给定的 tile/block 组合）：
  - period = block_h / gcd(step, block_h)
  - crossing_per_period = 每周期 crossing 次数
  - crossing_positions = 周期内哪些位置会 crossing

需要计算的：
  - num_tiles = ceil(input_h / step) 或类似公式

【关键问题：num_tiles 是变量还是常数？】

情况 1: input_h 是常数（workload 给定）
  → num_tiles 对于每个 (tile_h, step) 组合是常数
  → 可以预计算所有组合的 crossing 次数
  → 像现在一样用枚举 + 选择变量

情况 2: input_h 是变量（依赖于 tiling）
  → num_tiles 是 ILP 变量
  → 需要用整数变量表示 floor 和 mod
  → 复杂度增加

【情况 1 的实现（推荐）】

对于每个 (block_h, tile_h, step) 组合：
  1. 计算 num_tiles（常数）
  2. 计算 num_complete_periods = num_tiles // period
  3. 计算 remainder = num_tiles % period
  4. 计算 crossing_in_remainder（枚举）
  5. 总 crossing = num_complete_periods × crossing_per_period + crossing_in_remainder

然后用选择变量选择实际的组合：
  total_crossing = Σ combo_var[k] × crossing_count[k]

【情况 2 的实现（复杂）】

需要引入辅助整数变量：
  q = floor(num_tiles / period)  # 商
  r = num_tiles - q × period     # 余数
  
约束：
  num_tiles = q × period + r
  0 <= r < period
  q >= 0 (整数)
  r >= 0 (整数)

然后：
  crossing_in_remainder 需要用 indicator constraints：
  if r >= 1 and crossing_positions contains 0: add 1
  if r >= 2 and crossing_positions contains 1: add 1
  ...

这会引入很多二元变量和约束。
""")

print("=" * 70)
print("结论")
print("=" * 70)

print("""
【结论】

如果 input_h 对于 workload 是常数（通常是）：
  → 可以预计算每个 (block, tile, step) 组合的精确 crossing 次数
  → 不需要在 ILP 中表示 floor/mod
  → 只需要修改预计算函数 compute_input_crossing_ratio()

修改方案：

def compute_input_crossing_ratio_exact(
    block_h, tile_h, step, input_h
):
    '''精确计算 crossing ratio，考虑边界效应'''
    
    # 计算实际 tile 数量
    if tile_h >= input_h:
        num_tiles = 1
    else:
        num_tiles = (input_h - tile_h) // step + 1
    
    # 计算周期参数
    g = gcd(step, block_h)
    period = block_h // g
    
    # 计算周期内的 crossing 位置
    crossing_positions = set()
    for k in range(period):
        pos_mod = (k * step) % block_h
        if pos_mod + tile_h > block_h:
            crossing_positions.add(k % period)
    
    # 分解计算
    num_complete_periods = num_tiles // period
    remainder = num_tiles % period
    
    crossing_in_remainder = sum(1 for k in range(remainder) if k in crossing_positions)
    
    total_crossing = num_complete_periods * len(crossing_positions) + crossing_in_remainder
    
    # 但还需要考虑最后一个 tile 的边界效应！
    # 最后一个 tile 可能取不到完整的 tile_h
    # 这需要特殊处理...
    
    return total_crossing / num_tiles if num_tiles > 0 else 0.0

【关键】
还需要处理最后一个 tile 的边界效应：
  - 最后一个 tile 的实际大小可能 < tile_h
  - 这可能导致原本会 crossing 的 tile 不再 crossing
""")
