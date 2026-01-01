#!/usr/bin/env python3
"""详细解释 non_crossing_acts 的计算过程"""

import math

print('='*100)
print('ILP 模型代码详解: non_crossing_acts 的计算')
print('='*100)
print()

print('【1. 函数入口: _build_sequential_dram_crossing】')
print('='*100)
print('''
位置: src/pim_optimizer/model/row_activation.py 第 492-547 行

def _build_sequential_dram_crossing(
    model: gp.Model,
    tile_entries_list: tuple,    # 所有可能的 tile 大小选项 (元素数量)
    xu_vars: list,               # one-hot 变量，选择哪个 tile 大小
    element_bytes: float,        # 每个元素的字节数
    row_buffer_size_bytes: float,# DRAM row buffer 大小
    tensor_bytes: float,         # tensor 总字节数
    reuse_penalty: gp.Var,       # 重用惩罚因子 (ILP 变量)
    max_reuse_penalty: float,    # 最大重用惩罚
    w: int,                      # workload 索引
    t_id: int,                   # tensor ID (0=Input, 1=Weight, 2=Output)
) -> tuple[gp.Var, float]:
''')
print()

print('【2. 调用 precompute_tile_crossing_info 预计算】')
print('='*100)
print('''
位置: 第 526-528 行

non_crossing_acts_list, crossing_counts_list = precompute_tile_crossing_info(
    tile_entries_list, element_bytes, row_buffer_size_bytes, tensor_bytes
)

这个函数对每个可能的 tile 大小，预计算:
  - non_crossing_acts_list[k]: 第 k 种 tile 大小对应的 non-crossing activation 数
  - crossing_counts_list[k]: 第 k 种 tile 大小对应的 crossing tile 数量
''')
print()

print('【3. precompute_tile_crossing_info 函数详解】')
print('='*100)
print('''
位置: 第 17-56 行

def precompute_tile_crossing_info(
    tile_entries_list: list,      # 可能的 tile 大小列表 (元素数)
    element_bytes: float,         # 每元素字节数
    row_bytes: float,             # row buffer 字节数 (例如 1024)
    tensor_total_bytes: float,    # tensor 总字节数
) -> tuple[list[int], list[int]]:

    non_crossing_acts_list = []
    crossing_counts_list = []
    
    for te in tile_entries_list:
        # Step 3.1: 计算 tile 字节数
        tile_bytes = te * element_bytes
        
        # Step 3.2: 计算 tensor 中有多少个 tiles
        num_tiles = max(1, int(tensor_total_bytes / tile_bytes))
        
        # Step 3.3: 计算有多少 tiles 跨越 row 边界
        crossing_count = compute_dram_row_crossing_count(tile_bytes, row_bytes, num_tiles)
        
        # Step 3.4: 计算一个 row 能放多少个完整的 tile
        tiles_per_row = max(1, int(row_bytes / tile_bytes))
        
        # Step 3.5: non-crossing tiles 数量
        non_crossing_count = num_tiles - crossing_count
        
        # Step 3.6: 关键计算! non_crossing_acts
        non_crossing_acts = math.ceil(non_crossing_count / tiles_per_row)
        
        non_crossing_acts_list.append(non_crossing_acts)
        crossing_counts_list.append(crossing_count)
    
    return non_crossing_acts_list, crossing_counts_list
''')
print()

print('【4. compute_dram_row_crossing_count 函数详解】')
print('='*100)
print('''
位置: 第 243-285 行

def compute_dram_row_crossing_count(tile_bytes, row_bytes, num_tiles) -> int:
    """
    计算有多少个 tiles 跨越了 DRAM row 边界
    
    Crossing 发生条件:
      tile 的起始地址在 row A，结束地址在 row B (A != B)
      即: start_offset // row_bytes != (start_offset + tile_bytes - 1) // row_bytes
    """
    
    # 特殊情况处理
    if tile_bytes > row_bytes:
        return num_tiles  # 每个 tile 都跨越
    if tile_bytes == row_bytes:
        return 0  # 每个 tile 刚好占一个 row，不跨越
    
    # 使用 GCD 计算周期
    g = math.gcd(tile_bytes, row_bytes)
    period = row_bytes // g  # 周期长度
    
    # 计算一个周期内有多少 crossing
    threshold = row_bytes - tile_bytes + 1
    cross_count_per_period = period - math.ceil(threshold / g)
    
    # 分解为完整周期 + 余数
    num_complete_periods = num_tiles // period
    remainder_tiles = num_tiles % period
    
    # 余数部分逐个检查
    crossings_in_remainder = 0
    for i in range(remainder_tiles):
        start_offset = i * tile_bytes
        start_row = start_offset // row_bytes
        end_row = (start_offset + tile_bytes - 1) // row_bytes
        if end_row > start_row:
            crossings_in_remainder += 1
    
    return num_complete_periods * cross_count_per_period + crossings_in_remainder
''')
print()

print('【5. Output tensor 实例计算 (small-v2 workload)】')
print('='*100)

tile_entries = 128
element_bytes = 1
row_bytes = 1024
tensor_total_bytes = 4096

print(f'输入参数:')
print(f'  tile_entries = {tile_entries} (元素数: N=1 × K=2 × P=16 × Q=4)')
print(f'  element_bytes = {element_bytes}')
print(f'  row_bytes = {row_bytes}')
print(f'  tensor_total_bytes = {tensor_total_bytes} (N=1 × K=16 × P=16 × Q=16)')
print()

print('--- Step 3.1: 计算 tile_bytes ---')
tile_bytes = tile_entries * element_bytes
print(f'  tile_bytes = tile_entries × element_bytes')
print(f'             = {tile_entries} × {element_bytes}')
print(f'             = {tile_bytes} bytes')
print()

print('--- Step 3.2: 计算 num_tiles ---')
num_tiles = max(1, int(tensor_total_bytes / tile_bytes))
print(f'  num_tiles = int(tensor_total_bytes / tile_bytes)')
print(f'            = int({tensor_total_bytes} / {tile_bytes})')
print(f'            = {num_tiles} 个 tiles')
print()

print('--- Step 3.3: 计算 crossing_count ---')
print(f'  调用 compute_dram_row_crossing_count({tile_bytes}, {row_bytes}, {num_tiles})')
print()
print(f'  检查特殊情况:')
print(f'    tile_bytes ({tile_bytes}) > row_bytes ({row_bytes})? No')
print(f'    tile_bytes ({tile_bytes}) == row_bytes ({row_bytes})? No')
print()
print(f'  使用 GCD 方法计算:')
g = math.gcd(tile_bytes, row_bytes)
print(f'    g = gcd({tile_bytes}, {row_bytes}) = {g}')
period = row_bytes // g
print(f'    period = row_bytes // g = {row_bytes} // {g} = {period}')
threshold = row_bytes - tile_bytes + 1
print(f'    threshold = row_bytes - tile_bytes + 1 = {row_bytes} - {tile_bytes} + 1 = {threshold}')
cross_per_period = period - math.ceil(threshold / g)
print(f'    cross_per_period = period - ceil(threshold / g)')
print(f'                     = {period} - ceil({threshold} / {g})')
print(f'                     = {period} - {math.ceil(threshold / g)}')
print(f'                     = {cross_per_period}')
print()
print(f'  解释: 因为 tile_bytes={tile_bytes} 能整除 row_bytes={row_bytes}')
print(f'        每个 tile 都完整地位于某个 row 内，不会跨越边界')
crossing_count = 0
print(f'  crossing_count = {crossing_count}')
print()

print('--- Step 3.4: 计算 tiles_per_row ---')
tiles_per_row = max(1, int(row_bytes / tile_bytes))
print(f'  tiles_per_row = int(row_bytes / tile_bytes)')
print(f'                = int({row_bytes} / {tile_bytes})')
print(f'                = {tiles_per_row}')
print()
print(f'  含义: 每个 DRAM row 可以容纳 {tiles_per_row} 个完整的 tile')
print()

print('--- Step 3.5: 计算 non_crossing_count ---')
non_crossing_count = num_tiles - crossing_count
print(f'  non_crossing_count = num_tiles - crossing_count')
print(f'                     = {num_tiles} - {crossing_count}')
print(f'                     = {non_crossing_count}')
print()
print(f'  含义: {non_crossing_count} 个 tiles 不跨越 row 边界')
print()

print('--- Step 3.6: 计算 non_crossing_acts (关键!) ---')
non_crossing_acts = math.ceil(non_crossing_count / tiles_per_row)
print(f'  non_crossing_acts = ceil(non_crossing_count / tiles_per_row)')
print(f'                    = ceil({non_crossing_count} / {tiles_per_row})')
print(f'                    = ceil({non_crossing_count / tiles_per_row})')
print(f'                    = {non_crossing_acts}')
print()

print('【6. non_crossing_acts = 4 的物理含义】')
print('='*100)
print()
print('Sequential 模式下的 DRAM 地址布局:')
print()
print('  地址范围            | Tile 编号  | 所在 Row')
print('  --------------------|------------|----------')
for i in range(num_tiles):
    start = i * tile_bytes
    end = (i + 1) * tile_bytes - 1
    row = start // row_bytes
    print(f'  [{start:4d}, {end:4d}]       | tile {i:2d}    | Row {row}')
print()

print('按 Row 分组:')
print()
for row in range(non_crossing_acts):
    start_tile = row * tiles_per_row
    end_tile = min((row + 1) * tiles_per_row - 1, num_tiles - 1)
    start_addr = row * row_bytes
    end_addr = (row + 1) * row_bytes - 1
    print(f'  Row {row}: 地址 [{start_addr}, {end_addr}]')
    print(f'        包含 tiles {start_tile}-{end_tile}')
    print()

print('顺序访问时的 Row Buffer 行为:')
print()
print('  访问 tile 0 (Row 0):')
print('    Row Buffer 为空 → 激活 Row 0 → 第 1 次 activation')
print()
print('  访问 tile 1-7 (Row 0):')
print('    Row 0 已在 Row Buffer 中 → 命中，无需激活')
print()
print('  访问 tile 8 (Row 1):')
print('    Row Buffer 中是 Row 0，需要 Row 1')
print('    → 关闭 Row 0，激活 Row 1 → 第 2 次 activation')
print()
print('  访问 tile 9-15 (Row 1):')
print('    Row 1 已在 Row Buffer 中 → 命中，无需激活')
print()
print('  访问 tile 16 (Row 2):')
print('    → 激活 Row 2 → 第 3 次 activation')
print()
print('  访问 tile 17-23 (Row 2):')
print('    → 命中')
print()
print('  访问 tile 24 (Row 3):')
print('    → 激活 Row 3 → 第 4 次 activation')
print()
print('  访问 tile 25-31 (Row 3):')
print('    → 命中')
print()
print(f'总计: {non_crossing_acts} 次 row activation')
print()

print('【7. 公式含义总结】')
print('='*100)
print()
print('non_crossing_acts = ceil(non_crossing_count / tiles_per_row)')
print()
print('物理含义:')
print(f'  - {non_crossing_count} 个 non-crossing tiles')
print(f'  - 每 {tiles_per_row} 个 tiles 共享一个 row')
print(f'  - 所以这些 tiles 分布在 ceil({non_crossing_count}/{tiles_per_row}) = {non_crossing_acts} 个 rows 中')
print(f'  - 单次顺序遍历这 {num_tiles} 个 tiles，需要 {non_crossing_acts} 次 row activation')
print()
print('关键假设:')
print('  这个值假设的是【单次顺序遍历】，即 tile 0 → tile 1 → ... → tile 31')
print('  并且是【连续访问】，同一 row 内的 tiles 之间不会被其他 row 的访问打断')
