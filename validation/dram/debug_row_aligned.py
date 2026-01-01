"""Debug row_aligned implementation."""
import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

from validation.dram.trace_generator import TraceGenerator, DRAMConfig

# 检查 small workload 的布局信息
class MockMapping:
    def __init__(self):
        self.loop_bounds = {
            0: {'temporal': {4: 1, 5: 1, 6: 1}},  # Level 0
            1: {'temporal': {4: 1, 5: 1, 6: 1}},  # Level 1
            2: {'temporal': {4: 1, 5: 1, 6: 1}},  # Level 2 - RowBuffer
            3: {'temporal': {4: 16, 5: 16, 6: 1}}  # Level 3 - DRAM
        }
        self.permutation = {2: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}}
        self.layout = {0: 'row_aligned', 1: 'sequential', 2: 'sequential'}
        self.tile_info = {'block_h': 10, 'block_w': 10}
        self.bypass = {}
    
    def get_tile_size(self, level, dim):
        tile = 1
        for m in range(level + 1):
            if m in self.loop_bounds:
                for key in self.loop_bounds[m]:
                    tile *= self.loop_bounds[m][key].get(dim, 1)
        return tile
    
    def get_loop_order(self, level):
        return list(range(7))

class MockWorkload:
    N, K, C, P, Q, R, S = 1, 16, 16, 8, 8, 3, 3
    input_size = {'H': 10, 'W': 10}
    stride = (1, 1)
    dilation = (1, 1)

gen = TraceGenerator(DRAMConfig())
layout_info = gen._compute_data_layouts(MockMapping(), MockWorkload(), 10, 10)

print('=== Layout Info ===')
print(f'block_h={layout_info["block_h"]}, block_w={layout_info["block_w"]}')
print(f'block_size={layout_info["block_size"]}')
print(f'input_block_stride={layout_info["input_block_stride"]}')
print(f'nc_slice_stride={layout_info["nc_slice_stride"]}')
print(f'num_blocks_h={layout_info["num_blocks_h"]}, num_blocks_w={layout_info["num_blocks_w"]}')
print(f'row_size={layout_info["row_size"]}')
print(f'input_layout={layout_info["input_layout"]}')
print()

# 计算一些地址示例
print('=== Address Examples ===')
row_size = 1024
block_h, block_w = 10, 10
num_blocks_w = 1
nc_slice_stride = layout_info['nc_slice_stride']
input_block_stride = layout_info['input_block_stride']

print(f'nc_slice_stride = {nc_slice_stride}')
print(f'input_block_stride = {input_block_stride}')
print()

for c in [0, 1, 10, 15]:
    for h in [0, 9]:
        for w in [0, 9]:
            n = 0
            block_idx_h = h // block_h
            block_idx_w = w // block_w
            offset_h = h % block_h
            offset_w = w % block_w
            block_idx = block_idx_h * num_blocks_w + block_idx_w
            offset_in_block = offset_h * block_w + offset_w  # W inner
            
            idx = (n * 16 + c) * nc_slice_stride + block_idx * input_block_stride + offset_in_block
            row = idx // row_size
            col = idx % row_size
            print(f'Input[n={n},c={c:2},h={h},w={w}] -> idx={idx:5}, row={row:2}, col={col:3}')
    print()

# 问题分析
print('=== Problem Analysis ===')
print(f'Each (n,c) slice size = {num_blocks_w * 1 * block_h * block_w} elements')
print(f'Each (n,c) slice stride = {nc_slice_stride} elements')
print(f'row_size = {row_size} elements')
print()
print('For row_aligned to work correctly:')
print('  - Each RowBuffer tile should start at row boundary')
print('  - RowBuffer tile = block_h × block_w × C_rb × N_rb')
print()
print('Current issue:')
print('  - nc_slice_stride = 100 (not aligned to row_size)')
print('  - Different (n,c) slices cross row boundaries')
