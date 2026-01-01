#!/usr/bin/env python3
"""分析 crossing ratio 模型的根本问题"""

print("=" * 70)
print("Crossing Ratio 模型的根本问题")
print("=" * 70)

print("""
【当前模型的假设】

crossing ratio 计算假设：
  每个 tile 独立访问，如果 crossing，需要 2 次 row activation

row_act = num_tiles × (1 + crossing_ratio)

【问题：相邻 tile 有重叠！】

滑动窗口访问时，相邻 tile 共享数据：
  Tile 0: [0, 6)
  Tile 1: [4, 10)  ← 与 Tile 0 重叠 [4, 6)
  Tile 2: [8, 14)  ← 与 Tile 1 重叠 [8, 10)
  ...

Tile 1 的 [4, 6) 部分已经被 Tile 0 读取过了！
不需要额外的 row activation！
""")

print("=" * 70)
print("正确的 Row Activation 计算方式")
print("=" * 70)

print("""
【正确的模型】

Row activation 应该计算的是：
  "需要激活多少个 UNIQUE 的 DRAM rows"

而不是：
  "每个 tile 访问需要多少次 activation"

【例子】

假设：
  - 总 input 大小 = 100 bytes
  - row_buffer_size = 1024 bytes
  - 4 个 banks

无论有多少个 tile，无论 crossing ratio 是多少：
  - 实际需要的 row activation = ceil(100 / 1024) = 1 次 (per bank)
  - 如果数据分布在 4 个 bank，每个 bank 1 次

【Sequential Layout 的 Row Activation】

对于 Sequential 布局（数据连续存储）：

  row_act_per_bank = ceil(data_bytes_per_bank / row_buffer_size)

这与 crossing ratio 无关！
""")

print("=" * 70)
print("为什么 Crossing Ratio 是错误的概念？")
print("=" * 70)

print("""
【Crossing Ratio 的本意】

Crossing ratio 想要捕捉的是：
  "滑动窗口访问时，tile 跨越 block/row 边界的概率"

【但它忽略了】

1. 相邻 tile 有数据重叠 → 重叠部分已经被激活
2. Row buffer 缓存 → 最近激活的 row 可能还在 buffer 中
3. 顺序访问 → 整体只需要激活 ceil(total/row_size) 个 row

【Crossing 真正影响什么？】

Crossing 影响的是 Row-Aligned 布局的访问效率：
  - 如果 block 对齐到 DRAM row
  - 那么一个 block 内的访问只需要 1 次 row activation
  - 跨越 block 意味着跨越 DRAM row

但对于 Sequential 布局：
  - 数据是连续存储的
  - row activation = ceil(total_data / row_size)
  - 不存在 "crossing" 的概念
""")

print("=" * 70)
print("Tiny Workload 的正确计算")
print("=" * 70)

input_entries = 288
element_bytes = 2
input_bytes = input_entries * element_bytes
row_buffer_size = 1024
num_banks = 4

bytes_per_bank = input_bytes / num_banks
rows_per_bank = -(-int(bytes_per_bank) // row_buffer_size)  # ceil division

print(f"""
【Input 数据】
  entries = {input_entries}
  bytes = {input_bytes}
  
【DRAM 参数】
  row_buffer_size = {row_buffer_size} bytes
  num_banks = {num_banks}
  
【正确的 Row Activation】
  bytes_per_bank = {input_bytes} / {num_banks} = {bytes_per_bank} bytes
  rows_per_bank = ceil({bytes_per_bank} / {row_buffer_size}) = {rows_per_bank}
  
  总 row_act = {rows_per_bank} rows × {num_banks} banks = {rows_per_bank * num_banks} row activations

【如果我们要把这转换成 "entries/bank" 的单位】
  
  实际上，row_act 应该是 "次数"，不是 "entries"
  
  如果要用 entries 表示：
    row_act_entries = {rows_per_bank} × {row_buffer_size // element_bytes} = {rows_per_bank * row_buffer_size // element_bytes} entries/bank
    
  或者更简单：
    每个 bank 只有 {input_entries // num_banks} entries 的数据
    只需要 1 次 row activation
    row_act = {input_entries // num_banks} entries/bank (如果按数据量算)
""")

print("=" * 70)
print("模型应该如何修改？")
print("=" * 70)

print("""
【Sequential 布局的 Row Activation】

应该是：
  row_act = ceil(data_bytes / row_buffer_size) × row_buffer_size / element_bytes
  
或者更直接：
  row_act = data_entries  (如果数据能装进 row buffer)
  row_act = ceil(data_entries × element_bytes / row_buffer_size) × (row_buffer_size / element_bytes)  (否则)

【不需要 crossing ratio！】

Crossing ratio 是 Row-Aligned 布局的概念，
用于估计因为 tile 不对齐导致的额外 row activation。

但对于 Sequential 布局，数据是连续的，
只需要知道总数据量和 row buffer 大小。
""")
