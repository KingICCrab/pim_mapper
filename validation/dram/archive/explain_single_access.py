#!/usr/bin/env python3
"""分析 tiny workload 的实际访问模式"""

print("=" * 70)
print("你的关键观察：只有 Tile 0 访问了所有 input！")
print("=" * 70)

print("""
【Tiny Workload 的 DRAM Tiling】

从 mapping_results.txt:
  DRAM (level 3):
    Temporal: R=1, S=1, P=1, Q=1, C=1, K=1, N=1
    
所有维度的 DRAM tiling factor 都是 1！

这意味着：
  - 整个 Input 只需要从 DRAM 读取 **一次**
  - 没有第二次、第三次读取
  - 不存在 "滑动窗口" 的多次访问
""")

print("=" * 70)
print("实际的 Input 访问情况")
print("=" * 70)

input_h = 6
input_w = 6
input_c = 8
input_total = input_h * input_w * input_c
element_bytes = 2
input_bytes = input_total * element_bytes

row_buffer_size = 1024

print(f"""
【Input 大小】
  H × W × C = {input_h} × {input_w} × {input_c} = {input_total} entries
  总字节数 = {input_total} × {element_bytes} = {input_bytes} bytes

【DRAM Row Buffer】
  row_buffer_size = {row_buffer_size} bytes

【是否需要 crossing？】
  {input_bytes} bytes < {row_buffer_size} bytes
  整个 Input 能装进一个 DRAM row！
  
  → 只需要 1 次 row activation
  → 不存在 crossing 问题！
""")

print("=" * 70)
print("模型的问题")
print("=" * 70)

print("""
【当前模型的计算】

模型计算 crossing ratio 时假设：
  - 有多个 tile 的滑动窗口访问
  - 每个 tile 可能跨越 block 边界
  - 计算 crossing_ratio = 0.667

【但实际情况是】

当 DRAM tiling 全是 1 时：
  - 只有 "Tile 0" 这一次访问
  - Tile 0 读取整个 input
  - 整个 input 能装进一个 DRAM row
  - 不存在滑动窗口，不存在 crossing！

【正确的 row activation】

对于 tiny workload:
  Input = 288 entries = 576 bytes
  能装进 1 个 DRAM row (1024 bytes)
  分布在 4 个 bank
  
  理想的 row_act = 288 entries / 4 banks = 72 entries/bank
  
  但模型计算了：
  - crossing_ratio = 0.667
  - 额外的 crossing cost = 72 × 0.667 × 2 ≈ 96 entries/bank
  - 总 row_act = 72 + 96 = 168 entries/bank ← 这是错的！
""")

print("=" * 70)
print("结论")
print("=" * 70)

print("""
【你的理解是正确的】

当 DRAM tiling 全是 1 时：
1. 整个数据只读取一次
2. 不存在滑动窗口的多次访问
3. 不应该计算 crossing ratio

【模型的 Bug】

Row-Aligned 模式的 crossing ratio 计算不适用于
"数据只读取一次" 的情况。

模型应该检测：
  如果 DRAM 所有 relevant 维度 tiling == 1:
    不计算 crossing，直接用 base row activation

【正确的结果应该是】

  row_act = Input_entries / num_banks
         = 288 / 4
         = 72 entries/bank
         
  而不是 168 entries/bank (包含错误的 crossing cost)
""")
