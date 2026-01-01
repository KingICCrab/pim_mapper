#!/usr/bin/env python3
"""
验证结果分析与结论

基于 ILP → UniNDP 验证流程的实验结果
"""

print("="*70)
print("ILP → UniNDP 验证实验结果")  
print("="*70)

# 实验数据
results = {
    "1000x1000": {
        "filtered": {"size": 213, "best": 740, "worst": 945, "ratio": 1.3, "median": 781, "std": 67},
        "full": {"size": 30480, "best": 1041, "worst": 15168, "ratio": 14.6, "median": 3686, "std": 3417},
    },
    "2000x2000": {
        "filtered": {"size": 333, "best": 2861, "worst": 3431, "ratio": 1.2, "median": 2898, "std": 177},
        "full": {"size": 39134, "best": 2861, "worst": 142777, "ratio": 49.9, "median": 18001, "std": 38945},
    },
    "4096x4096": {
        "filtered": {"size": 437, "best": 11528, "worst": 14210, "ratio": 1.2, "median": 12749, "std": 870},
        # full 空间数据不完整，但我们有之前的数据
        "full": {"size": 46053, "best": "~15000", "worst": "~120000", "ratio": "~8x"},
    },
}

print()
print("┌──────────────────────────────────────────────────────────────────────┐")
print("│                        实验数据汇总                                  │")
print("├─────────────┬────────────────────────┬────────────────────────────────┤")
print("│   工作负载   │    过滤空间(高利用率)    │     完整空间(含低利用率)        │")
print("├─────────────┼────────────────────────┼────────────────────────────────┤")

for workload, data in results.items():
    f = data['filtered']
    full = data['full']
    print(f"│ {workload:>11} │ 策略数: {f['size']:>6}         │ 策略数: {full['size']:>6}               │")
    print(f"│             │ 最优: {f['best']:>8} cycles  │ 最优: {full['best'] if isinstance(full['best'], int) else full['best']:>8} cycles      │")
    print(f"│             │ 最差: {f['worst']:>8} cycles  │ 最差: {full['worst'] if isinstance(full['worst'], int) else full['worst']:>8} cycles      │")
    print(f"│             │ 差距比: {f['ratio']:>5.1f}x          │ 差距比: {full['ratio'] if isinstance(full['ratio'], float) else full['ratio']:>8}         │")
    print("├─────────────┼────────────────────────┼────────────────────────────────┤")

print("└──────────────────────────────────────────────────────────────────────┘")

print()
print("="*70)
print("关键发现")
print("="*70)

print("""
1. 【过滤空间 vs 完整空间】
   
   过滤空间(只保留高硬件利用率策略):
   - 策略数: 213-437 (较小)
   - 性能差距: 1.2x-1.3x (很小!)
   - 所有策略性能都差不多
   
   完整空间(包含低利用率策略):
   - 策略数: 30K-46K (很大)
   - 性能差距: 14.6x-49.9x (巨大!)
   - 最差策略比最优慢 50 倍

2. 【关键洞见】
   
   如果 ILP 优化器能确保:
   ✓ 高硬件利用率 (使用所有 512 PU)
   ✓ 合理的数据布局
   
   那么在过滤空间内，策略选择的影响只有 ~20%
   
   但如果 ILP 选择了低利用率策略:
   ✗ 可能导致 50x 的性能损失!

3. 【之前的 "400x 差距" 来源】
   
   之前验证发现的巨大差距(165x-429x)来自:
   - UniNDP baseline 使用了 mkl_Input_to_row[0][1]==8 的筛选条件
   - 这个条件选择了一个极差的低利用率策略
   - 不是映射策略的差距，而是利用率的差距!

4. 【ILP 优化器的真正价值】
   
   主要价值: 确保高硬件利用率
   - 这可以带来 15x-50x 的性能提升
   
   次要价值: 在高利用率策略中选择最优
   - 这只能带来 ~20% 的额外提升
   
   结论: ILP 必须正确建模硬件利用率约束!
""")

print()
print("="*70)
print("对 ILP 优化器的建议")
print("="*70)

print("""
1. 【必须约束】硬件利用率
   - 确保 ch × ra × de × pu 的乘积 = 512 (全部 PU)
   - 或者尽可能接近完全利用

2. 【Cost Model 改进】
   当前 Cost Model 问题:
   - 使用固定 16% 效率，无法区分策略好坏
   
   建议改进:
   - 效率 = f(硬件利用率, 数据布局, 内存访问模式)
   - 低利用率策略应该有更高的惩罚

3. 【验证方法】
   正确的验证流程:
   a) ILP 输出映射参数 (ch_l, pu_k, pu_l 等)
   b) 转换为 UniNDP 策略格式
   c) 用 UniNDP 模拟 ILP 策略
   d) 比较: ILP cycles vs 过滤空间最优 cycles
   e) 目标: ILP 应该在过滤空间最优的 20% 以内
""")

print()
print("="*70)
print("下一步实施计划")
print("="*70)

print("""
Phase 1: 添加硬件利用率约束
  - 在 ILP 中添加约束: total_PUs_used >= 0.9 * max_PUs
  - 这是最重要的优化目标

Phase 2: 实现 ILP → UniNDP 映射转换
  - ILP 输出: loop_bounds (分块大小)
  - 转换为: partition (ch, ra, de, pu 分割)
  - 难点: ILP 的分块需要与 UniNDP 的硬件单元对应

Phase 3: 端到端验证
  - 运行 ILP 优化
  - 转换映射
  - UniNDP 模拟
  - 比较结果

预期结果:
  - ILP 映射 cycles ≈ 过滤空间最优 cycles (差距 < 20%)
  - ILP 映射 cycles << 完整空间平均 cycles (差距 > 5x)
""")
