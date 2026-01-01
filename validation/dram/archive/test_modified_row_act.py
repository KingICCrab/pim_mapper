"""测试修改后的 Input Row Activation"""

from pim_optimizer.arch.pim_arch import PIMArchitecture
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer import PIMOptimizer

arch = PIMArchitecture()
workload = ConvWorkload(name='test', N=1, K=8, C=8, P=4, Q=4, R=3, S=3)

optimizer = PIMOptimizer(arch, verbose=False)
result = optimizer.optimize([workload])
model = optimizer.model

print('=== 修改后的 Input Row Activation ===')
print()

# 各个模式
var = model.getVarByName('INPUT_ROW_ACT_SEQ_(0)')
print(f'Sequential 模式: {var.X:.2f}' if var else 'Sequential 模式: N/A')

var = model.getVarByName('INPUT_ROW_ACT_RA_(0)')
print(f'Row-aligned 模式: {var.X:.2f}' if var else 'Row-aligned 模式: N/A')

var = model.getVarByName('INPUT_TOTAL_ROW_ACT_(0)')
print(f'最终选择: {var.X:.2f}' if var else '最终选择: N/A')

# Layout choice
seq = model.getVarByName('X_LAYOUT_(0,0,sequential)')
ra = model.getVarByName('X_LAYOUT_(0,0,row_aligned)')
layout = "Sequential" if seq and seq.X > 0.5 else "Row-aligned"
print(f'\n选择的布局: {layout}')

# Cycles breakdown
print('\n=== Row Activation Cycles 分解 ===')
var = model.getVarByName('ROW_ACTS_CYCLES_(0,0)')
print(f'Input: {var.X:.2f}' if var else 'Input: N/A')

var = model.getVarByName('ROW_ACTS_CYCLES_(0,1)')
print(f'Weight: {var.X:.2f}' if var else 'Weight: N/A')

var = model.getVarByName('ROW_ACTS_CYCLES_(0,2)')
print(f'Output: {var.X:.2f}' if var else 'Output: N/A')

# Total DRAM cycles
print('\n=== 总结 ===')
var = model.getVarByName('V_mem_cycles_(0,3)')
print(f'DRAM Cycles: {var.X:.2f}' if var else 'DRAM Cycles: N/A')

var = model.getVarByName('LATENCY')
print(f'总 LATENCY: {var.X:.2f}' if var else '总 LATENCY: N/A')

# 对比修改前后
print('\n=== 对比 ===')
print('修改前 (Row-aligned only):')
print('  Input Row Activation: 178.72')
print('  Input Cycles: 4467.93')
print('  总 DRAM Cycles: 8048')
print()
print('修改后 (Sequential mode selected):')
var_seq = model.getVarByName('INPUT_ROW_ACT_SEQ_(0)')
var_cyc = model.getVarByName('ROW_ACTS_CYCLES_(0,0)')
var_dram = model.getVarByName('V_mem_cycles_(0,3)')
if var_seq and var_cyc and var_dram:
    print(f'  Input Row Activation: {var_seq.X:.2f}')
    print(f'  Input Cycles: {var_cyc.X:.2f}')
    print(f'  总 DRAM Cycles: {var_dram.X:.2f}')
