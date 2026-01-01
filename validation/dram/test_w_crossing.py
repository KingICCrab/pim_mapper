#!/usr/bin/env python3
"""
测试添加 W 方向 block crossing 后的 ILP 计算
"""
import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.workload.conv import ConvWorkload

# ResNet-L1 workload
workload = ConvWorkload(
    R=7, S=7, P=56, Q=56, C=3, K=64, N=1,
    stride=(1, 1), dilation=(1, 1), path="ResNet-L1"
)

# Run optimization
opt = PIMOptimizer(
    arch_file='/Users/haochenzhao/Projects/pim_optimizer/examples/configs/arch.yaml',
    verbose=False,
    time_limit=60.0
)

print("=" * 80)
print("运行优化...")
print("=" * 80)

result = opt.optimize([workload])

print("\n" + "=" * 80)
print("优化完成，检查 Input Block Crossing 变量")
print("=" * 80)

# Check if new variables exist
model = opt.model
vars_obj = opt.vars
arch = opt.arch
w = 0

print("\n1. 检查 H 方向 crossing 变量:")
h_crossing_found = False
for var in model.getVars():
    if 'z_ibc_(' in var.VarName and '(0,' in var.VarName and var.X > 0.5:
        print(f"  {var.VarName} = {var.X}")
        h_crossing_found = True
if not h_crossing_found:
    print("  (未找到 H 方向 crossing 变量)")

print("\n2. 检查 W 方向 crossing 变量:")
w_crossing_found = False
for var in model.getVars():
    if 'z_ibc_w_' in var.VarName and '(0,' in var.VarName and var.X > 0.5:
        print(f"  {var.VarName} = {var.X}")
        w_crossing_found = True
if not w_crossing_found:
    print("  (未找到 W 方向 crossing 变量)")

print("\n3. 检查 block crossing 相关变量:")
for var in model.getVars():
    vn = var.VarName
    if 'input_block_crossing' in vn and '(0)' in vn:
        print(f"  {vn} = {var.X:.2f}")

print("\n4. 检查最终 row activations:")
for var in model.getVars():
    vn = var.VarName
    if 'row_acts_cycles_(0,0)' in vn:
        print(f"  {vn} = {var.X:.2f}")
        print(f"  对应 row acts = {var.X / 25.0:.2f}")

print("\n5. 分析 crossing count:")
# Get selected block_h, block_w
h_divisors = workload.hw_divisors['H']
w_divisors = workload.hw_divisors['W']

selected_h_idx = None
selected_w_idx = None

for i in range(len(h_divisors)):
    var = vars_obj.rowbuf_input_block_h.get((w, i))
    if var and var.X > 0.5:
        selected_h_idx = i
        print(f"  Selected H_rb index: {i}, value: {h_divisors[i]}")

for j in range(len(w_divisors)):
    var = vars_obj.rowbuf_input_block_w.get((w, j))
    if var and var.X > 0.5:
        selected_w_idx = j
        print(f"  Selected W_rb index: {j}, value: {w_divisors[j]}")

# Get selected P, Q, R, S factors at DRAM level
dram_level = arch.mem_idx.get("LocalDRAM")
s_temporal = 1

print(f"\n6. DRAM level tiling factors:")
for dim_idx, dim_name in enumerate(['R', 'S', 'P', 'Q']):
    divisors = workload.divisors[dim_idx]
    for factor_idx, factor in enumerate(divisors):
        var = vars_obj.xb.get((w, dram_level, s_temporal, dim_idx, factor_idx))
        if var and var.X > 0.5:
            tile_size = workload.bounds[dim_idx] // factor
            print(f"  {dim_name}: factor={factor}, tile_size={tile_size}")

print("\n" + "=" * 80)
print("完成")
print("=" * 80)
