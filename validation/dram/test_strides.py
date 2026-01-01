import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')
from pim_optimizer import PIMOptimizer
from pim_optimizer.workload import ConvWorkload

# 创建 small workload
workload = ConvWorkload('small', P=8, Q=8, C=16, K=16, R=3, S=3, N=1, stride=(1,1), dilation=(1,1))

# 运行优化器
optimizer = PIMOptimizer()
result = optimizer.optimize([workload], objective='latency')
mapping = result.mappings[0]

# 模拟 _build_loop_order
DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6
DIM_NAMES = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
input_dims = [DIM_N, DIM_C, DIM_Q, DIM_S, DIM_P, DIM_R]

print("Permutation at Level 3:", mapping.permutation.get(3, {}))
print("Permutation at Level 2:", mapping.permutation.get(2, {}))

loop_order = []
for m in [3, 2]:
    if m not in mapping.permutation:
        continue
    perm = mapping.permutation[m]
    sorted_perms = sorted(perm.items(), key=lambda x: x[0], reverse=True)
    print(f'Level {m} sorted_perms: {[(pos, DIM_NAMES[dim]) for pos, dim in sorted_perms]}')
    for p_level, dim in sorted_perms:
        if dim in input_dims:
            loop_order.append((m, dim))

print('input_loop_order:', [(lv, DIM_NAMES[d]) for lv, d in loop_order])
input_l3_dims_in_perm = [(lv, d) for (lv, d) in loop_order if lv == 3]
print('input_l3_dims_in_perm:', [(lv, DIM_NAMES[d]) for lv, d in input_l3_dims_in_perm])
print('reversed:', [(lv, DIM_NAMES[d]) for lv, d in reversed(input_l3_dims_in_perm)])

# 模拟 stride 计算
P_l3, Q_l3, C_l3, N_l3 = 2, 1, 8, 1  # 从 level3_factors
input_l3_tile_counts = {DIM_P: P_l3, DIM_Q: Q_l3, DIM_C: C_l3, DIM_N: N_l3}
input_aligned_tile_size = 1024

input_strides = {}
stride = input_aligned_tile_size
processed_dims = set()

print("\n--- Stride Calculation ---")
for (lv, dim) in reversed(input_l3_dims_in_perm):
    print(f'Processing ({lv}, {DIM_NAMES[dim]}): stride[({lv}, {DIM_NAMES[dim]})] = {stride}')
    if dim in input_l3_tile_counts:
        input_strides[(3, dim)] = stride
        stride *= input_l3_tile_counts[dim]
        print(f'  -> multiply by tile_count={input_l3_tile_counts[dim]}, new stride = {stride}')
        processed_dims.add(dim)

# Add remaining dims
print("\n--- Adding remaining dims ---")
for dim in [DIM_Q, DIM_P, DIM_C, DIM_N]:
    if dim not in processed_dims and dim in input_l3_tile_counts:
        print(f'Adding remaining dim {DIM_NAMES[dim]}: stride = {stride}')
        input_strides[(3, dim)] = stride
        stride *= input_l3_tile_counts[dim]
        processed_dims.add(dim)

print("\n--- Final strides ---")
for (lv, dim), s in input_strides.items():
    print(f'  stride[({lv}, {DIM_NAMES[dim]})] = {s}')
