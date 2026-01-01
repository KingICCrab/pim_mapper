"""
检查 Mapping 结果的合理性
"""
import yaml
import numpy as np
from pim_optimizer.arch import PIMArchitecture
from pim_optimizer.workload import ConvWorkload
from pim_optimizer import PIMOptimizer

arch = PIMArchitecture.from_yaml('examples/configs/arch.yaml')

print('=' * 80)
print('架构约束')
print('=' * 80)
print(f'PE Array: {arch.pe_array.dim.h} × {arch.pe_array.dim.w} = {arch.pe_array.dim.h * arch.pe_array.dim.w} PEs')

def stores_str(stores):
    types = []
    if stores[0]: types.append('Input')
    if stores[1]: types.append('Weight')
    if stores[2]: types.append('Output')
    return ' + '.join(types) if types else 'None'

print(f'PELocalBuffer: {arch.mem_entries[0]} entries per PE ({stores_str(arch.mem_stores_datatype[0])})')
print(f'GlobalBuffer: {arch.mem_entries[1]} entries ({stores_str(arch.mem_stores_datatype[1])})')
print(f'RowBuffer: {arch.mem_entries[2]} entries ({stores_str(arch.mem_stores_datatype[2])})')
print()

workloads = [
    ('tiny', ConvWorkload(R=3, S=3, P=4, Q=4, C=8, K=8, N=1)),
    ('small', ConvWorkload(R=3, S=3, P=14, Q=14, C=32, K=32, N=1)),
    ('medium_1x1', ConvWorkload(R=1, S=1, P=14, Q=14, C=64, K=64, N=1)),
]

for name, workload in workloads:
    print('=' * 80)
    print(f'Workload: {name}')
    print(f'Dimensions: R={workload.R}, S={workload.S}, P={workload.P}, Q={workload.Q}, C={workload.C}, K={workload.K}, N={workload.N}')
    print('=' * 80)
    
    optimizer = PIMOptimizer(arch=arch, verbose=False)
    result = optimizer.optimize([workload], enable_row_activation=True)
    
    vars = optimizer.vars
    dim_names = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
    
    # 提取各层的 tile factors
    tiles = {}
    for m in range(4):
        tiles[m] = {}
        if m == 0:
            s_range = 4
            spatial_names = ['Temporal', 'H-spatial', 'W-spatial', 'Internal']
        else:
            s_range = 2  # s=0 (dummy/spatial=1), s=1 (temporal)
            spatial_names = ['Spatial(=1)', 'Temporal']
        
        for s, sname in enumerate(spatial_names):
            tiles[m][sname] = {}
            for j, divs in enumerate(workload.divisors):
                for i, div in enumerate(divs):
                    if vars.xb[0, m, s, j, i].X > 0.5:
                        tiles[m][sname][dim_names[j]] = div
    
    # 1. 检查 PE 并行度
    print('\n【1. PE 并行度检查】')
    h_parallel = 1
    w_parallel = 1
    for dim in dim_names:
        h_parallel *= tiles[0].get('H-spatial', {}).get(dim, 1)
        w_parallel *= tiles[0].get('W-spatial', {}).get(dim, 1)
    
    print(f'  H-spatial: {tiles[0].get("H-spatial", {})} = {h_parallel} PEs')
    print(f'  W-spatial: {tiles[0].get("W-spatial", {})} = {w_parallel} PEs')
    print(f'  约束: H ≤ {arch.pe_array.dim.h}, W ≤ {arch.pe_array.dim.w}')
    print(f'  结果: {"✓" if h_parallel <= arch.pe_array.dim.h and w_parallel <= arch.pe_array.dim.w else "✗"}')
    
    # 2. 检查 Buffer 容量
    print('\n【2. Buffer 容量检查】')
    for m, mem_name in [(0, 'PELocalBuffer'), (1, 'GlobalBuffer'), (2, 'RowBuffer')]:
        # 累积到该层的 tile
        accum = {d: 1 for d in dim_names}
        for mm in range(m + 1):
            # For PELocalBuffer (m=0): only accumulate Temporal factors (per-PE tile)
            # Spatial factors are distributed across PEs
            if mm == 0 and m == 0:
                # Only use Temporal for per-PE calculation
                sname = 'Temporal'
                if sname in tiles[mm]:
                    for d in dim_names:
                        accum[d] *= tiles[mm][sname].get(d, 1)
            else:
                for sname in tiles[mm]:
                    for d in dim_names:
                        accum[d] *= tiles[mm][sname].get(d, 1)
        
        R, S, P, Q, C, K, N = [accum[d] for d in dim_names]
        
        # 计算各 datatype tile size
        H_in, W_in = P + R - 1, Q + S - 1
        input_size = H_in * W_in * C * N
        weight_size = R * S * C * K
        output_size = P * Q * K * N
        
        capacity = arch.mem_entries[m]
        stores = arch.mem_stores_datatype[m]
        
        per_pe_note = " (per-PE)" if m == 0 else ""
        print(f'\n  {mem_name} (容量={capacity}{per_pe_note}):')
        print(f'    累积: R={R}, S={S}, P={P}, Q={Q}, C={C}, K={K}, N={N}')
        
        all_ok = True
        if stores[0]:
            ok = input_size <= capacity
            all_ok = all_ok and ok
            print(f'    Input:  ({H_in}×{W_in}×{C}×{N}) = {input_size} {"✓" if ok else "✗"}')
        if stores[1]:
            ok = weight_size <= capacity
            all_ok = all_ok and ok
            print(f'    Weight: ({R}×{S}×{C}×{K}) = {weight_size} {"✓" if ok else "✗"}')
        if stores[2]:
            ok = output_size <= capacity
            all_ok = all_ok and ok
            print(f'    Output: ({P}×{Q}×{K}×{N}) = {output_size} {"✓" if ok else "✗"}')
    
    # 3. 检查维度因子乘积
    print('\n【3. 维度因子完整性检查】')
    for j, dim in enumerate(dim_names):
        total = 1
        for m in range(4):
            for sname in tiles[m]:
                total *= tiles[m][sname].get(dim, 1)
        expected = workload.bounds[j]
        ok = total == expected
        print(f'  {dim}: 各层乘积={total}, workload={expected} {"✓" if ok else "✗"}')
    
    # 4. 检查 DRAM 循环对 reuse 的影响
    print('\n【4. DRAM 循环分析】')
    dram_tiles = tiles[3].get('Temporal', {})
    print(f'  DRAM Temporal: {dram_tiles}')
    
    # Input 相关维度: R, S, P, Q, C, N (不含 K)
    # Weight 相关维度: R, S, C, K (不含 P, Q, N)
    # Output 相关维度: P, Q, K, N (不含 R, S, C)
    
    input_irrelevant = dram_tiles.get('K', 1)
    weight_irrelevant = dram_tiles.get('P', 1) * dram_tiles.get('Q', 1) * dram_tiles.get('N', 1)
    output_irrelevant = dram_tiles.get('R', 1) * dram_tiles.get('S', 1) * dram_tiles.get('C', 1)
    
    print(f'  Input reuse (K iterations): {input_irrelevant}')
    print(f'  Weight reuse (P×Q×N iterations): {weight_irrelevant}')
    print(f'  Output reuse (R×S×C iterations): {output_irrelevant}')
    
    print()
