"""
检查 Buffer 容量约束
"""
import yaml
import numpy as np
from pim_optimizer.arch import PIMArchitecture
from pim_optimizer.workload import ConvWorkload
from pim_optimizer import PIMOptimizer

# 架构
with open('examples/configs/arch.yaml', 'r') as f:
    arch_config = yaml.safe_load(f)
arch = PIMArchitecture(arch_config)

# Buffer 容量
buf_caps = {
    0: 64,    # PELocalBuffer
    1: 256,   # GlobalBuffer  
    2: 1024,  # RowBuffer
}
element_bytes = 1

print('=' * 80)
print('架构信息')
print('=' * 80)

print('\n【PE Array】')
print(f'  PE 阵列尺寸: {arch.pe_array.dim.h} × {arch.pe_array.dim.w} = {arch.pe_array.dim.h * arch.pe_array.dim.w} PEs')

print('\n【Memory Hierarchy】')
for i in range(arch.num_mems):
    name = arch.mem_name[i]
    entries = arch.mem_entries[i]
    entry_bytes = arch.entry_bytes
    size = entries * entry_bytes
    print(f'  Level {i}: {name}')
    print(f'    容量: {size} bytes ({size/1024:.2f} KB)' if size > 0 else f'    容量: 无限 (DRAM)')
    print(f'    存储 [Input, Weight, Output]: {arch.mem_stores_datatype[i]}')

print('\n【DRAM 参数】')
print(f'  Banks: {arch.dram_num_banks}')
print(f'  Row Buffer Size: {arch.dram_bank_row_buffer_size} bytes')
print(f'  Element Bytes: {element_bytes}')

workloads_config = [
    {'name': 'tiny', 'R': 3, 'S': 3, 'P': 4, 'Q': 4, 'C': 8, 'K': 8, 'N': 1},
    {'name': 'small', 'R': 3, 'S': 3, 'P': 14, 'Q': 14, 'C': 32, 'K': 32, 'N': 1},
    {'name': 'medium_1x1', 'R': 1, 'S': 1, 'P': 14, 'Q': 14, 'C': 64, 'K': 64, 'N': 1},
]

print('\n' + '=' * 80)
print('Buffer 容量检查')
print('=' * 80)

for wc in workloads_config:
    print(f"\n{'='*80}")
    print(f"Workload: {wc['name']}")
    print(f"  Dimensions: R={wc['R']}, S={wc['S']}, P={wc['P']}, Q={wc['Q']}, C={wc['C']}, K={wc['K']}, N={wc['N']}")
    print('='*80)
    
    workload = ConvWorkload(
        R=wc['R'], S=wc['S'], P=wc['P'], Q=wc['Q'],
        C=wc['C'], K=wc['K'], N=wc['N']
    )
    
    optimizer = PIMOptimizer(arch=arch, verbose=False)
    result = optimizer.optimize([workload], enable_row_activation=True)
    
    if not result.mappings:
        print(f'  优化失败')
        continue
    
    vars = optimizer.vars
    model = optimizer.model
    
    # 计算每层的累积 tile size
    for m in range(3):  # 只检查 PE, GlobalBuffer, RowBuffer
        mem_name = arch.mem_name[m]
        capacity = buf_caps[m]
        
        print(f"\n【{mem_name} (Level {m}) - 容量: {capacity} bytes】")
        
        # 计算到该层的累积 tile
        accum = {j: 1 for j in range(7)}
        for mm in range(m + 1):
            if mm == 0:
                s_range = 4
            else:
                s_range = 1
            for s in range(s_range):
                for j, divs in enumerate(workload.divisors):
                    for i, div in enumerate(divs):
                        if vars.xb[0, mm, s, j, i].X > 0.5:
                            accum[j] *= div
        
        R_t, S_t, P_t, Q_t, C_t, K_t, N_t = [accum[j] for j in range(7)]
        
        # Input tile size (with halo)
        H_in = P_t + R_t - 1
        W_in = Q_t + S_t - 1
        input_tile = H_in * W_in * C_t * N_t * element_bytes
        
        # Weight tile size
        weight_tile = R_t * S_t * C_t * K_t * element_bytes
        
        # Output tile size
        output_tile = P_t * Q_t * K_t * N_t * element_bytes
        
        stores = arch.mem_stores_datatype[m]
        total = 0
        details = []
        if stores[0]: 
            total += input_tile
            details.append(f'Input:{input_tile}')
        if stores[1]: 
            total += weight_tile
            details.append(f'Weight:{weight_tile}')
        if stores[2]: 
            total += output_tile
            details.append(f'Output:{output_tile}')
        
        print(f'  累积 Tile: R={R_t}, S={S_t}, P={P_t}, Q={Q_t}, C={C_t}, K={K_t}, N={N_t}')
        print(f'  Input Tile ({H_in}×{W_in}×{C_t}×{N_t}): {input_tile} bytes' + (' ← stored' if stores[0] else ' (not stored)'))
        print(f'  Weight Tile ({R_t}×{S_t}×{C_t}×{K_t}): {weight_tile} bytes' + (' ← stored' if stores[1] else ' (not stored)'))
        print(f'  Output Tile ({P_t}×{Q_t}×{K_t}×{N_t}): {output_tile} bytes' + (' ← stored' if stores[2] else ' (not stored)'))
        print(f'  总需求: {" + ".join(details)} = {total} bytes')
        print(f'  容量: {capacity} bytes')
        
        if total <= capacity:
            print(f'  ✓ 满足容量约束 ({total} ≤ {capacity})')
        else:
            print(f'  ✗ 超出容量! ({total} > {capacity})')
