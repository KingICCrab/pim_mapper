#!/usr/bin/env python3
"""
调试：检查 ILP 内部使用的所有参数
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')

from pim_optimizer.arch.pim_arch import PIMArchitecture
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer import PIMOptimizer
from pim_optimizer.model.expressions import compute_unique_input_size
from pim_optimizer.model.row_activation import compute_dram_row_crossing_ratio, compute_input_block_crossing_ratio


def debug_params(name, wl_params):
    """调试参数"""
    
    print("=" * 80)
    print(f"Workload: {name}")
    print("=" * 80)
    
    workload = ConvWorkload(**wl_params)
    arch = PIMArchitecture()
    
    # 检查 arch 参数
    print(f"\n【Architecture 参数】")
    
    # DRAM level index
    dram_level = arch.mem_idx.get("LocalDRAM")
    rowbuf_level = arch.mem_idx.get("RowBuffer")
    print(f"  DRAM level: {dram_level}")
    print(f"  RowBuffer level: {rowbuf_level}")
    
    # DRAM banks
    if hasattr(arch, 'mem_num_banks'):
        print(f"  mem_num_banks: {arch.mem_num_banks}")
        dram_banks = arch.mem_num_banks[dram_level]
        print(f"  DRAM banks: {dram_banks}")
    
    # Row buffer size
    if hasattr(arch, 'mem_row_buffer_size'):
        print(f"  mem_row_buffer_size: {arch.mem_row_buffer_size}")
        rb_size = arch.mem_row_buffer_size[rowbuf_level]
        print(f"  RowBuffer size: {rb_size}")
    
    # mem_entries
    print(f"  mem_entries: {arch.mem_entries}")
    rb_entries = arch.mem_entries[rowbuf_level]
    print(f"  RowBuffer entries: {rb_entries}")
    
    # element_bits
    if hasattr(arch, 'element_bits_per_dtype'):
        print(f"  element_bits_per_dtype: {arch.element_bits_per_dtype}")
    if hasattr(arch, 'default_element_bits'):
        print(f"  default_element_bits: {arch.default_element_bits}")
    
    # 计算 row_buffer_size_bytes (和 row_activation.py 一致的逻辑)
    row_buffer_size_bytes = None
    if hasattr(arch, "mem_row_buffer_size"):
        rb_size = arch.mem_row_buffer_size[rowbuf_level]
        if rb_size not in (None, 0):
            row_buffer_size_bytes = float(rb_size)
        else:
            rb_entries = arch.mem_entries[rowbuf_level]
            if rb_entries not in (None, 0, -1):
                row_buffer_size_bytes = float(rb_entries)
    
    if row_buffer_size_bytes is None:
        row_buffer_size_bytes = 1024.0
    
    print(f"\n  计算的 row_buffer_size_bytes: {row_buffer_size_bytes}")
    
    # 运行优化器
    optimizer = PIMOptimizer(arch, verbose=False)
    result = optimizer.optimize([workload])
    model = optimizer.model
    
    # 尝试获取模型中的常数
    print(f"\n【模型约束】")
    
    # 获取所有以 C_row_act_input 开头的约束
    for constr in model.getConstrs():
        if 'row_act_input' in constr.ConstrName.lower():
            print(f"  {constr.ConstrName}")
    
    print()


def main():
    test_workloads = [
        {"name": "tiny", "N": 1, "K": 8, "C": 8, "P": 4, "Q": 4, "R": 3, "S": 3},
    ]
    
    for wl in test_workloads:
        name = wl.pop('name')
        wl['name'] = name
        debug_params(name, wl)


if __name__ == "__main__":
    main()
