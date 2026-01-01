#!/usr/bin/env python3
"""
测试强制将 C=8 放在 RowBuffer 层的效果
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')

from pim_optimizer.arch.pim_arch import PIMArchitecture
from pim_optimizer.workload.conv import ConvWorkload
from pim_optimizer import PIMOptimizer
import gurobipy as gp

def test_default_tiling():
    """测试默认 tiling"""
    wl = ConvWorkload(name="tiny", N=1, K=8, C=8, P=4, Q=4, R=3, S=3)
    arch = PIMArchitecture.from_yaml('/Users/haochenzhao/Projects/pim_optimizer/examples/configs/arch.yaml')
    
    optimizer = PIMOptimizer(arch, verbose=False)
    result = optimizer.optimize([wl])
    model = optimizer.model
    
    print("=== Default Tiling ===")
    print(f"Objective (latency) = {model.ObjVal:.2f}")
    
    print("\nTiling at DRAM (m=3):")
    dim_names = ["R", "S", "P", "Q", "C", "K", "N"]
    for j in range(7):
        for i, d in enumerate(wl.divisors[j]):
            var = optimizer.vars.xb[0, 3, 1, j, i]
            if var.X > 0.5:
                print(f"  {dim_names[j]}={d}")
    
    print("\nRow activation cycles:")
    for t in range(3):
        tensor = ["Input", "Weight", "Output"][t]
        cycles_var = model.getVarByName(f"ROW_ACTS_CYCLES_(0,{t})")
        if cycles_var:
            print(f"  {tensor}: {cycles_var.X:.2f}")
    
    dram_cycles = model.getVarByName("V_mem_cycles_(0,3)")
    print(f"\nTotal DRAM cycles: {dram_cycles.X:.2f}")
    
    return model.ObjVal


def test_forced_c8_at_rowbuffer():
    """测试强制 C=8 在 RowBuffer"""
    wl = ConvWorkload(name="tiny", N=1, K=8, C=8, P=4, Q=4, R=3, S=3)
    arch = PIMArchitecture.from_yaml('/Users/haochenzhao/Projects/pim_optimizer/examples/configs/arch.yaml')
    
    optimizer = PIMOptimizer(arch, verbose=False)
    
    # 先调用 optimize 来初始化模型
    result = optimizer.optimize([wl])
    
    # 然后添加约束并重新求解
    model = optimizer.model
    vars = optimizer.vars
    
    # 找到 C 维度的变量
    # C's divisors are [1, 2, 4, 8], indices 0, 1, 2, 3
    # 强制 C=8 在 RowBuffer (m=2, s=1, j=4, i=3)
    # 强制 C=1 在 DRAM (m=3, s=1, j=4, i=0)
    
    print("\n=== Attempting to force C=8 at RowBuffer ===")
    
    # 固定变量值
    vars.xb[0, 2, 1, 4, 3].LB = 1
    vars.xb[0, 2, 1, 4, 3].UB = 1
    vars.xb[0, 3, 1, 4, 0].LB = 1
    vars.xb[0, 3, 1, 4, 0].UB = 1
    
    # 重新求解
    model.reset()
    model.optimize()
    
    print(f"Status: {model.Status}")
    if model.Status == gp.GRB.OPTIMAL:
        print(f"Objective (latency) = {model.ObjVal:.2f}")
        
        print("\nTiling at RowBuffer (m=2):")
        dim_names = ["R", "S", "P", "Q", "C", "K", "N"]
        for j in range(7):
            for i, d in enumerate(wl.divisors[j]):
                var = vars.xb[0, 2, 1, j, i]
                if var.X > 0.5:
                    print(f"  {dim_names[j]}={d}")
        
        print("\nTiling at DRAM (m=3):")
        for j in range(7):
            for i, d in enumerate(wl.divisors[j]):
                var = vars.xb[0, 3, 1, j, i]
                if var.X > 0.5:
                    print(f"  {dim_names[j]}={d}")
        
        print("\nRow activation cycles:")
        for t in range(3):
            tensor = ["Input", "Weight", "Output"][t]
            cycles_var = model.getVarByName(f"ROW_ACTS_CYCLES_(0,{t})")
            if cycles_var:
                print(f"  {tensor}: {cycles_var.X:.2f}")
        
        dram_cycles = model.getVarByName("V_mem_cycles_(0,3)")
        if dram_cycles:
            print(f"\nTotal DRAM cycles: {dram_cycles.X:.2f}")
        
        return model.ObjVal
    else:
        print("Model infeasible!")
        model.computeIIS()
        print("\nIIS Constraints:")
        for c in model.getConstrs():
            if c.IISConstr:
                print(f"  {c.ConstrName}")
        return None


if __name__ == "__main__":
    default_obj = test_default_tiling()
    forced_obj = test_forced_c8_at_rowbuffer()
    
    if forced_obj:
        print("\n" + "="*50)
        print(f"Default latency: {default_obj:.2f}")
        print(f"Forced C=8 at RowBuffer latency: {forced_obj:.2f}")
        print(f"Improvement: {(default_obj - forced_obj) / default_obj * 100:.1f}%")
