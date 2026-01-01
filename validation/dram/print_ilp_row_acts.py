#!/usr/bin/env python3
"""
打印ILP模型中row activation的各个变量值
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer/src')

from pim_optimizer import PIMOptimizer
from pim_optimizer.workload import ConvWorkload

def main():
    # ResNet-L1 workload (same as analysis.txt)
    workload = ConvWorkload(
        name="ResNet-L1",
        R=7, S=7, P=56, Q=56, C=3, K=64, N=1,
        stride=(1, 1),
        dilation=(1, 1)
    )
    
    # Create optimizer
    optimizer = PIMOptimizer(verbose=True, time_limit=300)
    result = optimizer.optimize([workload])
    
    if not result.mappings:
        print("No valid mapping found!")
        return
    
    mapping = result.mappings[0]
    
    # 打印基本信息
    print("\n" + "=" * 80)
    print("ILP Row Activation 各变量详细分解")
    print("=" * 80)
    
    # 获取模型变量
    model = optimizer.model
    vars_obj = optimizer.vars
    w = 0
    
    dim_names = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']
    tensor_names = {0: "Input", 1: "Weight", 2: "Output"}
    
    print("\n最终 ILP 结果:")
    print(f"  Input:  {mapping.metrics.get('row_activations_input', 0):.4f}")
    print(f"  Weight: {mapping.metrics.get('row_activations_weight', 0):.4f}")
    print(f"  Output: {mapping.metrics.get('row_activations_output', 0):.4f}")
    print(f"  Total:  {mapping.metrics.get('row_activations', 0):.4f}")
    
    # 遍历所有变量打印 row_act 相关的
    print("\n" + "=" * 80)
    print("ILP 模型中的 Row Activation 变量")
    print("=" * 80)
    
    for t_id in range(3):
        t_name = tensor_names[t_id]
        print(f"\n{'='*60}")
        print(f"Tensor: {t_name}")
        print(f"{'='*60}")
        
        # 1. reuse_penalty
        var_name = f"reuse_penalty_({w},{t_id})"
        var = model.getVarByName(var_name)
        if var is not None:
            print(f"\n  1. Reuse Penalty: {var.X:.4f}")
            print(f"     (Π {{irrelevant dims with xj=1}} bound_j)")
        
        # 2. row_acts_row_aligned
        var_name = f"row_acts_row_aligned_({w},{t_id})"
        var = model.getVarByName(var_name)
        if var is not None:
            print(f"\n  2. Row Acts (Row-Aligned mode): {var.X:.4f}")
            print(f"     (Π {{all dims with xj=1}} bound_j)")
        
        # 3. base_row_acts (for sequential)
        var_name = f"base_row_acts_({w},{t_id})"
        var = model.getVarByName(var_name)
        if var is not None:
            print(f"\n  3. Base Row Acts (Sequential): {var.X:.4f}")
            print(f"     (non_crossing_acts + 2 × crossing_count × reuse_penalty)")
        
        # 4. outer_irr_product
        var_name = f"outer_irr_prod_({w},{t_id})"
        var = model.getVarByName(var_name)
        if var is not None:
            print(f"\n  4. Outer Irrelevant Product: {var.X:.4f}")
            print(f"     (Π {{outer irrelevant dims with xj=1}} bound_j)")
        
        # 5. row_acts (sequential final)
        var_name = f"row_acts_({w},{t_id})"
        var = model.getVarByName(var_name)
        if var is not None:
            print(f"\n  5. Row Acts (Sequential final): {var.X:.4f}")
            print(f"     (base_row_acts × outer_irr_product)")
        
        # 6. seq_part (conditional)
        var_name = f"dram_crossing_seq_({w},{t_id})"
        var = model.getVarByName(var_name)
        if var is not None:
            print(f"\n  6. Sequential Part (conditional): {var.X:.4f}")
            print(f"     ((1 - row_aligned) × row_acts_seq)")
        
        # 7. aligned_part (conditional)
        var_name = f"row_aligned_acts_({w},{t_id})"
        var = model.getVarByName(var_name)
        if var is not None:
            print(f"\n  7. Row-Aligned Part (conditional): {var.X:.4f}")
            print(f"     (row_aligned × row_acts_row_aligned)")
        
        # 8. Input Block Crossing (only for Input)
        if t_id == 0:
            var_name = f"input_block_crossing_acts_({w})"
            var = model.getVarByName(var_name)
            if var is not None:
                print(f"\n  8. Input Block Crossing Acts: {var.X:.4f}")
                print(f"     (2 × aux_ibc_rp)")
            
            var_name = f"selected_ibc_count_({w})"
            var = model.getVarByName(var_name)
            if var is not None:
                print(f"\n     Selected IBC Count: {var.X:.4f}")
            
            # Print aux variable
            var_name = f"aux_ibc_rp_({w})"
            var = model.getVarByName(var_name)
            if var is not None:
                print(f"     aux_ibc_rp (selected_count × reuse_penalty): {var.X:.4f}")
            
            # Print reuse_penalty for verification
            rp_var = model.getVarByName(f"reuse_penalty_({w},0)")
            if rp_var is not None:
                print(f"     reuse_penalty (Input): {rp_var.X:.4f}")
                
            # Manual calculation
            ibc_count_var = model.getVarByName(f"selected_ibc_count_({w})")
            if ibc_count_var is not None and rp_var is not None:
                expected = 2 * ibc_count_var.X * rp_var.X
                print(f"     Expected: 2 × {ibc_count_var.X:.1f} × {rp_var.X:.1f} = {expected:.1f}")
        
        # 9. total_row_acts
        var_name = f"total_row_acts_({w},{t_id})"
        var = model.getVarByName(var_name)
        if var is not None:
            print(f"\n  9. Total Row Acts: {var.X:.4f}")
            print(f"     (seq_part + aligned_part + block_crossing)")
        
        # 10. scaled acts
        var_name = f"row_acts_scaled_({w},{t_id})"
        var = model.getVarByName(var_name)
        if var is not None:
            print(f"\n  10. Scaled Row Acts: {var.X:.4f}")
            print(f"     (total_row_acts × macs_scale_factor)")
        
        # 11. cycles
        var_name = f"row_acts_cycles_({w},{t_id})"
        var = model.getVarByName(var_name)
        if var is not None:
            print(f"\n  11. Row Acts Cycles: {var.X:.4f}")
            print(f"     (scaled_acts × activation_latency)")
    
    # 打印 xj 变量
    print("\n" + "=" * 80)
    print("xj 变量 (维度j是否有内层循环)")
    print("=" * 80)
    
    dram_level = 3
    rowbuf_level = 2
    s_temporal = 1
    
    for t_id in range(3):
        t_name = tensor_names[t_id]
        print(f"\n  {t_name}:")
        
        relevant_dims = [j for j in range(7) if workload.O[j][t_id] == 1]
        irrelevant_dims = [j for j in range(7) if workload.O[j][t_id] == 0]
        
        print(f"    Relevant dims: {[dim_names[j] for j in relevant_dims]}")
        print(f"    Irrelevant dims: {[dim_names[j] for j in irrelevant_dims]}")
        
        for j in range(7):
            xj_var = vars_obj.xj.get((w, t_id, rowbuf_level, dram_level, j))
            if xj_var is not None:
                relevant = "relevant" if workload.O[j][t_id] == 1 else "irrelevant"
                print(f"    {dim_names[j]}: xj = {int(xj_var.X)} ({relevant})")
    
    # 打印 DRAM factors
    print("\n" + "=" * 80)
    print("DRAM Level Factors (L3)")
    print("=" * 80)
    
    for j in range(7):
        for i, div in enumerate(workload.divisors[j]):
            xb_var = vars_obj.xb.get((w, dram_level, s_temporal, j, i))
            if xb_var is not None and xb_var.X > 0.5:
                print(f"  {dim_names[j]}: {div}")
    
    # 打印 layout choice
    print("\n" + "=" * 80)
    print("Layout Choice")
    print("=" * 80)
    
    for t_id in range(3):
        t_name = tensor_names[t_id]
        row_aligned_var = vars_obj.layout_choice.get((w, t_id, "row_aligned"))
        if row_aligned_var is not None:
            layout = "row_aligned" if row_aligned_var.X > 0.5 else "sequential"
            print(f"  {t_name}: {layout}")
        else:
            print(f"  {t_name}: sequential (no choice)")
    
    # 手工计算验证
    print("\n" + "=" * 80)
    print("手工计算验证 row_acts_row_aligned")
    print("=" * 80)
    
    for t_id in range(3):
        t_name = tensor_names[t_id]
        product = 1
        factors = []
        
        for j in range(7):
            xj_var = vars_obj.xj.get((w, t_id, rowbuf_level, dram_level, j))
            if xj_var is not None and xj_var.X > 0.5:
                # Find which divisor is selected
                for i, div in enumerate(workload.divisors[j]):
                    xb_var = vars_obj.xb.get((w, dram_level, s_temporal, j, i))
                    if xb_var is not None and xb_var.X > 0.5:
                        product *= div
                        factors.append(f"{dim_names[j]}={div}")
                        break
        
        print(f"\n  {t_name}:")
        print(f"    公式: Π_{{j: xj=1}} DRAM_factor[j]")
        print(f"    展开: {' × '.join(factors) if factors else '1'} = {product}")
        
        # 对比 ILP 变量
        var_name = f"row_acts_row_aligned_({w},{t_id})"
        var = model.getVarByName(var_name)
        if var is not None:
            print(f"    ILP 变量值: {var.X:.4f}")
            if abs(var.X - product) > 0.1:
                print(f"    ⚠️ 差异: {var.X - product:.4f}")

if __name__ == "__main__":
    main()
