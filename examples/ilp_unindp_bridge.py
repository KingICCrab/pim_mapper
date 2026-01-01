#!/usr/bin/env python3
"""
ILP → UniNDP 映射转换与验证

验证目标: ILP 生成的高利用率策略应接近 UniNDP 最优
"""

import sys
import os
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# 添加 UniNDP 路径并切换目录
UNINDP_PATH = "/Users/haochenzhao/Projects/UniNDP"
sys.path.insert(0, UNINDP_PATH)
os.chdir(UNINDP_PATH)

# UniNDP imports (必须在模块级别)
from frontend import *
from midend import *
from backend import *
from sim import sim
from tools import *
import tqdm as tqdm_module

# 时钟参数
tCK = 1
tREFI = 3900
tRFC = 350


def create_high_utilization_strategy(m: int, k: int, filtered_space: list) -> dict:
    """
    从过滤空间中选择高利用率策略
    
    由于 UniNDP 有特定的硬件约束，我们从有效空间中选择
    """
    if not filtered_space:
        return None
    
    best_partition = None
    best_score = 0
    
    for compute_level, pu_num, partition in filtered_space:
        ch, ra, de, pu = partition
        
        # 计算利用率得分
        utilization = (ch[0]*ch[1]*ch[2]*ch[3] * 
                       ra[0]*ra[1]*ra[2]*ra[3] * 
                       de[0]*de[1]*de[2]*de[3] * 
                       pu[0]*pu[1]*pu[2]*pu[3]) / 512
        
        # 偏好在 L 维度并行化的策略 (更好的数据重用)
        l_parallel = ch[2] * ra[2] * de[2] * pu[2]
        
        score = utilization * (1 + 0.1 * l_parallel)
        
        if score > best_score:
            best_score = score
            best_partition = {
                'ch': partition[0],
                'ra': partition[1],
                'de': partition[2],
                'pu': partition[3],
            }
    
    return best_partition


def simulate_unindp_strategy(strategy_dict: dict, m: int, k: int) -> float:
    """使用 UniNDP 模拟策略"""
    mm_size = (1, m, k, 1)
    SimConfig.read_from_yaml('./config/hbm-pim.yaml')
    SimConfig.pu_level = LEVEL.DE
    
    partition = (
        strategy_dict['ch'],
        strategy_dict['ra'],
        strategy_dict['de'],
        strategy_dict['pu'],
    )
    
    partition_tool = Partition(require_power_of_2=False)
    simd_k, mkl_list, simd_l, ml_list = partition_tool.mem_partition_mm(mm_size, partition)
    
    if not mkl_list or not ml_list:
        return float('inf')
    
    mkl_Input_to_row = mkl_list[0]
    ml_Out_to_row = ml_list[0]
    
    compute_level = LEVEL.DE
    pu_num = 8
    
    Codegen = hbmpim_verify
    
    mapping_tool = Mapping(require_power_of_2=False)
    hw_id_list = mapping_tool.assign_hw(partition)
    
    input_bank, input_row_offset, \
    weight_bank, weight_row_offset, \
    output_bank, output_row_offset = mapping_tool.assign_dram(
        pu_num, mkl_Input_to_row, ml_Out_to_row, partition
    )
    
    codegen_tool = Codegen(require_power_of_2=False)
    codegen_tool.set_gen()
    
    gen_code, inst_count, predict_result = codegen_tool.codegen(
        'mm', compute_level, pu_num, partition,
        simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row,
        hw_id_list, (input_bank, input_row_offset,
                     weight_bank, weight_row_offset,
                     output_bank, output_row_offset),
        cmd_threshold=0
    )
    
    sim_result = sim(gen_code, silent=True, sim_verify=1)
    sim_result = tCK * sim_result * (tRFC + tREFI) / tREFI
    
    return sim_result


def validate_ilp_strategy(m: int, k: int):
    """验证 ILP 生成的策略"""
    print(f"\n{'='*60}")
    print(f"验证工作负载: GEMM [{m} x {k}]")
    print(f"{'='*60}")
    
    # 初始化 SimConfig
    SimConfig.read_from_yaml('./config/hbm-pim.yaml')
    SimConfig.pu_level = LEVEL.DE
    
    # 先获取过滤空间
    print("\n[1] 获取有效策略空间...")
    mm_size = (1, m, k, 1)
    partition_tool = Partition(require_power_of_2=False)
    partition_space = partition_tool.get_partition_space_mm(mm_size)
    filtered_space = partition_tool.choose_from_partition_space_mm(partition_space)
    print(f"    过滤空间大小: {len(filtered_space)}")
    
    # 2. 创建高利用率策略 (从有效空间中选择)
    print("\n[2] 创建高利用率策略 (模拟 ILP 决策)...")
    ilp_strategy = create_high_utilization_strategy(m, k, filtered_space)
    
    if ilp_strategy:
        print(f"    分区: ch={ilp_strategy['ch']}, pu={ilp_strategy['pu']}")
        ch = ilp_strategy['ch']
        pu = ilp_strategy['pu']
        utilization = (ch[0]*ch[1]*ch[2]*ch[3] * pu[0]*pu[1]*pu[2]*pu[3]) / (64 * 8)
        print(f"    硬件利用率: {utilization*100:.1f}%")
    else:
        print("    无法创建策略")
        return None, None
    
    # 3. 用 UniNDP 模拟 ILP 策略
    print("\n[3] UniNDP 模拟 ILP 策略...")
    try:
        ilp_cycles = simulate_unindp_strategy(ilp_strategy, m, k)
        print(f"    ILP 策略 cycles: {ilp_cycles:,.0f}")
    except Exception as e:
        print(f"    模拟失败: {e}")
        import traceback
        traceback.print_exc()
        ilp_cycles = None
    
    # 4. 采样其他策略获取基准
    print("\n[4] 采样过滤空间获取基准...")
    
    samples = random.sample(filtered_space, min(100, len(filtered_space)))
    
    best_baseline = float('inf')
    worst_baseline = 0
    all_cycles = []
    
    for compute_level, pu_num, partition in tqdm_module.tqdm(samples, desc="    采样"):
        try:
            strategy = {
                'ch': partition[0],
                'ra': partition[1],
                'de': partition[2],
                'pu': partition[3],
            }
            sample_cycles = simulate_unindp_strategy(strategy, m, k)
            all_cycles.append(sample_cycles)
            if sample_cycles < best_baseline:
                best_baseline = sample_cycles
            if sample_cycles > worst_baseline:
                worst_baseline = sample_cycles
        except:
            continue
    
    print(f"    采样最优 cycles: {best_baseline:,.0f}")
    print(f"    采样最差 cycles: {worst_baseline:,.0f}")
    print(f"    采样范围比: {worst_baseline/best_baseline:.1f}x")
    
    # 5. 结果
    print("\n[5] 结果比较")
    print("-" * 40)
    
    if ilp_cycles and best_baseline < float('inf'):
        ratio = ilp_cycles / best_baseline
        print(f"    ILP 策略 cycles:  {ilp_cycles:,.0f}")
        print(f"    采样最优 cycles:  {best_baseline:,.0f}")
        print(f"    比值: {ratio:.2f}x")
        
        if ratio < 1.3:
            print("    ✅ ILP 策略接近最优 (差距 < 30%)")
        elif ratio < 1.5:
            print("    ⚠️ ILP 策略可接受 (差距 < 50%)")
        else:
            print("    ❌ ILP 策略需要优化")
    
    return ilp_cycles, best_baseline


def main():
    print("="*70)
    print("ILP → UniNDP 映射转换与验证")
    print("="*70)
    print()
    print("验证目标: ILP 生成的高利用率策略应接近 UniNDP 最优")
    print()
    
    workloads = [
        (512, 512),
        (1000, 1000),
        (2000, 2000),
        (4096, 4096),
        (1024, 2048),
        (2048, 1024),
    ]
    
    results = []
    
    for m, k in workloads:
        try:
            ilp_cycles, best_cycles = validate_ilp_strategy(m, k)
            results.append({
                'workload': f'{m}x{k}',
                'ilp': ilp_cycles,
                'best': best_cycles,
                'ratio': ilp_cycles / best_cycles if ilp_cycles and best_cycles else None
            })
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print()
    
    print("{:<12} {:>12} {:>12} {:>8}".format("工作负载", "ILP cycles", "最优 cycles", "比值"))
    print("-"*48)
    for r in results:
        if r['ratio']:
            print("{:<12} {:>12,.0f} {:>12,.0f} {:>8.2f}x".format(
                r['workload'], r['ilp'], r['best'], r['ratio']))
    
    print()
    print("结论:")
    print("  如果比值 < 1.3x → ILP 策略有效")
    print("  如果比值 > 1.3x → 需要改进 ILP 分区策略")


if __name__ == "__main__":
    main()
