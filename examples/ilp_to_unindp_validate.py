#!/usr/bin/env python3
"""
ILP → UniNDP 验证流程

这个脚本实现方案 B：直接验证 ILP 优化器找到的映射策略

流程:
1. 运行 ILP 优化器，获取最优映射
2. 将 ILP 映射转换为 UniNDP 策略格式
3. 用 UniNDP 模拟 ILP 策略
4. 比较: ILP 策略 vs 随机策略 vs 穷举最优

验证目标:
- ILP 找到的策略应该接近穷举最优
- ILP 策略应该显著优于随机策略
"""

import sys
import os
import random
import time

# 添加 UniNDP 路径
UNINDP_PATH = "/Users/haochenzhao/Projects/UniNDP"
sys.path.insert(0, UNINDP_PATH)
os.chdir(UNINDP_PATH)

from frontend import *
from midend import *
from backend import *
from sim import sim
from tools import *
import tqdm

# 时钟参数
tCK = 1
tREFI = 3900
tRFC = 350


def get_design_space(m, k, allow_under_utilize=False):
    """获取 UniNDP 的完整设计空间
    
    Args:
        allow_under_utilize: 如果 True，允许硬件利用率低的策略
    """
    mm_size = (1, m, k, 1)
    SimConfig.read_from_yaml('./config/hbm-pim.yaml')
    SimConfig.pu_level = LEVEL.DE
    
    partition_tool = Partition(require_power_of_2=False)
    partition_space = partition_tool.get_partition_space_mm(mm_size)
    
    # 是否过滤低利用率的策略
    if not allow_under_utilize:
        partition_space = partition_tool.choose_from_partition_space_mm(partition_space)
    
    design_space = []
    for index in range(len(partition_space)):
        compute_level, pu_num, partition = partition_space[index]
        simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row = \
            partition_tool.mem_partition_mm(mm_size, partition)
        
        for input_choice in reversed(mkl_Input_to_row):
            for output_choice in reversed(ml_Out_to_row):
                design_space.append({
                    'compute_level': compute_level,
                    'pu_num': pu_num,
                    'partition': partition,
                    'simd_k': simd_k,
                    'mkl_Input_to_row': input_choice,
                    'simd_l': simd_l,
                    'ml_Out_to_row': output_choice,
                })
    
    return design_space


def simulate_strategy(strategy, m, k, require_power_of_2=False):
    """模拟单个策略，返回 cycles"""
    mm_size = (1, m, k, 1)
    
    compute_level = strategy['compute_level']
    pu_num = strategy['pu_num']
    partition = strategy['partition']
    simd_k = strategy['simd_k']
    mkl_Input_to_row = strategy['mkl_Input_to_row']
    simd_l = strategy['simd_l']
    ml_Out_to_row = strategy['ml_Out_to_row']
    
    # Codegen
    Codegen = hbmpim_verify
    
    # A. hw mapping
    mapping_tool = Mapping(require_power_of_2=require_power_of_2)
    hw_id_list = mapping_tool.assign_hw(partition)
    
    # B. dram mapping
    input_bank, input_row_offset, \
    weight_bank, weight_row_offset, \
    output_bank, output_row_offset = mapping_tool.assign_dram(
        pu_num, mkl_Input_to_row, ml_Out_to_row, partition
    )
    
    # D. Codegen
    codegen_tool = Codegen(require_power_of_2=require_power_of_2)
    codegen_tool.set_gen()
    
    gen_code, inst_count, predict_result = codegen_tool.codegen(
        'mm', compute_level, pu_num, partition,
        simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row,
        hw_id_list, (input_bank, input_row_offset,
                     weight_bank, weight_row_offset,
                     output_bank, output_row_offset),
        cmd_threshold=0
    )
    
    # E. simulation
    sim_result = sim(gen_code, silent=True, sim_verify=1)
    sim_result = tCK * sim_result * (tRFC + tREFI) / tREFI
    
    return sim_result


def extract_partition_params(strategy):
    """提取策略的关键参数"""
    partition = strategy['partition']
    ch, ra, de, pu = partition
    
    return {
        'ch_m': ch[0], 'ch_k': ch[1], 'ch_l': ch[2],
        'ra_m': ra[0], 'ra_k': ra[1], 'ra_l': ra[2],
        'de_m': de[0], 'de_k': de[1], 'de_l': de[2],
        'pu_m': pu[0], 'pu_k': pu[1], 'pu_l': pu[2],
        'k_to_row': strategy['mkl_Input_to_row'][0][1] if strategy['mkl_Input_to_row'] else 0,
    }


def find_best_strategy(design_space, m, k, sample_size=100):
    """从设计空间中找最优策略 (通过采样)"""
    if len(design_space) <= sample_size:
        samples = design_space
    else:
        samples = random.sample(design_space, sample_size)
    
    best_strategy = None
    best_cycles = float('inf')
    all_cycles = []
    
    print(f"  采样 {len(samples)} 个策略...")
    for strategy in tqdm.tqdm(samples, desc="  Finding best"):
        try:
            cycles = simulate_strategy(strategy, m, k)
            all_cycles.append(cycles)
            if cycles < best_cycles:
                best_cycles = cycles
                best_strategy = strategy
        except Exception as e:
            continue
    
    return best_strategy, best_cycles, all_cycles


def find_worst_strategy(design_space, m, k, sample_size=50):
    """从设计空间中找最差策略 (通过采样)"""
    if len(design_space) <= sample_size:
        samples = design_space
    else:
        samples = random.sample(design_space, sample_size)
    
    worst_strategy = None
    worst_cycles = 0
    
    for strategy in tqdm.tqdm(samples, desc="  Finding worst"):
        try:
            cycles = simulate_strategy(strategy, m, k)
            if cycles > worst_cycles:
                worst_cycles = cycles
                worst_strategy = strategy
        except Exception as e:
            continue
    
    return worst_strategy, worst_cycles


def get_random_strategies(design_space, m, k, num_samples=10):
    """获取随机策略的性能"""
    samples = random.sample(design_space, min(num_samples, len(design_space)))
    
    results = []
    for strategy in samples:
        try:
            cycles = simulate_strategy(strategy, m, k)
            results.append(cycles)
        except:
            continue
    
    return results


def ilp_mapping_to_unindp_strategy(ilp_mapping, design_space, m, k):
    """
    将 ILP 映射转换为 UniNDP 策略
    
    ILP 优化器的核心决策:
    - loop_bounds: 各层的分块大小 (ch_l, pu_k, pu_l 等)
    - permutation: 循环顺序
    - bypass: 数据旁路
    
    UniNDP 策略参数:
    - partition: [ch, ra, de, pu] 每个是 (m, k, l, b) 的分割
    - mkl_Input_to_row: 输入数据的行映射
    - ml_Out_to_row: 输出数据的行映射
    """
    # 从 ILP mapping 提取关键参数
    # ILP mapping.loop_bounds[m]["spatial"/"temporal"][dim] = bound
    
    # 这里我们需要找到设计空间中与 ILP 决策最接近的策略
    # ILP 决策的核心是分块大小，需要映射到 UniNDP 的 partition 参数
    
    # 方法: 基于 ILP 的分块因子搜索匹配的 UniNDP 策略
    
    # 获取 ILP 的核心参数 (假设 GEMM: M, K, N 维度)
    # loop_bounds[0] 是 PE 层
    # loop_bounds[1] 是 LocalBuffer
    # loop_bounds[2] 是 GlobalBuffer
    # loop_bounds[3] 是 DRAM
    
    if ilp_mapping is None:
        return None, "No ILP mapping provided"
    
    # 提取 ILP 的空间分块 (用于 PIM)
    loop_bounds = ilp_mapping.loop_bounds
    
    # 计算 ILP 期望的并行度
    # 在 PIM 架构中，空间分块决定了如何利用并行单元
    
    # 找到最匹配的 UniNDP 策略
    best_match = None
    best_score = float('inf')
    
    # 简化版本：直接搜索设计空间中性能最好的策略
    # 这相当于验证 ILP 的优化目标是否正确
    
    # 基于 ILP 的分块参数进行筛选
    # 例如: 如果 ILP 选择 ch_l=64, 则在 UniNDP 中找 ch[2]=64 的策略
    
    # 返回最佳匹配的策略
    return None, "Conversion not fully implemented - using search-based validation"


def run_ilp_optimizer(m, k):
    """
    运行 ILP 优化器
    
    注意: 这需要实际调用 pim_optimizer
    """
    try:
        # 切换回 pim_optimizer 目录
        original_dir = os.getcwd()
        os.chdir("/Users/haochenzhao/Projects/pim_optimizer")
        
        sys.path.insert(0, "/Users/haochenzhao/Projects/pim_optimizer/src")
        
        from pim_optimizer import PIMOptimizer, PIMArchitecture
        from pim_optimizer.workload import ConvWorkload
        
        # 创建 GEMM 工作负载
        workload = ConvWorkload(
            name=f"GEMM_{m}x{k}",
            bounds=[1, 1, 1, 1, m, k, 1],  # [R, S, P, Q, C, K, N]
            divisors=None,  # Will be auto-generated
        )
        
        # 使用默认架构或加载配置
        arch = PIMArchitecture()
        
        optimizer = PIMOptimizer(
            arch=arch,
            verbose=False,
            time_limit=60.0,
        )
        
        result = optimizer.optimize([workload])
        
        os.chdir(original_dir)
        
        if result.mappings:
            return result.mappings[0]
        return None
        
    except Exception as e:
        print(f"  ILP optimizer error: {e}")
        os.chdir(original_dir)
        return None


def validate_workload(m, k, sample_size=100):
    """验证单个工作负载"""
    print(f"\n{'='*60}")
    print(f"验证工作负载: GEMM [{m} x {k}]")
    print(f"{'='*60}")
    
    results = {}
    
    # 测试两种设计空间
    for allow_under in [False, True]:
        label = "完整空间(含低利用率)" if allow_under else "过滤空间(高利用率)"
        print(f"\n[设计空间] {label}")
        
        # 1. 获取设计空间
        design_space = get_design_space(m, k, allow_under_utilize=allow_under)
        print(f"    设计空间大小: {len(design_space)}")
        
        if len(design_space) == 0:
            print("    错误: 设计空间为空!")
            continue
        
        # 2. 找最优和最差策略
        best_strategy, best_cycles, all_cycles = find_best_strategy(
            design_space, m, k, min(sample_size, len(design_space))
        )
        
        # 从采样中获取最差
        if all_cycles:
            worst_cycles = max(all_cycles)
        else:
            worst_strategy, worst_cycles = find_worst_strategy(design_space, m, k, sample_size // 2)
        
        print(f"    最优 cycles: {best_cycles:,.0f}")
        print(f"    最差 cycles: {worst_cycles:,.0f}")
        print(f"    差距比: {worst_cycles / best_cycles:.1f}x")
        
        # 统计分布
        if all_cycles:
            import numpy as np
            arr = np.array(all_cycles)
            print(f"    中位数: {np.median(arr):,.0f}")
            print(f"    标准差: {np.std(arr):,.0f}")
        
        key = "full" if allow_under else "filtered"
        results[key] = {
            'space_size': len(design_space),
            'best': best_cycles,
            'worst': worst_cycles,
            'ratio': worst_cycles / best_cycles,
            'all_cycles': all_cycles,
        }
    
    # 3. 运行 ILP 优化器 (如果可用)
    print("\n[ILP 优化器]")
    ilp_mapping = run_ilp_optimizer(m, k)
    
    if ilp_mapping:
        print("    ILP 优化完成!")
        print(f"    ILP 预测延迟: {ilp_mapping.latency:.2e}")
        results['ilp_latency'] = ilp_mapping.latency
    else:
        print("    ILP 优化器未能返回结果 (需要 gurobipy)")
    
    # 4. 结果汇总
    print("\n[结果汇总]")
    print("-" * 40)
    
    results['workload'] = f"{m}x{k}"
    
    return results


def main():
    print("="*70)
    print("ILP → UniNDP 验证流程")
    print("="*70)
    print()
    print("验证目标:")
    print("1. ILP 找到的策略应该接近穷举最优")
    print("2. 不同策略间的性能差距有多大")
    print("3. ILP 的优化目标是否正确")
    print()
    
    # 测试工作负载 - 包含之前验证中发现性能差距大的工作负载
    workloads = [
        (1000, 1000),   # 小
        (2000, 2000),   # 之前发现 17x 差距
        (4096, 4096),   # 之前发现 8x 差距
        (1024, 2048),   # 非方阵
    ]
    
    all_results = []
    
    for m, k in workloads:
        try:
            result = validate_workload(m, k, sample_size=200)  # 增加采样数
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"Error processing {m}x{k}: {e}")
            import traceback
            traceback.print_exc()
    
    # 最终汇总
    print("\n" + "="*70)
    print("最终汇总")
    print("="*70)
    
    print("\n{:<12} {:>12} {:>12} {:>10} {:>12} {:>12} {:>10}".format(
        "工作负载", "过滤-最优", "过滤-最差", "过滤比", "完整-最优", "完整-最差", "完整比"))
    print("-"*82)
    
    for r in all_results:
        filtered = r.get('filtered', {})
        full = r.get('full', {})
        
        print("{:<12} {:>12,} {:>12,} {:>10.1f}x {:>12,} {:>12,} {:>10.1f}x".format(
            r['workload'],
            int(filtered.get('best', 0)),
            int(filtered.get('worst', 0)),
            filtered.get('ratio', 0),
            int(full.get('best', 0)),
            int(full.get('worst', 0)),
            full.get('ratio', 0),
        ))
    
    print("\n" + "="*70)
    print("关键发现")
    print("="*70)
    print()
    print("1. 过滤空间 vs 完整空间:")
    print("   - 过滤空间只保留高硬件利用率的策略")
    print("   - 完整空间包含所有可能策略 (包括低利用率)")
    print()
    print("2. 性能差距分析:")
    print("   - 如果完整空间的差距比 >> 过滤空间，说明:")
    print("     低利用率策略性能很差，ILP 需要优化利用率")
    print()
    print("3. ILP 优化器的目标:")
    print("   - 应该找到接近过滤空间最优的策略")
    print("   - 或者找到比过滤空间更好的策略 (通过更智能的映射)")
    print()
    

if __name__ == "__main__":
    main()
