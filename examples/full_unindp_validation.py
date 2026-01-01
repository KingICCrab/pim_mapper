#!/usr/bin/env python3
"""
完整的 UniNDP 模拟验证 - 使用对齐配置

运行完整的编译+模拟流程，获得实际 cycle 数
"""

import sys
import os

# 切换到 UniNDP 目录
os.chdir('/Users/haochenzhao/Projects/UniNDP')
sys.path.insert(0, '/Users/haochenzhao/Projects/UniNDP')

from frontend import *
from midend import *
from backend import *
from sim import sim
from tools import *
import tqdm

# 禁用 tqdm 进度条
def tqdm_replacement(iterable_object, *args, **kwargs):
    return iterable_object
tqdm.tqdm = tqdm_replacement

print("="*70)
print("UniNDP 完整模拟验证 (对齐配置)")
print("="*70)

# 加载对齐后的配置
print("\n加载配置: pim-optimizer-aligned.yaml")
SimConfig.read_from_yaml('./config/pim-optimizer-aligned.yaml')

# 配置确认
total_macs = SimConfig.ch * SimConfig.de_pu[0] * SimConfig.de_pu_w[2]
print(f"总 MACs: {total_macs} (= 256 PEs × 8 MACs)")

# 使用 hbmpim codegen
Codegen = hbmpim

# 测试用例
test_cases = [
    (64, 64, 64, 1),
    (128, 128, 128, 1),
    (256, 256, 256, 1),
]

results = []

for M, K, N, B in test_cases:
    print(f"\n{'='*70}")
    print(f"测试: GEMM M={M}, K={K}, N={N}")
    print("="*70)
    
    mm_size = (M * B, K, N, 1)
    total_ops = M * K * N * B
    
    try:
        # 前端: 生成调度空间
        print("1. 前端: 生成调度空间...")
        schedules = Scheduler(mm_size, 'mm', 
                              po2=True,
                              allow_under_ultize=False)
        print(f"   生成 {len(schedules)} 个调度")
        
        if len(schedules) == 0:
            print("   无有效调度，跳过")
            continue
        
        # 中端: 编译
        print("2. 中端: 编译...")
        compile_results = []
        for i, sch in enumerate(schedules[:10]):  # 只测试前10个
            try:
                res = Codegen(sch, 'mm', mm_size, 0, 0)
                compile_results.append((i, sch, res))
            except:
                pass
        
        print(f"   成功编译 {len(compile_results)} 个")
        
        if len(compile_results) == 0:
            print("   无成功编译，跳过")
            continue
        
        # 后端: 模拟
        print("3. 后端: 模拟...")
        sim_results = []
        for idx, sch, res in compile_results[:5]:  # 只模拟前5个
            try:
                cycles = sim(res)
                sim_results.append({
                    'schedule_idx': idx,
                    'cycles': cycles,
                    'schedule': sch
                })
            except Exception as e:
                print(f"   模拟失败: {e}")
        
        print(f"   成功模拟 {len(sim_results)} 个")
        
        if len(sim_results) > 0:
            # 找最优
            best = min(sim_results, key=lambda x: x['cycles'])
            worst = max(sim_results, key=lambda x: x['cycles'])
            
            print(f"\n结果:")
            print(f"  总 MACs: {total_ops}")
            print(f"  理论最小 cycles: {total_ops / total_macs:.0f}")
            print(f"  实际最优 cycles: {best['cycles']}")
            print(f"  实际最差 cycles: {worst['cycles']}")
            print(f"  硬件利用率: {total_ops / (best['cycles'] * total_macs) * 100:.1f}%")
            
            results.append({
                'workload': f'GEMM_{M}x{K}x{N}',
                'total_ops': total_ops,
                'theory_min': total_ops / total_macs,
                'best_cycles': best['cycles'],
                'worst_cycles': worst['cycles'],
                'utilization': total_ops / (best['cycles'] * total_macs) * 100
            })
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

# 总结
print("\n" + "="*70)
print("总结")
print("="*70)

if results:
    print(f"\n{'Workload':<20} {'Total Ops':<12} {'Theory':<10} {'Best':<10} {'Util%':<8}")
    print("-" * 60)
    for r in results:
        print(f"{r['workload']:<20} {r['total_ops']:<12} {r['theory_min']:<10.0f} {r['best_cycles']:<10} {r['utilization']:<8.1f}")
else:
    print("无有效结果")
