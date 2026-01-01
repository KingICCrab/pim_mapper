#!/usr/bin/env python3
"""
合法 Mapping 验证实验

核心约束 (来自 ILP 模型):
=========================================
1. 维度分解约束 (Dimension Factorization)
   - 所有层的 tiling 因子乘积 = 原始维度
   - P_l0 × P_l1 × P_l2 × P_l3 = P (对所有维度)

2. Buffer 容量约束 (Buffer Capacity)
   - 每层 buffer 中的 tile 大小 <= buffer 容量
   - Input tile: (P_buffer + R - 1) × (Q_buffer + S - 1) × C_buffer
   - Weight tile: R_buffer × S_buffer × C_buffer × K_buffer
   - Output tile: P_buffer × Q_buffer × K_buffer

3. PE Array 约束 (PE Parallelism)
   - H 方向并行度 <= PE_H
   - W 方向并行度 <= PE_W
   - Internal 并行度 <= compute_unit.num_macs

4. 数据布局约束 (Data Layout)
   - block_h >= P_buffer + R - 1 (一个 h tile 不能跨太多 block)
   - block_w >= Q_buffer + S - 1 (一个 w tile 不能跨太多 block)
   - 实际上放宽：只要 block 是合理的 H_in, W_in 的因子

=========================================
简化假设 (用于验证实验):
- PE Array: 16 × 16 (H=16, W=16)
- GlobalBuffer: 64KB = 16384 entries (4 bytes per entry)
- RowBuffer: 1KB = 256 entries
- LocalDRAM: 无限大
- Level 0 (PE) 空间并行, Level 1-3 时间循环
"""

import sys
sys.path.insert(0, '/Users/haochenzhao/Projects/pim_optimizer')

import math
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ==========================================
# 硬件参数 (简化的 PIM 架构)
# ==========================================
@dataclass
class HardwareConfig:
    """硬件配置"""
    pe_h: int = 16
    pe_w: int = 16
    global_buffer_entries: int = 16384  # 64KB / 4 bytes
    row_buffer_entries: int = 256       # 1KB / 4 bytes
    # LocalDRAM: unlimited


@dataclass
class Workload:
    """Conv Workload"""
    name: str
    N: int  # Batch
    K: int  # Output channels
    C: int  # Input channels
    P: int  # Output height
    Q: int  # Output width
    R: int  # Filter height
    S: int  # Filter width
    stride: int = 1
    
    @property
    def H_in(self) -> int:
        return self.P + self.R - 1
    
    @property
    def W_in(self) -> int:
        return self.Q + self.S - 1
    
    def get_divisors(self, dim_value: int) -> List[int]:
        """获取一个维度的所有因子"""
        divisors = []
        for i in range(1, dim_value + 1):
            if dim_value % i == 0:
                divisors.append(i)
        return divisors


@dataclass
class Mapping:
    """Mapping 配置"""
    name: str
    
    # DRAM level (L3) tiling
    P_l3: int
    Q_l3: int
    C_l3: int
    K_l3: int
    
    # RowBuffer level (L2) tiling - 主要是 R, S
    R_l2: int = 1
    S_l2: int = 1
    
    # GlobalBuffer level (L1) - 隐式，由 L2, L3 推导
    # Buffer tile = Total / (L2 × L3)
    
    # PE level (L0) spatial - 简化为全 1
    # 假设所有空间并行都在 L0
    
    # 数据布局
    block_h: int = 31
    block_w: int = 31
    
    # Workload reference (for validation)
    workload: Optional[Workload] = None
    
    @property
    def P_buffer(self) -> int:
        if self.workload:
            return self.workload.P // self.P_l3
        return 0
    
    @property
    def Q_buffer(self) -> int:
        if self.workload:
            return self.workload.Q // self.Q_l3
        return 0
    
    @property
    def C_buffer(self) -> int:
        if self.workload:
            return self.workload.C // self.C_l3
        return 0
    
    @property
    def K_buffer(self) -> int:
        if self.workload:
            return self.workload.K // self.K_l3
        return 0
    
    @property
    def R_buffer(self) -> int:
        if self.workload:
            return self.workload.R // self.R_l2
        return 0
    
    @property
    def S_buffer(self) -> int:
        if self.workload:
            return self.workload.S // self.S_l2
        return 0


def validate_mapping(mapping: Mapping, hw: HardwareConfig) -> Tuple[bool, List[str]]:
    """
    验证 mapping 是否满足所有约束
    
    Returns:
        (is_valid, list of violation messages)
    """
    violations = []
    workload = mapping.workload
    
    if workload is None:
        return False, ["No workload attached"]
    
    # ==========================================
    # 1. 维度分解约束
    # ==========================================
    # P: P_l3 × P_buffer = P
    if mapping.P_l3 * mapping.P_buffer != workload.P:
        violations.append(f"P factorization: {mapping.P_l3} × {mapping.P_buffer} != {workload.P}")
    
    # Q: Q_l3 × Q_buffer = Q
    if mapping.Q_l3 * mapping.Q_buffer != workload.Q:
        violations.append(f"Q factorization: {mapping.Q_l3} × {mapping.Q_buffer} != {workload.Q}")
    
    # C: C_l3 × C_buffer = C
    if mapping.C_l3 * mapping.C_buffer != workload.C:
        violations.append(f"C factorization: {mapping.C_l3} × {mapping.C_buffer} != {workload.C}")
    
    # K: K_l3 × K_buffer = K
    if mapping.K_l3 * mapping.K_buffer != workload.K:
        violations.append(f"K factorization: {mapping.K_l3} × {mapping.K_buffer} != {workload.K}")
    
    # R: R_l2 × R_buffer = R
    if mapping.R_l2 * mapping.R_buffer != workload.R:
        violations.append(f"R factorization: {mapping.R_l2} × {mapping.R_buffer} != {workload.R}")
    
    # S: S_l2 × S_buffer = S
    if mapping.S_l2 * mapping.S_buffer != workload.S:
        violations.append(f"S factorization: {mapping.S_l2} × {mapping.S_buffer} != {workload.S}")
    
    # ==========================================
    # 2. Buffer 容量约束 (GlobalBuffer)
    # ==========================================
    # Input tile size: (P_buffer + R_buffer - 1) × (Q_buffer + S_buffer - 1) × C_buffer
    # 注意：Input tile 需要考虑 halo (滤波器大小)
    input_tile_h = mapping.P_buffer + mapping.R_buffer - 1
    input_tile_w = mapping.Q_buffer + mapping.S_buffer - 1
    input_tile_size = input_tile_h * input_tile_w * mapping.C_buffer
    
    # Weight tile size: R_buffer × S_buffer × C_buffer × K_buffer
    weight_tile_size = mapping.R_buffer * mapping.S_buffer * mapping.C_buffer * mapping.K_buffer
    
    # Output tile size: P_buffer × Q_buffer × K_buffer
    output_tile_size = mapping.P_buffer * mapping.Q_buffer * mapping.K_buffer
    
    # 检查是否超过 GlobalBuffer 容量
    # 假设三种数据类型可以同时存储
    total_buffer_usage = input_tile_size + weight_tile_size + output_tile_size
    
    if total_buffer_usage > hw.global_buffer_entries:
        violations.append(
            f"GlobalBuffer overflow: Input({input_tile_size}) + Weight({weight_tile_size}) + "
            f"Output({output_tile_size}) = {total_buffer_usage} > {hw.global_buffer_entries}"
        )
    
    # ==========================================
    # 3. 数据布局约束 (block size)
    # ==========================================
    # block_h 和 block_w 必须是 H_in, W_in 的合理因子或者足够大
    # 这里简化为：block 必须 >= 1 且 <= H_in/W_in
    if mapping.block_h < 1 or mapping.block_h > workload.H_in:
        violations.append(f"Invalid block_h: {mapping.block_h} (should be 1 ~ {workload.H_in})")
    
    if mapping.block_w < 1 or mapping.block_w > workload.W_in:
        violations.append(f"Invalid block_w: {mapping.block_w} (should be 1 ~ {workload.W_in})")
    
    is_valid = len(violations) == 0
    return is_valid, violations


def generate_valid_mappings(workload: Workload, hw: HardwareConfig, max_count: int = 30) -> List[Mapping]:
    """
    生成合法的 mappings
    
    策略:
    1. 枚举 P_l3, Q_l3, C_l3, K_l3 的因子组合
    2. 对每个组合，计算 buffer tile 大小
    3. 过滤掉不满足 buffer 约束的组合
    4. 对 R_l2 尝试 1 和 R
    """
    valid_mappings = []
    
    # 获取各维度的因子
    P_divisors = workload.get_divisors(workload.P)
    Q_divisors = workload.get_divisors(workload.Q)
    C_divisors = workload.get_divisors(workload.C)
    K_divisors = workload.get_divisors(workload.K)
    R_divisors = workload.get_divisors(workload.R)
    
    # 限制搜索空间
    P_divisors = [d for d in P_divisors if d >= 1][:6]
    Q_divisors = [d for d in Q_divisors if d >= 1][:6]
    C_divisors = [d for d in C_divisors if d >= 1][:4]
    K_divisors = [d for d in K_divisors if d >= 1][:4]
    
    count = 0
    
    for P_l3 in P_divisors:
        for Q_l3 in Q_divisors:
            for C_l3 in C_divisors:
                for K_l3 in K_divisors:
                    for R_l2 in [1, workload.R] if workload.R > 1 else [1]:
                        if count >= max_count:
                            return valid_mappings
                        
                        # 计算 buffer tiles
                        P_buffer = workload.P // P_l3
                        Q_buffer = workload.Q // Q_l3
                        C_buffer = workload.C // C_l3
                        K_buffer = workload.K // K_l3
                        R_buffer = workload.R // R_l2
                        S_buffer = workload.S  # S_l2 = 1
                        
                        # 计算 buffer 使用量
                        input_tile = (P_buffer + R_buffer - 1) * (Q_buffer + S_buffer - 1) * C_buffer
                        weight_tile = R_buffer * S_buffer * C_buffer * K_buffer
                        output_tile = P_buffer * Q_buffer * K_buffer
                        total = input_tile + weight_tile + output_tile
                        
                        if total > hw.global_buffer_entries:
                            continue  # 不满足 buffer 约束
                        
                        # 尝试几种 block 配置
                        block_configs = [
                            (workload.H_in, workload.W_in),  # 单 block
                            (max(1, workload.H_in // 2), workload.W_in),
                            (workload.H_in, max(1, workload.W_in // 2)),
                        ]
                        
                        for block_h, block_w in block_configs:
                            name = f"P{P_l3}_Q{Q_l3}_C{C_l3}_K{K_l3}_R{R_l2}_b{block_h}x{block_w}"
                            mapping = Mapping(
                                name=name,
                                P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3,
                                R_l2=R_l2, S_l2=1,
                                block_h=block_h, block_w=block_w,
                                workload=workload
                            )
                            
                            # 再次验证
                            is_valid, _ = validate_mapping(mapping, hw)
                            if is_valid:
                                valid_mappings.append(mapping)
                                count += 1
                                
                                if count >= max_count:
                                    return valid_mappings
    
    return valid_mappings


def compute_exact_row_switches(mapping: Mapping) -> int:
    """精确模拟计算 row switches"""
    workload = mapping.workload
    
    num_h_blocks = math.ceil(workload.H_in / mapping.block_h)
    num_w_blocks = math.ceil(workload.W_in / mapping.block_w)
    
    row_switches = 0
    last_block = None
    
    # 循环顺序: K -> C -> P -> Q -> R (从外到内)
    for k in range(mapping.K_l3):
        for c in range(mapping.C_l3):
            for p in range(mapping.P_l3):
                for q in range(mapping.Q_l3):
                    for r in range(mapping.R_l2):
                        # 计算 h, w 范围
                        h = p * mapping.P_buffer + r
                        w_start = q * mapping.Q_buffer
                        w_end = q * mapping.Q_buffer + mapping.Q_buffer + workload.S - 2
                        
                        # 确定访问的 blocks
                        hb = min(h // mapping.block_h, num_h_blocks - 1)
                        wb_start = min(w_start // mapping.block_w, num_w_blocks - 1)
                        wb_end = min(w_end // mapping.block_w, num_w_blocks - 1)
                        
                        for wb in range(wb_start, wb_end + 1):
                            curr_block = (c, hb, wb)
                            if last_block is not None and curr_block != last_block:
                                row_switches += 1
                            last_block = curr_block
    
    return row_switches


def compute_formula_prediction(mapping: Mapping) -> int:
    """简化公式预测 row switches"""
    workload = mapping.workload
    
    num_h_blocks = math.ceil(workload.H_in / mapping.block_h)
    num_w_blocks = math.ceil(workload.W_in / mapping.block_w)
    
    # 计算 Q tiles 中跨 w_block 的数量
    q_crossing = 0
    multi_block = 0
    for q in range(mapping.Q_l3):
        w_start = q * mapping.Q_buffer
        w_end = w_start + mapping.Q_buffer + workload.S - 2
        wb_start = w_start // mapping.block_w
        wb_end = min(w_end // mapping.block_w, num_w_blocks - 1)
        blocks_in_q = wb_end - wb_start
        multi_block += blocks_in_q
        if blocks_in_q > 0:
            q_crossing += 1
    
    multi_block *= mapping.P_l3 * mapping.R_l2 * mapping.C_l3 * mapping.K_l3
    
    # R switches (只在 multi-block Q tiles 中有)
    r_switches = q_crossing * mapping.P_l3 * max(0, mapping.R_l2 - 1) * mapping.C_l3 * mapping.K_l3
    
    # P switches
    p_switches = max(0, mapping.P_l3 - 1) * mapping.C_l3 * mapping.K_l3
    
    # Q switches (Q 变化跨 w_block 边界)
    q_switches_count = 0
    for q in range(1, mapping.Q_l3):
        w_end_prev = (q - 1) * mapping.Q_buffer + mapping.Q_buffer + workload.S - 2
        wb_prev = min(w_end_prev // mapping.block_w, num_w_blocks - 1)
        wb_curr = q * mapping.Q_buffer // mapping.block_w
        if wb_prev != wb_curr:
            q_switches_count += 1
    q_switches = q_switches_count * mapping.P_l3 * mapping.R_l2 * mapping.C_l3 * mapping.K_l3
    
    # C & K switches
    c_switches = max(0, mapping.C_l3 - 1) * mapping.K_l3
    k_switches = max(0, mapping.K_l3 - 1)
    
    return multi_block + r_switches + p_switches + q_switches + c_switches + k_switches


def compute_current_ilp(mapping: Mapping, BC: int = 10) -> int:
    """当前 ILP 模型 (不含 R_l2)"""
    return (mapping.P_l3 * mapping.Q_l3 * mapping.C_l3 + BC) * mapping.K_l3


def run_workload_validation(workload: Workload, hw: HardwareConfig):
    """验证单个 workload"""
    print(f"\n{'='*80}")
    print(f"Workload: {workload.name}")
    print(f"  N={workload.N}, K={workload.K}, C={workload.C}")
    print(f"  P={workload.P}, Q={workload.Q}, R={workload.R}, S={workload.S}")
    print(f"  H_in={workload.H_in}, W_in={workload.W_in}")
    print(f"{'='*80}")
    
    # 生成合法 mappings
    mappings = generate_valid_mappings(workload, hw, max_count=30)
    print(f"\n生成了 {len(mappings)} 个合法 mapping")
    
    if len(mappings) < 3:
        print("  ⚠️ 合法 mapping 数量太少，跳过此 workload")
        return None
    
    # 验证并计算
    results = []
    for m in mappings:
        is_valid, violations = validate_mapping(m, hw)
        if not is_valid:
            print(f"  ⚠️ {m.name} 不合法: {violations}")
            continue
        
        exact = compute_exact_row_switches(m)
        formula = compute_formula_prediction(m)
        ilp = compute_current_ilp(m)
        
        results.append({
            'name': m.name,
            'mapping': m,
            'exact': exact,
            'formula': formula,
            'ilp': ilp,
        })
    
    if len(results) < 3:
        print("  ⚠️ 有效结果太少，跳过此 workload")
        return None
    
    # 显示结果 (按 exact 排序)
    print(f"\n{'Mapping':<40} {'Exact':>8} {'Formula':>8} {'ILP':>8} {'F_err%':>8} {'I_err%':>8}")
    print("-" * 85)
    
    for r in sorted(results, key=lambda x: x['exact']):
        f_err = abs(r['formula'] - r['exact']) / max(1, r['exact']) * 100
        i_err = abs(r['ilp'] - r['exact']) / max(1, r['exact']) * 100
        print(f"{r['name']:<40} {r['exact']:>8} {r['formula']:>8} {r['ilp']:>8} {f_err:>7.1f}% {i_err:>7.1f}%")
    
    # 计算相关系数
    exact_vals = [r['exact'] for r in results]
    formula_vals = [r['formula'] for r in results]
    ilp_vals = [r['ilp'] for r in results]
    
    corr_f, p_f = stats.spearmanr(exact_vals, formula_vals)
    corr_i, p_i = stats.spearmanr(exact_vals, ilp_vals)
    
    print(f"\nSpearman 排序相关系数:")
    print(f"  Formula vs Exact: ρ = {corr_f:.3f} (p = {p_f:.4f})")
    print(f"  Current ILP vs Exact: ρ = {corr_i:.3f} (p = {p_i:.4f})")
    
    avg_f_err = np.mean([abs(r['formula'] - r['exact']) / max(1, r['exact']) * 100 for r in results])
    avg_i_err = np.mean([abs(r['ilp'] - r['exact']) / max(1, r['exact']) * 100 for r in results])
    
    print(f"\n平均误差:")
    print(f"  Formula: {avg_f_err:.1f}%")
    print(f"  Current ILP: {avg_i_err:.1f}%")
    
    return {
        'name': workload.name,
        'num_mappings': len(results),
        'corr_formula': corr_f,
        'corr_ilp': corr_i,
        'avg_err_formula': avg_f_err,
        'avg_err_ilp': avg_i_err,
    }


def main():
    print("="*80)
    print("合法 Mapping Row Activation 验证实验")
    print("="*80)
    
    hw = HardwareConfig()
    print(f"\n硬件配置:")
    print(f"  PE Array: {hw.pe_h} × {hw.pe_w}")
    print(f"  GlobalBuffer: {hw.global_buffer_entries} entries ({hw.global_buffer_entries * 4 // 1024} KB)")
    print(f"  RowBuffer: {hw.row_buffer_entries} entries ({hw.row_buffer_entries * 4 // 1024} KB)")
    
    # 定义测试 workloads
    workloads = [
        Workload(name='ResNet-L1 (7x7)', N=1, K=64, C=3, P=56, Q=56, R=7, S=7),
        Workload(name='ResNet-3x3', N=1, K=64, C=64, P=56, Q=56, R=3, S=3),
        Workload(name='ResNet-1x1', N=1, K=256, C=64, P=56, Q=56, R=1, S=1),
        Workload(name='VGG-conv1', N=1, K=64, C=3, P=224, Q=224, R=3, S=3),
        Workload(name='VGG-conv5', N=1, K=512, C=512, P=14, Q=14, R=3, S=3),
        Workload(name='MobileNet-dw', N=1, K=32, C=32, P=112, Q=112, R=3, S=3),
    ]
    
    all_results = []
    
    for workload in workloads:
        result = run_workload_validation(workload, hw)
        if result:
            all_results.append(result)
    
    # 总结
    if all_results:
        print("\n" + "="*80)
        print("总体验证结果汇总")
        print("="*80)
        
        print(f"\n{'Workload':<20} {'#Maps':>6} {'ρ(Form)':>10} {'ρ(ILP)':>10} {'AvgErr(F)':>10} {'AvgErr(I)':>10}")
        print("-"*70)
        
        for r in all_results:
            print(f"{r['name']:<20} {r['num_mappings']:>6} {r['corr_formula']:>10.3f} {r['corr_ilp']:>10.3f} {r['avg_err_formula']:>9.1f}% {r['avg_err_ilp']:>9.1f}%")
        
        avg_corr_f = np.mean([r['corr_formula'] for r in all_results])
        avg_corr_i = np.mean([r['corr_ilp'] for r in all_results])
        
        print("-"*70)
        print(f"{'平均':>26} {avg_corr_f:>10.3f} {avg_corr_i:>10.3f}")
        
        print("\n" + "="*80)
        print("结论")
        print("="*80)
        
        if avg_corr_f > 0.9:
            print("✓ Formula 模型与精确值高度相关 (ρ > 0.9)")
        elif avg_corr_f > 0.7:
            print("△ Formula 模型与精确值中度相关 (0.7 < ρ < 0.9)")
        else:
            print("✗ Formula 模型相关性较低 (ρ < 0.7)")
        
        if avg_corr_f > avg_corr_i + 0.05:
            print(f"✓ Formula (ρ={avg_corr_f:.3f}) 显著优于 Current ILP (ρ={avg_corr_i:.3f})")
        elif avg_corr_f > avg_corr_i:
            print(f"△ Formula (ρ={avg_corr_f:.3f}) 略优于 Current ILP (ρ={avg_corr_i:.3f})")
        else:
            print(f"✗ Current ILP (ρ={avg_corr_i:.3f}) >= Formula (ρ={avg_corr_f:.3f})")


if __name__ == '__main__':
    main()
