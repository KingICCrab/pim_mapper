#!/usr/bin/env python3
"""
MappingEnumerator: 统一的 mapping 空间枚举器

支持枚举模式:
- L3 only: 只枚举 L3 层 tiling (L0=L1=L2=1)
- L3+L2: 枚举 L3 和 L2 两层 tiling (L0=L1=1)
- Full: 枚举所有 4 层 tiling
"""

from itertools import permutations, product
from typing import List, Dict, Tuple, Iterator, Optional
from enum import Enum
import random

from .config import (
    DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N, 
    DIM_NAMES, DEFAULT_PERMUTATION,
    ArchConfig, WorkloadConfig
)
from .mapping_config import MappingConfig
from .constraints import MappingConstraintChecker
from .utils import get_divisors, get_2level_decompositions, get_factor_decompositions


class TilingMode(Enum):
    """Tiling 枚举模式"""
    L3_L2 = "l3_l2"          # 枚举 L3 和 L2 两层
    FULL = "full"            # 枚举所有 4 层


class MappingEnumerator:
    """统一的 Mapping 空间枚举器"""
    
    def __init__(self, workload: WorkloadConfig, arch: ArchConfig = None,
                 validate_constraints: bool = True):
        """
        Args:
            workload: Workload 配置
            arch: 架构配置（用于约束检查）
            validate_constraints: 是否检查约束
        """
        self.workload = workload
        self.arch = arch or ArchConfig()
        self.validate_constraints = validate_constraints
        
        if validate_constraints:
            self.checker = MappingConstraintChecker(workload, self.arch)
        else:
            self.checker = None
        
        # 计算各维度的因子
        self._divisors = {
            'R': get_divisors(workload.R),
            'S': get_divisors(workload.S),
            'P': get_divisors(workload.P),
            'Q': get_divisors(workload.Q),
            'C': get_divisors(workload.C),
            'K': get_divisors(workload.K),
            'N': get_divisors(workload.N),
        }
        
        # 标准 4D permutation (P, Q, C, K)
        self._perms_4d = list(permutations([DIM_P, DIM_Q, DIM_C, DIM_K]))
        # 扩展为 7D: (R, S) + 4D + (N)
        self._perms_7d = [(DIM_R, DIM_S) + p + (DIM_N,) for p in self._perms_4d]
        
        # Layout 选项
        self._layouts = ['sequential', 'row_aligned']
        
        # Block sizes 候选: H 和 W 的 divisors 的所有组合
        # 合法性由 MappingConstraintChecker 检查
        h_divs = get_divisors(workload.H)
        w_divs = get_divisors(workload.W)
        self._candidate_block_sizes = [(bh, bw) for bh in h_divs for bw in w_divs]
    
    # ==================== 枚举方法 ====================
    
    def enumerate(self,
                  mode: TilingMode = TilingMode.L3_L2,
                  vary_permutation: bool = True,
                  vary_layout: bool = True,
                  vary_block_size: bool = True,
                  sample_tiling: int = None,
                  sample_total: int = None,
                  filter_invalid: bool = True) -> Iterator[MappingConfig]:
        """统一的枚举接口
        
        Args:
            mode: Tiling 枚举模式
            vary_permutation: 是否枚举 permutation (24 种)
            vary_layout: 是否枚举 layout 组合 (8 种)
            vary_block_size: 是否枚举 block size
            sample_tiling: 从 tiling 组合中采样数量
            sample_total: 总采样数量
            filter_invalid: 是否过滤不合法的 mapping
            
        Yields:
            MappingConfig 实例
        """
        # 根据模式获取 tiling 组合
        if mode == TilingMode.L3_L2:
            tilings = self._get_l3_l2_tilings()
        else:  # FULL
            tilings = self._get_full_tilings()
        
        # 采样 tiling
        if sample_tiling and len(tilings) > sample_tiling:
            tilings = random.sample(tilings, sample_tiling)
        
        # Permutation 列表
        perms = self._perms_7d if vary_permutation else [DEFAULT_PERMUTATION]
        
        # Layout 组合
        if vary_layout:
            layouts = list(product(self._layouts, repeat=3))
        else:
            layouts = [('sequential', 'sequential', 'sequential')]
        
        # Block sizes
        blocks = self._candidate_block_sizes if vary_block_size else [(1, 1)]
        
        # 生成配置
        all_configs = []
        
        for tiling in tilings:
            for perm in perms:
                for in_lay, w_lay, out_lay in layouts:
                    for bh, bw in blocks:
                        config = self._make_config(tiling, perm, in_lay, w_lay, out_lay, bh, bw, mode)
                        
                        if filter_invalid and self.checker:
                            if self.checker.is_valid(config):
                                all_configs.append(config)
                        else:
                            all_configs.append(config)
        
        # 总采样
        if sample_total and len(all_configs) > sample_total:
            all_configs = random.sample(all_configs, sample_total)
        
        for config in all_configs:
            yield config
    
    def _get_l3_l2_tilings(self) -> List[Tuple]:
        """获取 L3+L2 的 tiling 组合"""
        wl = self.workload
        
        # 每个维度的 2 层分解: (L2, L3)
        decomps = {
            'R': get_2level_decompositions(wl.R),
            'S': get_2level_decompositions(wl.S),
            'P': get_2level_decompositions(wl.P),
            'Q': get_2level_decompositions(wl.Q),
            'C': get_2level_decompositions(wl.C),
            'K': get_2level_decompositions(wl.K),
            'N': get_2level_decompositions(wl.N),
        }
        
        return list(product(
            decomps['R'], decomps['S'], decomps['P'], decomps['Q'],
            decomps['C'], decomps['K'], decomps['N']
        ))
    
    def _get_full_tilings(self) -> List[Tuple]:
        """获取全 4 层的 tiling 组合"""
        wl = self.workload
        
        # 每个维度的 4 层分解: (L0, L1, L2, L3)
        decomps = {
            'R': get_factor_decompositions(wl.R, 4),
            'S': get_factor_decompositions(wl.S, 4),
            'P': get_factor_decompositions(wl.P, 4),
            'Q': get_factor_decompositions(wl.Q, 4),
            'C': get_factor_decompositions(wl.C, 4),
            'K': get_factor_decompositions(wl.K, 4),
            'N': get_factor_decompositions(wl.N, 4),
        }
        
        return list(product(
            decomps['R'], decomps['S'], decomps['P'], decomps['Q'],
            decomps['C'], decomps['K'], decomps['N']
        ))
    
    def _make_config(self, tiling, perm, in_lay, w_lay, out_lay, bh, bw, 
                     mode: TilingMode) -> MappingConfig:
        """从 tiling 元组创建 MappingConfig"""
        
        if mode == TilingMode.L3_L2:
            # tiling = ((R_l2, R_l3), (S_l2, S_l3), ..., (N_l2, N_l3))
            R_l2, R_l3 = tiling[0]
            S_l2, S_l3 = tiling[1]
            P_l2, P_l3 = tiling[2]
            Q_l2, Q_l3 = tiling[3]
            C_l2, C_l3 = tiling[4]
            K_l2, K_l3 = tiling[5]
            N_l2, N_l3 = tiling[6]
            
            # L0 = 原始维度 / (L2 × L3)，L1 = 1
            wl = self.workload
            R_l0 = wl.R // (R_l2 * R_l3)
            S_l0 = wl.S // (S_l2 * S_l3)
            P_l0 = wl.P // (P_l2 * P_l3)
            Q_l0 = wl.Q // (Q_l2 * Q_l3)
            C_l0 = wl.C // (C_l2 * C_l3)
            K_l0 = wl.K // (K_l2 * K_l3)
            N_l0 = wl.N // (N_l2 * N_l3)
            
            return MappingConfig(
                R_l3=R_l3, S_l3=S_l3, P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3, N_l3=N_l3,
                R_l2=R_l2, S_l2=S_l2, P_l2=P_l2, Q_l2=Q_l2, C_l2=C_l2, K_l2=K_l2, N_l2=N_l2,
                R_l0=R_l0, S_l0=S_l0, P_l0=P_l0, Q_l0=Q_l0, C_l0=C_l0, K_l0=K_l0, N_l0=N_l0,
                permutation_l3=perm,
                input_layout=in_lay, weight_layout=w_lay, output_layout=out_lay,
                block_h=bh, block_w=bw
            )
        
        else:  # FULL
            # tiling = ((R_l0, R_l1, R_l2, R_l3), ..., (N_l0, N_l1, N_l2, N_l3))
            R_l0, R_l1, R_l2, R_l3 = tiling[0]
            S_l0, S_l1, S_l2, S_l3 = tiling[1]
            P_l0, P_l1, P_l2, P_l3 = tiling[2]
            Q_l0, Q_l1, Q_l2, Q_l3 = tiling[3]
            C_l0, C_l1, C_l2, C_l3 = tiling[4]
            K_l0, K_l1, K_l2, K_l3 = tiling[5]
            N_l0, N_l1, N_l2, N_l3 = tiling[6]
            
            return MappingConfig(
                R_l3=R_l3, S_l3=S_l3, P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3, N_l3=N_l3,
                R_l2=R_l2, S_l2=S_l2, P_l2=P_l2, Q_l2=Q_l2, C_l2=C_l2, K_l2=K_l2, N_l2=N_l2,
                R_l1=R_l1, S_l1=S_l1, P_l1=P_l1, Q_l1=Q_l1, C_l1=C_l1, K_l1=K_l1, N_l1=N_l1,
                R_l0=R_l0, S_l0=S_l0, P_l0=P_l0, Q_l0=Q_l0, C_l0=C_l0, K_l0=K_l0, N_l0=N_l0,
                permutation_l3=perm,
                input_layout=in_lay, weight_layout=w_lay, output_layout=out_lay,
                block_h=bh, block_w=bw
            )
    
    # ==================== 统计方法 ====================
    
    def count_space(self,
                    mode: TilingMode = TilingMode.L3_L2,
                    vary_permutation: bool = True,
                    vary_layout: bool = True,
                    vary_block_size: bool = True) -> Dict[str, int]:
        """计算空间大小（不实际枚举）"""
        
        if mode == TilingMode.L3_L2:
            # 2 层分解数 = divisor 数
            tiling_count = 1
            for dim in ['R', 'S', 'P', 'Q', 'C', 'K', 'N']:
                tiling_count *= len(self._divisors[dim])
        else:  # FULL
            # 估算：4 层分解数约为 divisor 数的立方（粗略估计）
            tiling_count = 1
            for dim in ['R', 'S', 'P', 'Q', 'C', 'K', 'N']:
                d = len(self._divisors[dim])
                # 实际分解数比 d^3 小，但用这个作为上界
                tiling_count *= len(get_factor_decompositions(getattr(self.workload, dim), 4))
        
        perm_count = 24 if vary_permutation else 1
        layout_count = 8 if vary_layout else 1
        block_count = len(self._candidate_block_sizes) if vary_block_size else 1
        
        return {
            'tiling_combos': tiling_count,
            'permutation_count': perm_count,
            'layout_combos': layout_count,
            'block_size_count': block_count,
            'total': tiling_count * perm_count * layout_count * block_count
        }
    
    def summary(self):
        """打印空间摘要"""
        wl = self.workload
        print("=" * 60)
        print("MappingEnumerator Summary")
        print("=" * 60)
        print(f"Workload: P={wl.P}, Q={wl.Q}, C={wl.C}, K={wl.K}, R={wl.R}, S={wl.S}, N={wl.N}")
        print(f"Arch: row_buffer={self.arch.row_buffer_bytes}B, global_buffer={self.arch.global_buffer_bytes}B")
        print(f"Constraint validation: {self.validate_constraints}")
        print()
        
        for mode in [TilingMode.L3_L2]:  # 不计算 FULL，太慢
            counts = self.count_space(mode, vary_permutation=True, vary_layout=True, vary_block_size=True)
            print(f"{mode.value}:")
            print(f"  Tiling: {counts['tiling_combos']:,}, Total: {counts['total']:,}")
