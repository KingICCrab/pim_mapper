#!/usr/bin/env python3
"""
MappingConfig: 单个 mapping 配置的数据类

Memory hierarchy (4 levels):
- Level 0: PE (register)       - 最内层，计算单元
- Level 1: GlobalBuffer (SRAM) - 片上缓存
- Level 2: RowBuffer           - DRAM row buffer
- Level 3: LocalDRAM           - 最外层，主存
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .config import DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N, DIM_NAMES, DEFAULT_PERMUTATION


@dataclass
class MappingConfig:
    """一个具体的 mapping 配置
    
    包含:
    - 4 层级的 tiling factors (L0-L3, 每层 7 个维度)
    - 各层级的 permutation
    - Layout (sequential / row_aligned)
    - Block size (用于 row_aligned layout)
    """
    
    # ==================== Level 3: DRAM level ====================
    R_l3: int = 1
    S_l3: int = 1
    P_l3: int = 1
    Q_l3: int = 1
    C_l3: int = 1
    K_l3: int = 1
    N_l3: int = 1
    
    # ==================== Level 2: RowBuffer level ====================
    R_l2: int = 1
    S_l2: int = 1
    P_l2: int = 1
    Q_l2: int = 1
    C_l2: int = 1
    K_l2: int = 1
    N_l2: int = 1
    
    # ==================== Level 1: GlobalBuffer (SRAM) level ====================
    R_l1: int = 1
    S_l1: int = 1
    P_l1: int = 1
    Q_l1: int = 1
    C_l1: int = 1
    K_l1: int = 1
    N_l1: int = 1
    
    # ==================== Level 0: PE level ====================
    R_l0: int = 1
    S_l0: int = 1
    P_l0: int = 1
    Q_l0: int = 1
    C_l0: int = 1
    K_l0: int = 1
    N_l0: int = 1
    
    # ==================== Permutation ====================
    permutation_l3: Tuple[int, ...] = DEFAULT_PERMUTATION
    permutation_l2: Tuple[int, ...] = ()  # 空表示使用 L3 的顺序
    
    # ==================== Layout ====================
    input_layout: str = 'sequential'
    weight_layout: str = 'sequential'
    output_layout: str = 'sequential'
    
    # Block size for row_aligned layout
    block_h: int = 1
    block_w: int = 1
    
    # ==================== 获取 tiling factors ====================
    
    def get_factors(self, level: int) -> Dict[int, int]:
        """返回指定层级的 tiling factors 字典"""
        if level == 3:
            return {DIM_R: self.R_l3, DIM_S: self.S_l3, DIM_P: self.P_l3, DIM_Q: self.Q_l3,
                    DIM_C: self.C_l3, DIM_K: self.K_l3, DIM_N: self.N_l3}
        elif level == 2:
            return {DIM_R: self.R_l2, DIM_S: self.S_l2, DIM_P: self.P_l2, DIM_Q: self.Q_l2,
                    DIM_C: self.C_l2, DIM_K: self.K_l2, DIM_N: self.N_l2}
        elif level == 1:
            return {DIM_R: self.R_l1, DIM_S: self.S_l1, DIM_P: self.P_l1, DIM_Q: self.Q_l1,
                    DIM_C: self.C_l1, DIM_K: self.K_l1, DIM_N: self.N_l1}
        elif level == 0:
            return {DIM_R: self.R_l0, DIM_S: self.S_l0, DIM_P: self.P_l0, DIM_Q: self.Q_l0,
                    DIM_C: self.C_l0, DIM_K: self.K_l0, DIM_N: self.N_l0}
        else:
            raise ValueError(f"Invalid level: {level}, must be 0-3")
    
    def get_total_factor(self, dim: int) -> int:
        """计算某维度的总 tiling factor (L0 * L1 * L2 * L3)"""
        factors = {
            DIM_R: self.R_l0 * self.R_l1 * self.R_l2 * self.R_l3,
            DIM_S: self.S_l0 * self.S_l1 * self.S_l2 * self.S_l3,
            DIM_P: self.P_l0 * self.P_l1 * self.P_l2 * self.P_l3,
            DIM_Q: self.Q_l0 * self.Q_l1 * self.Q_l2 * self.Q_l3,
            DIM_C: self.C_l0 * self.C_l1 * self.C_l2 * self.C_l3,
            DIM_K: self.K_l0 * self.K_l1 * self.K_l2 * self.K_l3,
            DIM_N: self.N_l0 * self.N_l1 * self.N_l2 * self.N_l3,
        }
        return factors.get(dim, 1)
    
    def get_permutation(self, level: int) -> Tuple[int, ...]:
        """获取指定层级的 permutation"""
        if level == 3:
            return self.permutation_l3
        elif level == 2:
            return self.permutation_l2 if self.permutation_l2 else self.permutation_l3
        else:
            # L0, L1 使用 L3 的 permutation
            return self.permutation_l3
    
    # ==================== 验证 ====================
    
    def validate_tiling(self, workload) -> Tuple[bool, List[str]]:
        """验证 tiling factors 是否与 workload 匹配
        
        检查: L0 * L1 * L2 * L3 == workload dimension
        """
        errors = []
        dims = {
            'R': (getattr(workload, 'R', 1), self.get_total_factor(DIM_R)),
            'S': (getattr(workload, 'S', 1), self.get_total_factor(DIM_S)),
            'P': (getattr(workload, 'P', 1), self.get_total_factor(DIM_P)),
            'Q': (getattr(workload, 'Q', 1), self.get_total_factor(DIM_Q)),
            'C': (getattr(workload, 'C', 1), self.get_total_factor(DIM_C)),
            'K': (getattr(workload, 'K', 1), self.get_total_factor(DIM_K)),
            'N': (getattr(workload, 'N', 1), self.get_total_factor(DIM_N)),
        }
        
        for name, (workload_val, total_factor) in dims.items():
            if total_factor != workload_val:
                errors.append(f"{name}: workload={workload_val}, total_factor={total_factor}")
        
        return len(errors) == 0, errors
    
    # ==================== 字符串表示 ====================
    
    def __str__(self):
        """完整字符串，显示所有非 1 的 tiling factors"""
        parts = []
        for level, prefix in [(3, 'L3'), (2, 'L2'), (1, 'L1'), (0, 'L0')]:
            factors = self.get_factors(level)
            non_one = {d: v for d, v in factors.items() if v > 1}
            if non_one:
                factor_str = ''.join(f"{DIM_NAMES[d]}{v}" for d, v in non_one.items())
                parts.append(f"{prefix}[{factor_str}]")
        
        tiling_str = '_'.join(parts) if parts else "all1"
        perm_str = '-'.join(DIM_NAMES[d] for d in self.permutation_l3)
        return (f"{tiling_str}_perm({perm_str})_"
                f"layout({self.input_layout[0]}{self.weight_layout[0]}{self.output_layout[0]})_"
                f"blk({self.block_h}x{self.block_w})")
    
    def short_str(self) -> str:
        """简短字符串，只显示非 1 的 factors"""
        parts = []
        
        for level, prefix in [(3, 'L3'), (2, 'L2'), (1, 'L1')]:
            factors = self.get_factors(level)
            factor_parts = [f"{DIM_NAMES[d]}{v}" for d, v in factors.items() if v > 1]
            if factor_parts:
                parts.append(f"{prefix}[{''.join(factor_parts)}]")
        
        if not parts:
            parts.append("all1")
        
        perm_str = '-'.join(DIM_NAMES[d] for d in self.permutation_l3)
        return (f"{'_'.join(parts)}_perm({perm_str})_"
                f"layout({self.input_layout[0]}{self.weight_layout[0]}{self.output_layout[0]})_"
                f"blk({self.block_h}x{self.block_w})")
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            # L3 factors
            'R_l3': self.R_l3, 'S_l3': self.S_l3, 'P_l3': self.P_l3, 'Q_l3': self.Q_l3,
            'C_l3': self.C_l3, 'K_l3': self.K_l3, 'N_l3': self.N_l3,
            # L2 factors
            'R_l2': self.R_l2, 'S_l2': self.S_l2, 'P_l2': self.P_l2, 'Q_l2': self.Q_l2,
            'C_l2': self.C_l2, 'K_l2': self.K_l2, 'N_l2': self.N_l2,
            # L1 factors
            'R_l1': self.R_l1, 'S_l1': self.S_l1, 'P_l1': self.P_l1, 'Q_l1': self.Q_l1,
            'C_l1': self.C_l1, 'K_l1': self.K_l1, 'N_l1': self.N_l1,
            # L0 factors
            'R_l0': self.R_l0, 'S_l0': self.S_l0, 'P_l0': self.P_l0, 'Q_l0': self.Q_l0,
            'C_l0': self.C_l0, 'K_l0': self.K_l0, 'N_l0': self.N_l0,
            # Other
            'permutation_l3': '-'.join(DIM_NAMES[d] for d in self.permutation_l3),
            'input_layout': self.input_layout,
            'weight_layout': self.weight_layout,
            'output_layout': self.output_layout,
            'block_h': self.block_h,
            'block_w': self.block_w
        }
