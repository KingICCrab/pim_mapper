#!/usr/bin/env python3
"""
Mapping Space: 枚举卷积 workload 的合法 mapping 空间

合法 Mapping 的 5 个自由度:
1. Tiling Factors (L3): P, Q, C, K 的因子
2. Permutation: L3 层循环顺序
3. Layout: 每个 tensor (input/weight/output) 的布局 (sequential/row_aligned)
4. Block Size: block_h, block_w 用于 row_aligned layout

ILP 约束 (来自 src/pim_optimizer/model/constraints.py):
1. Dimension factorization: L3 factor 必须能整除对应维度
2. Block size constraint: block_h * block_w <= row_buffer_bytes
3. Buffer capacity constraints: 每个 tensor 的 tile 不能超过 buffer 容量
4. Input tile in RowBuffer: Input tile 必须能放入 RowBuffer
"""

from itertools import permutations, product
from typing import List, Dict, Tuple, Iterator, Optional
from dataclasses import dataclass, field

# 维度常量
DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6
DIM_NAMES = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']


@dataclass
class ArchConfig:
    """架构配置参数"""
    # DRAM 配置
    row_buffer_bytes: int = 1024      # RowBuffer 大小 (bytes)
    num_banks: int = 4                # Bank 数量
    num_rows: int = 16384             # 每个 bank 的 row 数量
    element_size: int = 1             # 每个元素的字节数
    
    # Buffer 配置 (简化: 只考虑 RowBuffer 和 GlobalBuffer)
    global_buffer_bytes: int = 256 * 4  # GlobalBuffer 大小 (256KB)
    
    # PE Array 配置 (对于 row activation 验证可以忽略)
    pe_array_h: int = 16
    pe_array_w: int = 16
    
    def __post_init__(self):
        self.row_buffer_elements = self.row_buffer_bytes // self.element_size
        self.global_buffer_elements = self.global_buffer_bytes // self.element_size


@dataclass 
class WorkloadConfig:
    """Workload 配置参数"""
    # 必需参数 (无默认值的放前面)
    P: int  # Output height
    Q: int  # Output width
    K: int  # Output channels
    C: int  # Input channels
    H: int  # Input height
    W: int  # Input width
    
    # 可选参数 (有默认值)
    N: int = 1  # Batch size
    R: int = 3  # Filter height
    S: int = 3  # Filter width
    stride_h: int = 1
    stride_w: int = 1
    dilation_h: int = 1
    dilation_w: int = 1


def get_divisors(n: int) -> List[int]:
    """获取 n 的所有因子"""
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted(divisors)


def get_factor_decompositions(n: int, num_levels: int = 4) -> List[Tuple[int, ...]]:
    """获取 n 的所有 num_levels 层因子分解
    
    返回所有满足 f0 * f1 * ... * f_{num_levels-1} = n 的组合
    
    Args:
        n: 要分解的数
        num_levels: 层级数量，默认 4 (L0, L1, L2, L3)
        
    Returns:
        所有因子分解的列表，每个元素是 (f0, f1, ..., f_{num_levels-1})
    """
    if num_levels == 1:
        return [(n,)]
    
    divisors = get_divisors(n)
    results = []
    
    for d in divisors:
        # d 作为第一层的因子，剩余 n/d 分解到后面的层
        remaining = n // d
        sub_decomps = get_factor_decompositions(remaining, num_levels - 1)
        for sub in sub_decomps:
            results.append((d,) + sub)
    
    return results


@dataclass
class MappingConfig:
    """一个具体的 mapping 配置
    
    与 trace_generator.py 的对齐说明:
    - trace_generator 期望 mapping.layout = {0: str, 1: str, 2: str}
    - trace_generator 期望 mapping.tile_info = {'block_h': int, 'block_w': int}
    - trace_generator 期望 mapping.permutation = {level: {pos: dim}}，包含全部 7 个维度
    - trace_generator 期望 mapping.loop_bounds = {level: {key: {dim: bound}}}
    
    Memory hierarchy (4 levels):
    - Level 0: PE (register)       - 最内层，计算单元
    - Level 1: GlobalBuffer (SRAM) - 片上缓存
    - Level 2: RowBuffer           - DRAM row buffer
    - Level 3: LocalDRAM           - 最外层，主存
    
    维度定义 (DIM_R=0, DIM_S=1, DIM_P=2, DIM_Q=3, DIM_C=4, DIM_K=5, DIM_N=6):
    - R, S: Filter 维度
    - P, Q: Output 空间维度
    - C: Input channel
    - K: Output channel
    - N: Batch
    
    本类提供属性方法来适配这些接口。
    """
    # ==================== Level 3: DRAM level tiling factors ====================
    R_l3: int = 1  # Filter height (通常不 tile)
    S_l3: int = 1  # Filter width (通常不 tile)
    P_l3: int = 1  # Output height
    Q_l3: int = 1  # Output width
    C_l3: int = 1  # Input channels
    K_l3: int = 1  # Output channels
    N_l3: int = 1  # Batch size (通常不 tile)
    
    # ==================== Level 2: RowBuffer level tiling factors ====================
    R_l2: int = 1
    S_l2: int = 1
    P_l2: int = 1
    Q_l2: int = 1
    C_l2: int = 1
    K_l2: int = 1
    N_l2: int = 1
    
    # ==================== Level 1: GlobalBuffer (SRAM) level tiling factors ====================
    R_l1: int = 1
    S_l1: int = 1
    P_l1: int = 1
    Q_l1: int = 1
    C_l1: int = 1
    K_l1: int = 1
    N_l1: int = 1
    
    # ==================== Level 0: PE level tiling factors ====================
    R_l0: int = 1
    S_l0: int = 1
    P_l0: int = 1
    Q_l0: int = 1
    C_l0: int = 1
    K_l0: int = 1
    N_l0: int = 1
    
    # ==================== Permutation for each level ====================
    # Permutation: tuple of all 7 dims from inner to outer
    # e.g., (DIM_R, DIM_S, DIM_Q, DIM_P, DIM_C, DIM_K, DIM_N)
    permutation_l3: Tuple[int, ...] = (DIM_R, DIM_S, DIM_Q, DIM_P, DIM_C, DIM_K, DIM_N)
    permutation_l2: Tuple[int, ...] = ()  # 默认为空，表示使用 L3 的顺序
    permutation_l1: Tuple[int, ...] = ()
    
    # 兼容旧接口：permutation 默认指向 L3
    permutation: Tuple[int, ...] = field(default=None)
    
    # Layout for each tensor: 'sequential' or 'row_aligned'
    input_layout: str = 'sequential'
    weight_layout: str = 'sequential'
    output_layout: str = 'sequential'
    
    # Block size for row_aligned layout
    block_h: int = 1
    block_w: int = 1
    
    def __post_init__(self):
        """初始化后处理"""
        # 如果 permutation 未设置，使用 permutation_l3
        if self.permutation is None:
            object.__setattr__(self, 'permutation', self.permutation_l3)
    
    @classmethod
    def from_workload(cls, workload, 
                      l3_factors: Dict[str, int] = None,
                      l2_factors: Dict[str, int] = None,
                      l1_factors: Dict[str, int] = None,
                      l0_factors: Dict[str, int] = None,
                      permutation_l3: Tuple[int, ...] = None,
                      permutation_l2: Tuple[int, ...] = None,
                      permutation_l1: Tuple[int, ...] = None,
                      input_layout: str = 'sequential',
                      weight_layout: str = 'sequential', 
                      output_layout: str = 'sequential',
                      block_h: int = 1, block_w: int = 1,
                      auto_complete: bool = True) -> 'MappingConfig':
        """从 workload 创建多层级配置
        
        Args:
            workload: 包含 R, S, P, Q, C, K, N 的 workload 对象
            l3_factors: L3 (DRAM) tiling factors，格式 {'P': 2, 'Q': 2, ...}
            l2_factors: L2 (RowBuffer) tiling factors
            l1_factors: L1 (GlobalBuffer) tiling factors
            l0_factors: L0 (PE) tiling factors
            permutation_l3, permutation_l2, permutation_l1: 各层级循环顺序
            input_layout, weight_layout, output_layout: tensor 布局
            block_h, block_w: row_aligned 布局的 block 大小
            auto_complete: 如果 True，自动计算未指定层级的 factors
                          使得 L0 * L1 * L2 * L3 = workload dimension
            
        Returns:
            MappingConfig 实例
            
        Note:
            如果 auto_complete=True 且某层级未指定，会自动填充：
            - 优先使用指定的 factors
            - 剩余的 factor 放入最低未指定层级
        """
        R = getattr(workload, 'R', 1)
        S = getattr(workload, 'S', 1)
        P = getattr(workload, 'P', 1)
        Q = getattr(workload, 'Q', 1)
        C = getattr(workload, 'C', 1)
        K = getattr(workload, 'K', 1)
        N = getattr(workload, 'N', 1)
        
        # 初始化各层级 factors，默认全 1
        l3 = l3_factors or {}
        l2 = l2_factors or {}
        l1 = l1_factors or {}
        l0 = l0_factors or {}
        
        # 解析各层级的 factors
        factors = {
            3: {d: l3.get(d, 1) for d in ['R', 'S', 'P', 'Q', 'C', 'K', 'N']},
            2: {d: l2.get(d, 1) for d in ['R', 'S', 'P', 'Q', 'C', 'K', 'N']},
            1: {d: l1.get(d, 1) for d in ['R', 'S', 'P', 'Q', 'C', 'K', 'N']},
            0: {d: l0.get(d, 1) for d in ['R', 'S', 'P', 'Q', 'C', 'K', 'N']},
        }
        
        # 如果 auto_complete，计算剩余 factor
        if auto_complete:
            workload_dims = {'R': R, 'S': S, 'P': P, 'Q': Q, 'C': C, 'K': K, 'N': N}
            for dim_name, total in workload_dims.items():
                # 计算已指定的乘积
                specified_product = (factors[3][dim_name] * factors[2][dim_name] * 
                                    factors[1][dim_name] * factors[0][dim_name])
                
                if specified_product == 1:
                    # 没有任何层级指定，全部放入 L0
                    factors[0][dim_name] = total
                elif specified_product != total:
                    # 有部分指定，计算剩余
                    remaining = total // specified_product
                    if remaining > 1:
                        # 找到第一个为 1 的层级，放入剩余 factor
                        for level in [0, 1, 2, 3]:
                            if factors[level][dim_name] == 1:
                                factors[level][dim_name] = remaining
                                break
        
        # 默认 permutation
        if permutation_l3 is None:
            permutation_l3 = (DIM_R, DIM_S, DIM_Q, DIM_P, DIM_C, DIM_K, DIM_N)
        
        return cls(
            R_l3=factors[3]['R'], S_l3=factors[3]['S'], 
            P_l3=factors[3]['P'], Q_l3=factors[3]['Q'], 
            C_l3=factors[3]['C'], K_l3=factors[3]['K'], N_l3=factors[3]['N'],
            R_l2=factors[2]['R'], S_l2=factors[2]['S'], 
            P_l2=factors[2]['P'], Q_l2=factors[2]['Q'], 
            C_l2=factors[2]['C'], K_l2=factors[2]['K'], N_l2=factors[2]['N'],
            R_l1=factors[1]['R'], S_l1=factors[1]['S'], 
            P_l1=factors[1]['P'], Q_l1=factors[1]['Q'], 
            C_l1=factors[1]['C'], K_l1=factors[1]['K'], N_l1=factors[1]['N'],
            R_l0=factors[0]['R'], S_l0=factors[0]['S'], 
            P_l0=factors[0]['P'], Q_l0=factors[0]['Q'], 
            C_l0=factors[0]['C'], K_l0=factors[0]['K'], N_l0=factors[0]['N'],
            permutation_l3=permutation_l3,
            permutation_l2=permutation_l2 or (),
            permutation_l1=permutation_l1 or (),
            input_layout=input_layout,
            weight_layout=weight_layout,
            output_layout=output_layout,
            block_h=block_h,
            block_w=block_w
        )
    
    def get_permutation(self, level: int) -> Tuple[int, ...]:
        """获取指定层级的 permutation"""
        if level == 3:
            return self.permutation_l3
        elif level == 2:
            return self.permutation_l2 if self.permutation_l2 else self.permutation_l3
        elif level == 1:
            return self.permutation_l1 if self.permutation_l1 else (self.permutation_l2 if self.permutation_l2 else self.permutation_l3)
        else:
            return self.permutation_l3

    def __str__(self):
        perm_str = '-'.join(DIM_NAMES[d] for d in self.permutation_l3)
        return (f"L3[R{self.R_l3}S{self.S_l3}P{self.P_l3}Q{self.Q_l3}C{self.C_l3}K{self.K_l3}N{self.N_l3}]_"
                f"perm({perm_str})_"
                f"layout({self.input_layout[0]}{self.weight_layout[0]}{self.output_layout[0]})_"
                f"blk({self.block_h}x{self.block_w})")
    
    def short_str(self):
        """简短的字符串表示，只显示非 1 的 tiling factors"""
        parts = []
        # L3 factors
        l3_parts = []
        if self.R_l3 > 1: l3_parts.append(f"R{self.R_l3}")
        if self.S_l3 > 1: l3_parts.append(f"S{self.S_l3}")
        if self.P_l3 > 1: l3_parts.append(f"P{self.P_l3}")
        if self.Q_l3 > 1: l3_parts.append(f"Q{self.Q_l3}")
        if self.C_l3 > 1: l3_parts.append(f"C{self.C_l3}")
        if self.K_l3 > 1: l3_parts.append(f"K{self.K_l3}")
        if self.N_l3 > 1: l3_parts.append(f"N{self.N_l3}")
        if l3_parts:
            parts.append(f"L3[{''.join(l3_parts)}]")
        
        # L2 factors (if any non-1)
        l2_parts = []
        if self.R_l2 > 1: l2_parts.append(f"R{self.R_l2}")
        if self.S_l2 > 1: l2_parts.append(f"S{self.S_l2}")
        if self.P_l2 > 1: l2_parts.append(f"P{self.P_l2}")
        if self.Q_l2 > 1: l2_parts.append(f"Q{self.Q_l2}")
        if self.C_l2 > 1: l2_parts.append(f"C{self.C_l2}")
        if self.K_l2 > 1: l2_parts.append(f"K{self.K_l2}")
        if self.N_l2 > 1: l2_parts.append(f"N{self.N_l2}")
        if l2_parts:
            parts.append(f"L2[{''.join(l2_parts)}]")
        
        # L1 factors (if any non-1)
        l1_parts = []
        if self.R_l1 > 1: l1_parts.append(f"R{self.R_l1}")
        if self.S_l1 > 1: l1_parts.append(f"S{self.S_l1}")
        if self.P_l1 > 1: l1_parts.append(f"P{self.P_l1}")
        if self.Q_l1 > 1: l1_parts.append(f"Q{self.Q_l1}")
        if self.C_l1 > 1: l1_parts.append(f"C{self.C_l1}")
        if self.K_l1 > 1: l1_parts.append(f"K{self.K_l1}")
        if self.N_l1 > 1: l1_parts.append(f"N{self.N_l1}")
        if l1_parts:
            parts.append(f"L1[{''.join(l1_parts)}]")
        
        if not parts:
            parts.append("all1")
        
        perm_str = '-'.join(DIM_NAMES[d] for d in self.permutation_l3)
        return (f"{'_'.join(parts)}_perm({perm_str})_"
                f"layout({self.input_layout[0]}{self.weight_layout[0]}{self.output_layout[0]})_"
                f"blk({self.block_h}x{self.block_w})")
    
    def to_dict(self):
        return {
            # L3 factors
            'R_l3': self.R_l3, 'S_l3': self.S_l3,
            'P_l3': self.P_l3, 'Q_l3': self.Q_l3, 
            'C_l3': self.C_l3, 'K_l3': self.K_l3, 'N_l3': self.N_l3,
            # L2 factors
            'R_l2': self.R_l2, 'S_l2': self.S_l2,
            'P_l2': self.P_l2, 'Q_l2': self.Q_l2, 
            'C_l2': self.C_l2, 'K_l2': self.K_l2, 'N_l2': self.N_l2,
            # L1 factors
            'R_l1': self.R_l1, 'S_l1': self.S_l1,
            'P_l1': self.P_l1, 'Q_l1': self.Q_l1, 
            'C_l1': self.C_l1, 'K_l1': self.K_l1, 'N_l1': self.N_l1,
            # L0 factors
            'R_l0': self.R_l0, 'S_l0': self.S_l0,
            'P_l0': self.P_l0, 'Q_l0': self.Q_l0, 
            'C_l0': self.C_l0, 'K_l0': self.K_l0, 'N_l0': self.N_l0,
            # Permutation
            'permutation_l3': '-'.join(DIM_NAMES[d] for d in self.permutation_l3),
            # Layout
            'input_layout': self.input_layout,
            'weight_layout': self.weight_layout,
            'output_layout': self.output_layout,
            'block_h': self.block_h,
            'block_w': self.block_w
        }
    
    def get_factors(self, level: int) -> Dict[int, int]:
        """返回指定层级的 tiling factors 字典
        
        Args:
            level: 0=PE, 1=GlobalBuffer, 2=RowBuffer, 3=DRAM
        """
        if level == 3:
            return {
                DIM_R: self.R_l3, DIM_S: self.S_l3,
                DIM_P: self.P_l3, DIM_Q: self.Q_l3,
                DIM_C: self.C_l3, DIM_K: self.K_l3, DIM_N: self.N_l3
            }
        elif level == 2:
            return {
                DIM_R: self.R_l2, DIM_S: self.S_l2,
                DIM_P: self.P_l2, DIM_Q: self.Q_l2,
                DIM_C: self.C_l2, DIM_K: self.K_l2, DIM_N: self.N_l2
            }
        elif level == 1:
            return {
                DIM_R: self.R_l1, DIM_S: self.S_l1,
                DIM_P: self.P_l1, DIM_Q: self.Q_l1,
                DIM_C: self.C_l1, DIM_K: self.K_l1, DIM_N: self.N_l1
            }
        elif level == 0:
            return {
                DIM_R: self.R_l0, DIM_S: self.S_l0,
                DIM_P: self.P_l0, DIM_Q: self.Q_l0,
                DIM_C: self.C_l0, DIM_K: self.K_l0, DIM_N: self.N_l0
            }
        else:
            raise ValueError(f"Invalid level: {level}, must be 0-3")
    
    def get_l3_factors(self) -> Dict[int, int]:
        """返回 L3 tiling factors（兼容旧接口）"""
        return self.get_factors(3)
    
    # ==================== trace_generator 兼容属性 ====================
    
    @property
    def layout(self) -> Dict[int, str]:
        """返回 trace_generator 期望的 layout 字典格式
        
        Returns:
            {0: input_layout, 1: weight_layout, 2: output_layout}
        """
        return {
            0: self.input_layout,
            1: self.weight_layout,
            2: self.output_layout
        }
    
    @property
    def tile_info(self) -> Dict[str, int]:
        """返回 trace_generator 期望的 tile_info 字典格式
        
        Returns:
            {'block_h': block_h, 'block_w': block_w}
        """
        return {
            'block_h': self.block_h,
            'block_w': self.block_w
        }
    
    def get_permutation_dict(self, level: int = 3) -> Dict[int, int]:
        """返回 trace_generator 期望的 permutation 字典格式
        
        Args:
            level: 循环层级 (0=PE, 1=GlobalBuffer, 2=RowBuffer, 3=DRAM)
            
        Returns:
            {pos: dim} 格式的字典，包含全部 7 个维度
        """
        if level == 3:
            perm = self.permutation_l3
        elif level == 2:
            perm = self.permutation_l2 if self.permutation_l2 else self.permutation_l3
        elif level == 1:
            perm = self.permutation_l1 if self.permutation_l1 else self.permutation_l3
        else:
            perm = self.permutation_l3
        return {i: dim for i, dim in enumerate(perm)}
    
    def build_loop_bounds(self, workload=None) -> Dict[int, Dict]:
        """构建 trace_generator 期望的 loop_bounds 结构
        
        Memory hierarchy:
        - Level 0: PE (register)      ─┬─ buffer_tile (SRAM内复用)
        - Level 1: GlobalBuffer (SRAM) ─┘
        - Level 2: RowBuffer           ─┬─ dram_loops (DRAM访问循环)
        - Level 3: LocalDRAM           ─┘
        
        Args:
            workload: 包含 R, S, P, Q, C, K, N 等属性的 workload 对象（可选）
                      如果提供，用于验证 tiling factors；否则直接使用配置值
            
        Returns:
            {level: {key: {dim: bound}}} 格式的字典
        """
        loop_bounds = {}
        
        # Level 0: PE level - 使用 L0 factors
        # 注意: Level 0 使用特殊的 'H', 'W', 'Internal', 'temporal' 格式
        loop_bounds[0] = {
            'H': {
                DIM_P: self.P_l0,
                DIM_R: self.R_l0,
            },
            'W': {
                DIM_Q: self.Q_l0,
                DIM_S: self.S_l0,
            },
            'Internal': {
                DIM_C: self.C_l0,
            },
            'temporal': {
                DIM_K: self.K_l0,
                DIM_N: self.N_l0,
            }
        }
        
        # Level 1: GlobalBuffer (SRAM) - 使用 L1 factors
        loop_bounds[1] = {
            'spatial': {},
            'temporal': {
                DIM_R: self.R_l1,
                DIM_S: self.S_l1,
                DIM_P: self.P_l1,
                DIM_Q: self.Q_l1,
                DIM_C: self.C_l1,
                DIM_K: self.K_l1,
                DIM_N: self.N_l1
            }
        }
        
        # Level 2: RowBuffer - 使用 L2 factors
        loop_bounds[2] = {
            'spatial': {},
            'temporal': {
                DIM_R: self.R_l2,
                DIM_S: self.S_l2,
                DIM_P: self.P_l2,
                DIM_Q: self.Q_l2,
                DIM_C: self.C_l2,
                DIM_K: self.K_l2,
                DIM_N: self.N_l2
            }
        }
        
        # Level 3: DRAM - 使用 L3 factors
        loop_bounds[3] = {
            'spatial': {},
            'temporal': {
                DIM_R: self.R_l3,
                DIM_S: self.S_l3,
                DIM_P: self.P_l3,
                DIM_Q: self.Q_l3,
                DIM_C: self.C_l3,
                DIM_K: self.K_l3,
                DIM_N: self.N_l3
            }
        }
        
        return loop_bounds
    
    def build_permutation_dict(self) -> Dict[int, Dict[int, int]]:
        """构建 trace_generator 期望的 permutation 结构
        
        Returns:
            {level: {pos: dim}} 格式的字典
        """
        result = {}
        
        # Level 1: 使用 permutation_l1，如果为空则用 permutation_l3
        perm_l1 = self.permutation_l1 if self.permutation_l1 else self.permutation_l3
        result[1] = {i: dim for i, dim in enumerate(perm_l1)} if perm_l1 else {}
        
        # Level 2: 使用 permutation_l2，如果为空则用 permutation_l3
        perm_l2 = self.permutation_l2 if self.permutation_l2 else self.permutation_l3
        result[2] = {i: dim for i, dim in enumerate(perm_l2)} if perm_l2 else {}
        
        # Level 3: 使用 permutation_l3
        result[3] = {i: dim for i, dim in enumerate(self.permutation_l3)}
        
        return result
    
    def to_trace_generator_mapping(self, workload=None):
        """创建一个完全兼容 trace_generator 的 mapping 对象
        
        Args:
            workload: 包含卷积参数的 workload 对象（可选）
            
        Returns:
            一个具有 loop_bounds, permutation, layout, tile_info 属性的对象
        """
        class TraceGeneratorCompatibleMapping:
            """trace_generator 兼容的 Mapping 类"""
            pass
        
        mapping = TraceGeneratorCompatibleMapping()
        mapping.loop_bounds = self.build_loop_bounds(workload)
        mapping.permutation = self.build_permutation_dict()
        mapping.layout = self.layout
        mapping.tile_info = self.tile_info
        
        # 添加 get_tile_size 方法
        def get_tile_size(level, dim):
            if level not in mapping.loop_bounds:
                return 1
            level_bounds = mapping.loop_bounds[level]
            factor = 1
            for key in ['spatial', 'temporal', 'H', 'W', 'Internal']:
                if key in level_bounds and dim in level_bounds[key]:
                    factor *= level_bounds[key][dim]
            return factor
        
        mapping.get_tile_size = get_tile_size
        
        return mapping
    
    def compute_total_factors(self, dim: int) -> int:
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
    
    def validate_tiling(self, workload) -> Tuple[bool, List[str]]:
        """验证 tiling factors 是否与 workload 匹配
        
        检查: L0 * L1 * L2 * L3 == workload dimension
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        R = getattr(workload, 'R', 1)
        S = getattr(workload, 'S', 1)
        P = getattr(workload, 'P', 1)
        Q = getattr(workload, 'Q', 1)
        C = getattr(workload, 'C', 1)
        K = getattr(workload, 'K', 1)
        N = getattr(workload, 'N', 1)
        
        checks = [
            ('R', R, self.compute_total_factors(DIM_R)),
            ('S', S, self.compute_total_factors(DIM_S)),
            ('P', P, self.compute_total_factors(DIM_P)),
            ('Q', Q, self.compute_total_factors(DIM_Q)),
            ('C', C, self.compute_total_factors(DIM_C)),
            ('K', K, self.compute_total_factors(DIM_K)),
            ('N', N, self.compute_total_factors(DIM_N)),
        ]
        
        for dim_name, workload_val, total_factor in checks:
            if total_factor != workload_val:
                errors.append(f"{dim_name}: workload={workload_val}, total_factor={total_factor}")
        
        return len(errors) == 0, errors
    
    @classmethod
    def enumerate_all_mappings(cls, workload,
                               vary_permutation: bool = True,
                               vary_layout: bool = True,
                               vary_block_size: bool = True,
                               permutation_list: List[Tuple[int, ...]] = None,
                               layout_list: List[str] = None,
                               block_sizes: List[Tuple[int, int]] = None,
                               max_block_area: int = 1024,
                               sample_tiling: int = None,
                               sample_total: int = None) -> Iterator['MappingConfig']:
        """枚举给定 workload 的所有合法 mapping
        
        对于每个维度，枚举所有满足 L0 * L1 * L2 * L3 = dim 的因子分解。
        
        Args:
            workload: 包含 R, S, P, Q, C, K, N 的 workload 对象
            vary_permutation: 是否枚举不同 permutation
            vary_layout: 是否枚举不同 layout 组合
            vary_block_size: 是否枚举不同 block_size
            permutation_list: 自定义 permutation 列表，None 表示使用默认
            layout_list: 自定义 layout 列表，默认 ['sequential', 'row_aligned']
            block_sizes: 自定义 block_size 列表，None 表示自动生成
            max_block_area: block_h * block_w 的最大值
            sample_tiling: 从 tiling 组合中采样数量，None 表示不采样
            sample_total: 总采样数量，None 表示不采样
            
        Yields:
            MappingConfig 实例
        """
        import random
        
        # 获取 workload 维度
        R = getattr(workload, 'R', 1)
        S = getattr(workload, 'S', 1)
        P = getattr(workload, 'P', 1)
        Q = getattr(workload, 'Q', 1)
        C = getattr(workload, 'C', 1)
        K = getattr(workload, 'K', 1)
        N = getattr(workload, 'N', 1)
        H = getattr(workload, 'H', P)
        W = getattr(workload, 'W', Q)
        
        # 获取每个维度的所有因子分解 (L0, L1, L2, L3)
        R_decomps = get_factor_decompositions(R, 4)
        S_decomps = get_factor_decompositions(S, 4)
        P_decomps = get_factor_decompositions(P, 4)
        Q_decomps = get_factor_decompositions(Q, 4)
        C_decomps = get_factor_decompositions(C, 4)
        K_decomps = get_factor_decompositions(K, 4)
        N_decomps = get_factor_decompositions(N, 4)
        
        # 生成所有 tiling 组合
        all_tilings = list(product(R_decomps, S_decomps, P_decomps, Q_decomps, 
                                   C_decomps, K_decomps, N_decomps))
        
        # 采样 tiling
        if sample_tiling and len(all_tilings) > sample_tiling:
            all_tilings = random.sample(all_tilings, sample_tiling)
        
        # Permutation 列表
        if permutation_list is None:
            if vary_permutation:
                # 默认只排列 P, Q, C, K，R, S, N 固定
                perm_4d = list(permutations([DIM_P, DIM_Q, DIM_C, DIM_K]))
                permutation_list = [(DIM_R, DIM_S) + p + (DIM_N,) for p in perm_4d]
            else:
                permutation_list = [(DIM_R, DIM_S, DIM_Q, DIM_P, DIM_C, DIM_K, DIM_N)]
        
        # Layout 列表
        if layout_list is None:
            layout_list = ['sequential', 'row_aligned'] if vary_layout else ['sequential']
        
        if vary_layout:
            layout_combos = list(product(layout_list, repeat=3))
        else:
            layout_combos = [('sequential', 'sequential', 'sequential')]
        
        # Block size 列表
        if block_sizes is None:
            if vary_block_size:
                block_sizes = []
                h_divs = get_divisors(H)
                w_divs = get_divisors(W)
                for bh in h_divs:
                    for bw in w_divs:
                        if bh * bw <= max_block_area:
                            block_sizes.append((bh, bw))
                if not block_sizes:
                    block_sizes = [(1, 1)]
            else:
                block_sizes = [(1, 1)]
        
        # 生成所有配置
        all_configs = []
        
        for tiling in all_tilings:
            R_l0, R_l1, R_l2, R_l3 = tiling[0]
            S_l0, S_l1, S_l2, S_l3 = tiling[1]
            P_l0, P_l1, P_l2, P_l3 = tiling[2]
            Q_l0, Q_l1, Q_l2, Q_l3 = tiling[3]
            C_l0, C_l1, C_l2, C_l3 = tiling[4]
            K_l0, K_l1, K_l2, K_l3 = tiling[5]
            N_l0, N_l1, N_l2, N_l3 = tiling[6]
            
            for perm in permutation_list:
                for in_lay, w_lay, out_lay in layout_combos:
                    for bh, bw in block_sizes:
                        config = cls(
                            R_l3=R_l3, S_l3=S_l3, P_l3=P_l3, Q_l3=Q_l3,
                            C_l3=C_l3, K_l3=K_l3, N_l3=N_l3,
                            R_l2=R_l2, S_l2=S_l2, P_l2=P_l2, Q_l2=Q_l2,
                            C_l2=C_l2, K_l2=K_l2, N_l2=N_l2,
                            R_l1=R_l1, S_l1=S_l1, P_l1=P_l1, Q_l1=Q_l1,
                            C_l1=C_l1, K_l1=K_l1, N_l1=N_l1,
                            R_l0=R_l0, S_l0=S_l0, P_l0=P_l0, Q_l0=Q_l0,
                            C_l0=C_l0, K_l0=K_l0, N_l0=N_l0,
                            permutation_l3=perm,
                            input_layout=in_lay,
                            weight_layout=w_lay,
                            output_layout=out_lay,
                            block_h=bh,
                            block_w=bw
                        )
                        all_configs.append(config)
        
        # 总采样
        if sample_total and len(all_configs) > sample_total:
            all_configs = random.sample(all_configs, sample_total)
        
        for config in all_configs:
            yield config
    
    @classmethod
    def count_all_mappings(cls, workload,
                           vary_permutation: bool = True,
                           vary_layout: bool = True,
                           vary_block_size: bool = True) -> Dict[str, int]:
        """计算给定 workload 的合法 mapping 总数（不实际枚举）
        
        Returns:
            包含各维度分解数量和总数的字典
        """
        R = getattr(workload, 'R', 1)
        S = getattr(workload, 'S', 1)
        P = getattr(workload, 'P', 1)
        Q = getattr(workload, 'Q', 1)
        C = getattr(workload, 'C', 1)
        K = getattr(workload, 'K', 1)
        N = getattr(workload, 'N', 1)
        
        # 计算每个维度的分解数量
        counts = {
            'R_decomps': len(get_factor_decompositions(R, 4)),
            'S_decomps': len(get_factor_decompositions(S, 4)),
            'P_decomps': len(get_factor_decompositions(P, 4)),
            'Q_decomps': len(get_factor_decompositions(Q, 4)),
            'C_decomps': len(get_factor_decompositions(C, 4)),
            'K_decomps': len(get_factor_decompositions(K, 4)),
            'N_decomps': len(get_factor_decompositions(N, 4)),
        }
        
        # 计算 tiling 组合数
        tiling_count = 1
        for key in counts:
            tiling_count *= counts[key]
        counts['tiling_combos'] = tiling_count
        
        # Permutation 数量 (4! = 24)
        counts['permutation_count'] = 24 if vary_permutation else 1
        
        # Layout 组合数 (2^3 = 8)
        counts['layout_combos'] = 8 if vary_layout else 1
        
        # 总数
        counts['total'] = tiling_count * counts['permutation_count'] * counts['layout_combos']
        
        return counts


class MappingConstraintChecker:
    """ILP 约束检查器
    
    检查 mapping 是否满足所有 ILP 模型约束，包括:
    1. Tiling 因子约束: 因子必须能整除对应维度
    2. Block size 约束: block_h * block_w <= row_buffer_bytes
    3. RowBuffer Input Block 约束: Input tile 在 RowBuffer 中的约束
    4. Buffer 容量约束: 每个 tensor 的 tile 不能超过 buffer 容量
    """
    
    def __init__(self, workload: WorkloadConfig, arch: ArchConfig):
        self.workload = workload
        self.arch = arch
    
    def check_all_constraints(self, config: MappingConfig) -> Tuple[bool, List[str]]:
        """检查所有约束
        
        Returns:
            (is_valid, violation_reasons): 是否合法及违反的约束列表
        """
        violations = []
        
        # 1. Tiling 因子约束
        if not self._check_tiling_divisibility(config):
            violations.append("Tiling factor not divisible")
        
        # 2. Block size 约束
        block_violation = self._check_block_size_constraint(config)
        if block_violation:
            violations.append(block_violation)
        
        # 3. RowBuffer Input Block 约束
        rb_violation = self._check_rowbuffer_input_constraint(config)
        if rb_violation:
            violations.append(rb_violation)
        
        # 4. Buffer 容量约束
        buf_violations = self._check_buffer_capacity_constraints(config)
        violations.extend(buf_violations)
        
        return len(violations) == 0, violations
    
    def _check_tiling_divisibility(self, config: MappingConfig) -> bool:
        """检查所有层级的 tiling factor 乘积是否等于 workload 维度
        
        验证: L0 * L1 * L2 * L3 == workload dimension
        """
        is_valid, _ = config.validate_tiling(self.workload)
        return is_valid
    
    def _check_block_size_constraint(self, config: MappingConfig) -> Optional[str]:
        """检查 block_h * block_w <= row_buffer_bytes
        
        这是 ILP 中的 RowBuffer Input Block 约束:
        block_h 和 block_w 定义了 Input 在 row_aligned 布局时的 2D block 大小
        """
        block_area = config.block_h * config.block_w * self.arch.element_size
        if block_area > self.arch.row_buffer_bytes:
            return (f"Block size violation: {config.block_h}x{config.block_w}={block_area} bytes "
                    f"> row_buffer={self.arch.row_buffer_bytes} bytes")
        return None
    
    def _check_rowbuffer_input_constraint(self, config: MappingConfig) -> Optional[str]:
        """检查 RowBuffer 中 Input tile 的约束
        
        对于 row_aligned layout:
        - Input tile 在 RowBuffer 中的大小 = P_tile * Q_tile * C_tile (考虑 stride)
        - 注意: block_h × block_w 不是 Input tile size，而是地址对齐的 block
        
        对于 sequential layout:
        - 需要确保 tile 可以有效访问
        """
        wl = self.workload
        
        # L2 tile (RowBuffer level) - 直接使用 config 中的 L2 factors
        P_l2 = config.P_l2
        Q_l2 = config.Q_l2
        C_l2 = config.C_l2
        K_l2 = config.K_l2
        
        # Input tile 大小 (考虑滑动窗口)
        # Input height for P_l2 outputs: P_l2 + (R-1) * dilation
        # Input width for Q_l2 outputs: Q_l2 + (S-1) * dilation
        input_h_l2 = (P_l2 - 1) * wl.stride_h + (wl.R - 1) * wl.dilation_h + 1
        input_w_l2 = (Q_l2 - 1) * wl.stride_w + (wl.S - 1) * wl.dilation_w + 1
        
        # Input tile 在 RowBuffer 中的大小
        input_tile_size = input_h_l2 * input_w_l2 * C_l2 * self.arch.element_size
        
        # Weight tile 在 RowBuffer 中的大小
        weight_tile_size = wl.R * wl.S * C_l2 * K_l2 * self.arch.element_size
        
        # Output tile 在 RowBuffer 中的大小
        output_tile_size = P_l2 * Q_l2 * K_l2 * self.arch.element_size
        
        # 检查是否超过 RowBuffer 容量
        # 注意: 实际 ILP 模型中，不同 tensor 可能在不同 bank，这里简化为单独检查
        row_buffer_capacity = self.arch.row_buffer_bytes
        
        # 对于 row_aligned layout，Input 需要额外的对齐空间
        if config.input_layout == 'row_aligned':
            # row_aligned 下，每个 row 只能存储 block_h * block_w 个元素
            # 需要更多 row 来存储 tile
            blocks_per_h = (input_h_l2 + config.block_h - 1) // config.block_h
            blocks_per_w = (input_w_l2 + config.block_w - 1) // config.block_w
            blocks_needed = blocks_per_h * blocks_per_w * C_l2
            # 每个 block 占用一个 row 的部分空间
            effective_input_size = blocks_needed * config.block_h * config.block_w * self.arch.element_size
        else:
            effective_input_size = input_tile_size
        
        # 返回违反约束的信息 (如果有)
        # 注意: 这是简化的检查，实际 ILP 有更复杂的 buffer 分配
        return None  # 简化: 暂不检查 tile size，因为这需要知道完整的 memory hierarchy
    
    def _check_buffer_capacity_constraints(self, config: MappingConfig) -> List[str]:
        """检查各级 buffer 的容量约束
        
        检查 GlobalBuffer(L1) 能否容纳所有 tensor 的 L1 tile
        L1 tile = L0 * L1 factors（存储在 SRAM 中的部分）
        """
        violations = []
        wl = self.workload
        
        # 计算 L1 tile 大小（L0 * L1 乘积）
        P_tile = config.P_l0 * config.P_l1
        Q_tile = config.Q_l0 * config.Q_l1
        C_tile = config.C_l0 * config.C_l1
        K_tile = config.K_l0 * config.K_l1
        R_tile = config.R_l0 * config.R_l1
        S_tile = config.S_l0 * config.S_l1
        
        # Input L1 tile (考虑滑动窗口扩展)
        input_h_tile = (P_tile - 1) * wl.stride_h + (R_tile - 1) * wl.dilation_h + 1
        input_w_tile = (Q_tile - 1) * wl.stride_w + (S_tile - 1) * wl.dilation_w + 1
        input_tile_size = input_h_tile * input_w_tile * C_tile * self.arch.element_size
        
        # Weight L1 tile
        weight_tile_size = R_tile * S_tile * C_tile * K_tile * self.arch.element_size
        
        # Output L1 tile
        output_tile_size = P_tile * Q_tile * K_tile * self.arch.element_size
        
        # GlobalBuffer 需要同时容纳 Input, Weight, Output 的 tile
        total_tile_size = input_tile_size + weight_tile_size + output_tile_size
        
        if total_tile_size > self.arch.global_buffer_bytes:
            violations.append(
                f"GlobalBuffer overflow: tiles={total_tile_size} bytes > "
                f"capacity={self.arch.global_buffer_bytes} bytes "
                f"(Input={input_tile_size}, Weight={weight_tile_size}, Output={output_tile_size})"
            )
        
        return violations
    
    def is_valid(self, config: MappingConfig) -> bool:
        """快速检查 mapping 是否合法"""
        valid, _ = self.check_all_constraints(config)
        return valid
    
    def get_violation_summary(self, config: MappingConfig) -> str:
        """获取违反约束的摘要"""
        valid, violations = self.check_all_constraints(config)
        if valid:
            return "Valid mapping"
        return "; ".join(violations)


class MappingSpace:
    """合法 Mapping 空间的枚举器
    
    支持 ILP 约束检查，只枚举满足所有约束的合法 mapping
    支持全部 7 个维度：R, S, P, Q, C, K, N
    """
    
    def __init__(self, P: int, Q: int, C: int, K: int, H: int, W: int, 
                 row_buffer_bytes: int = 1024, element_size: int = 1,
                 R: int = 3, S: int = 3, N: int = 1,
                 stride: int = 1, dilation: int = 1,
                 global_buffer_bytes: int = 256 * 1024,
                 validate_constraints: bool = True):
        """
        Args:
            P, Q, C, K: workload 主要维度
            H, W: input tensor 的高度和宽度
            R, S: Filter 维度
            N: Batch size
            row_buffer_bytes: DRAM row buffer 大小
            element_size: 每个元素的字节数
            stride, dilation: Conv 参数 (可以是 int 或 tuple)
            global_buffer_bytes: GlobalBuffer 大小
            validate_constraints: 是否启用 ILP 约束检查
        """
        self.P = P
        self.Q = Q
        self.C = C
        self.K = K
        self.H = H
        self.W = W
        self.R = R
        self.S = S
        
        # 处理 stride 和 dilation (可能是 int 或 tuple)
        if isinstance(stride, tuple):
            stride_h, stride_w = stride[0], stride[1] if len(stride) > 1 else stride[0]
        else:
            stride_h, stride_w = stride, stride
        
        if isinstance(dilation, tuple):
            dilation_h, dilation_w = dilation[0], dilation[1] if len(dilation) > 1 else dilation[0]
        else:
            dilation_h, dilation_w = dilation, dilation
        
        self.stride = (stride_h, stride_w)
        self.dilation = (dilation_h, dilation_w)
        self.row_buffer_bytes = row_buffer_bytes
        self.element_size = element_size
        self.global_buffer_bytes = global_buffer_bytes
        self.validate_constraints = validate_constraints
        self.N = N  # 添加 Batch size
        
        # 创建约束检查器
        self.workload_config = WorkloadConfig(
            P=P, Q=Q, K=K, C=C, H=H, W=W, R=R, S=S, N=N,
            stride_h=stride_h, stride_w=stride_w,
            dilation_h=dilation_h, dilation_w=dilation_w
        )
        self.arch_config = ArchConfig(
            row_buffer_bytes=row_buffer_bytes,
            element_size=element_size,
            global_buffer_bytes=global_buffer_bytes
        )
        self.constraint_checker = MappingConstraintChecker(
            self.workload_config, self.arch_config
        )
        
        # 计算每个维度的因子（全部 7 个维度）
        self.R_divisors = get_divisors(R)
        self.S_divisors = get_divisors(S)
        self.P_divisors = get_divisors(P)
        self.Q_divisors = get_divisors(Q)
        self.C_divisors = get_divisors(C)
        self.K_divisors = get_divisors(K)
        self.N_divisors = get_divisors(N)
        
        # 可能的 block sizes (必须满足 block_h * block_w <= row_buffer)
        row_elements = row_buffer_bytes // element_size
        self.valid_block_sizes = self._compute_valid_block_sizes(row_elements)
        
        # 可能的 permutation（全部 7 个维度的排列）
        # 注意：完整排列有 7! = 5040 种，可能需要采样
        self.all_permutations_7d = list(permutations([DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N]))
        # 简化版：只排列 P, Q, C, K 四个维度，R, S, N 固定在外层
        self.all_permutations_4d = list(permutations([DIM_P, DIM_Q, DIM_C, DIM_K]))
        # 默认使用 4D 排列（兼容旧代码）
        self.all_permutations = self.all_permutations_4d
        
        # 可能的 layout
        self.layouts = ['sequential', 'row_aligned']
    
    def _compute_valid_block_sizes(self, row_elements: int) -> List[Tuple[int, int]]:
        """计算合法的 block_h, block_w 组合
        
        ILP 约束: block_h * block_w <= row_buffer_bytes
        """
        valid = []
        h_divisors = get_divisors(self.H)
        w_divisors = get_divisors(self.W)
        
        for bh in h_divisors:
            for bw in w_divisors:
                # 关键约束: block_h * block_w <= row_buffer (in elements)
                if bh * bw <= row_elements:
                    valid.append((bh, bw))
        
        # 如果没有合法的 block size，使用最小的
        if not valid:
            valid.append((1, 1))
        
        return sorted(set(valid))
    
    def get_default_permutation_7d(self) -> Tuple[int, ...]:
        """返回默认的 7 维 permutation（从内到外）
        
        默认顺序：R, S, Q, P, C, K, N
        - R, S 在最内层（filter 维度，通常不 tile）
        - Q, P 在中间（空间维度）
        - C, K 在外层（channel 维度）
        - N 在最外层（batch 维度，通常不 tile）
        """
        return (DIM_R, DIM_S, DIM_Q, DIM_P, DIM_C, DIM_K, DIM_N)
    
    def extend_4d_to_7d_permutation(self, perm_4d: Tuple[int, ...]) -> Tuple[int, ...]:
        """将 4 维 permutation 扩展为 7 维
        
        在 4D permutation 的基础上，在最内层添加 R, S，在最外层添加 N
        
        Args:
            perm_4d: 4 维 permutation (P, Q, C, K 的排列)
            
        Returns:
            7 维 permutation
        """
        # R, S 在最内层，N 在最外层
        return (DIM_R, DIM_S) + perm_4d + (DIM_N,)
    
    def is_valid_mapping(self, config: MappingConfig) -> bool:
        """检查 mapping 是否满足所有 ILP 约束"""
        if not self.validate_constraints:
            return True
        return self.constraint_checker.is_valid(config)
    
    def get_constraint_violations(self, config: MappingConfig) -> List[str]:
        """获取 mapping 违反的约束列表"""
        _, violations = self.constraint_checker.check_all_constraints(config)
        return violations
    
    def get_space_size(self, 
                       vary_tiling: bool = True,
                       vary_permutation: bool = True,
                       vary_layout: bool = True,
                       vary_block_size: bool = True) -> int:
        """计算 mapping 空间大小"""
        size = 1
        if vary_tiling:
            size *= (len(self.R_divisors) * len(self.S_divisors) * 
                     len(self.P_divisors) * len(self.Q_divisors) * 
                     len(self.C_divisors) * len(self.K_divisors) *
                     len(self.N_divisors))
        if vary_permutation:
            size *= len(self.all_permutations)
        if vary_layout:
            size *= len(self.layouts) ** 3
        if vary_block_size:
            size *= len(self.valid_block_sizes)
        return size
    
    def enumerate_permutations(self, 
                               P_l3: int = 1, Q_l3: int = 1, C_l3: int = 1, K_l3: int = 1,
                               R_l3: int = 1, S_l3: int = 1, N_l3: int = 1,
                               layout: Tuple[str, str, str] = ('row_aligned', 'sequential', 'row_aligned'),
                               block_size: Tuple[int, int] = None,
                               use_7d_permutation: bool = True,
                               filter_invalid: bool = True) -> Iterator[MappingConfig]:
        """
        固定 tiling 和 layout，只枚举所有 permutation
        
        Args:
            R_l3, S_l3, P_l3, Q_l3, C_l3, K_l3, N_l3: 固定的 L3 tiling factors（全部 7 维）
            layout: (input_layout, weight_layout, output_layout)
            block_size: (block_h, block_w), 如果 None 则使用合法的最大值
            use_7d_permutation: 如果 True，使用完整 7 维 permutation；否则使用 4 维并自动扩展
            filter_invalid: 是否过滤不合法的 mapping
        """
        if block_size is None:
            # 选择一个合法的 block size
            if self.valid_block_sizes:
                block_size = self.valid_block_sizes[-1]  # 最大的合法 block
            else:
                block_size = (1, 1)
        
        # 选择 permutation 列表
        if use_7d_permutation:
            perm_list = self.all_permutations_7d
        else:
            perm_list = self.all_permutations_4d
        
        for perm in perm_list:
            # 如果是 4D permutation，扩展为 7D
            if len(perm) == 4:
                perm = self.extend_4d_to_7d_permutation(perm)
            
            config = MappingConfig(
                R_l3=R_l3, S_l3=S_l3,
                P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3,
                N_l3=N_l3,
                permutation=perm,
                input_layout=layout[0],
                weight_layout=layout[1],
                output_layout=layout[2],
                block_h=block_size[0],
                block_w=block_size[1]
            )
            if filter_invalid and self.validate_constraints:
                if self.is_valid_mapping(config):
                    yield config
            else:
                yield config
    
    def enumerate_tiling_and_permutation(self,
                                         layout: Tuple[str, str, str] = ('row_aligned', 'sequential', 'row_aligned'),
                                         block_size: Tuple[int, int] = None,
                                         sample_tiling: int = None,
                                         use_7d_permutation: bool = False,
                                         filter_invalid: bool = True) -> Iterator[MappingConfig]:
        """
        枚举 tiling 和 permutation，固定 layout 和 block_size
        
        Args:
            layout: (input_layout, weight_layout, output_layout)
            block_size: (block_h, block_w)
            sample_tiling: 如果设置，则从 tiling 组合中均匀采样
            use_7d_permutation: 如果 True，使用完整 7 维 permutation
            filter_invalid: 是否过滤不合法的 mapping
        """
        if block_size is None:
            if self.valid_block_sizes:
                block_size = self.valid_block_sizes[-1]
            else:
                block_size = (1, 1)
        
        # 生成所有 tiling 组合（目前只使用 P, Q, C, K，R, S, N 默认为 1）
        tiling_combos = list(product(self.P_divisors, self.Q_divisors, 
                                     self.C_divisors, self.K_divisors))
        
        # 如果需要采样
        if sample_tiling and len(tiling_combos) > sample_tiling:
            import random
            tiling_combos = random.sample(tiling_combos, sample_tiling)
        
        # 选择 permutation 列表
        if use_7d_permutation:
            perm_list = self.all_permutations_7d
        else:
            perm_list = self.all_permutations_4d
        
        for P_l3, Q_l3, C_l3, K_l3 in tiling_combos:
            for perm in perm_list:
                # 如果是 4D permutation，扩展为 7D
                if len(perm) == 4:
                    perm = self.extend_4d_to_7d_permutation(perm)
                
                config = MappingConfig(
                    R_l3=1, S_l3=1,  # Filter 维度默认不 tile
                    P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3,
                    N_l3=1,  # Batch 维度默认不 tile
                    permutation=perm,
                    input_layout=layout[0],
                    weight_layout=layout[1],
                    output_layout=layout[2],
                    block_h=block_size[0],
                    block_w=block_size[1]
                )
                if filter_invalid and self.validate_constraints:
                    if self.is_valid_mapping(config):
                        yield config
                else:
                    yield config
    
    def enumerate_full_space(self, 
                             sample_size: int = None,
                             tiling_sample: int = None,
                             use_7d_permutation: bool = False,
                             filter_invalid: bool = True) -> Iterator[MappingConfig]:
        """
        枚举完整的 mapping 空间
        
        Args:
            sample_size: 总采样数量
            tiling_sample: 每个 (layout, block_size) 组合中采样的 tiling 数量
            use_7d_permutation: 如果 True，使用完整 7 维 permutation
            filter_invalid: 是否过滤不合法的 mapping
        """
        all_configs = []
        
        # 所有 tiling 组合（目前只使用 P, Q, C, K）
        tiling_combos = list(product(self.P_divisors, self.Q_divisors,
                                     self.C_divisors, self.K_divisors))
        
        if tiling_sample and len(tiling_combos) > tiling_sample:
            import random
            tiling_combos = random.sample(tiling_combos, tiling_sample)
        
        # 所有 layout 组合
        layout_combos = list(product(self.layouts, repeat=3))
        
        # 选择 permutation 列表
        if use_7d_permutation:
            perm_list = self.all_permutations_7d
        else:
            perm_list = self.all_permutations_4d
        
        for P_l3, Q_l3, C_l3, K_l3 in tiling_combos:
            for perm in perm_list:
                # 如果是 4D permutation，扩展为 7D
                if len(perm) == 4:
                    perm = self.extend_4d_to_7d_permutation(perm)
                
                for in_lay, w_lay, out_lay in layout_combos:
                    for bh, bw in self.valid_block_sizes:
                        config = MappingConfig(
                            R_l3=1, S_l3=1,  # Filter 维度默认不 tile
                            P_l3=P_l3, Q_l3=Q_l3, C_l3=C_l3, K_l3=K_l3,
                            N_l3=1,  # Batch 维度默认不 tile
                            permutation=perm,
                            input_layout=in_lay,
                            weight_layout=w_lay,
                            output_layout=out_lay,
                            block_h=bh,
                            block_w=bw
                        )
                        # 过滤非法 mapping
                        if filter_invalid and self.validate_constraints:
                            if self.is_valid_mapping(config):
                                all_configs.append(config)
                        else:
                            all_configs.append(config)
        
        # 如果需要采样
        if sample_size and len(all_configs) > sample_size:
            import random
            all_configs = random.sample(all_configs, sample_size)
        
        for config in all_configs:
            yield config
    
    def enumerate_valid_mappings(self, 
                                 sample_size: int = None,
                                 use_7d_permutation: bool = False,
                                 verbose: bool = False) -> Iterator[MappingConfig]:
        """
        只枚举满足所有 ILP 约束的合法 mapping
        
        Args:
            sample_size: 采样数量
            use_7d_permutation: 如果 True，使用完整 7 维 permutation
            verbose: 是否打印过滤统计
        """
        total_generated = 0
        valid_count = 0
        
        for config in self.enumerate_full_space(sample_size=None, use_7d_permutation=use_7d_permutation, filter_invalid=False):
            total_generated += 1
            if self.is_valid_mapping(config):
                valid_count += 1
                if sample_size is None or valid_count <= sample_size:
                    yield config
                if sample_size and valid_count >= sample_size:
                    break
        
        if verbose:
            print(f"Constraint filtering: {valid_count}/{total_generated} valid "
                  f"({100*valid_count/max(1,total_generated):.1f}%)")
    
    def summary(self):
        """打印 mapping 空间的摘要"""
        print("=" * 60)
        print("Mapping Space Summary")
        print("=" * 60)
        print(f"Workload: P={self.P}, Q={self.Q}, C={self.C}, K={self.K}, H={self.H}, W={self.W}")
        print(f"Filter: R={self.R}, S={self.S}, stride={self.stride}, dilation={self.dilation}")
        print(f"Row buffer: {self.row_buffer_bytes} bytes")
        print(f"Global buffer: {self.global_buffer_bytes} bytes")
        print(f"Constraint validation: {'Enabled' if self.validate_constraints else 'Disabled'}")
        print()
        print("Tiling factors:")
        print(f"  P divisors ({len(self.P_divisors)}): {self.P_divisors}")
        print(f"  Q divisors ({len(self.Q_divisors)}): {self.Q_divisors}")
        print(f"  C divisors ({len(self.C_divisors)}): {self.C_divisors}")
        print(f"  K divisors ({len(self.K_divisors)}): {self.K_divisors}")
        print()
        print(f"Permutations: {len(self.all_permutations)}")
        print(f"Layouts: {len(self.layouts)}^3 = {len(self.layouts)**3}")
        print(f"Valid block sizes: {len(self.valid_block_sizes)}")
        print(f"  Block sizes: {self.valid_block_sizes}")
        print()
        print("Space sizes (before constraint filtering):")
        print(f"  Permutation only: {self.get_space_size(False, True, False, False)}")
        print(f"  Tiling + Permutation: {self.get_space_size(True, True, False, False)}")
        print(f"  Full space: {self.get_space_size(True, True, True, True)}")
    
    def count_valid_mappings(self, sample_size: int = 1000) -> Tuple[int, int]:
        """统计合法 mapping 的数量 (采样估计)
        
        Returns:
            (valid_count, total_count): 合法数量和总数量
        """
        valid = 0
        total = 0
        for config in self.enumerate_full_space(sample_size=sample_size, filter_invalid=False):
            total += 1
            if self.is_valid_mapping(config):
                valid += 1
        return valid, total
    
    def enumerate_l3_l2_space(self,
                              vary_permutation: bool = True,
                              vary_layout: bool = True,
                              vary_block_size: bool = True,
                              sample_tiling: int = None,
                              sample_total: int = None,
                              filter_invalid: bool = True) -> Iterator[MappingConfig]:
        """枚举 L3 + L2 两层的 tiling 空间
        
        对于每个维度，枚举所有满足 L2 * L3 = dim 的因子分解。
        L0 和 L1 默认为 1。
        
        Args:
            vary_permutation: 是否枚举不同 permutation (4! = 24 种)
            vary_layout: 是否枚举不同 layout 组合 (2^3 = 8 种)
            vary_block_size: 是否枚举不同 block_size
            sample_tiling: 从 tiling 组合中采样数量
            sample_total: 总采样数量
            filter_invalid: 是否过滤不合法的 mapping
            
        Yields:
            MappingConfig 实例（L0=L1=1, 只有 L2 和 L3 有 tiling）
        """
        import random
        
        # 获取每个维度的 2 层因子分解 (L2, L3)
        def get_2level_decomps(n: int) -> List[Tuple[int, int]]:
            """获取 n 的所有 2 层因子分解: L2 * L3 = n"""
            divisors = get_divisors(n)
            return [(d, n // d) for d in divisors]  # (L2, L3)
        
        R_decomps = get_2level_decomps(self.R)
        S_decomps = get_2level_decomps(self.S)
        P_decomps = get_2level_decomps(self.P)
        Q_decomps = get_2level_decomps(self.Q)
        C_decomps = get_2level_decomps(self.C)
        K_decomps = get_2level_decomps(self.K)
        N_decomps = get_2level_decomps(self.N)
        
        # 生成所有 tiling 组合
        all_tilings = list(product(R_decomps, S_decomps, P_decomps, Q_decomps,
                                   C_decomps, K_decomps, N_decomps))
        
        # 采样 tiling
        if sample_tiling and len(all_tilings) > sample_tiling:
            all_tilings = random.sample(all_tilings, sample_tiling)
        
        # Permutation 列表
        if vary_permutation:
            perm_list = [self.extend_4d_to_7d_permutation(p) for p in self.all_permutations_4d]
        else:
            perm_list = [self.get_default_permutation_7d()]
        
        # Layout 组合
        if vary_layout:
            layout_combos = list(product(self.layouts, repeat=3))
        else:
            layout_combos = [('sequential', 'sequential', 'sequential')]
        
        # Block size 列表
        if vary_block_size:
            block_sizes = self.valid_block_sizes
        else:
            block_sizes = [(1, 1)]
        
        # 生成所有配置
        all_configs = []
        
        for tiling in all_tilings:
            R_l2, R_l3 = tiling[0]
            S_l2, S_l3 = tiling[1]
            P_l2, P_l3 = tiling[2]
            Q_l2, Q_l3 = tiling[3]
            C_l2, C_l3 = tiling[4]
            K_l2, K_l3 = tiling[5]
            N_l2, N_l3 = tiling[6]
            
            for perm in perm_list:
                for in_lay, w_lay, out_lay in layout_combos:
                    for bh, bw in block_sizes:
                        config = MappingConfig(
                            # L3 factors
                            R_l3=R_l3, S_l3=S_l3, P_l3=P_l3, Q_l3=Q_l3,
                            C_l3=C_l3, K_l3=K_l3, N_l3=N_l3,
                            # L2 factors
                            R_l2=R_l2, S_l2=S_l2, P_l2=P_l2, Q_l2=Q_l2,
                            C_l2=C_l2, K_l2=K_l2, N_l2=N_l2,
                            # L1 and L0 = 1 (default)
                            permutation_l3=perm,
                            input_layout=in_lay,
                            weight_layout=w_lay,
                            output_layout=out_lay,
                            block_h=bh,
                            block_w=bw
                        )
                        
                        if filter_invalid and self.validate_constraints:
                            if self.is_valid_mapping(config):
                                all_configs.append(config)
                        else:
                            all_configs.append(config)
        
        # 总采样
        if sample_total and len(all_configs) > sample_total:
            all_configs = random.sample(all_configs, sample_total)
        
        for config in all_configs:
            yield config
    
    def count_l3_l2_space(self,
                          vary_permutation: bool = True,
                          vary_layout: bool = True,
                          vary_block_size: bool = True) -> Dict[str, int]:
        """计算 L3+L2 mapping 空间大小（不实际枚举）
        
        Returns:
            包含各维度分解数量和总数的字典
        """
        counts = {
            'R_decomps': len(get_divisors(self.R)),  # 2层分解数 = divisor数
            'S_decomps': len(get_divisors(self.S)),
            'P_decomps': len(get_divisors(self.P)),
            'Q_decomps': len(get_divisors(self.Q)),
            'C_decomps': len(get_divisors(self.C)),
            'K_decomps': len(get_divisors(self.K)),
            'N_decomps': len(get_divisors(self.N)),
        }
        
        # tiling 组合数
        tiling_count = 1
        for key in counts:
            tiling_count *= counts[key]
        counts['tiling_combos'] = tiling_count
        
        # Permutation 数量 (4! = 24)
        counts['permutation_count'] = 24 if vary_permutation else 1
        
        # Layout 组合数 (2^3 = 8)
        counts['layout_combos'] = 8 if vary_layout else 1
        
        # Block size 数量
        counts['block_size_count'] = len(self.valid_block_sizes) if vary_block_size else 1
        
        # 总数
        counts['total'] = (tiling_count * counts['permutation_count'] * 
                          counts['layout_combos'] * counts['block_size_count'])
        
        return counts


if __name__ == "__main__":
    print("=" * 70)
    print("测试 ILP 约束检查器")
    print("=" * 70)
    
    # 测试: ResNet-L1
    space = MappingSpace(
        P=56, Q=56, C=3, K=64, H=62, W=62,
        R=3, S=3, stride=1, dilation=1,
        row_buffer_bytes=1024,
        global_buffer_bytes=256*1024,
        validate_constraints=True
    )
    space.summary()
    
    # 测试约束检查
    print("\n" + "=" * 70)
    print("测试单个 mapping 的约束检查")
    print("=" * 70)
    
    # 测试一个合法的 mapping
    valid_config = MappingConfig(
        P_l3=8, Q_l3=8, C_l3=3, K_l3=8,
        permutation=(DIM_P, DIM_Q, DIM_C, DIM_K),
        input_layout='sequential',
        weight_layout='sequential',
        output_layout='sequential',
        block_h=2, block_w=2
    )
    is_valid, violations = space.constraint_checker.check_all_constraints(valid_config)
    print(f"\nConfig: {valid_config}")
    print(f"Valid: {is_valid}")
    if violations:
        print(f"Violations: {violations}")
    
    # 测试一个非法的 mapping (block size 太大)
    invalid_config = MappingConfig(
        P_l3=8, Q_l3=8, C_l3=3, K_l3=8,
        permutation=(DIM_P, DIM_Q, DIM_C, DIM_K),
        input_layout='row_aligned',
        weight_layout='sequential',
        output_layout='row_aligned',
        block_h=62, block_w=62  # 62*62=3844 > 1024
    )
    is_valid, violations = space.constraint_checker.check_all_constraints(invalid_config)
    print(f"\nConfig: {invalid_config}")
    print(f"Valid: {is_valid}")
    if violations:
        print(f"Violations: {violations}")
    
    # 统计合法 mapping 比例
    print("\n" + "=" * 70)
    print("统计合法 mapping 比例")
    print("=" * 70)
    
    valid_count, total_count = space.count_valid_mappings(sample_size=500)
    print(f"采样 {total_count} 个 mapping:")
    print(f"  合法: {valid_count} ({100*valid_count/max(1,total_count):.1f}%)")
    print(f"  非法: {total_count - valid_count} ({100*(total_count-valid_count)/max(1,total_count):.1f}%)")
    
    # 枚举一些合法 mapping
    print("\n" + "=" * 70)
    print("枚举合法 mapping 示例")
    print("=" * 70)
    
    for i, config in enumerate(space.enumerate_valid_mappings(sample_size=5)):
        print(f"  {i+1}: {config}")
