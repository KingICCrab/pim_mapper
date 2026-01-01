#!/usr/bin/env python3
"""
MappingConstraintChecker: 检查 mapping 是否满足硬件约束

约束类型:
1. Tiling 因子约束: L0 * L1 * L2 * L3 = workload dimension
2. Block size 约束: block_h * block_w <= row_buffer_bytes
3. Buffer 容量约束: 每个 tensor 的 tile 不能超过对应层级的 buffer 容量

Memory Hierarchy:
- Level 0 (PE): 存储 L0 tile
- Level 1 (GlobalBuffer): 存储 L0*L1 tile  
- Level 2 (RowBuffer): 存储 L0*L1*L2 tile
"""

from typing import List, Tuple, Optional, Dict

from .config import ArchConfig, WorkloadConfig
from .mapping_config import MappingConfig


class MappingConstraintChecker:
    """ILP 约束检查器
    
    按照 ILP 模型的方式检查约束:
    1. 每个 tensor 分开检查容量约束
    2. 对每个 memory level 分别检查
    3. RowBuffer 有特殊的 tile 大小限制
    """
    
    def __init__(self, workload: WorkloadConfig, arch: ArchConfig):
        self.workload = workload
        self.arch = arch
    
    def check_all(self, config: MappingConfig) -> Tuple[bool, List[str]]:
        """检查所有约束
        
        Returns:
            (is_valid, violation_reasons)
        """
        violations = []
        
        # 1. Tiling 因子约束
        tiling_ok, tiling_errors = config.validate_tiling(self.workload)
        if not tiling_ok:
            violations.extend([f"Tiling: {e}" for e in tiling_errors])
        
        # 2. R/S 维度约束 (必须在 L2, L3 为 1)
        rs_errors = self._check_rs_constraint(config)
        violations.extend(rs_errors)
        
        # 3. Block size 约束
        block_err = self._check_block_size(config)
        if block_err:
            violations.append(block_err)
        
        # 4. Buffer 容量约束 - 每层、每个 tensor 分别检查
        buf_errors = self._check_buffer_capacity_per_level(config)
        violations.extend(buf_errors)
        
        return len(violations) == 0, violations
    
    def _check_rs_constraint(self, config: MappingConfig) -> List[str]:
        """检查 R/S 维度在 L2 和 L3 层级必须为 1
        
        这是 Row Activation 公式的关键约束:
        - R/S 循环只能在 Buffer Level (L1) 或更低层级展开
        - 如果 R/S 在 DRAM Level (L3) 或 Row Buffer Level (L2) 展开,
          会导致 Kernel 窗口在高层级滑动，严重破坏 Input 数据局部性
        - 公式假设整个 Kernel 窗口 (R×S) 在加载 Input Tile 时是原子操作
        """
        violations = []
        
        # 检查 L2 层级
        if config.R_l2 != 1:
            violations.append(
                f"R/S constraint: R_l2={config.R_l2} must be 1 "
                f"(R loop cannot be at RowBuffer level)"
            )
        if config.S_l2 != 1:
            violations.append(
                f"R/S constraint: S_l2={config.S_l2} must be 1 "
                f"(S loop cannot be at RowBuffer level)"
            )
        
        # 检查 L3 层级
        if config.R_l3 != 1:
            violations.append(
                f"R/S constraint: R_l3={config.R_l3} must be 1 "
                f"(R loop cannot be at DRAM level)"
            )
        if config.S_l3 != 1:
            violations.append(
                f"R/S constraint: S_l3={config.S_l3} must be 1 "
                f"(S loop cannot be at DRAM level)"
            )
        
        return violations
    
    def _check_block_size(self, config: MappingConfig) -> Optional[str]:
        """检查 block_h * block_w <= row_buffer_bytes"""
        block_area = config.block_h * config.block_w * self.arch.element_size
        if block_area > self.arch.row_buffer_bytes:
            return (f"Block size: {config.block_h}x{config.block_w}={block_area} bytes "
                    f"> row_buffer={self.arch.row_buffer_bytes} bytes")
        return None
    
    def _compute_tile_sizes(self, config: MappingConfig, level: int) -> Dict[str, int]:
        """计算指定层级的 tile 大小 (elements)
        
        Args:
            config: MappingConfig
            level: 0=PE, 1=GlobalBuffer, 2=RowBuffer
            
        Returns:
            {'input': size, 'weight': size, 'output': size}
        """
        wl = self.workload
        
        # 计算累积 tiling factors (从 L0 累积到指定 level)
        if level == 0:
            # L0 tile
            P_tile = config.P_l0
            Q_tile = config.Q_l0
            C_tile = config.C_l0
            K_tile = config.K_l0
            R_tile = config.R_l0
            S_tile = config.S_l0
            N_tile = config.N_l0
        elif level == 1:
            # L0 * L1 tile
            P_tile = config.P_l0 * config.P_l1
            Q_tile = config.Q_l0 * config.Q_l1
            C_tile = config.C_l0 * config.C_l1
            K_tile = config.K_l0 * config.K_l1
            R_tile = config.R_l0 * config.R_l1
            S_tile = config.S_l0 * config.S_l1
            N_tile = config.N_l0 * config.N_l1
        elif level == 2:
            # L0 * L1 * L2 tile
            P_tile = config.P_l0 * config.P_l1 * config.P_l2
            Q_tile = config.Q_l0 * config.Q_l1 * config.Q_l2
            C_tile = config.C_l0 * config.C_l1 * config.C_l2
            K_tile = config.K_l0 * config.K_l1 * config.K_l2
            R_tile = config.R_l0 * config.R_l1 * config.R_l2
            S_tile = config.S_l0 * config.S_l1 * config.S_l2
            N_tile = config.N_l0 * config.N_l1 * config.N_l2
        else:
            raise ValueError(f"Invalid level: {level}")
        
        # Input tile: 考虑滑动窗口的 unique 元素
        # unique_h = (P_tile - 1) * stride + (R_tile - 1) * dilation + 1
        input_h = self._compute_unique_input_size(
            wl.stride_h, wl.dilation_h, P_tile, R_tile
        )
        input_w = self._compute_unique_input_size(
            wl.stride_w, wl.dilation_w, Q_tile, S_tile
        )
        input_size = input_h * input_w * C_tile * N_tile
        
        # Weight tile: R * S * C * K
        weight_size = R_tile * S_tile * C_tile * K_tile
        
        # Output tile: P * Q * K * N
        output_size = P_tile * Q_tile * K_tile * N_tile
        
        return {
            'input': input_size,
            'weight': weight_size,
            'output': output_size
        }
    
    def _compute_unique_input_size(self, stride: int, dilation: int, 
                                    spatial_tile: int, filter_tile: int) -> int:
        """计算 Input tile 的 unique 元素数量
        
        考虑滑动窗口重叠:
        unique = (spatial_tile - 1) * stride + (filter_tile - 1) * dilation + 1
        """
        return (spatial_tile - 1) * stride + (filter_tile - 1) * dilation + 1
    
    def _check_buffer_capacity_per_level(self, config: MappingConfig) -> List[str]:
        """检查每个 memory level 的每个 tensor 容量约束
        
        ILP 模型方式: 每个 tensor 分开检查，每层分别检查
        
        特殊处理 (row_aligned layout):
        1. Input 在 RowBuffer: block_h × block_w × C_tile × N_tile <= row_buffer_size
        2. block_h >= H_tile, block_w >= W_tile (防止 tile 超过 block)
        """
        violations = []
        elem_size = self.arch.element_size
        wl = self.workload
        
        # Level 1: GlobalBuffer 检查
        l1_tiles = self._compute_tile_sizes(config, level=1)
        l1_capacity = self.arch.global_buffer_bytes // elem_size
        
        for tensor_name, tile_size in l1_tiles.items():
            if tile_size > l1_capacity:
                violations.append(
                    f"GlobalBuffer {tensor_name}: {tile_size} elements "
                    f"> capacity {l1_capacity} elements"
                )
        
        # Level 2: RowBuffer 检查
        l2_capacity = self.arch.row_buffer_bytes // elem_size
        
        # 计算 RowBuffer level 的 tile factors
        P_tile_l2 = config.P_l0 * config.P_l1 * config.P_l2
        Q_tile_l2 = config.Q_l0 * config.Q_l1 * config.Q_l2
        R_tile_l2 = config.R_l0 * config.R_l1 * config.R_l2
        S_tile_l2 = config.S_l0 * config.S_l1 * config.S_l2
        C_tile_l2 = config.C_l0 * config.C_l1 * config.C_l2
        N_tile_l2 = config.N_l0 * config.N_l1 * config.N_l2
        
        # 计算 Input 的 H_tile 和 W_tile (unique 元素)
        H_tile = self._compute_unique_input_size(wl.stride_h, wl.dilation_h, P_tile_l2, R_tile_l2)
        W_tile = self._compute_unique_input_size(wl.stride_w, wl.dilation_w, Q_tile_l2, S_tile_l2)
        
        # Input 在 RowBuffer: 取决于 layout
        if config.input_layout == 'row_aligned':
            # 约束 1: block_h × block_w × C_tile × N_tile <= row_buffer_size
            input_rowbuf_size = config.block_h * config.block_w * C_tile_l2 * N_tile_l2
            
            if input_rowbuf_size > l2_capacity:
                violations.append(
                    f"RowBuffer input (row_aligned): "
                    f"block_h({config.block_h}) × block_w({config.block_w}) × C({C_tile_l2}) × N({N_tile_l2}) "
                    f"= {input_rowbuf_size} elements > capacity {l2_capacity} elements"
                )
            
            # 约束 2: block_h >= H_tile
            if config.block_h < H_tile:
                violations.append(
                    f"RowBuffer input: block_h({config.block_h}) < H_tile({H_tile}), "
                    f"tile exceeds block height"
                )
            
            # 约束 3: block_w >= W_tile
            if config.block_w < W_tile:
                violations.append(
                    f"RowBuffer input: block_w({config.block_w}) < W_tile({W_tile}), "
                    f"tile exceeds block width"
                )
        else:
            # sequential: 正常 tile 大小
            l2_tiles = self._compute_tile_sizes(config, level=2)
            if l2_tiles['input'] > l2_capacity:
                violations.append(
                    f"RowBuffer input: {l2_tiles['input']} elements "
                    f"> capacity {l2_capacity} elements"
                )
        
        # Weight 和 Output 在 RowBuffer: 正常 tile 大小
        l2_tiles = self._compute_tile_sizes(config, level=2)
        
        if l2_tiles['weight'] > l2_capacity:
            violations.append(
                f"RowBuffer weight: {l2_tiles['weight']} elements "
                f"> capacity {l2_capacity} elements"
            )
        
        if l2_tiles['output'] > l2_capacity:
            violations.append(
                f"RowBuffer output: {l2_tiles['output']} elements "
                f"> capacity {l2_capacity} elements"
            )
        
        return violations
    
    def is_valid(self, config: MappingConfig) -> bool:
        """快速检查是否合法"""
        valid, _ = self.check_all(config)
        return valid
    
    def get_tile_summary(self, config: MappingConfig) -> Dict:
        """获取各层级 tile 大小摘要（用于调试）"""
        return {
            'L0': self._compute_tile_sizes(config, level=0),
            'L1': self._compute_tile_sizes(config, level=1),
            'L2': self._compute_tile_sizes(config, level=2),
            'capacities': {
                'GlobalBuffer': self.arch.global_buffer_bytes // self.arch.element_size,
                'RowBuffer': self.arch.row_buffer_bytes // self.arch.element_size,
            }
        }
