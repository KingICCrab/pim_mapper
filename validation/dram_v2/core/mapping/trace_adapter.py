#!/usr/bin/env python3
"""
TraceAdapter: 将 MappingConfig 适配为 trace_generator 期望的格式

trace_generator 期望的接口:
- mapping.layout = {0: str, 1: str, 2: str}
- mapping.tile_info = {'block_h': int, 'block_w': int}
- mapping.permutation = {level: {pos: dim}}
- mapping.loop_bounds = {level: {key: {dim: bound}}}
"""

from typing import Dict

from .config import DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N
from .mapping_config import MappingConfig


class TraceGeneratorMapping:
    """trace_generator 兼容的 Mapping 类"""
    
    def __init__(self, config: MappingConfig):
        self.config = config
        self._build()
    
    def _build(self):
        """构建 trace_generator 需要的属性"""
        self.layout = self._build_layout()
        self.tile_info = self._build_tile_info()
        self.permutation = self._build_permutation()
        self.loop_bounds = self._build_loop_bounds()
    
    def _build_layout(self) -> Dict[int, str]:
        """构建 layout 字典: {0: input, 1: weight, 2: output}"""
        return {
            0: self.config.input_layout,
            1: self.config.weight_layout,
            2: self.config.output_layout
        }
    
    def _build_tile_info(self) -> Dict[str, int]:
        """构建 tile_info 字典"""
        return {
            'block_h': self.config.block_h,
            'block_w': self.config.block_w
        }
    
    def _build_permutation(self) -> Dict[int, Dict[int, int]]:
        """构建 permutation 字典: {level: {pos: dim}}"""
        result = {}
        for level in [1, 2, 3]:
            perm = self.config.get_permutation(level)
            result[level] = {i: dim for i, dim in enumerate(perm)}
        return result
    
    def _build_loop_bounds(self) -> Dict[int, Dict]:
        """构建 loop_bounds 字典
        
        Memory hierarchy:
        - Level 0: PE - 特殊格式 (H, W, Internal, temporal)
        - Level 1-3: 标准格式 (spatial, temporal)
        """
        cfg = self.config
        
        loop_bounds = {}
        
        # Level 0: PE level - 特殊格式
        loop_bounds[0] = {
            'H': {DIM_P: cfg.P_l0, DIM_R: cfg.R_l0},
            'W': {DIM_Q: cfg.Q_l0, DIM_S: cfg.S_l0},
            'Internal': {DIM_C: cfg.C_l0},
            'temporal': {DIM_K: cfg.K_l0, DIM_N: cfg.N_l0}
        }
        
        # Level 1: GlobalBuffer (SRAM)
        loop_bounds[1] = {
            'spatial': {},
            'temporal': {
                DIM_R: cfg.R_l1, DIM_S: cfg.S_l1,
                DIM_P: cfg.P_l1, DIM_Q: cfg.Q_l1,
                DIM_C: cfg.C_l1, DIM_K: cfg.K_l1, DIM_N: cfg.N_l1
            }
        }
        
        # Level 2: RowBuffer
        loop_bounds[2] = {
            'spatial': {},
            'temporal': {
                DIM_R: cfg.R_l2, DIM_S: cfg.S_l2,
                DIM_P: cfg.P_l2, DIM_Q: cfg.Q_l2,
                DIM_C: cfg.C_l2, DIM_K: cfg.K_l2, DIM_N: cfg.N_l2
            }
        }
        
        # Level 3: DRAM
        loop_bounds[3] = {
            'spatial': {},
            'temporal': {
                DIM_R: cfg.R_l3, DIM_S: cfg.S_l3,
                DIM_P: cfg.P_l3, DIM_Q: cfg.Q_l3,
                DIM_C: cfg.C_l3, DIM_K: cfg.K_l3, DIM_N: cfg.N_l3
            }
        }
        
        return loop_bounds
    
    def get_tile_size(self, level: int, dim: int) -> int:
        """获取某层级某维度的 tile size"""
        if level not in self.loop_bounds:
            return 1
        level_bounds = self.loop_bounds[level]
        factor = 1
        for key in ['spatial', 'temporal', 'H', 'W', 'Internal']:
            if key in level_bounds and dim in level_bounds[key]:
                factor *= level_bounds[key][dim]
        return factor


def to_trace_generator_mapping(config: MappingConfig) -> TraceGeneratorMapping:
    """将 MappingConfig 转换为 trace_generator 兼容的 mapping"""
    return TraceGeneratorMapping(config)
