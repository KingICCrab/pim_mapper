#!/usr/bin/env python3
"""
Mapping Space 模块

提供 CNN 卷积 mapping 的配置、枚举和约束检查功能。

主要类:
- MappingConfig: 单个 mapping 配置
- MappingEnumerator: 统一的空间枚举器
- MappingConstraintChecker: 约束检查器

配置类:
- ArchConfig: 架构配置
- WorkloadConfig: Workload 配置

工具:
- to_trace_generator_mapping(): 适配 trace_generator
"""

# 配置和常量
from .config import (
    DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N,
    DIM_NAMES, DEFAULT_PERMUTATION,
    ArchConfig, WorkloadConfig
)

# Mapping 配置
from .mapping_config import MappingConfig

# 枚举器
from .enumerator import MappingEnumerator, TilingMode

# 约束检查
from .constraints import MappingConstraintChecker

# trace_generator 适配
from .trace_adapter import to_trace_generator_mapping, TraceGeneratorMapping

# 工具函数
from .utils import get_divisors, get_2level_decompositions, get_factor_decompositions

__all__ = [
    # 常量
    'DIM_R', 'DIM_S', 'DIM_P', 'DIM_Q', 'DIM_C', 'DIM_K', 'DIM_N',
    'DIM_NAMES', 'DEFAULT_PERMUTATION',
    # 配置
    'ArchConfig', 'WorkloadConfig',
    # Mapping
    'MappingConfig',
    # 枚举
    'MappingEnumerator', 'TilingMode',
    # 约束
    'MappingConstraintChecker',
    # 适配
    'to_trace_generator_mapping', 'TraceGeneratorMapping',
    # 工具
    'get_divisors', 'get_2level_decompositions', 'get_factor_decompositions',
]
