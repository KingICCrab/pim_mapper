"""
Core module for DRAM Row Activation validation.

This module provides:
- MappingSpace: Enumerate valid mapping configurations
- MappingConfig: A specific mapping configuration
"""

from .mapping_space import MappingSpace, MappingConfig, DIM_NAMES, get_divisors

__all__ = [
    'MappingSpace',
    'MappingConfig',
    'DIM_NAMES',
    'get_divisors',
]

