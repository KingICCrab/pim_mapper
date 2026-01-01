"""
Experiments module.

This module contains validation and analysis experiments.
"""

# 延迟导入以避免模块循环依赖
__all__ = [
    'run_validation',
    'run_permutation_analysis',
]

def __getattr__(name):
    if name == 'run_validation':
        from .validate_formula import run_validation
        return run_validation
    elif name == 'run_permutation_analysis':
        from .analyze_permutation import run_permutation_analysis
        return run_permutation_analysis
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
