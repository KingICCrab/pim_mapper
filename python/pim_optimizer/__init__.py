"""
PIM Optimizer - Dataflow mapping optimization for PIM architectures.

This package provides an ILP-based optimizer for finding optimal dataflow
mappings for convolutional workloads on Processing-In-Memory architectures.

Main components:
- PIMOptimizer: Main optimizer class
- PIMArchitecture: Architecture definition
- ConvWorkload: Convolution workload definition
- Mapping: Mapping result representation

Quick Start:
    from pim_optimizer import PIMOptimizer, PIMArchitecture, ConvWorkload
    
    arch = PIMArchitecture.from_yaml("arch.yaml")
    workload = ConvWorkload(R=3, S=3, P=56, Q=56, C=64, K=128, N=1)
    
    optimizer = PIMOptimizer(arch=arch)
    result = optimizer.optimize([workload])
    
    print(result.mappings[0].pretty_print())
"""

__version__ = "0.1.0"

from pim_optimizer.arch import PIMArchitecture, MemoryLevel, MemoryHierarchy
from pim_optimizer.workload import ConvWorkload
from pim_optimizer.mapping import Mapping, OptimizationResult
from pim_optimizer.optimizer import PIMOptimizer, run_optimization

__all__ = [
    # Main classes
    "PIMOptimizer",
    "PIMArchitecture",
    "ConvWorkload",
    "Mapping",
    "OptimizationResult",
    
    # Helper classes
    "MemoryLevel",
    "MemoryHierarchy",
    
    # Functions
    "run_optimization",
]
