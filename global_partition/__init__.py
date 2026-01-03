"""
Global Partition Optimizer for Neural Network Dataflow.

This module solves the global partition optimization problem:
Given a neural network with multiple layers, find the optimal partition
scheme for each layer such that the total cost (computation + data redistribution)
is minimized.

Key insight: The output partition of layer i determines the input data layout
of layer i+1. This creates a propagation constraint that must be considered
globally rather than layer-by-layer.

Two approaches are provided:
1. Graph-based (shortest path): Transforms problem to DAG shortest path
2. ILP-based: Uses Integer Linear Programming for exact solution

The ILP approach is inspired by LEMON (Memory-Aware DNN Algorithm-Hardware
Mapping via Integer Linear Programming, CF'23).
"""

# Core abstractions
from .partition_state import PartitionState, PartitionScheme, PartitionDim
from .partition_graph import PartitionGraph, PartitionNode, PartitionEdge

# ILP-based optimizer (recommended for exact solutions)
from .ilp_optimizer import (
    GlobalPartitionILPOptimizer,
    PartDim,
    PartitionChoice,
    LayerPartitionConfig,
    RedistributionType,
    create_optimizer_from_nn_dataflow,
)

# Cost models
from .nn_dataflow_cost import (
    NNDataflowCostModel,
    SimpleCostModel,
    create_cost_model_from_nn_dataflow,
)

# Legacy imports (for backward compatibility)
try:
    from .cost_model import CostModel
    from .redistribution_cost import RedistributionCost
except ImportError:
    pass

__all__ = [
    # Core
    'PartitionState',
    'PartitionScheme',
    'PartitionDim',
    'PartitionGraph',
    'PartitionNode',
    'PartitionEdge',

    # ILP Optimizer
    'GlobalPartitionILPOptimizer',
    'PartDim',
    'PartitionChoice',
    'LayerPartitionConfig',
    'RedistributionType',
    'create_optimizer_from_nn_dataflow',

    # Cost Models
    'NNDataflowCostModel',
    'SimpleCostModel',
    'create_cost_model_from_nn_dataflow',
]
