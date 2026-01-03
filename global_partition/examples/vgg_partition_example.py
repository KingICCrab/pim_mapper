"""
Example: Global Partition Optimization for VGG Network

This example demonstrates how to use the ILP-based global partition optimizer
to find the optimal partition scheme for VGG-16 network.

The key insight is that partition choices are NOT independent across layers:
- Layer i's output channel (K) partition affects Layer i+1's input distribution
- This "partition propagation" constraint makes greedy optimization suboptimal
- ILP finds the globally optimal solution considering all layer interactions
"""

from nn_dataflow.core import InputLayer, ConvLayer, FCLayer, PoolingLayer
from nn_dataflow.core import Network
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_simple_vgg():
    """Create a simplified VGG-like network for demonstration."""
    nn = Network('SimpleVGG')
    nn.set_input_layer(InputLayer(3, 224))

    # First block: 3 -> 64 channels
    nn.add('conv1_1', ConvLayer(3, 64, 224, 3))
    nn.add('conv1_2', ConvLayer(64, 64, 224, 3))
    nn.add('pool1', PoolingLayer(64, 112, 2))

    # Second block: 64 -> 128 channels
    nn.add('conv2_1', ConvLayer(64, 128, 112, 3))
    nn.add('conv2_2', ConvLayer(128, 128, 112, 3))
    nn.add('pool2', PoolingLayer(128, 56, 2))

    # Third block: 128 -> 256 channels
    nn.add('conv3_1', ConvLayer(128, 256, 56, 3))
    nn.add('conv3_2', ConvLayer(256, 256, 56, 3))
    nn.add('pool3', PoolingLayer(256, 28, 2))

    # Fourth block: 256 -> 512 channels
    nn.add('conv4_1', ConvLayer(256, 512, 28, 3))
    nn.add('conv4_2', ConvLayer(512, 512, 28, 3))
    nn.add('pool4', PoolingLayer(512, 14, 2))

    return nn


def create_mini_network():
    """Create a minimal network for quick testing."""
    nn = Network('MiniNet')
    nn.set_input_layer(InputLayer(3, 32))

    nn.add('conv1', ConvLayer(3, 16, 32, 3))
    nn.add('conv2', ConvLayer(16, 32, 32, 3))
    nn.add('pool1', PoolingLayer(32, 16, 2))
    nn.add('conv3', ConvLayer(32, 64, 16, 3))
    nn.add('conv4', ConvLayer(64, 64, 16, 3))

    return nn


class MockResource:
    """Mock resource object for demonstration."""

    class MockRegion:
        class MockDim:
            def __init__(self, h, w):
                self.h = h
                self.w = w

            def size(self):
                return self.h * self.w

        def __init__(self, h, w):
            self.dim = self.MockDim(h, w)

    def __init__(self, nodes_h=4, nodes_w=4):
        self.proc_region = self.MockRegion(nodes_h, nodes_w)
        self.dim_nodes = self.proc_region.dim


def demo_without_ilp_solver():
    """
    Demonstrate the global partition problem without requiring ILP solver.

    This shows:
    1. How partition choices propagate between layers
    2. Why greedy optimization is suboptimal
    3. The structure of the ILP formulation
    """
    print("="*70)
    print("Global Partition Optimization Demo (No ILP Solver Required)")
    print("="*70)

    # Create a mini network
    network = create_mini_network()
    resource = MockResource(4, 4)  # 4x4 = 16 nodes

    print(f"\nNetwork: {network.net_name}")
    print(f"Total nodes available: {resource.dim_nodes.size()}")
    print("\nLayers:")
    for name in network:
        layer = network[name]
        if hasattr(layer, 'nifm'):
            print(f"  {name}: {layer.nifm} -> {layer.nofm} channels, "
                  f"{layer.hofm}x{layer.wofm} spatial")

    # Illustrate partition propagation
    print("\n" + "-"*70)
    print("Partition Propagation Illustration")
    print("-"*70)

    print("""
Consider layers conv1 and conv2:
  conv1: 3 -> 16 channels
  conv2: 16 -> 32 channels

If we partition conv1's output channels (K=16) by 4:
  - conv1's output is distributed: K/4 = 4 channels per node
  - conv2's input (C=16) is already distributed across 4 nodes
  
Options for conv2:
  1. Keep K-partition=4: Output also distributed by 4
     → Propagates to conv3
  2. Use different partition: Requires data redistribution
     → Additional communication cost
     
This is why partition choices are NOT independent!
""")

    # Show partition choice space
    print("-"*70)
    print("Partition Choice Space")
    print("-"*70)

    from global_partition.ilp_optimizer import (
        LayerPartitionConfig, PartitionChoice, PartDim
    )

    for name in network:
        layer = network[name]
        config = LayerPartitionConfig(name, layer, 0)

        print(f"\n{name}:")
        print(
            f"  Valid K factors: {config.valid_factors.get(PartDim.OUTP, [1])[:5]}...")
        print(
            f"  Valid H factors: {config.valid_factors.get(PartDim.OFMP_H, [1])[:5]}...")

        # Show a few partition choices
        choices = []
        for k in [1, 2, 4]:
            if k in config.valid_factors.get(PartDim.OUTP, []):
                choices.append(PartitionChoice({PartDim.OUTP: k}))
        for h in [2, 4]:
            if h in config.valid_factors.get(PartDim.OFMP_H, []):
                choices.append(PartitionChoice({PartDim.OFMP_H: h}))

        print(f"  Example choices: {choices[:4]}")

    # Show ILP formulation structure
    print("\n" + "-"*70)
    print("ILP Formulation Structure")
    print("-"*70)

    print("""
Decision Variables:
  x[l, c] ∈ {0, 1}  : Layer l uses partition choice c

Constraints:
  1. Σ_c x[l,c] = 1  for all layers l
     (Each layer uses exactly one partition)
     
  2. nodes(c) ≤ total_nodes  when x[l,c] = 1
     (Partition doesn't exceed available nodes)
     
  3. Partition propagation constraints (implicit in cost)
     K-partition of layer l affects C-distribution of layer l+1

Objective:
  Minimize Σ_l (compute_cost[l,c] * x[l,c]) +
           Σ_l (redist_cost[l,c,c'] * x[l,c] * x[l+1,c'])
           
  Where redist_cost captures the K→C propagation penalty
""")

    # Demonstrate greedy vs global optimal
    print("-"*70)
    print("Greedy vs Global Optimal")
    print("-"*70)

    print("""
Greedy approach (like nn_dataflow):
  - Optimize each layer independently
  - Pick best partition for conv1, then conv2, etc.
  - Problem: Doesn't consider inter-layer costs

Example:
  Layer    Greedy Choice    Cost
  conv1    K=4             100
  conv2    H=4             120   + redist_cost(K→H) = 50
  conv3    K=2             80    + redist_cost(H→K) = 40
  Total greedy: 100 + 170 + 120 = 390

Global optimal (ILP):
  Layer    Optimal Choice   Cost
  conv1    K=4             100
  conv2    K=4             130   + redist_cost = 0 (propagates!)
  conv3    K=4             90    + redist_cost = 0
  Total optimal: 100 + 130 + 90 = 320

The global solution is better because it considers partition propagation!
""")


def demo_with_ilp_solver():
    """
    Full demonstration with ILP solver (requires gurobipy or pulp).
    """
    print("="*70)
    print("Global Partition Optimization with ILP Solver")
    print("="*70)

    try:
        from global_partition.ilp_optimizer import GlobalPartitionILPOptimizer
        from global_partition.nn_dataflow_cost import SimpleCostModel
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure global_partition module is in the path.")
        return

    # Create network and resource
    network = create_mini_network()
    resource = MockResource(4, 4)

    print(f"\nNetwork: {network.net_name}")
    print(f"Nodes: {resource.dim_nodes.size()}")

    # Create optimizer
    try:
        optimizer = GlobalPartitionILPOptimizer(
            network=network,
            resource=resource,
            batch_size=1,
            max_partitions_per_dim=8,
            solver='auto'  # Will use pulp if gurobi not available
        )

        print(f"\nUsing solver: {optimizer.solver}")
        print(f"Layers: {len(optimizer.layers)}")
        for l, choices in enumerate(optimizer.partition_choices):
            print(f"  Layer {l}: {len(choices)} partition choices")

        # Run optimization
        print("\nRunning ILP optimization...")
        solution = optimizer.optimize(time_limit=60, verbose=False)

        # Print results
        optimizer.print_solution(solution)

    except ImportError as e:
        print(f"\nILP solver not available: {e}")
        print("Install gurobipy (commercial) or pulp (open-source):")
        print("  pip install pulp")
        print("\nFalling back to demonstration mode...")
        demo_without_ilp_solver()


def compare_greedy_vs_global():
    """
    Compare greedy (layer-by-layer) vs global (ILP) optimization.
    """
    print("="*70)
    print("Comparison: Greedy vs Global Optimization")
    print("="*70)

    from global_partition.ilp_optimizer import (
        LayerPartitionConfig, PartitionChoice, PartDim
    )
    from global_partition.nn_dataflow_cost import SimpleCostModel

    network = create_mini_network()
    cost_model = SimpleCostModel()

    # Build layer configs
    configs = []
    for idx, name in enumerate(network):
        layer = network[name]
        configs.append(LayerPartitionConfig(name, layer, idx))

    # Greedy: pick best partition for each layer independently
    print("\n--- Greedy Approach ---")
    greedy_solution = []
    greedy_total = 0

    for idx, config in enumerate(configs):
        best_cost = float('inf')
        best_choice = None

        # Try different partitions
        for k_factor in [1, 2, 4]:
            choice = PartitionChoice({PartDim.OUTP: k_factor})
            cost = cost_model.compute_cost(config.layer, choice)
            if cost < best_cost:
                best_cost = cost
                best_choice = choice

        greedy_solution.append((config.layer_name, best_choice))

        # Add redistribution cost
        if idx > 0:
            prev_config = configs[idx - 1]
            prev_choice = greedy_solution[idx - 1][1]
            output_size = prev_config.nofm * prev_config.hofm * prev_config.wofm
            redist = cost_model.redistribution_cost(
                prev_config, config, prev_choice, best_choice, output_size)
            best_cost += redist

        greedy_total += best_cost
        print(f"  {config.layer_name}: {best_choice} -> cost={best_cost:.2f}")

    print(f"\nGreedy total: {greedy_total:.2f}")

    # Global: consider all interactions (simplified enumeration for demo)
    print("\n--- Global Optimization (Simplified) ---")
    print("Considering partition propagation...")

    # For demo, try consistent K-partition across all layers
    consistent_total = 0
    for k_factor in [1, 2, 4]:
        total = 0
        for idx, config in enumerate(configs):
            choice = PartitionChoice({PartDim.OUTP: k_factor})
            cost = cost_model.compute_cost(config.layer, choice)
            total += cost

        if consistent_total == 0 or total < consistent_total:
            consistent_total = total
            best_k = k_factor

    print(f"  Best consistent K-partition: K={best_k}")
    print(f"  Cost (no redistribution!): {consistent_total:.2f}")

    print(
        f"\nImprovement: {(greedy_total - consistent_total)/greedy_total*100:.1f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Global Partition Optimization Demo")
    parser.add_argument('--mode', choices=['demo', 'full', 'compare'],
                        default='demo',
                        help='Demo mode: demo (no solver), full (with solver), compare')

    args = parser.parse_args()

    if args.mode == 'demo':
        demo_without_ilp_solver()
    elif args.mode == 'full':
        demo_with_ilp_solver()
    elif args.mode == 'compare':
        compare_greedy_vs_global()
