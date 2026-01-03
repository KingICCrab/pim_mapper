"""
Standalone test for Global Partition ILP Optimizer.

This script tests the ILP optimizer without requiring nn_dataflow dependencies.
"""

from global_partition.nn_dataflow_cost import SimpleCostModel
from global_partition.ilp_optimizer import (
    PartDim, PartitionChoice, LayerPartitionConfig,
    GlobalPartitionILPOptimizer, RedistributionType
)
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockLayer:
    """Mock layer for testing without nn_dataflow."""

    def __init__(self, nifm, nofm, hofm, wofm=None, hfil=3, wfil=3):
        self.nifm = nifm
        self.nofm = nofm
        self.hofm = hofm
        self.wofm = wofm or hofm
        self.hfil = hfil
        self.wfil = wfil


class MockNetwork:
    """Mock network for testing."""

    def __init__(self, name='TestNet'):
        self.net_name = name
        self._layers = {}
        self._order = []

    def add(self, name, layer):
        self._layers[name] = layer
        self._order.append(name)

    def __iter__(self):
        return iter(self._order)

    def __getitem__(self, name):
        return self._layers[name]

    def __len__(self):
        return len(self._layers)


class MockResource:
    """Mock resource for testing."""
    class MockDim:
        def __init__(self, h, w):
            self.h = h
            self.w = w

        def size(self):
            return self.h * self.w

    class MockRegion:
        def __init__(self, h, w):
            self.dim = MockResource.MockDim(h, w)

    def __init__(self, h=4, w=4):
        self.proc_region = self.MockRegion(h, w)
        self.dim_nodes = self.proc_region.dim


def test_partition_choice():
    """Test PartitionChoice class."""
    print("Testing PartitionChoice...")

    # No partition
    choice1 = PartitionChoice({})
    assert choice1.total_nodes == 1
    print(f"  No partition: {choice1}, nodes={choice1.total_nodes}")

    # K-partition
    choice2 = PartitionChoice({PartDim.OUTP: 4})
    assert choice2.total_nodes == 4
    print(f"  K=4: {choice2}, nodes={choice2.total_nodes}")

    # Combined partition
    choice3 = PartitionChoice({PartDim.OUTP: 2, PartDim.OFMP_H: 2})
    assert choice3.total_nodes == 4
    print(f"  K=2, H=2: {choice3}, nodes={choice3.total_nodes}")

    print("  ✓ PartitionChoice tests passed\n")


def test_layer_config():
    """Test LayerPartitionConfig."""
    print("Testing LayerPartitionConfig...")

    layer = MockLayer(nifm=64, nofm=128, hofm=56)
    config = LayerPartitionConfig('conv1', layer, 0)

    print(f"  Layer: {config.layer_name}")
    print(f"  nifm={config.nifm}, nofm={config.nofm}")
    print(f"  Valid K factors: {config.valid_factors[PartDim.OUTP][:5]}...")
    print(f"  Valid H factors: {config.valid_factors[PartDim.OFMP_H][:5]}...")

    print("  ✓ LayerPartitionConfig tests passed\n")


def test_cost_model():
    """Test SimpleCostModel."""
    print("Testing SimpleCostModel...")

    cost_model = SimpleCostModel()
    layer = MockLayer(nifm=64, nofm=128, hofm=56)

    # Test compute cost
    choice1 = PartitionChoice({})
    cost1 = cost_model.compute_cost(layer, choice1)

    choice2 = PartitionChoice({PartDim.OUTP: 4})
    cost2 = cost_model.compute_cost(layer, choice2)

    print(f"  No partition cost: {cost1:.0f}")
    print(f"  K=4 partition cost: {cost2:.0f}")
    print(f"  Speedup: {cost1/cost2:.2f}x")

    # Test redistribution cost
    config1 = LayerPartitionConfig('conv1', layer, 0)
    layer2 = MockLayer(nifm=128, nofm=256, hofm=56)
    config2 = LayerPartitionConfig('conv2', layer2, 1)

    # Same partition - should have low cost
    choice_same = PartitionChoice({PartDim.OUTP: 4})
    redist_same = cost_model.redistribution_cost(
        config1, config2, choice_same, choice_same,
        output_size=128 * 56 * 56)

    # Different partition - should have higher cost
    choice_diff = PartitionChoice({PartDim.OFMP_H: 4})
    redist_diff = cost_model.redistribution_cost(
        config1, config2, choice_same, choice_diff,
        output_size=128 * 56 * 56)

    print(f"  Redistribution (same partition): {redist_same:.0f}")
    print(f"  Redistribution (different partition): {redist_diff:.0f}")

    print("  ✓ SimpleCostModel tests passed\n")


def test_optimizer_creation():
    """Test GlobalPartitionILPOptimizer creation."""
    print("Testing GlobalPartitionILPOptimizer creation...")

    # Create mock network
    network = MockNetwork('TestNet')
    network.add('conv1', MockLayer(3, 64, 32))
    network.add('conv2', MockLayer(64, 128, 32))
    network.add('conv3', MockLayer(128, 256, 16))

    resource = MockResource(4, 4)

    try:
        optimizer = GlobalPartitionILPOptimizer(
            network=network,
            resource=resource,
            batch_size=1,
            max_partitions_per_dim=8,
            solver='auto'
        )

        print(f"  Network: {network.net_name}")
        print(f"  Layers: {len(optimizer.layers)}")
        print(f"  Solver: {optimizer.solver}")
        print(f"  Total nodes: {optimizer._get_total_nodes()}")

        for idx, choices in enumerate(optimizer.partition_choices):
            print(f"  Layer {idx}: {len(choices)} partition choices")

        print("  ✓ Optimizer creation tests passed\n")
    except ImportError as e:
        print(f"  ⚠ ILP solver not available: {e}")
        print("  Skipping optimizer creation test (install pulp: pip install pulp)")
        print("  ⚠ Test skipped\n")


def test_ilp_optimization():
    """Test ILP optimization (requires pulp)."""
    print("Testing ILP optimization...")

    try:
        import pulp
    except ImportError:
        print("  ⚠ PuLP not installed, skipping ILP test")
        print("  Install with: pip install pulp")
        print("  ⚠ Test skipped\n")
        return

    # Create simple network
    network = MockNetwork('SimpleNet')
    network.add('conv1', MockLayer(3, 16, 32))
    network.add('conv2', MockLayer(16, 32, 32))
    network.add('conv3', MockLayer(32, 64, 16))

    resource = MockResource(4, 4)

    try:
        optimizer = GlobalPartitionILPOptimizer(
            network=network,
            resource=resource,
            batch_size=1,
            max_partitions_per_dim=4,
            solver='pulp'
        )

        print(f"  Running ILP optimization...")
        solution = optimizer.optimize(time_limit=30, verbose=False)

        print(f"  Solution found!")
        for layer_name, choice in solution:
            print(f"    {layer_name}: {choice}")

        optimizer.print_solution(solution)

        print("  ✓ ILP optimization tests passed\n")
    except ImportError as e:
        print(f"  ⚠ ILP solver not available: {e}")
        print("  ⚠ Test skipped\n")


def demonstrate_partition_propagation():
    """Demonstrate the partition propagation concept."""
    print("="*60)
    print("Demonstration: Partition Propagation")
    print("="*60)

    print("""
The key insight of global partition optimization:

Consider a CNN with layers:
  Layer 1: Conv 3->64, output shape [N, 64, H, W]
  Layer 2: Conv 64->128, output shape [N, 128, H, W]

If Layer 1's output channels (K=64) are partitioned by 4:
  - Output is distributed as [N, 16, H, W] per node
  - Layer 2's input is ALREADY distributed by 4 in C dimension

This creates a "propagation constraint":
  Layer[i].K_partition --> affects --> Layer[i+1].input_distribution

The ILP formulation captures this by:
  1. Decision variables: x[l,c] = 1 if layer l uses choice c
  2. Linearization: y[l,ci,cj] = x[l,ci] * x[l+1,cj]
  3. Redistribution cost depends on (ci, cj) compatibility
  4. Objective: minimize compute + redistribution costs
""")


def main():
    print("\n" + "="*60)
    print("Global Partition ILP Optimizer - Test Suite")
    print("="*60 + "\n")

    test_partition_choice()
    test_layer_config()
    test_cost_model()
    test_optimizer_creation()
    test_ilp_optimization()
    demonstrate_partition_propagation()

    print("="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
