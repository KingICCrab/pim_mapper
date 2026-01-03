"""
Redistribution Cost Analysis.

Detailed analysis of data redistribution costs between different partition schemes.
This is critical for understanding the trade-offs in global partition optimization.
"""

import math
from enum import Enum
from .partition_state import PartitionDim


class RedistributionPattern(Enum):
    """Types of data redistribution patterns."""
    NONE = 0            # No redistribution needed
    LOCAL = 1           # Data rearrangement within same node
    MULTICAST = 2       # One-to-many broadcast
    GATHER = 3          # Many-to-one collection
    ALL_TO_ALL = 4      # All-to-all permutation
    ALL_REDUCE = 5      # Reduction across all nodes
    PARTIAL_REDUCE = 6  # Reduction across subset of nodes


class RedistributionAnalyzer:
    """
    Analyzes data redistribution requirements between partition schemes.
    """

    def __init__(self, num_nodes, noc_topology='mesh',
                 noc_bandwidth=128e9, word_size=2):
        """
        Initialize the analyzer.

        Args:
            num_nodes: Total number of compute nodes
            noc_topology: NoC topology ('mesh', 'torus', 'crossbar')
            noc_bandwidth: Bandwidth per link in bytes/s
            word_size: Data word size in bytes
        """
        self.num_nodes = num_nodes
        self.noc_topology = noc_topology
        self.noc_bandwidth = noc_bandwidth
        self.word_size = word_size

        # Compute topology parameters
        self.mesh_dim = int(math.sqrt(num_nodes))
        if self.mesh_dim ** 2 != num_nodes:
            # Non-square, use rectangular
            for h in range(int(math.sqrt(num_nodes)), 0, -1):
                if num_nodes % h == 0:
                    self.mesh_h = h
                    self.mesh_w = num_nodes // h
                    break
        else:
            self.mesh_h = self.mesh_dim
            self.mesh_w = self.mesh_dim

    def analyze(self, src_layer, src_scheme, dst_layer, dst_scheme, batch_size=1):
        """
        Analyze redistribution between two partition schemes.

        Returns:
            RedistributionResult with detailed analysis
        """
        # Get data layout information
        src_output = src_scheme.output_layout()
        dst_input = dst_scheme.output_layout()

        # Get output data size
        data_size = self._output_data_size(src_layer, batch_size)

        # Determine redistribution pattern
        pattern = self._determine_pattern(src_output, dst_input,
                                          src_layer, dst_layer)

        # Compute communication cost
        comm_cost = self._compute_comm_cost(pattern, data_size,
                                            src_scheme, dst_scheme)

        # Create result
        result = RedistributionResult(
            pattern=pattern,
            src_scheme=src_scheme,
            dst_scheme=dst_scheme,
            data_size=data_size,
            comm_cost=comm_cost,
            details=self._get_details(pattern, src_output, dst_input)
        )

        return result

    def _determine_pattern(self, src_output, dst_input, src_layer, dst_layer):
        """Determine the redistribution pattern needed."""
        # Check for input channel partition (requires reduction)
        if dst_input.get(PartitionDim.IN_CH, 1) > 1:
            return RedistributionPattern.ALL_REDUCE

        # Compare layouts dimension by dimension
        changes = []
        for dim in [PartitionDim.BATCH, PartitionDim.OUT_CH,
                    PartitionDim.OUT_H, PartitionDim.OUT_W]:
            src_factor = src_output.get(dim, 1)
            dst_factor = dst_input.get(dim, 1)
            if src_factor != dst_factor:
                changes.append((dim, src_factor, dst_factor))

        if not changes:
            return RedistributionPattern.NONE

        if len(changes) == 1:
            dim, src_f, dst_f = changes[0]
            if src_f == 1 and dst_f > 1:
                return RedistributionPattern.MULTICAST
            elif src_f > 1 and dst_f == 1:
                return RedistributionPattern.GATHER
            elif src_f > 1 and dst_f > 1:
                return RedistributionPattern.ALL_TO_ALL

        # Multiple dimension changes
        return RedistributionPattern.ALL_TO_ALL

    def _compute_comm_cost(self, pattern, data_size, src_scheme, dst_scheme):
        """Compute communication cost based on pattern."""
        n = self.num_nodes

        if pattern == RedistributionPattern.NONE:
            return 0

        elif pattern == RedistributionPattern.LOCAL:
            # Only local memory access
            return data_size / self.noc_bandwidth * 0.1

        elif pattern == RedistributionPattern.MULTICAST:
            # Broadcast from src partitions to dst partitions
            src_parts = src_scheme.total_partitions
            dst_parts = dst_scheme.total_partitions
            fan_out = dst_parts // src_parts

            if self.noc_topology == 'crossbar':
                return data_size / self.noc_bandwidth
            else:
                # Mesh: log(fan_out) hops
                hops = math.ceil(math.log2(fan_out + 1))
                return data_size * hops / self.noc_bandwidth

        elif pattern == RedistributionPattern.GATHER:
            # Collect from src partitions to dst partitions
            src_parts = src_scheme.total_partitions
            dst_parts = dst_scheme.total_partitions
            fan_in = src_parts // dst_parts

            if self.noc_topology == 'crossbar':
                return data_size / self.noc_bandwidth
            else:
                hops = math.ceil(math.log2(fan_in + 1))
                return data_size * hops / self.noc_bandwidth

        elif pattern == RedistributionPattern.ALL_TO_ALL:
            # All-to-all permutation
            if self.noc_topology == 'crossbar':
                return data_size * (n - 1) / n / self.noc_bandwidth
            else:
                # Mesh: average hop distance
                avg_hops = (self.mesh_h + self.mesh_w) / 3
                return data_size * (n - 1) / n * avg_hops / self.noc_bandwidth

        elif pattern == RedistributionPattern.ALL_REDUCE:
            # All-reduce (e.g., for input channel partition)
            if self.noc_topology == 'crossbar':
                return data_size * 2 * math.log2(n) / self.noc_bandwidth
            else:
                # Ring all-reduce
                return data_size * 2 * (n - 1) / n / self.noc_bandwidth

        else:
            # Default
            return data_size * (n - 1) / n / self.noc_bandwidth

    def _output_data_size(self, layer, batch_size):
        """Get output data size for a layer."""
        if hasattr(layer, 'total_ofmap_size'):
            return layer.total_ofmap_size(batch_size, self.word_size)

        if hasattr(layer, 'nofm'):
            hofm = getattr(layer, 'hofm', 1)
            wofm = getattr(layer, 'wofm', 1)
            return batch_size * layer.nofm * hofm * wofm * self.word_size

        return 0

    def _get_details(self, pattern, src_output, dst_input):
        """Get detailed information about the redistribution."""
        return {
            'pattern': pattern.name,
            'src_output': {d.name: v for d, v in src_output.items() if v > 1},
            'dst_input': {d.name: v for d, v in dst_input.items() if v > 1},
        }


class RedistributionResult:
    """Result of redistribution analysis."""

    def __init__(self, pattern, src_scheme, dst_scheme,
                 data_size, comm_cost, details=None):
        self.pattern = pattern
        self.src_scheme = src_scheme
        self.dst_scheme = dst_scheme
        self.data_size = data_size
        self.comm_cost = comm_cost
        self.details = details or {}

    @property
    def is_zero_cost(self):
        """Check if redistribution has zero cost."""
        return self.pattern == RedistributionPattern.NONE

    def __repr__(self):
        return (f"RedistributionResult(pattern={self.pattern.name}, "
                f"cost={self.comm_cost:.2e})")


def compute_propagation_constraint(prev_scheme, layer):
    """
    Compute the constraint on current layer's partition based on
    the previous layer's partition scheme.

    This captures the key insight: prev's output partition determines
    the data layout that current layer receives.

    Args:
        prev_scheme: PartitionScheme of previous layer
        layer: Current layer

    Returns:
        dict: Constraints on current layer's partition
    """
    prev_output = prev_scheme.output_layout()

    constraints = {
        'preferred_batch': prev_output.get(PartitionDim.BATCH, 1),
        'preferred_spatial': (
            prev_output.get(PartitionDim.OUT_H, 1),
            prev_output.get(PartitionDim.OUT_W, 1)
        ),
        'input_channel_layout': prev_output.get(PartitionDim.OUT_CH, 1),
    }

    return constraints
