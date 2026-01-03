"""
Partition State representation.

Each layer has a partition state that describes how its data is distributed
across compute nodes. The key insight is:

    Layer[i].output_partition == Layer[i+1].input_data_layout

This creates the propagation constraint for global optimization.
"""

from enum import IntEnum
from collections import namedtuple
import itertools


class PartitionDim(IntEnum):
    """Partition dimensions for a convolutional layer."""
    BATCH = 0      # Batch dimension (N)
    OUT_CH = 1     # Output channel dimension (K)
    OUT_H = 2      # Output height dimension (H)
    OUT_W = 3      # Output width dimension (W)
    IN_CH = 4      # Input channel dimension (C) - requires reduction
    NUM = 5


class PartitionScheme(namedtuple('PartitionScheme', ['dims', 'factors'])):
    """
    A partition scheme specifies how a layer is partitioned.

    dims: tuple of PartitionDim, specifying which dimensions are partitioned
    factors: tuple of ints, specifying the partition factor for each dim

    Example:
        dims = (PartitionDim.OUT_CH, PartitionDim.OUT_H)
        factors = (4, 2)
        means: partition K into 4 parts, H into 2 parts -> 8 nodes total
    """

    def __new__(cls, dims, factors):
        if len(dims) != len(factors):
            raise ValueError("dims and factors must have same length")
        return super().__new__(cls, tuple(dims), tuple(factors))

    @property
    def total_partitions(self):
        """Total number of partitions (nodes used)."""
        result = 1
        for f in self.factors:
            result *= f
        return result

    def output_layout(self):
        """
        Get the output data layout resulting from this partition.

        Returns a dict mapping PartitionDim to factor.
        """
        layout = {dim: 1 for dim in PartitionDim if dim != PartitionDim.NUM}
        for dim, factor in zip(self.dims, self.factors):
            layout[dim] = factor
        return layout

    def input_requirement(self):
        """
        Get the input data requirement for this partition.

        For most dimensions, input requirement = output partition.
        But IN_CH partition requires data from all partitions (reduction).
        """
        req = self.output_layout()
        # Input channel partition doesn't affect input layout directly
        # but requires reduction after computation
        return req

    def __repr__(self):
        parts = []
        for dim, factor in zip(self.dims, self.factors):
            if factor > 1:
                parts.append(f"{dim.name}={factor}")
        return f"PartitionScheme({', '.join(parts)})"


class PartitionState:
    """
    Represents the partition state of a layer in the network.

    Attributes:
        layer_name: Name of the layer
        scheme: PartitionScheme for this layer
        input_layout: Expected input data layout (from previous layer)
        output_layout: Output data layout (for next layer)
    """

    def __init__(self, layer_name, scheme, input_layout=None):
        self.layer_name = layer_name
        self.scheme = scheme
        self.input_layout = input_layout
        self._output_layout = None

    @property
    def output_layout(self):
        """Output data layout is determined by the partition scheme."""
        if self._output_layout is None:
            self._output_layout = self.scheme.output_layout()
        return self._output_layout

    def needs_redistribution(self, prev_output_layout):
        """
        Check if data redistribution is needed from previous layer.

        Redistribution is needed when:
        1. Previous output layout doesn't match current input requirement
        2. Dimension partition factors are different

        Args:
            prev_output_layout: dict mapping PartitionDim to factor

        Returns:
            bool: True if redistribution is needed
        """
        curr_req = self.scheme.input_requirement()

        for dim in PartitionDim:
            if dim == PartitionDim.NUM:
                continue
            prev_factor = prev_output_layout.get(dim, 1)
            curr_factor = curr_req.get(dim, 1)

            # Special case: IN_CH in current layer
            # Maps to OUT_CH of previous layer
            if dim == PartitionDim.IN_CH:
                # Current layer's input channels = previous layer's output channels
                # Check if OUT_CH partition matches
                if prev_output_layout.get(PartitionDim.OUT_CH, 1) != curr_factor:
                    return True
            elif prev_factor != curr_factor:
                return True

        return False

    def redistribution_type(self, prev_output_layout):
        """
        Determine the type of redistribution needed.

        Returns:
            str: 'none', 'all_to_all', 'gather', 'scatter', or 'reduce'
        """
        if not self.needs_redistribution(prev_output_layout):
            return 'none'

        curr_req = self.scheme.input_requirement()

        # Check for reduction (IN_CH partition in current layer)
        for dim, factor in zip(self.scheme.dims, self.scheme.factors):
            if dim == PartitionDim.IN_CH and factor > 1:
                return 'reduce'

        # Check for all-to-all
        # This happens when partition dimensions change
        prev_dims = set(d for d in PartitionDim
                        if prev_output_layout.get(d, 1) > 1)
        curr_dims = set(d for d in PartitionDim
                        if curr_req.get(d, 1) > 1)

        if prev_dims != curr_dims:
            return 'all_to_all'

        # Check for gather/scatter (same dims but different factors)
        total_prev = 1
        total_curr = 1
        for dim in prev_dims:
            total_prev *= prev_output_layout.get(dim, 1)
            total_curr *= curr_req.get(dim, 1)

        if total_curr < total_prev:
            return 'gather'
        elif total_curr > total_prev:
            return 'scatter'

        return 'all_to_all'


def generate_partition_schemes(num_nodes, max_dims=2):
    """
    Generate all valid partition schemes for a given number of nodes.

    Args:
        num_nodes: Total number of compute nodes
        max_dims: Maximum number of dimensions to partition simultaneously

    Yields:
        PartitionScheme instances
    """
    # Get all ways to factorize num_nodes
    factors = factorize(num_nodes)

    # Partition dimensions (excluding NUM)
    dims = [d for d in PartitionDim if d != PartitionDim.NUM]

    for num_parts in range(1, min(max_dims + 1, len(dims) + 1)):
        for selected_dims in itertools.combinations(dims, num_parts):
            for factor_combo in factors_for_dims(num_nodes, num_parts):
                yield PartitionScheme(selected_dims, factor_combo)


def factorize(n):
    """Get all factorizations of n."""
    if n == 1:
        return [(1,)]

    result = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            result.append((i, n // i))
            if i != n // i:
                result.append((n // i, i))
    return result


def factors_for_dims(n, num_dims):
    """Generate all ways to factor n into num_dims parts."""
    if num_dims == 1:
        yield (n,)
        return

    for i in range(1, n + 1):
        if n % i == 0:
            for rest in factors_for_dims(n // i, num_dims - 1):
                yield (i,) + rest
