"""
Cost Model for partition optimization.

This module provides cost estimation for:
1. Computation cost for a given partition scheme
2. Data redistribution cost between different partitions
"""

import math
from .partition_state import PartitionDim, PartitionScheme


class CostModel:
    """
    Cost model for partition optimization.

    Provides methods to estimate:
    - Computation cost for a layer with a given partition
    - Data redistribution cost between consecutive layers
    """

    def __init__(self,
                 num_nodes,
                 memory_bandwidth=256,  # GB/s
                 noc_bandwidth=128,     # GB/s
                 compute_throughput=1e12,  # FLOPS
                 word_size=2):          # bytes (FP16)
        """
        Initialize the cost model.

        Args:
            num_nodes: Number of compute nodes
            memory_bandwidth: Memory bandwidth per node (GB/s)
            noc_bandwidth: Network-on-chip bandwidth (GB/s)
            compute_throughput: Compute throughput per node (FLOPS)
            word_size: Word size in bytes
        """
        self.num_nodes = num_nodes
        self.memory_bandwidth = memory_bandwidth * 1e9  # Convert to bytes/s
        self.noc_bandwidth = noc_bandwidth * 1e9
        self.compute_throughput = compute_throughput
        self.word_size = word_size

    def compute_cost(self, layer, scheme, batch_size=1):
        """
        Compute the computation cost for a layer with given partition.

        Cost includes:
        - Compute time (MACs / throughput per partition)
        - Memory access time (data movement to/from memory)

        Returns cost in cycles/time units.
        """
        # Total operations
        total_ops = self._layer_ops(layer, batch_size)

        # Operations per partition
        ops_per_partition = total_ops / scheme.total_partitions

        # Compute time
        compute_time = ops_per_partition / self.compute_throughput

        # Memory access (simplified model)
        # Each partition needs to access its portion of data
        data_size = self._partition_data_size(layer, scheme, batch_size)
        memory_time = data_size / self.memory_bandwidth

        # Return the maximum of compute and memory (they overlap)
        return max(compute_time, memory_time)

    def redistribution_cost(self, src_layer, src_scheme,
                            dst_layer, dst_scheme, batch_size=1):
        """
        Compute the data redistribution cost between two partitions.

        This is the key function for global optimization - it captures
        the cost of partition mismatch between consecutive layers.

        Returns cost in cycles/time units.
        """
        # Get output layout of source and input requirement of destination
        src_output = src_scheme.output_layout()
        dst_input = self._input_layout_requirement(
            dst_scheme, src_layer, dst_layer)

        # Check if redistribution is needed
        if self._layouts_match(src_output, dst_input, src_layer, dst_layer):
            return 0

        # Compute redistribution cost based on type
        redist_type = self._redistribution_type(src_output, dst_input,
                                                src_layer, dst_layer)

        # Data size to redistribute
        data_size = self._output_data_size(src_layer, batch_size)

        if redist_type == 'none':
            return 0
        elif redist_type == 'local':
            # Data stays on same node, just rearranged
            return data_size / self.memory_bandwidth * 0.1
        elif redist_type == 'all_to_all':
            # All-to-all communication
            # Each node sends and receives data
            # Cost = data_size * (N-1) / N / bandwidth
            n = self.num_nodes
            return data_size * (n - 1) / n / self.noc_bandwidth
        elif redist_type == 'gather':
            # Gather to fewer nodes
            return data_size / self.noc_bandwidth
        elif redist_type == 'scatter':
            # Scatter to more nodes
            return data_size / self.noc_bandwidth
        elif redist_type == 'reduce':
            # Reduce operation (e.g., for input channel partition)
            return data_size / self.noc_bandwidth * math.log2(self.num_nodes)
        else:
            # Default: assume all-to-all
            return data_size / self.noc_bandwidth

    def _layer_ops(self, layer, batch_size):
        """Get total operations for a layer."""
        if hasattr(layer, 'total_ops'):
            return layer.total_ops() * batch_size

        # Estimate for ConvLayer
        if hasattr(layer, 'nofm') and hasattr(layer, 'nifm'):
            nofm = layer.nofm
            nifm = layer.nifm
            hofm = getattr(layer, 'hofm', 1)
            wofm = getattr(layer, 'wofm', 1)
            hfil = getattr(layer, 'hfil', 1)
            wfil = getattr(layer, 'wfil', 1)

            # MACs = batch * K * H * W * C * R * S
            return batch_size * nofm * hofm * wofm * nifm * hfil * wfil * 2

        return 1e6  # Default

    def _partition_data_size(self, layer, scheme, batch_size):
        """Get data size per partition."""
        # Total data = input + output + weights
        total = 0

        # Input feature map
        if hasattr(layer, 'nifm'):
            hifm = getattr(layer, 'hifm', 1)
            wifm = getattr(layer, 'wifm', 1)
            total += batch_size * layer.nifm * hifm * wifm

        # Output feature map
        if hasattr(layer, 'nofm'):
            hofm = getattr(layer, 'hofm', 1)
            wofm = getattr(layer, 'wofm', 1)
            total += batch_size * layer.nofm * hofm * wofm

        # Weights
        if hasattr(layer, 'nofm') and hasattr(layer, 'nifm'):
            hfil = getattr(layer, 'hfil', 1)
            wfil = getattr(layer, 'wfil', 1)
            total += layer.nofm * layer.nifm * hfil * wfil

        # Per partition
        return total * self.word_size / scheme.total_partitions

    def _output_data_size(self, layer, batch_size):
        """Get output data size for a layer."""
        if hasattr(layer, 'nofm'):
            hofm = getattr(layer, 'hofm', 1)
            wofm = getattr(layer, 'wofm', 1)
            return batch_size * layer.nofm * hofm * wofm * self.word_size
        return 0

    def _input_layout_requirement(self, dst_scheme, src_layer, dst_layer):
        """
        Get the input layout requirement for destination partition.

        Key insight: dst's input channels = src's output channels
        So OUT_CH partition of src must match what dst needs for its IN_CH
        """
        req = dst_scheme.output_layout().copy()

        # Map dimensions appropriately
        # dst's input = src's output, so spatial dims and batch should match
        # but channel dims need mapping: dst.IN_CH <- src.OUT_CH

        return req

    def _layouts_match(self, src_output, dst_input, src_layer, dst_layer):
        """Check if two layouts are compatible without redistribution."""
        # Check batch dimension
        if src_output.get(PartitionDim.BATCH, 1) != dst_input.get(PartitionDim.BATCH, 1):
            return False

        # Check spatial dimensions
        if src_output.get(PartitionDim.OUT_H, 1) != dst_input.get(PartitionDim.OUT_H, 1):
            return False
        if src_output.get(PartitionDim.OUT_W, 1) != dst_input.get(PartitionDim.OUT_W, 1):
            return False

        # Check channel mapping: src.OUT_CH should match dst's channel handling
        src_out_ch = src_output.get(PartitionDim.OUT_CH, 1)
        dst_in_ch = dst_input.get(PartitionDim.IN_CH, 1)

        # If dst partitions on IN_CH, it needs data from all src OUT_CH partitions
        # This is always a mismatch (requires reduction)
        if dst_in_ch > 1:
            return False

        # If src partitions on OUT_CH, dst receives partitioned data
        # This is OK if dst doesn't partition on IN_CH

        return True

    def _redistribution_type(self, src_output, dst_input, src_layer, dst_layer):
        """Determine the type of redistribution needed."""
        # Check for input channel partition (requires reduction)
        if dst_input.get(PartitionDim.IN_CH, 1) > 1:
            return 'reduce'

        # Count partition changes
        changes = 0
        for dim in PartitionDim:
            if dim == PartitionDim.NUM or dim == PartitionDim.IN_CH:
                continue
            src_factor = src_output.get(dim, 1)
            dst_factor = dst_input.get(dim, 1)
            if src_factor != dst_factor:
                changes += 1

        if changes == 0:
            return 'none'
        elif changes == 1:
            # Single dimension change
            for dim in PartitionDim:
                if dim == PartitionDim.NUM or dim == PartitionDim.IN_CH:
                    continue
                src_factor = src_output.get(dim, 1)
                dst_factor = dst_input.get(dim, 1)
                if src_factor != dst_factor:
                    if dst_factor > src_factor:
                        return 'scatter'
                    else:
                        return 'gather'

        return 'all_to_all'


class RedistributionCost:
    """
    Detailed redistribution cost analysis.

    Provides breakdown of redistribution costs for debugging and analysis.
    """

    def __init__(self, src_scheme, dst_scheme, data_size, cost_model):
        self.src_scheme = src_scheme
        self.dst_scheme = dst_scheme
        self.data_size = data_size
        self.cost_model = cost_model

        self._analyze()

    def _analyze(self):
        """Analyze the redistribution in detail."""
        src_output = self.src_scheme.output_layout()
        dst_input = self.dst_scheme.output_layout()

        self.src_partitions = self.src_scheme.total_partitions
        self.dst_partitions = self.dst_scheme.total_partitions

        # Determine which dimensions change
        self.changed_dims = []
        for dim in PartitionDim:
            if dim == PartitionDim.NUM:
                continue
            src_factor = src_output.get(dim, 1)
            dst_factor = dst_input.get(dim, 1)
            if src_factor != dst_factor:
                self.changed_dims.append((dim, src_factor, dst_factor))

        # Compute communication volume
        self.comm_volume = self._compute_comm_volume()

    def _compute_comm_volume(self):
        """Compute the total communication volume."""
        if not self.changed_dims:
            return 0

        # For each element, determine how many nodes need it
        # This depends on the partition change pattern

        # Simplified model: assume all-to-all for now
        return self.data_size * (self.cost_model.num_nodes - 1) / self.cost_model.num_nodes

    def summary(self):
        """Get a summary of the redistribution."""
        return {
            'src_scheme': str(self.src_scheme),
            'dst_scheme': str(self.dst_scheme),
            'src_partitions': self.src_partitions,
            'dst_partitions': self.dst_partitions,
            'changed_dims': [(d.name, s, d_) for d, s, d_ in self.changed_dims],
            'data_size_bytes': self.data_size,
            'comm_volume_bytes': self.comm_volume,
        }
