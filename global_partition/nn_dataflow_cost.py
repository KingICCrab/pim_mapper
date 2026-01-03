"""
Cost model integration with nn_dataflow.

This module provides cost estimation for partitioning decisions,
integrating with nn_dataflow's existing cost model.
"""

from typing import Dict, Optional, Tuple
from enum import IntEnum

# Import nn_dataflow modules
try:
    from nn_dataflow.core import mem_hier_enum as me
    from nn_dataflow.core.cost import Cost
    HAS_NN_DATAFLOW = True
except ImportError:
    HAS_NN_DATAFLOW = False


class DataCategory(IntEnum):
    """Data categories for cost estimation."""
    IFM = 0   # Input feature maps
    OFM = 1   # Output feature maps
    FIL = 2   # Filters/Weights
    NUM = 3


class NNDataflowCostModel:
    """
    Cost model that integrates with nn_dataflow's cost estimation.

    This provides a unified interface for:
    1. Compute cost estimation
    2. Memory access cost estimation  
    3. Communication (redistribution) cost estimation
    """

    def __init__(self, cost: Optional['Cost'] = None,
                 mac_op_cost: float = 1.0,
                 mem_costs: Optional[Tuple[float, ...]] = None,
                 noc_hop_cost: float = 1.0):
        """
        Args:
            cost: nn_dataflow Cost object (if available)
            mac_op_cost: Cost per MAC operation
            mem_costs: Tuple of memory costs (REGF, ITCN, GBUF, DRAM)
            noc_hop_cost: Cost per NoC hop
        """
        if cost is not None and HAS_NN_DATAFLOW:
            self.mac_op_cost = cost.mac_op
            self.mem_costs = cost.mem_hier
            self.noc_hop_cost = cost.noc_hop
        else:
            self.mac_op_cost = mac_op_cost
            # Default memory costs: REGF < ITCN < GBUF < DRAM
            self.mem_costs = mem_costs or (1.0, 2.0, 6.0, 200.0)
            self.noc_hop_cost = noc_hop_cost

    def compute_cost(self, layer, partition_choice) -> float:
        """
        Estimate computation cost for a layer with given partition.

        Args:
            layer: nn_dataflow Layer object
            partition_choice: PartitionChoice object

        Returns:
            Estimated computation cost
        """
        # Get layer dimensions
        nifm = getattr(layer, 'nifm', 1)
        nofm = getattr(layer, 'nofm', 1)
        hofm = getattr(layer, 'hofm', 1)
        wofm = getattr(layer, 'wofm', 1)
        hfil = getattr(layer, 'hfil', 1)
        wfil = getattr(layer, 'wfil', 1)

        # Total MACs
        total_macs = nifm * nofm * hofm * wofm * hfil * wfil

        # Partition factors
        from .ilp_optimizer import PartDim
        batch_factor = partition_choice.get_factor(PartDim.BATCH)
        outp_factor = partition_choice.get_factor(PartDim.OUTP)
        ofmp_h_factor = partition_choice.get_factor(PartDim.OFMP_H)
        ofmp_w_factor = partition_choice.get_factor(PartDim.OFMP_W)
        inpp_factor = partition_choice.get_factor(PartDim.INPP)

        total_partitions = partition_choice.total_nodes

        # MACs per partition
        macs_per_partition = total_macs / total_partitions

        # MAC cost
        mac_cost = macs_per_partition * self.mac_op_cost

        # Memory access cost estimation
        # This depends on data reuse within each partition
        mem_access_cost = self._estimate_mem_access_cost(
            layer, partition_choice, macs_per_partition)

        # Overhead for INPP partitioning (requires reduction)
        reduction_overhead = 0
        if inpp_factor > 1:
            # Partial sums need to be reduced
            partial_sum_size = nofm * hofm * wofm / \
                (outp_factor * ofmp_h_factor * ofmp_w_factor)
            reduction_overhead = partial_sum_size * 2 * \
                (inpp_factor - 1) * self.noc_hop_cost

        return mac_cost + mem_access_cost + reduction_overhead

    def _estimate_mem_access_cost(self, layer, partition_choice, macs_per_partition):
        """Estimate memory access cost for a partition."""
        from .ilp_optimizer import PartDim

        nifm = getattr(layer, 'nifm', 1)
        nofm = getattr(layer, 'nofm', 1)
        hofm = getattr(layer, 'hofm', 1)
        wofm = getattr(layer, 'wofm', 1)
        hfil = getattr(layer, 'hfil', 1)
        wfil = getattr(layer, 'wfil', 1)

        # Partition factors
        outp_factor = partition_choice.get_factor(PartDim.OUTP)
        ofmp_h_factor = partition_choice.get_factor(PartDim.OFMP_H)
        ofmp_w_factor = partition_choice.get_factor(PartDim.OFMP_W)
        inpp_factor = partition_choice.get_factor(PartDim.INPP)
        batch_factor = partition_choice.get_factor(PartDim.BATCH)

        # Data sizes per partition
        # IFM: depends on spatial and input channel partitioning
        hifm_per_part = hofm / ofmp_h_factor + (hfil - 1)  # Halo
        wifm_per_part = wofm / ofmp_w_factor + (wfil - 1)
        ifm_per_part = hifm_per_part * wifm_per_part * (nifm / inpp_factor)

        # OFM: depends on spatial and output channel partitioning
        hofm_per_part = hofm / ofmp_h_factor
        wofm_per_part = wofm / ofmp_w_factor
        ofm_per_part = hofm_per_part * wofm_per_part * (nofm / outp_factor)

        # FIL: depends on channel partitioning
        fil_per_part = hfil * wfil * \
            (nifm / inpp_factor) * (nofm / outp_factor)

        # Estimate memory level access
        # Assume data fits in global buffer
        gbuf_cost = self.mem_costs[2] if len(self.mem_costs) > 2 else 6.0

        # Simple cost model: each data element accessed multiple times
        # IFM reuse = nofm/outp_factor (across output channels)
        ifm_reuse = nofm / outp_factor
        ofm_reuse = nifm / inpp_factor * hfil * wfil  # partial sum accumulation
        fil_reuse = hofm_per_part * wofm_per_part  # across spatial locations

        mem_cost = (
            ifm_per_part / ifm_reuse * gbuf_cost +
            ofm_per_part / ofm_reuse * gbuf_cost +
            fil_per_part / fil_reuse * gbuf_cost
        )

        return mem_cost

    def redistribution_cost(self, src_config, dst_config,
                            src_choice, dst_choice,
                            output_size: int) -> float:
        """
        Estimate data redistribution cost between layers.

        Args:
            src_config: Source layer config
            dst_config: Destination layer config
            src_choice: Partition choice for source layer
            dst_choice: Partition choice for destination layer
            output_size: Size of output data (number of elements)

        Returns:
            Redistribution cost
        """
        from .ilp_optimizer import PartDim, RedistributionType

        # Get partition factors
        src_outp = src_choice.get_factor(PartDim.OUTP)
        src_ofmp_h = src_choice.get_factor(PartDim.OFMP_H)
        src_ofmp_w = src_choice.get_factor(PartDim.OFMP_W)
        src_batch = src_choice.get_factor(PartDim.BATCH)
        src_inpp = src_choice.get_factor(PartDim.INPP)

        dst_outp = dst_choice.get_factor(PartDim.OUTP)
        dst_ofmp_h = dst_choice.get_factor(PartDim.OFMP_H)
        dst_ofmp_w = dst_choice.get_factor(PartDim.OFMP_W)
        dst_batch = dst_choice.get_factor(PartDim.BATCH)
        dst_inpp = dst_choice.get_factor(PartDim.INPP)

        # Determine redistribution type and cost
        # Case 1: INPP partition requires all-reduce
        if src_inpp > 1:
            # Ring all-reduce cost: 2 * (n-1)/n * data_size
            return output_size * 2 * (src_inpp - 1) / src_inpp * self.noc_hop_cost

        # Case 2: Check if output channel partition matches
        # src's K-partition affects dst's input distribution
        if src_outp > 1:
            if dst_outp == src_outp:
                # Output channel partition propagates, data stays local
                return 0
            elif dst_inpp > 1 and dst_inpp == src_outp:
                # Matches: src K-partition -> dst C-partition
                return 0
            else:
                # Need all-to-all redistribution
                total_nodes = max(src_choice.total_nodes,
                                  dst_choice.total_nodes)
                return output_size * (total_nodes - 1) / total_nodes * self.noc_hop_cost

        # Case 3: Spatial partition changes
        if (src_ofmp_h != dst_ofmp_h or src_ofmp_w != dst_ofmp_w):
            # Need to redistribute spatially
            src_nodes = src_ofmp_h * src_ofmp_w
            dst_nodes = dst_ofmp_h * dst_ofmp_w
            if src_nodes == 1:
                # Scatter
                return output_size * (dst_nodes - 1) / dst_nodes * self.noc_hop_cost
            elif dst_nodes == 1:
                # Gather
                return output_size * (src_nodes - 1) / src_nodes * self.noc_hop_cost
            else:
                # All-to-all
                total_nodes = max(src_nodes, dst_nodes)
                return output_size * (total_nodes - 1) / total_nodes * self.noc_hop_cost

        # Case 4: Batch partition changes
        if src_batch != dst_batch:
            total_nodes = max(src_batch, dst_batch)
            return output_size * (total_nodes - 1) / total_nodes * self.noc_hop_cost

        # No redistribution needed
        return 0

    def total_network_cost(self, solution, layer_configs) -> Dict:
        """
        Calculate total network cost for a partition solution.

        Args:
            solution: List of (layer_name, partition_choice) tuples
            layer_configs: List of LayerPartitionConfig objects

        Returns:
            Dictionary with cost breakdown
        """
        total_compute = 0
        total_redist = 0
        layer_costs = []

        for idx, (layer_name, choice) in enumerate(solution):
            config = layer_configs[idx]

            # Compute cost
            compute = self.compute_cost(config.layer, choice)
            total_compute += compute

            # Redistribution cost (from previous layer)
            redist = 0
            if idx > 0:
                prev_name, prev_choice = solution[idx - 1]
                prev_config = layer_configs[idx - 1]
                output_size = (prev_config.nofm * prev_config.hofm *
                               prev_config.wofm)
                redist = self.redistribution_cost(
                    prev_config, config, prev_choice, choice, output_size)
            total_redist += redist

            layer_costs.append({
                'layer': layer_name,
                'partition': str(choice),
                'compute_cost': compute,
                'redistribution_cost': redist
            })

        return {
            'total_compute_cost': total_compute,
            'total_redistribution_cost': total_redist,
            'total_cost': total_compute + total_redist,
            'layer_costs': layer_costs
        }


class SimpleCostModel(NNDataflowCostModel):
    """
    Simplified cost model for testing and demonstration.

    Uses simple heuristics instead of detailed cost estimation.
    """

    def __init__(self, nodes_per_dimension=None):
        """
        Args:
            nodes_per_dimension: Available nodes in each dimension (h, w)
        """
        super().__init__()
        self.nodes_per_dim = nodes_per_dimension or (4, 4)

    def compute_cost(self, layer, partition_choice) -> float:
        """Simple compute cost: MACs / parallelism."""
        nifm = getattr(layer, 'nifm', 1)
        nofm = getattr(layer, 'nofm', 1)
        hofm = getattr(layer, 'hofm', 1)
        wofm = getattr(layer, 'wofm', 1)
        hfil = getattr(layer, 'hfil', 1)
        wfil = getattr(layer, 'wfil', 1)

        total_macs = nifm * nofm * hofm * wofm * hfil * wfil
        parallelism = partition_choice.total_nodes

        # Efficiency loss for INPP partitioning
        from .ilp_optimizer import PartDim
        inpp = partition_choice.get_factor(PartDim.INPP)
        efficiency = 1.0 / (1.0 + 0.1 * (inpp - 1)) if inpp > 1 else 1.0

        return total_macs / parallelism / efficiency

    def redistribution_cost(self, src_config, dst_config,
                            src_choice, dst_choice,
                            output_size: int) -> float:
        """Simple redistribution cost model."""
        from .ilp_optimizer import PartDim

        # Check if partitions are compatible
        src_k = src_choice.get_factor(PartDim.OUTP)
        dst_c_related = dst_choice.get_factor(PartDim.OUTP)  # Simplified

        # If source K-partition matches destination needs, no redistribution
        if src_choice.total_nodes == dst_choice.total_nodes:
            # Check spatial compatibility
            src_h = src_choice.get_factor(PartDim.OFMP_H)
            src_w = src_choice.get_factor(PartDim.OFMP_W)
            dst_h = dst_choice.get_factor(PartDim.OFMP_H)
            dst_w = dst_choice.get_factor(PartDim.OFMP_W)

            if src_h == dst_h and src_w == dst_w:
                return 0

        # Cost proportional to data movement
        nodes = max(src_choice.total_nodes, dst_choice.total_nodes)
        return output_size * (nodes - 1) / nodes


def create_cost_model_from_nn_dataflow(cost=None, **kwargs):
    """
    Factory function to create cost model from nn_dataflow objects.

    Args:
        cost: nn_dataflow.Cost object
        **kwargs: Additional arguments for NNDataflowCostModel

    Returns:
        NNDataflowCostModel instance
    """
    return NNDataflowCostModel(cost=cost, **kwargs)
