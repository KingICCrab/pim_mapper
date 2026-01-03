"""
Partition Graph representation.

The partition graph captures the dependency between layer partitions:
- Nodes: (layer_name, partition_scheme) pairs
- Edges: Data flow dependencies with redistribution costs

This formulation transforms the global partition optimization into
a shortest path problem on the partition graph.
"""

from collections import defaultdict
import itertools

from .partition_state import (
    PartitionScheme, PartitionState, PartitionDim,
    generate_partition_schemes
)


class PartitionNode:
    """
    A node in the partition graph.

    Represents a specific partition choice for a layer.
    """

    def __init__(self, layer_name, layer, scheme, layer_idx):
        self.layer_name = layer_name
        self.layer = layer
        self.scheme = scheme
        self.layer_idx = layer_idx

        # Compute cost is fixed for a given partition
        self._compute_cost = None

    @property
    def node_id(self):
        return (self.layer_name, self.scheme)

    def compute_cost(self, cost_model):
        """Get the computation cost for this partition."""
        if self._compute_cost is None:
            self._compute_cost = cost_model.compute_cost(
                self.layer, self.scheme)
        return self._compute_cost

    def output_layout(self):
        """Get the output data layout."""
        return self.scheme.output_layout()

    def __repr__(self):
        return f"PartitionNode({self.layer_name}, {self.scheme})"

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        if isinstance(other, PartitionNode):
            return self.node_id == other.node_id
        return False


class PartitionEdge:
    """
    An edge in the partition graph.

    Represents the data flow from one layer's partition to another,
    including any redistribution cost.
    """

    def __init__(self, src_node, dst_node, redistribution_cost=0):
        self.src_node = src_node
        self.dst_node = dst_node
        self.redistribution_cost = redistribution_cost

    @property
    def edge_id(self):
        return (self.src_node.node_id, self.dst_node.node_id)

    def total_cost(self):
        """Total cost of this edge (redistribution only, compute is on nodes)."""
        return self.redistribution_cost

    def __repr__(self):
        return f"PartitionEdge({self.src_node.layer_name} -> {self.dst_node.layer_name}, cost={self.redistribution_cost})"


class PartitionGraph:
    """
    A graph representing all possible partition choices for a network.

    Structure:
    - Virtual source node connects to all partition choices of the first layer
    - Virtual sink node receives from all partition choices of the last layer
    - Between consecutive layers, edges connect compatible partitions

    The optimal global partition is the shortest path from source to sink.
    """

    def __init__(self, network, num_nodes, cost_model, batch_size=1):
        """
        Build the partition graph for a network.

        Args:
            network: nn_dataflow Network object
            num_nodes: Number of compute nodes
            cost_model: CostModel instance for computing costs
            batch_size: Batch size for the computation
        """
        self.network = network
        self.num_nodes = num_nodes
        self.cost_model = cost_model
        self.batch_size = batch_size

        # Graph structure
        self.nodes = {}  # node_id -> PartitionNode
        self.edges = {}  # edge_id -> PartitionEdge
        self.adj_list = defaultdict(list)  # node_id -> list of edge_ids
        # node_id -> list of incoming edge_ids
        self.reverse_adj = defaultdict(list)

        # Layer order (topological)
        self.layer_order = list(network)
        self.layer_to_idx = {l: i for i, l in enumerate(self.layer_order)}

        # Nodes grouped by layer
        self.nodes_by_layer = defaultdict(list)

        # Virtual source and sink
        self.source_id = ('__SOURCE__', None)
        self.sink_id = ('__SINK__', None)

        # Build the graph
        self._build_graph()

    def _build_graph(self):
        """Build the partition graph."""
        # Generate partition nodes for each layer
        for layer_idx, layer_name in enumerate(self.layer_order):
            layer = self.network[layer_name]

            # Generate all valid partition schemes
            for scheme in self._generate_valid_schemes(layer):
                node = PartitionNode(layer_name, layer, scheme, layer_idx)
                self.nodes[node.node_id] = node
                self.nodes_by_layer[layer_name].append(node)

        # Create edges between consecutive layers
        for layer_idx in range(len(self.layer_order)):
            layer_name = self.layer_order[layer_idx]

            # Get previous layers
            if layer_idx == 0:
                prev_layers = []
            else:
                prev_layers = list(self.network.prevs(layer_name))

            current_nodes = self.nodes_by_layer[layer_name]

            if not prev_layers:
                # Connect from source
                for dst_node in current_nodes:
                    edge = PartitionEdge(None, dst_node, 0)
                    edge_id = (self.source_id, dst_node.node_id)
                    self.edges[edge_id] = edge
                    self.adj_list[self.source_id].append(edge_id)
                    self.reverse_adj[dst_node.node_id].append(edge_id)
            else:
                # Connect from previous layer(s)
                for prev_layer in prev_layers:
                    prev_nodes = self.nodes_by_layer[prev_layer]

                    for src_node in prev_nodes:
                        for dst_node in current_nodes:
                            # Compute redistribution cost
                            redist_cost = self.cost_model.redistribution_cost(
                                src_node.layer, src_node.scheme,
                                dst_node.layer, dst_node.scheme,
                                self.batch_size
                            )

                            edge = PartitionEdge(
                                src_node, dst_node, redist_cost)
                            edge_id = (src_node.node_id, dst_node.node_id)
                            self.edges[edge_id] = edge
                            self.adj_list[src_node.node_id].append(edge_id)
                            self.reverse_adj[dst_node.node_id].append(edge_id)

        # Connect last layer to sink
        last_layer = self.layer_order[-1]
        for src_node in self.nodes_by_layer[last_layer]:
            edge = PartitionEdge(src_node, None, 0)
            edge_id = (src_node.node_id, self.sink_id)
            self.edges[edge_id] = edge
            self.adj_list[src_node.node_id].append(edge_id)
            self.reverse_adj[self.sink_id].append(edge_id)

    def _generate_valid_schemes(self, layer):
        """Generate valid partition schemes for a layer."""
        # Get layer properties
        has_batch = self.batch_size > 1
        has_spatial = hasattr(layer, 'hofm') and layer.hofm > 1
        has_out_ch = hasattr(layer, 'nofm') and layer.nofm > 1
        has_in_ch = hasattr(layer, 'nifm') and layer.nifm > 1

        # Generate schemes
        for scheme in generate_partition_schemes(self.num_nodes, max_dims=2):
            # Validate scheme against layer properties
            valid = True

            for dim, factor in zip(scheme.dims, scheme.factors):
                if factor == 1:
                    continue

                if dim == PartitionDim.BATCH and not has_batch:
                    valid = False
                    break
                if dim == PartitionDim.BATCH and self.batch_size % factor != 0:
                    valid = False
                    break
                if dim == PartitionDim.OUT_CH and hasattr(layer, 'nofm'):
                    if layer.nofm % factor != 0:
                        valid = False
                        break
                if dim == PartitionDim.OUT_H and hasattr(layer, 'hofm'):
                    if layer.hofm < factor:
                        valid = False
                        break
                if dim == PartitionDim.OUT_W and hasattr(layer, 'wofm'):
                    if layer.wofm < factor:
                        valid = False
                        break
                if dim == PartitionDim.IN_CH and hasattr(layer, 'nifm'):
                    if layer.nifm % factor != 0:
                        valid = False
                        break

            if valid:
                yield scheme

    def get_node(self, node_id):
        """Get a node by its ID."""
        if node_id in (self.source_id, self.sink_id):
            return None
        return self.nodes.get(node_id)

    def get_edge(self, edge_id):
        """Get an edge by its ID."""
        return self.edges.get(edge_id)

    def successors(self, node_id):
        """Get successor edges from a node."""
        return [self.edges[eid] for eid in self.adj_list[node_id]]

    def predecessors(self, node_id):
        """Get predecessor edges to a node."""
        return [self.edges[eid] for eid in self.reverse_adj[node_id]]

    def num_partition_nodes(self):
        """Total number of partition nodes (excluding source/sink)."""
        return len(self.nodes)

    def num_edges(self):
        """Total number of edges."""
        return len(self.edges)

    def layer_partition_count(self, layer_name):
        """Number of partition options for a layer."""
        return len(self.nodes_by_layer[layer_name])

    def __repr__(self):
        return (f"PartitionGraph(layers={len(self.layer_order)}, "
                f"nodes={self.num_partition_nodes()}, "
                f"edges={self.num_edges()})")
