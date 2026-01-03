from typing import List, Dict, Tuple, Optional
import math
from .data_layout import DataLayout, PhysicalDimension, LayoutConstraint, DimensionType
from .layout_propagator_v2 import LayoutNode


class TilingGenerator:
    """
    Generates DataLayouts and LoopNests based on hardware constraints (Two-level Tiling).
    """

    def __init__(self, array_size: int = 16):
        self.array_size = array_size

    def generate_hardware_aware_layout(self,
                                       logical_shape: Dict[str, int],
                                       layer_type: str = 'Conv') -> DataLayout:
        """
        Heuristically select the optimal layout based on Hardware Constraints (Array Size).

        Logic:
        1. Identify dimensions that can be spatially unrolled to fill the PE Array.
           - For Conv: Output Channels (K/C_out) and Input Channels (C/C_in).
        2. Create a layout that places these Spatial dimensions in the innermost position.
           - This ensures single-cycle parallel access (or efficient multicasting).
        3. Order outer dimensions to maximize reuse (Temporal Locality).
        """

        # Default NCHW fallback
        order = ['N', 'C', 'H', 'W']

        # Check for Channel Parallelism (Systolic / Tensor Core style)
        C = logical_shape.get('C', 1)
        # Note: 'K' (Output Channel) is usually mapped to 'C' in input tensor of next layer,
        # or 'N' in filter tensor. Here we assume we are defining the layout of the ACTIVATION tensor.
        # So we care about C (Channels).

        # If C is large enough, split it for spatial parallelism
        if C >= self.array_size:
            # Split C -> C_out, C_in
            # Layout: [N, C_out, H, W, C_in(Spatial)]
            order = ['N', 'C_out', 'H', 'W', 'C_in']

        # TODO: Support Spatial H/W splitting for very large images (e.g. 4K)
        # if H >= self.array_size: ...

        return self.create_layout(logical_shape, order)

    def create_layout(self,
                      logical_shape: Dict[str, int],
                      order: List[str],
                      blocked_dims: List[str] = None) -> DataLayout:
        """
        Create a DataLayout.

        Args:
            logical_shape: e.g. {'N': 1, 'C': 64, 'H': 32, 'W': 32}
            order: Physical order of dimensions, e.g. ['N', 'C', 'H', 'W'] or ['N', 'C_out', 'H', 'W', 'C_in']
            blocked_dims: List of logical dimensions to block (split into _out and _in). 
                          Used if 'order' contains simple names but we want implicit blocking.
                          (Actually, it's better if 'order' explicitly specifies the split names).

        Returns:
            DataLayout object.
        """
        physical_dims = []

        # Helper to find logical parent
        def get_logical(name):
            if name.endswith('_in') or name.endswith('_out'):
                return name.rsplit('_', 1)[0]
            return name

        for dim_name in order:
            logical_name = get_logical(dim_name)
            full_size = logical_shape.get(logical_name, 1)

            dim_type = DimensionType.TEMPORAL

            if dim_name.endswith('_in'):
                # Inner block -> Candidate for Spatial Unrolling
                if full_size < self.array_size:
                    size = full_size
                else:
                    size = self.array_size
                    # Heuristic: Inner blocks of size 'array_size' are likely Spatial
                    dim_type = DimensionType.SPATIAL
            elif dim_name.endswith('_out'):
                # Outer block
                if full_size < self.array_size:
                    size = 1
                else:
                    size = math.ceil(full_size / self.array_size)
            else:
                # Full dimension
                size = full_size

            pd = PhysicalDimension(
                name=dim_name, logical_dim=logical_name, size=size, dim_type=dim_type)
            physical_dims.append(pd)

        # 1 byte for simplicity
        return DataLayout(logical_shape=logical_shape, ordering=physical_dims, element_size=1)

    def create_loop_nest(self,
                         logical_shape: Dict[str, int],
                         loop_order: List[str]) -> List[Tuple[str, int]]:
        """
        Create a Loop Nest (list of dim, extent tuples).

        Args:
            logical_shape: e.g. {'N': 1, 'C': 64, 'H': 32, 'W': 32}
            loop_order: e.g. ['N', 'C_out', 'H', 'W', 'C_in']

        Returns:
            List of (dim_name, extent).
        """
        nest = []
        for dim_name in loop_order:
            logical_name = dim_name.rsplit(
                '_', 1)[0] if '_' in dim_name else dim_name
            full_size = logical_shape.get(logical_name, 1)

            if dim_name.endswith('_in'):
                if full_size < self.array_size:
                    extent = full_size
                else:
                    extent = self.array_size
            elif dim_name.endswith('_out'):
                if full_size < self.array_size:
                    extent = 1
                else:
                    extent = math.ceil(full_size / self.array_size)
            else:
                extent = full_size

            nest.append((dim_name, extent))
        return nest


class ScenarioBuilder:
    """
    Builds graph scenarios for testing layout propagation.
    """

    def __init__(self):
        self.nodes: Dict[str, LayoutNode] = {}

    def create_node(self, name: str, op_type: str, is_sensitive: bool = True) -> LayoutNode:
        node = LayoutNode(op_name=name, op_type=op_type,
                          is_sensitive=is_sensitive)
        self.nodes[name] = node
        return node

    def link_nodes(self, src_name: str, dst_name: str):
        src = self.nodes[src_name]
        dst = self.nodes[dst_name]
        src.successors.append(dst)
        dst.predecessors.append(src)

    def build_chain(self, node_configs: List[Dict]) -> List[LayoutNode]:
        """
        Build a linear chain of nodes.
        node_configs: List of dicts with keys 'name', 'op_type', 'is_sensitive'.
        """
        created_nodes = []
        for i, config in enumerate(node_configs):
            node = self.create_node(
                config['name'], config['op_type'], config.get('is_sensitive', True))
            created_nodes.append(node)
            if i > 0:
                self.link_nodes(node_configs[i-1]['name'], config['name'])
        return created_nodes
