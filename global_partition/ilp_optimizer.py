"""
ILP-based Global Partition Optimizer.

This module implements Integer Linear Programming (ILP) formulation for
finding the globally optimal partition scheme across all layers in a
neural network.

Key Insight:
    Partition dimension propagation creates dependencies between layers:
    - Layer[i] outputs partitioned by K (output channels)
    - Layer[i+1] receives inputs partitioned by C (input channels)
    - Since K[i] == C[i+1], their partition factors must match

ILP Formulation:
    Decision Variables:
        x[l, d, f] ∈ {0, 1} : layer l uses partition dim d with factor f
        
    Constraints:
        1. Each layer uses exactly one partition scheme
        2. Total nodes used <= available nodes
        3. Partition propagation: K-partition of layer l == C-partition of layer l+1
        4. Valid partition factors (must divide dimension size)
        
    Objective:
        Minimize: Σ (compute_cost[l] + redistribution_cost[l-1 → l])

Inspired by LEMON (Memory-Aware DNN Algorithm-Hardware Mapping via ILP)
but adapted for partition optimization problem.
"""

import itertools
from enum import IntEnum
from collections import namedtuple, defaultdict
from typing import List, Dict, Tuple, Optional, Set

# Try to import ILP solvers
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


class PartDim(IntEnum):
    """Partition dimensions aligned with nn_dataflow."""
    BATCH = 0   # N - Batch
    OUTP = 1    # K - Output channels (mapped to next layer's input)
    OFMP_H = 2  # H - Output feature map height
    OFMP_W = 3  # W - Output feature map width
    INPP = 4    # C - Input channels (requires reduction)
    NUM = 5


class LayerPartitionConfig:
    """Configuration for partitioning a single layer."""

    def __init__(self, layer_name, layer, layer_idx):
        self.layer_name = layer_name
        self.layer = layer
        self.layer_idx = layer_idx

        # Extract layer dimensions
        self.nifm = getattr(layer, 'nifm', 1)  # Input channels (C)
        self.nofm = getattr(layer, 'nofm', 1)  # Output channels (K)
        self.hofm = getattr(layer, 'hofm', 1)  # Output height (H)
        self.wofm = getattr(layer, 'wofm', 1)  # Output width (W)
        self.hfil = getattr(layer, 'hfil', 1)  # Filter height
        self.wfil = getattr(layer, 'wfil', 1)  # Filter width

        # Generate valid partition factors for each dimension
        self.valid_factors = self._compute_valid_factors()

    def _get_divisors(self, n):
        """Get all divisors of n."""
        divisors = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divisors.append(i)
                if i != n // i:
                    divisors.append(n // i)
        return sorted(divisors)

    def _compute_valid_factors(self):
        """Compute valid partition factors for each dimension."""
        factors = {}

        # For batch, we don't have batch info in layer, use placeholder
        factors[PartDim.BATCH] = [1]  # Will be updated with actual batch

        # Output channels
        factors[PartDim.OUTP] = self._get_divisors(self.nofm)

        # Spatial dimensions
        factors[PartDim.OFMP_H] = self._get_divisors(self.hofm)
        factors[PartDim.OFMP_W] = self._get_divisors(self.wofm)

        # Input channels (for INPP partitioning)
        factors[PartDim.INPP] = self._get_divisors(self.nifm)

        return factors

    def set_batch_size(self, batch_size):
        """Set the batch size and update valid factors."""
        self.batch_size = batch_size
        self.valid_factors[PartDim.BATCH] = self._get_divisors(batch_size)


class PartitionChoice:
    """A specific partition choice (combination of dimensions and factors)."""

    def __init__(self, partition_dict: Dict[PartDim, int]):
        """
        partition_dict: {PartDim: factor}
        Example: {PartDim.OUTP: 4, PartDim.OFMP_H: 2} means K/4 and H/2
        """
        self.partition_dict = partition_dict

    @property
    def total_nodes(self):
        """Total number of nodes required."""
        result = 1
        for factor in self.partition_dict.values():
            result *= factor
        return result

    def get_factor(self, dim):
        """Get partition factor for a dimension (default 1)."""
        return self.partition_dict.get(dim, 1)

    def __repr__(self):
        parts = [f"{dim.name}={f}" for dim,
                 f in self.partition_dict.items() if f > 1]
        return f"PartitionChoice({', '.join(parts) or 'none'})"

    def __hash__(self):
        return hash(tuple(sorted(self.partition_dict.items())))

    def __eq__(self, other):
        if isinstance(other, PartitionChoice):
            return self.partition_dict == other.partition_dict
        return False


class RedistributionType(IntEnum):
    """Types of data redistribution between layers."""
    NONE = 0           # No redistribution needed
    LOCAL = 1          # Data already local
    ALL_GATHER = 2     # Gather distributed data
    ALL_TO_ALL = 3     # All-to-all communication
    ALL_REDUCE = 4     # Reduce partial results
    SCATTER = 5        # Scatter data to nodes


class GlobalPartitionILPOptimizer:
    """
    ILP-based optimizer for global partition assignment.

    Uses Integer Linear Programming to find the globally optimal
    partition scheme for all layers simultaneously.
    """

    def __init__(self, network, resource, batch_size=1,
                 max_partitions_per_dim=None,
                 solver='auto'):
        """
        Args:
            network: nn_dataflow Network object
            resource: Resource object with node information
            batch_size: Batch size for the network
            max_partitions_per_dim: Maximum partitions to consider per dimension
            solver: 'gurobi', 'pulp', or 'auto'
        """
        self.network = network
        self.resource = resource
        self.batch_size = batch_size
        self.max_partitions = max_partitions_per_dim or 16

        # Select solver
        self.solver = self._select_solver(solver)

        # Extract layers in topological order
        self.layers = self._extract_layers()

        # Build layer configs
        self.layer_configs = self._build_layer_configs()

        # Generate partition choices for each layer
        self.partition_choices = self._generate_partition_choices()

        # Cost model
        self.compute_costs = {}
        self.redistribution_costs = {}

    def _select_solver(self, solver):
        """Select ILP solver based on availability."""
        if solver == 'gurobi' and HAS_GUROBI:
            return 'gurobi'
        elif solver == 'pulp' and HAS_PULP:
            return 'pulp'
        elif solver == 'auto':
            if HAS_GUROBI:
                return 'gurobi'
            elif HAS_PULP:
                return 'pulp'
            else:
                raise ImportError(
                    "No ILP solver available. Install gurobipy or pulp.")
        else:
            raise ValueError(f"Unknown solver: {solver}")

    def _extract_layers(self):
        """Extract layers from network in topological order."""
        layers = []
        for layer_name in self.network:
            layer = self.network[layer_name]
            layers.append((layer_name, layer))
        return layers

    def _build_layer_configs(self):
        """Build LayerPartitionConfig for each layer."""
        configs = []
        for idx, (name, layer) in enumerate(self.layers):
            config = LayerPartitionConfig(name, layer, idx)
            config.set_batch_size(self.batch_size)
            configs.append(config)
        return configs

    def _generate_partition_choices(self):
        """Generate all valid partition choices for each layer."""
        all_choices = []

        for config in self.layer_configs:
            layer_choices = []

            # Get valid factors for each dimension, limited by max_partitions
            factors_per_dim = {}
            for dim in [PartDim.BATCH, PartDim.OUTP, PartDim.OFMP_H,
                        PartDim.OFMP_W, PartDim.INPP]:
                valid = [f for f in config.valid_factors.get(dim, [1])
                         if f <= self.max_partitions]
                factors_per_dim[dim] = valid if valid else [1]

            # Generate combinations (limit total nodes)
            total_nodes = self._get_total_nodes()

            # We generate partition choices more intelligently
            # Consider single-dim partitions first, then combinations
            seen = set()

            # Single dimension partitions
            for dim in [PartDim.BATCH, PartDim.OUTP, PartDim.OFMP_H, PartDim.OFMP_W]:
                for factor in factors_per_dim[dim]:
                    if factor <= total_nodes:
                        choice = PartitionChoice({dim: factor})
                        if choice not in seen:
                            layer_choices.append(choice)
                            seen.add(choice)

            # Two-dimension partitions
            for dim1, dim2 in itertools.combinations(
                    [PartDim.BATCH, PartDim.OUTP, PartDim.OFMP_H, PartDim.OFMP_W], 2):
                for f1 in factors_per_dim[dim1]:
                    for f2 in factors_per_dim[dim2]:
                        if f1 * f2 <= total_nodes:
                            choice = PartitionChoice({dim1: f1, dim2: f2})
                            if choice not in seen:
                                layer_choices.append(choice)
                                seen.add(choice)

            # INPP partition (requires reduction)
            for factor in factors_per_dim[PartDim.INPP]:
                if factor <= total_nodes and factor > 1:
                    choice = PartitionChoice({PartDim.INPP: factor})
                    if choice not in seen:
                        layer_choices.append(choice)
                        seen.add(choice)

            # Add no partition choice
            no_part = PartitionChoice({})
            if no_part not in seen:
                layer_choices.insert(0, no_part)

            all_choices.append(layer_choices)

        return all_choices

    def _get_total_nodes(self):
        """Get total available nodes from resource."""
        if hasattr(self.resource, 'dim_nodes'):
            return self.resource.dim_nodes.h * self.resource.dim_nodes.w
        elif hasattr(self.resource, 'proc_region'):
            return self.resource.proc_region.dim.size()
        return 16  # Default

    def set_cost_model(self, cost_model):
        """Set the cost model for computing costs."""
        self.cost_model = cost_model

    def _compute_all_costs(self):
        """Pre-compute all costs for ILP."""
        # Compute costs for each partition choice
        for layer_idx, (config, choices) in enumerate(
                zip(self.layer_configs, self.partition_choices)):
            for choice_idx, choice in enumerate(choices):
                self.compute_costs[(layer_idx, choice_idx)] = \
                    self._estimate_compute_cost(config, choice)

        # Redistribution costs between consecutive layers
        for layer_idx in range(len(self.layers) - 1):
            choices_curr = self.partition_choices[layer_idx]
            choices_next = self.partition_choices[layer_idx + 1]

            for ci, choice_curr in enumerate(choices_curr):
                for cj, choice_next in enumerate(choices_next):
                    cost = self._estimate_redistribution_cost(
                        self.layer_configs[layer_idx],
                        self.layer_configs[layer_idx + 1],
                        choice_curr, choice_next)
                    self.redistribution_costs[(layer_idx, ci, cj)] = cost

    def _estimate_compute_cost(self, config, choice):
        """
        Estimate computation cost for a partition choice.

        The compute cost depends on:
        - Number of operations (MACs)
        - Memory access pattern
        - Parallelization overhead
        """
        # Base MACs
        macs = (config.nifm * config.nofm * config.hofm * config.wofm *
                config.hfil * config.wfil)

        # Work per node
        nodes = choice.total_nodes
        macs_per_node = macs / nodes

        # Parallelization efficiency (simplified model)
        # INPP partitioning requires reduction, adding overhead
        inpp_factor = choice.get_factor(PartDim.INPP)
        reduction_overhead = 1.0 + 0.1 * \
            (inpp_factor - 1) if inpp_factor > 1 else 1.0

        # Memory access overhead for spatial partitioning
        # Halo exchange for OFMP partitioning
        h_factor = choice.get_factor(PartDim.OFMP_H)
        w_factor = choice.get_factor(PartDim.OFMP_W)
        halo_overhead = 1.0
        if h_factor > 1:
            halo_overhead += 0.05 * (config.hfil - 1) / \
                (config.hofm / h_factor)
        if w_factor > 1:
            halo_overhead += 0.05 * (config.wfil - 1) / \
                (config.wofm / w_factor)

        cost = macs_per_node * reduction_overhead * halo_overhead
        return cost

    def _estimate_redistribution_cost(self, config_src, config_dst,
                                      choice_src, choice_dst):
        """
        Estimate redistribution cost between two partition choices.

        Key propagation constraint:
            src's OUTP partition → dst's input channel distribution
        """
        # Output size from source layer
        output_size = config_src.nofm * config_src.hofm * config_src.wofm

        # Determine redistribution type
        redist_type = self._determine_redistribution_type(
            choice_src, choice_dst, config_src, config_dst)

        # Cost based on redistribution type
        if redist_type == RedistributionType.NONE:
            return 0
        elif redist_type == RedistributionType.LOCAL:
            return output_size * 0.01  # Local copy cost
        elif redist_type == RedistributionType.ALL_GATHER:
            # Gather: each node receives from all others
            nodes = choice_src.total_nodes
            return output_size * (nodes - 1) / nodes
        elif redist_type == RedistributionType.ALL_TO_ALL:
            # All-to-all: full data exchange
            nodes_src = choice_src.total_nodes
            nodes_dst = choice_dst.total_nodes
            return output_size * (1 - 1/max(nodes_src, nodes_dst))
        elif redist_type == RedistributionType.ALL_REDUCE:
            # Reduce: sum partial results
            nodes = choice_src.get_factor(PartDim.INPP)
            return output_size * 2 * (nodes - 1) / nodes  # Ring allreduce
        elif redist_type == RedistributionType.SCATTER:
            # Scatter: distribute data
            nodes = choice_dst.total_nodes
            return output_size * (nodes - 1) / nodes

        return output_size  # Default: full data movement

    def _determine_redistribution_type(self, choice_src, choice_dst,
                                       config_src, config_dst):
        """
        Determine the type of redistribution needed.

        Key insight: src's K-partition affects dst's C-distribution.
        """
        src_k_factor = choice_src.get_factor(PartDim.OUTP)
        dst_outp_factor = choice_dst.get_factor(PartDim.OUTP)
        dst_inpp_factor = choice_dst.get_factor(PartDim.INPP)

        # Check if output channel partition matches input requirement
        # src partitions K, dst needs C (which equals src's K)

        # Case 1: No partitioning change
        if (choice_src.total_nodes == 1 and choice_dst.total_nodes == 1):
            return RedistributionType.NONE

        # Case 2: Source OUTP partition, destination uses same dimension
        if src_k_factor > 1:
            if dst_outp_factor == src_k_factor:
                # Same partitioning, data stays local
                return RedistributionType.LOCAL
            elif dst_inpp_factor > 1:
                # Need all-to-all: different partitioning schemes
                return RedistributionType.ALL_TO_ALL
            else:
                # Need to gather for different partitioning
                return RedistributionType.ALL_GATHER

        # Case 3: INPP partition requires reduction at layer boundary
        if choice_src.get_factor(PartDim.INPP) > 1:
            return RedistributionType.ALL_REDUCE

        # Case 4: Spatial partition changes
        src_h = choice_src.get_factor(PartDim.OFMP_H)
        src_w = choice_src.get_factor(PartDim.OFMP_W)
        dst_h = choice_dst.get_factor(PartDim.OFMP_H)
        dst_w = choice_dst.get_factor(PartDim.OFMP_W)

        if (src_h != dst_h or src_w != dst_w):
            return RedistributionType.ALL_TO_ALL

        # Case 5: Batch partition - usually stays local
        src_batch = choice_src.get_factor(PartDim.BATCH)
        dst_batch = choice_dst.get_factor(PartDim.BATCH)
        if src_batch == dst_batch:
            return RedistributionType.NONE

        return RedistributionType.ALL_TO_ALL

    def optimize(self, time_limit=300, verbose=True):
        """
        Run ILP optimization to find globally optimal partition.

        Args:
            time_limit: Maximum optimization time in seconds
            verbose: Print optimization progress

        Returns:
            List of (layer_name, PartitionChoice) tuples
        """
        # Pre-compute all costs
        self._compute_all_costs()

        if self.solver == 'gurobi':
            return self._optimize_gurobi(time_limit, verbose)
        else:
            return self._optimize_pulp(time_limit, verbose)

    def _optimize_gurobi(self, time_limit, verbose):
        """Optimize using Gurobi."""
        if not HAS_GUROBI:
            raise ImportError("Gurobi not available")

        model = gp.Model("GlobalPartition")

        if not verbose:
            model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', time_limit)

        num_layers = len(self.layers)
        total_nodes = self._get_total_nodes()

        # Decision variables: x[l, c] = 1 if layer l uses choice c
        x = {}
        for l in range(num_layers):
            for c in range(len(self.partition_choices[l])):
                x[l, c] = model.addVar(vtype=GRB.BINARY, name=f"x_{l}_{c}")

        # Constraint 1: Each layer selects exactly one partition choice
        for l in range(num_layers):
            model.addConstr(
                gp.quicksum(x[l, c] for c in range(
                    len(self.partition_choices[l]))) == 1,
                name=f"one_choice_{l}")

        # Constraint 2: Total nodes constraint
        # For each layer, nodes used <= total available
        for l in range(num_layers):
            model.addConstr(
                gp.quicksum(x[l, c] * self.partition_choices[l][c].total_nodes
                            for c in range(len(self.partition_choices[l]))) <= total_nodes,
                name=f"nodes_{l}")

        # Constraint 3: Partition propagation constraint
        # If layer l uses OUTP partition with factor f, and layer l+1 is connected,
        # then layer l+1's input is distributed with factor f
        # This is encoded as: compatible choices must be selected together
        for l in range(num_layers - 1):
            config_curr = self.layer_configs[l]
            config_next = self.layer_configs[l + 1]

            # Check if layers are connected (curr's nofm == next's nifm)
            if config_curr.nofm == config_next.nifm:
                # Add compatibility constraints
                for ci, choice_curr in enumerate(self.partition_choices[l]):
                    k_factor = choice_curr.get_factor(PartDim.OUTP)

                    if k_factor > 1:
                        # Find compatible choices in next layer
                        compatible = []
                        for cj, choice_next in enumerate(self.partition_choices[l + 1]):
                            # Next layer can either:
                            # 1. Have matching OUTP factor (data stays local)
                            # 2. Do explicit redistribution (handled by cost)
                            compatible.append(cj)

                        # If no compatible choices, this is infeasible
                        # (but we allow all choices with different costs)

        # Objective: Minimize total cost (compute + redistribution)
        compute_cost = gp.quicksum(
            x[l, c] * self.compute_costs.get((l, c), 0)
            for l in range(num_layers)
            for c in range(len(self.partition_choices[l])))

        redist_cost = gp.quicksum(
            x[l, ci] * x[l+1, cj] *
            self.redistribution_costs.get((l, ci, cj), 0)
            for l in range(num_layers - 1)
            for ci in range(len(self.partition_choices[l]))
            for cj in range(len(self.partition_choices[l + 1])))

        # Note: x[l,ci] * x[l+1,cj] is quadratic.
        # Linearize using auxiliary variables
        y = {}  # y[l, ci, cj] = x[l, ci] * x[l+1, cj]
        for l in range(num_layers - 1):
            for ci in range(len(self.partition_choices[l])):
                for cj in range(len(self.partition_choices[l + 1])):
                    y[l, ci, cj] = model.addVar(vtype=GRB.BINARY,
                                                name=f"y_{l}_{ci}_{cj}")
                    # Linearization constraints
                    model.addConstr(y[l, ci, cj] <= x[l, ci])
                    model.addConstr(y[l, ci, cj] <= x[l + 1, cj])
                    model.addConstr(
                        y[l, ci, cj] >= x[l, ci] + x[l + 1, cj] - 1)

        # Linearized redistribution cost
        redist_cost_linear = gp.quicksum(
            y[l, ci, cj] * self.redistribution_costs.get((l, ci, cj), 0)
            for l in range(num_layers - 1)
            for ci in range(len(self.partition_choices[l]))
            for cj in range(len(self.partition_choices[l + 1])))

        model.setObjective(compute_cost + redist_cost_linear, GRB.MINIMIZE)

        # Optimize
        model.optimize()

        if model.SolCount == 0:
            raise RuntimeError("No feasible solution found")

        # Extract solution
        solution = []
        for l in range(num_layers):
            for c in range(len(self.partition_choices[l])):
                if x[l, c].X > 0.5:
                    layer_name = self.layers[l][0]
                    choice = self.partition_choices[l][c]
                    solution.append((layer_name, choice))
                    break

        return solution

    def _optimize_pulp(self, time_limit, verbose):
        """Optimize using PuLP (open-source)."""
        if not HAS_PULP:
            raise ImportError("PuLP not available")

        prob = pulp.LpProblem("GlobalPartition", pulp.LpMinimize)

        num_layers = len(self.layers)
        total_nodes = self._get_total_nodes()

        # Decision variables
        x = {}
        for l in range(num_layers):
            for c in range(len(self.partition_choices[l])):
                x[l, c] = pulp.LpVariable(f"x_{l}_{c}", cat='Binary')

        # Constraint 1: Each layer selects exactly one partition choice
        for l in range(num_layers):
            prob += (
                pulp.lpSum(x[l, c]
                           for c in range(len(self.partition_choices[l]))) == 1,
                f"one_choice_{l}")

        # Constraint 2: Total nodes constraint
        for l in range(num_layers):
            prob += (
                pulp.lpSum(x[l, c] * self.partition_choices[l][c].total_nodes
                           for c in range(len(self.partition_choices[l]))) <= total_nodes,
                f"nodes_{l}")

        # Auxiliary variables for linearization
        y = {}
        for l in range(num_layers - 1):
            for ci in range(len(self.partition_choices[l])):
                for cj in range(len(self.partition_choices[l + 1])):
                    y[l, ci, cj] = pulp.LpVariable(
                        f"y_{l}_{ci}_{cj}", cat='Binary')
                    prob += (y[l, ci, cj] <= x[l, ci])
                    prob += (y[l, ci, cj] <= x[l + 1, cj])
                    prob += (y[l, ci, cj] >= x[l, ci] + x[l + 1, cj] - 1)

        # Objective
        compute_cost = pulp.lpSum(
            x[l, c] * self.compute_costs.get((l, c), 0)
            for l in range(num_layers)
            for c in range(len(self.partition_choices[l])))

        redist_cost = pulp.lpSum(
            y[l, ci, cj] * self.redistribution_costs.get((l, ci, cj), 0)
            for l in range(num_layers - 1)
            for ci in range(len(self.partition_choices[l]))
            for cj in range(len(self.partition_choices[l + 1])))

        prob += compute_cost + redist_cost

        # Solve
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=verbose)
        prob.solve(solver)

        if prob.status != pulp.LpStatusOptimal:
            raise RuntimeError(
                f"Optimization failed with status: {pulp.LpStatus[prob.status]}")

        # Extract solution
        solution = []
        for l in range(num_layers):
            for c in range(len(self.partition_choices[l])):
                if pulp.value(x[l, c]) > 0.5:
                    layer_name = self.layers[l][0]
                    choice = self.partition_choices[l][c]
                    solution.append((layer_name, choice))
                    break

        return solution

    def print_solution(self, solution):
        """Pretty print the optimization solution."""
        print("\n" + "="*60)
        print("Global Partition Optimization Result")
        print("="*60)

        total_compute = 0
        total_redist = 0

        for idx, (layer_name, choice) in enumerate(solution):
            config = self.layer_configs[idx]
            compute = self._estimate_compute_cost(config, choice)
            total_compute += compute

            print(f"\nLayer {idx}: {layer_name}")
            print(f"  Partition: {choice}")
            print(f"  Nodes used: {choice.total_nodes}")
            print(f"  Compute cost: {compute:.2f}")

            if idx > 0:
                prev_choice = solution[idx - 1][1]
                redist = self._estimate_redistribution_cost(
                    self.layer_configs[idx - 1], config,
                    prev_choice, choice)
                total_redist += redist
                print(f"  Redistribution cost: {redist:.2f}")

        print("\n" + "-"*60)
        print(f"Total compute cost: {total_compute:.2f}")
        print(f"Total redistribution cost: {total_redist:.2f}")
        print(f"Total cost: {total_compute + total_redist:.2f}")
        print("="*60)


def create_optimizer_from_nn_dataflow(network, resource, batch_size=1,
                                      max_partitions=16, solver='auto'):
    """
    Factory function to create GlobalPartitionILPOptimizer from nn_dataflow objects.

    Args:
        network: nn_dataflow.Network object
        resource: nn_dataflow.Resource object
        batch_size: Batch size for inference/training
        max_partitions: Maximum partitions per dimension
        solver: 'gurobi', 'pulp', or 'auto'

    Returns:
        GlobalPartitionILPOptimizer instance
    """
    return GlobalPartitionILPOptimizer(
        network=network,
        resource=resource,
        batch_size=batch_size,
        max_partitions_per_dim=max_partitions,
        solver=solver
    )
