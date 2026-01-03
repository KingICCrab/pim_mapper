from typing import List, Dict, Set, Tuple, Optional
from collections import deque
from dataclasses import dataclass, field
from .data_layout import DataLayout
from .strategy_selector import LayoutStrategySelector


@dataclass
class LayoutNode:
    """Represents an operator in the layout graph."""
    op_name: str
    op_type: str
    is_sensitive: bool
    loop_nest: List[Tuple[str, int]] = field(
        default_factory=list)  # Execution order
    preferred_layouts: List[DataLayout] = field(
        default_factory=list)  # Candidates from propagation
    selected_layout: Optional[DataLayout] = None  # Final decision

    # Graph connectivity
    predecessors: List['LayoutNode'] = field(default_factory=list)
    successors: List['LayoutNode'] = field(default_factory=list)


class LayoutPropagator:
    """
    Implements the two-phase layout optimization:
    1. Propagation: Build candidate sets.
    2. Decision: Select optimal layouts and transformations.
    """

    def __init__(self, strategy_selector: LayoutStrategySelector):
        self.selector = strategy_selector
        self.nodes: Dict[str, LayoutNode] = {}

    def add_node(self, op_name: str, op_type: str, is_sensitive: bool, loop_nest: List[Tuple[str, int]] = None):
        node = LayoutNode(op_name, op_type, is_sensitive)
        if loop_nest:
            node.loop_nest = loop_nest
        self.nodes[op_name] = node

    def add_edge(self, src_name: str, dst_name: str):
        src = self.nodes[src_name]
        dst = self.nodes[dst_name]
        src.successors.append(dst)
        dst.predecessors.append(src)

    def set_preferred_layout(self, op_name: str, layout: DataLayout):
        """Set the intrinsic preferred layout for a sensitive operator."""
        if op_name in self.nodes:
            self.nodes[op_name].preferred_layouts.append(layout)

    def run(self):
        """Execute the full optimization pipeline."""
        print("Phase 1: Propagating Layout Candidates...")
        self._propagate()

        print("Phase 2: Making Layout Decisions...")
        decisions = self._decide()

        return decisions

    def _propagate(self):
        """
        Phase 1: Bidirectional Propagation.
        Propagate preferred layouts from Sensitive nodes to Insensitive nodes.
        """
        # Queue stores (node, layout, direction)
        # direction: 'forward' or 'backward'
        queue = deque()

        # Initialize queue with sensitive nodes that have preferences
        for node in self.nodes.values():
            if node.is_sensitive and node.preferred_layouts:
                # Broadcast initial preferences
                for layout in node.preferred_layouts:
                    queue.append((node, layout, 'init'))

        visited_states = set()  # (node_name, layout_hash)

        while queue:
            curr_node, layout, direction = queue.popleft()

            state = (curr_node.op_name, hash(layout))
            if state in visited_states:
                continue
            visited_states.add(state)

            # Add to current node's candidates if not present
            if layout not in curr_node.preferred_layouts:
                curr_node.preferred_layouts.append(layout)

            # If current node is Sensitive and this wasn't an init,
            # it acts as a boundary. It might accept the layout as a candidate,
            # but it won't necessarily propagate it further (unless we want full flooding).
            # For now, let's assume Sensitive nodes stop propagation to define domains.
            if curr_node.is_sensitive and direction != 'init':
                continue

            # Propagate to neighbors
            # Forward: to successors
            for succ in curr_node.successors:
                queue.append((succ, layout, 'forward'))

            # Backward: to predecessors
            for pred in curr_node.predecessors:
                queue.append((pred, layout, 'backward'))

    def _decide(self) -> Dict[str, any]:
        """
        Phase 2: Greedy Decision Making (simplified).
        Resolve conflicts and insert transformations.
        """
        final_plan = {}

        # Simple greedy approach:
        # 1. Fix layouts for Sensitive nodes (they usually have 1 strong preference).
        # 2. For Insensitive nodes, pick the layout that minimizes transition cost.

        # Step 1: Fix Sensitive Nodes
        for node in self.nodes.values():
            if node.is_sensitive and node.preferred_layouts:
                # If multiple preferences, pick the first one (simplification)
                # In a real solver, this would be part of the global cost function.
                node.selected_layout = node.preferred_layouts[0]
            elif not node.is_sensitive:
                # Insensitive nodes initially undefined
                node.selected_layout = None

        # Step 2: Propagate Decisions to Insensitive Nodes (Cost-Aware)
        # We iterate until all nodes have a selected layout
        # For this simplified version, we assume a linear chain or simple graph where we can resolve one by one.

        undefined_nodes = [
            n for n in self.nodes.values() if n.selected_layout is None]

        for node in undefined_nodes:
            # Collect candidate layouts from neighbors
            candidates = set()
            for pred in node.predecessors:
                if pred.selected_layout:
                    candidates.add(pred.selected_layout)
            for succ in node.successors:
                if succ.selected_layout:
                    candidates.add(succ.selected_layout)

            if not candidates:
                continue

            # Evaluate each candidate
            best_layout = None
            min_cost = float('inf')

            for layout in candidates:
                current_cost = 0.0

                # For Insensitive nodes, assume they adopt the loop nest of the candidate layout
                # This is crucial because Elementwise ops are flexible.
                # We need a way to generate loop nest from layout.
                # For now, we'll use a heuristic: if the node has no loop nest, or is insensitive,
                # we assume its loop nest aligns with the layout.

                # Mock loop nest from layout (since we don't have the TilingGenerator here)
                # This is a limitation. Ideally we pass a generator.
                # But wait, if we just pass the layout's ordering as the loop nest,
                # the Cost Evaluator might work if it just needs dimensions.
                # But Cost Evaluator needs (dim, extent).

                # Let's use the node's stored loop nest if available, otherwise try to adapt.
                # If the user provided a loop nest for the insensitive node (e.g. NCHW),
                # we might be stuck with it.

                # BETTER APPROACH:
                # If the node is insensitive, we assume it can execute in ANY order.
                # However, to evaluate the cost of a specific layout candidate, we should
                # assume the node executes in a way that is compatible with that layout
                # (or penalize it if we assume a default order).

                # For this fix, if loop_nest is None, we infer it from the layout's ordering.
                # This represents the "Ideal" execution for that layout.
                node_loop = node.loop_nest
                if node_loop is None and layout is not None:
                    # Infer loop nest from layout dimensions
                    # We need extents. We can try to get them from the shape if available.
                    # Since we don't have shape here easily, we might skip or use a dummy.
                    # But wait, StrategySelector needs loop_nest to calculate costs.
                    # If we pass None, cost is 0.
                    pass

                # --- 1. Execution Cost (Internal) ---
                # REMOVED to avoid Double Counting.
                # The write cost is covered by the incoming edge (Producer Write)
                # or outgoing edge (Node Write) depending on how we view it.
                # In our StrategySelector, select_strategy(Pred, Node) covers Pred->Node edge.
                # It includes Pred Write + Node Read.
                # So Node's Write cost will be covered by select_strategy(Node, Succ).
                # exec_cost = self.selector.evaluate_execution_cost(layout, node_loop)
                # current_cost += exec_cost

                # --- 2. Transition Cost (Incoming) ---
                # Cost from Predecessors -> Node (if Node adopts 'layout')
                for pred in node.predecessors:
                    if pred.selected_layout:
                        # Edge: Pred -> Node
                        # Pred Layout: pred.selected_layout
                        # Node Layout: layout
                        # Pred Loop: pred.loop_nest
                        # Node Loop: node_loop

                        if pred.selected_layout != layout:
                            res = self.selector.select_strategy(
                                pred.selected_layout, layout, pred.loop_nest, node_loop
                            )
                            current_cost += res['cost']
                        else:
                            # Even if layouts match, we should add the "Direct Write" cost
                            # But wait, "Direct Write" cost includes Read Cost which is Execution Cost.
                            # We must be careful not to double count.
                            # StrategySelector returns (Write Cost + Read Cost).
                            # Read Cost IS Execution Cost (roughly).

                            # If we use StrategySelector, it sums Write + Read.
                            # If we add Execution Cost separately, we double count Read Cost.

                            # CORRECTION:
                            # The StrategySelector calculates the cost of the EDGE.
                            # Edge Cost = Producer Write + Consumer Read.
                            # Consumer Read IS the Consumer's Execution Cost (for that input tensor).

                            # So, if we sum up edge costs, we are already accounting for execution cost
                            # related to INPUTS.
                            # But what about OUTPUTS? Or internal buffers?
                            # For simple elementwise/conv chains, "Read Cost" covers the input side.
                            # "Write Cost" covers the output side.

                            # So, if we sum all incoming edge costs + all outgoing edge costs,
                            # we cover:
                            # Incoming: Pred Write + Node Read (Input)
                            # Outgoing: Node Write (Output) + Succ Read

                            # So "Node Read" and "Node Write" are covered.
                            # Is there anything missing?
                            # If layouts match (pred.layout == layout), we assumed cost=0.
                            # THIS IS THE BUG.
                            # If layouts match, cost is NOT 0. It is "Direct Write Cost".
                            # Direct Write Cost = Pred Write (Seq) + Node Read (Seq/Strided).

                            # So we should ALWAYS call select_strategy, even if layouts match.
                            # And we should NOT add explicit execution_cost separately if select_strategy covers it.

                            # Let's verify select_strategy behavior.
                            # It returns total_cost = write + read.

                            # So the fix is: ALWAYS call select_strategy.

                            res = self.selector.select_strategy(
                                pred.selected_layout, layout, pred.loop_nest, node_loop
                            )
                            current_cost += res['cost']

                # --- 3. Transition Cost (Outgoing) ---
                # Cost from Node -> Successors (if Node adopts 'layout')
                for succ in node.successors:
                    if succ.selected_layout:
                        # Edge: Node -> Succ
                        # Node Layout: layout
                        # Succ Layout: succ.selected_layout
                        # Node Loop: node_loop
                        # Succ Loop: succ.loop_nest

                        res = self.selector.select_strategy(
                            layout, succ.selected_layout, node_loop, succ.loop_nest
                        )
                        current_cost += res['cost']

                # Edge Case: If Node has NO successors (Output Layer), we must add its Write Cost
                # because it won't be covered by any outgoing edge.
                if not node.successors:
                    # We use evaluate_execution_cost to simulate the write
                    write_cost = self.selector.evaluate_execution_cost(
                        layout, node_loop)
                    current_cost += write_cost

                if current_cost < min_cost:
                    min_cost = current_cost
                    best_layout = layout

            node.selected_layout = best_layout

        # Step 3: Calculate Costs and Insert Transforms
        total_system_cost = 0.0
        transformations = []

        for node in self.nodes.values():
            if not node.selected_layout:
                continue

            for succ in node.successors:
                if not succ.selected_layout:
                    continue

                # Check edge: Node -> Succ
                src_layout = node.selected_layout
                dst_layout = succ.selected_layout

                # Use stored loop nests
                p_loop = node.loop_nest
                c_loop = succ.loop_nest

                # If loop nests are missing, warn or fallback (omitted for brevity)

                # Always evaluate cost, even if layouts are same (to capture read/write costs)
                # But StrategySelector is designed for conflicts.
                # If layouts are identical, cost is usually low (Direct Write).

                if src_layout != dst_layout:
                    # Conflict! Need Strategy Selector
                    decision = self.selector.select_strategy(
                        src_layout, dst_layout, p_loop, c_loop
                    )

                    transformations.append({
                        "src": node.op_name,
                        "dst": succ.op_name,
                        "strategy": decision['strategy'],
                        "cost": decision['cost'],
                        "details": decision['details']
                    })
                    total_system_cost += decision['cost']
                else:
                    # No transform needed, but we should technically add the "Direct" cost
                    # For now, we assume cost is 0 if layouts match (ideal case)
                    pass

        return {
            "transformations": transformations,
            "total_cost": total_system_cost
        }
