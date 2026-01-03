from typing import List, Dict, Tuple, Optional
from .data_layout import DataLayout
from .layout_cost import LayoutCostEvaluator


class LayoutStrategySelector:
    """
    Selects the optimal layout strategy for a Producer-Consumer pair.

    Strategies:
    1. Direct Write (No Transform): Producer writes in its natural order. Consumer adapts.
    2. Transform-on-Write: Producer transforms data to Consumer's preferred layout during write.

    (Strategy 3 "Propagation" is handled by the global optimizer, not locally here)
    """

    def __init__(self, sram_buffer_size: int = 1024 * 1024):  # e.g., 1MB Global Buffer
        self.sram_buffer_size = sram_buffer_size

    def select_strategy(self,
                        producer_layout: DataLayout,
                        consumer_preferred_layout: DataLayout,
                        producer_loop_nest: List[Tuple[str, int]],
                        consumer_loop_nest: List[Tuple[str, int]]) -> Dict[str, any]:
        """
        Compare strategies and return the best one.

        Args:
            producer_layout: The layout resulting from Producer's natural write order.
            consumer_preferred_layout: The layout Consumer wants to read.
            producer_loop_nest: The loop nest generating the output (for write cost).
            consumer_loop_nest: The loop nest consuming the input (for read cost).

        Returns:
            Dict with:
            - 'strategy': 'direct_write' or 'transform_on_write'
            - 'selected_layout': The DataLayout object to be used in DRAM.
            - 'cost': Total cost (Write + Read).
            - 'details': Breakdown of costs.
        """

        # --- Strategy 1: Direct Write (No Transform) ---
        # DRAM Layout = Producer's Natural Layout
        # Write Cost: Low (Sequential)
        # Read Cost: High (Consumer reads from Producer's layout)

        evaluator_direct = LayoutCostEvaluator(producer_layout)

        # Write Cost (Producer writing to its own layout) -> Should be near 1.0
        if producer_loop_nest:
            write_cost_direct = evaluator_direct.evaluate_loop_nest(producer_loop_nest)[
                'total_cost']
        else:
            write_cost_direct = 1.0  # Assume perfect adaptation

        # Read Cost (Consumer reading from Producer's layout)
        if consumer_loop_nest:
            read_cost_direct = evaluator_direct.evaluate_loop_nest(consumer_loop_nest)[
                'total_cost']
        else:
            # If consumer has no loop nest (Insensitive), it adapts to Producer's layout.
            # So it reads sequentially.
            read_cost_direct = 1.0

        total_cost_direct = write_cost_direct + read_cost_direct

        # --- Strategy 2: Transform-on-Write ---
        # DRAM Layout = Consumer's Preferred Layout
        # Write Cost: High (Producer writes to Consumer's layout) - BUT mitigated by SRAM
        # Read Cost: Low (Sequential)

        evaluator_transform = LayoutCostEvaluator(consumer_preferred_layout)

        # Raw Write Cost (without SRAM buffering)
        if producer_loop_nest:
            raw_write_metrics = evaluator_transform.evaluate_loop_nest(
                producer_loop_nest)
            raw_write_cost = raw_write_metrics['total_cost']
        else:
            raw_write_cost = 1.0

        # Apply SRAM Buffering Model
        # ... (omitted for brevity, same as before)
        buffered_write_cost = raw_write_cost

        # Read Cost (Consumer reading from its own preferred layout) -> Should be near 1.0
        if consumer_loop_nest:
            read_cost_transform = evaluator_transform.evaluate_loop_nest(consumer_loop_nest)[
                'total_cost']
        else:
            read_cost_transform = 1.0

        total_cost_transform = buffered_write_cost + read_cost_transform

        # --- Decision ---

        details = {
            "direct": {
                "write": write_cost_direct,
                "read": read_cost_direct,
                "total": total_cost_direct,
                "layout": producer_layout
            },
            "transform": {
                "write": buffered_write_cost,
                "read": read_cost_transform,
                "total": total_cost_transform,
                "layout": consumer_preferred_layout
            }
        }

        if total_cost_transform < total_cost_direct:
            return {
                "strategy": "transform_on_write",
                "selected_layout": consumer_preferred_layout,
                "cost": total_cost_transform,
                "details": details
            }
        else:
            return {
                "strategy": "direct_write",
                "selected_layout": producer_layout,
                "cost": total_cost_direct,
                "details": details
            }

    def evaluate_execution_cost(self, layout: DataLayout, loop_nest: List[Tuple[str, int]]) -> float:
        """
        Evaluate the cost of executing a loop nest on a specific layout.
        This represents the "Internal Execution Cost" of an operator.
        """
        if not layout or not loop_nest:
            return 0.0

        evaluator = LayoutCostEvaluator(layout)
        metrics = evaluator.evaluate_loop_nest(loop_nest)
        return metrics['total_cost']
