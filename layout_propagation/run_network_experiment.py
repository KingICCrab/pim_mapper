from layout_propagation.strategy_selector import LayoutStrategySelector
from layout_propagation.layout_propagator_v2 import LayoutPropagator
from layout_propagation.experiment_utils import TilingGenerator
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_network_experiment():
    print("=== Starting Network Layout Propagation Experiment ===")
    print("Scenario: Conv1 (NCHW) -> ReLU (Insensitive) -> Conv2 (Blocked)")

    # 1. Setup
    tiler = TilingGenerator(array_size=16)
    selector = LayoutStrategySelector()
    propagator = LayoutPropagator(selector)

    shape = {'N': 1, 'C': 64, 'H': 32, 'W': 32}

    # 2. Define Layouts
    layout_nchw = tiler.create_layout(shape, ['N', 'C', 'H', 'W'])
    layout_blocked = tiler.create_layout(
        shape, ['N', 'C_out', 'H', 'W', 'C_in'])

    # 3. Define Loop Nests
    # Conv1: Output Stationary (N, C, H, W)
    loop_conv1 = tiler.create_loop_nest(shape, ['N', 'C', 'H', 'W'])

    # Conv2: Input Stationary (N, C_out, H, W, C_in)
    loop_conv2 = tiler.create_loop_nest(
        shape, ['N', 'C_out', 'H', 'W', 'C_in'])

    # ReLU: Elementwise. Usually matches the input loop order.
    # But since it's flexible, we can give it a generic loop nest or let it adapt.
    # For this experiment, let's assume ReLU executes in the same order as its input layout.
    # However, the Propagator needs a fixed loop nest for cost evaluation *if* we treat it as a fixed node.
    # But ReLU is insensitive.
    # Let's give ReLU the NCHW loop nest for now. If it adopts Blocked layout,
    # the cost model will evaluate (Blocked Layout + NCHW Loop) which might be bad.
    # Ideally, ReLU's loop nest should match its selected layout.
    # For this simplified experiment, we'll assign it NCHW loop.
    loop_relu = tiler.create_loop_nest(shape, ['N', 'C', 'H', 'W'])

    # 4. Build Graph
    # Add Nodes
    propagator.add_node("Conv1", "Conv", is_sensitive=True,
                        loop_nest=loop_conv1)
    propagator.add_node("ReLU", "ReLU", is_sensitive=False,
                        loop_nest=loop_relu)
    propagator.add_node("Conv2", "Conv", is_sensitive=True,
                        loop_nest=loop_conv2)

    # Add Edges
    propagator.add_edge("Conv1", "ReLU")
    propagator.add_edge("ReLU", "Conv2")

    # Set Preferences
    propagator.set_preferred_layout("Conv1", layout_nchw)
    propagator.set_preferred_layout("Conv2", layout_blocked)

    # 5. Run Propagation (Scenario 1: ReLU uses NCHW Loop)
    print("\n--- Running Propagation (Scenario 1: ReLU uses NCHW Loop) ---")
    decisions = propagator.run()

    # Analyze Results 1
    relu_node = propagator.nodes["ReLU"]
    print(f"ReLU Selected Layout: {relu_node.selected_layout.ordering}")
    if relu_node.selected_layout == layout_blocked:
        print("ReLU adopted BLOCKED layout.")
    else:
        print("ReLU adopted NCHW layout.")
    print(f"Total System Cost: {decisions['total_cost']:.4f}")

    # 6. Run Propagation (Scenario 2: ReLU uses Blocked Loop)
    print("\n--- Running Propagation (Scenario 2: ReLU uses Blocked Loop) ---")
    # Reset nodes
    propagator.nodes = {}
    # Re-add nodes with new loop nest for ReLU
    # We assume ReLU can adapt its execution order to match the blocked layout
    loop_relu_blocked = tiler.create_loop_nest(
        shape, ['N', 'C_out', 'H', 'W', 'C_in'])

    propagator.add_node("Conv1", "Conv", is_sensitive=True,
                        loop_nest=loop_conv1)
    propagator.add_node("ReLU", "ReLU", is_sensitive=False,
                        loop_nest=loop_relu_blocked)
    propagator.add_node("Conv2", "Conv", is_sensitive=True,
                        loop_nest=loop_conv2)

    propagator.add_edge("Conv1", "ReLU")
    propagator.add_edge("ReLU", "Conv2")

    propagator.set_preferred_layout("Conv1", layout_nchw)
    propagator.set_preferred_layout("Conv2", layout_blocked)

    decisions_2 = propagator.run()

    # Analyze Results 2
    relu_node_2 = propagator.nodes["ReLU"]
    print(f"ReLU Selected Layout: {relu_node_2.selected_layout.ordering}")
    if relu_node_2.selected_layout == layout_blocked:
        print("ReLU adopted BLOCKED layout.")
    else:
        print("ReLU adopted NCHW layout.")
    print(f"Total System Cost: {decisions_2['total_cost']:.4f}")


if __name__ == "__main__":
    run_network_experiment()
