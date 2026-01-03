from layout_propagation.strategy_selector import LayoutStrategySelector
from layout_propagation.layout_propagator_v2 import LayoutPropagator
from layout_propagation.experiment_utils import TilingGenerator, ScenarioBuilder
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_experiment():
    print("=== Starting Layout Propagation Experiment (No Cache) ===")

    # 1. Setup
    # Array Size 16 means the hardware prefers operating on 16 elements at a time.
    tiler = TilingGenerator(array_size=16)
    builder = ScenarioBuilder()

    # Logical Shape: N=1, C=64, H=32, W=32
    # Total size: 64*32*32 = 65536 elements.
    shape = {'N': 1, 'C': 64, 'H': 32, 'W': 32}

    # 2. Create Scenario: Conv1 (Producer) -> Conv2 (Consumer)
    # Producer: Prefers Linear NCHW (Standard Output)
    # Consumer: Prefers Blocked N(C/16)HW(16) (Systolic Input)

    node_configs = [
        {'name': 'Conv1', 'op_type': 'Conv', 'is_sensitive': True},
        {'name': 'Conv2', 'op_type': 'Conv', 'is_sensitive': True}
    ]
    nodes = builder.build_chain(node_configs)
    conv1, conv2 = nodes[0], nodes[1]

    # 3. Define Layouts
    # Linear NCHW: N, C, H, W (W is innermost)
    layout_nchw = tiler.create_layout(shape, ['N', 'C', 'H', 'W'])

    # Blocked NCHWc: N, C_out, H, W, C_in (C_in is innermost)
    # This represents a layout optimized for channel-parallel processing (e.g. Systolic Array)
    layout_blocked = tiler.create_layout(
        shape, ['N', 'C_out', 'H', 'W', 'C_in'])

    # Assign Preferences
    conv1.preferred_layouts = [layout_nchw]
    conv2.preferred_layouts = [layout_blocked]

    print(f"Conv1 Preference: {layout_nchw}")
    print(f"Conv2 Preference: {layout_blocked}")

    # 4. Define Loop Nests (Execution Order)
    # Conv1 (Producer) writes in NCHW order (Output Stationary)
    # Loop: N, C, H, W (W is innermost)
    loop_nest_conv1 = tiler.create_loop_nest(shape, ['N', 'C', 'H', 'W'])

    # Conv2 (Consumer) reads in Blocked order (Input Stationary / Tiled)
    # Loop: N, C_out, H, W, C_in (C_in is innermost)
    loop_nest_conv2 = tiler.create_loop_nest(
        shape, ['N', 'C_out', 'H', 'W', 'C_in'])

    # 5. Run Strategy Selection
    selector = LayoutStrategySelector()

    print("\n--- Evaluating Strategies ---")

    # The selector compares:
    # 1. Direct Write: Use Producer's Layout (NCHW).
    #    - Producer Write Cost (Seq) + Consumer Read Cost (Strided)
    # 2. Transform Write: Use Consumer's Layout (Blocked).
    #    - Producer Write Cost (Strided) + Consumer Read Cost (Seq)

    result = selector.select_strategy(
        producer_layout=layout_nchw,
        consumer_preferred_layout=layout_blocked,
        producer_loop_nest=loop_nest_conv1,
        consumer_loop_nest=loop_nest_conv2
    )

    print(f"\nStrategy Selection Result: {result['strategy'].upper()}")
    print(f"Selected Layout: {result['selected_layout'].ordering}")
    print(f"Total Cost: {result['cost']:.4f}")
    print("\nDetailed Breakdown:")
    print(f"  Direct Strategy (Layout=NCHW):")
    print(f"    Write Cost (Seq): {result['details']['direct']['write']:.4f}")
    print(
        f"    Read Cost (Strided): {result['details']['direct']['read']:.4f}")
    print(f"    Total: {result['details']['direct']['total']:.4f}")

    print(f"  Transform Strategy (Layout=Blocked):")
    print(
        f"    Write Cost (Strided): {result['details']['transform']['write']:.4f}")
    print(f"    Read Cost (Seq): {result['details']['transform']['read']:.4f}")
    print(f"    Total: {result['details']['transform']['total']:.4f}")

    # Analysis
    if result['strategy'] == 'transform_on_write':
        print("\n[Conclusion] The system chose to Transform on Write.")
        print(
            "This means the penalty of the Consumer reading from a bad layout (Row Misses)")
        print("was higher than the penalty of the Producer writing to a bad layout (Burst Inefficiency).")
    else:
        print("\n[Conclusion] The system chose Direct Write.")
        print("This means the transformation overhead was deemed too high.")


if __name__ == "__main__":
    run_experiment()
