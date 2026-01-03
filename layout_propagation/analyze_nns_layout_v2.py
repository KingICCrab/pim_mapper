from layout_propagation.strategy_selector import LayoutStrategySelector
from layout_propagation.layout_propagator_v2 import LayoutPropagator
from layout_propagation.experiment_utils import TilingGenerator
import os
import sys
import re
import importlib.util
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- Parsing Logic (Adapted from analyze_nns_layout.py) ---


def parse_layer_call(line: str) -> Tuple[str, str, Dict]:
    """
    Parses a layer definition line.
    Returns (layer_name, layer_type, params)
    """
    # Match NN.add('name', LayerType(...))
    pattern = r"NN\.add\(['\"]([^'\"]+)['\"],\s*(\w+)\(([^)]*)\)"
    match = re.search(pattern, line)

    if not match:
        # Try simpler pattern
        pattern2 = r"NN\.add\(['\"]([^'\"]+)['\"]"
        match2 = re.search(pattern2, line)
        if match2:
            type_match = re.search(
                r"(ConvLayer|FCLayer|PoolingLayer|EltwiseLayer|LocalRegionLayer)\(([^)]*)\)", line)
            if type_match:
                return match2.group(1), type_match.group(1), parse_args(type_match.group(2))
        return None, None, None

    return match.group(1), match.group(2), parse_args(match.group(3))


def parse_args(args_str: str) -> Dict:
    params = {}
    if not args_str.strip():
        return params

    args = [a.strip() for a in args_str.split(',')]
    nums = []
    for arg in args:
        if '=' in arg:
            continue  # Skip kwargs for now
        try:
            nums.append(int(arg))
        except ValueError:
            pass

    # Heuristic mapping based on typical nn_dataflow usage
    # Conv: nifm, nofm, hofm, sfil
    if len(nums) >= 3:
        params['nifm'] = nums[0]
        params['nofm'] = nums[1]
        params['hofm'] = nums[2]

    return params


def load_network_layers(filepath: str) -> List[Tuple[str, str, Dict]]:
    layers = []
    with open(filepath, 'r') as f:
        for line in f:
            name, ltype, params = parse_layer_call(line)
            if name:
                layers.append((name, ltype, params))
    return layers

# --- Analysis Logic ---


def analyze_network(name: str, layers: List[Tuple[str, str, Dict]]):
    print(f"\nAnalyzing Network: {name} ({len(layers)} layers)")

    # 1. Setup
    tiler = TilingGenerator(array_size=16)
    selector = LayoutStrategySelector()
    propagator = LayoutPropagator(selector)

    # 2. Build Graph
    # We assume a linear chain for simplicity in this parser,
    # although real networks have branches.
    # The regex parser extracts layers in order.
    # For branching networks (ResNet/Inception), this linear assumption is weak,
    # but sufficient to test Layout Propagation on the *sequence* of layers.

    prev_node_name = None

    for i, (layer_name, layer_type, params) in enumerate(layers):
        # Determine Shape
        # Default to some reasonable shape if parsing failed
        C = params.get('nifm', 64)
        K = params.get('nofm', 64)
        H = params.get('hofm', 32)
        W = H  # Assume square

        shape = {'N': 1, 'C': C, 'H': H, 'W': W}

        # Determine Type and Sensitivity
        is_sensitive = False
        op_type = "Generic"

        if layer_type in ['ConvLayer', 'FCLayer']:
            op_type = "Conv"  # Treat FC as Conv
            is_sensitive = True
        elif layer_type in ['PoolingLayer']:
            op_type = "Pool"
            is_sensitive = True
        elif layer_type in ['EltwiseLayer']:
            op_type = "Eltwise"
            is_sensitive = False

        # Create Loop Nest & Layout Preference
        loop_nest = None
        preferred_layout = None

        if is_sensitive:
            # Sensitive ops prefer Blocked Layout and Blocked Loop
            # We use K (Output Channels) for Conv
            # TilingGenerator uses 'C' for input, so we map K to C_out/C_in logic?
            # TilingGenerator is simple. Let's just use 'C' to represent the "Channel" dim of interest.
            # For Conv, the output layout matters for the NEXT layer.
            # But here we are defining the PREFERENCE of THIS layer.
            # A Conv layer prefers to WRITE its output in a specific way?
            # Or READ its input?
            # The Propagator defines "Preferred Layout" as what the node wants for its DATA.
            # Usually this means "Input Layout".

            # Let's assume:
            # Conv/FC prefers Blocked Input.
            preferred_layout = tiler.create_layout(
                shape, ['N', 'C_out', 'H', 'W', 'C_in'])
            loop_nest = tiler.create_loop_nest(
                shape, ['N', 'C_out', 'H', 'W', 'C_in'])
        else:
            # Insensitive ops (ReLU) have no intrinsic preference.
            # But we need a loop nest for execution cost.
            # Let's give it a default NCHW loop nest.
            # (As seen in experiment, this might bias it against Blocked,
            # but the Propagator should handle it if we implemented the "adapt loop nest" logic?
            # Wait, in the previous step we decided to pass `node_loop` to `evaluate_execution_cost`.
            # If we pass NCHW loop, it will penalize Blocked layout.
            # Ideally, Eltwise ops should have `loop_nest=None` and let the Propagator
            # pick the best loop nest matching the layout.
            # BUT, my Propagator implementation currently uses `node.loop_nest` if present.
            # I should set it to None for Eltwise!)
            loop_nest = None

        propagator.add_node(layer_name, op_type, is_sensitive, loop_nest)

        if preferred_layout:
            propagator.set_preferred_layout(layer_name, preferred_layout)

        # Link to previous
        if prev_node_name:
            propagator.add_edge(prev_node_name, layer_name)

        prev_node_name = layer_name

    # 3. Run Propagation
    try:
        decisions = propagator.run()

        print(f"  Total Cost: {decisions['total_cost']:.4f}")
        print(f"  Transformations: {len(decisions['transformations'])}")
        for t in decisions['transformations']:
            print(
                f"    {t['src']} -> {t['dst']}: {t['strategy']} ({t['cost']:.2f})")

    except Exception as e:
        print(f"  Error analyzing {name}: {e}")

# --- Main ---


def main():
    nns_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'nn_dataflow', 'nns')

    # List of networks to analyze
    networks = [
        'alex_net.py',
        'vgg_net.py',
        'resnet50.py',
        'googlenet.py'
    ]

    print(f"Scanning {nns_dir}...")

    for net_file in networks:
        filepath = os.path.join(nns_dir, net_file)
        if os.path.exists(filepath):
            layers = load_network_layers(filepath)
            if layers:
                analyze_network(net_file, layers)
            else:
                print(f"Skipping {net_file}: No layers found (parsing issue?)")
        else:
            print(f"Skipping {net_file}: File not found")


if __name__ == "__main__":
    main()
