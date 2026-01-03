from nn_dataflow.core import ConvLayer, FCLayer, PoolingLayer, EltwiseLayer, LocalRegionLayer
import importlib.util
from layout_propagation.strategy_selector import LayoutStrategySelector
from layout_propagation.layout_propagator_v2 import LayoutPropagator
from layout_propagation.experiment_utils import TilingGenerator
import os
import sys
import re
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- Parsing Logic (Dynamic Import) ---


def load_network(filepath: str):
    module_name = os.path.basename(filepath).replace('.py', '')
    try:
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        if hasattr(module, 'NN'):
            return module.NN
    except Exception as e:
        print(f"Failed to load {module_name}: {e}")
    return None

# --- Analysis Logic ---


def run_analysis(network, mode: str) -> float:
    """
    Run analysis for a given mode.
    mode: 'Linear' (NCHW) or 'Optimized' (Blocked/Propagated)
    """
    tiler = TilingGenerator(array_size=16)
    selector = LayoutStrategySelector()
    propagator = LayoutPropagator(selector)

    # Add Input Layer
    input_layer = network.input_layer()
    # Input layer is typically fixed layout (e.g. NCHW) or flexible.
    # For simplicity, we treat it as a generic node with fixed NCHW layout if Linear, or flexible if Optimized.
    # Actually, input data usually comes in NCHW.

    # Add all nodes
    all_layers = list(network.layer_dict.items())
    # Add external layers if any (treat as inputs)
    for name, layer in network.ext_dict.items():
        all_layers.append((name, layer))

    for layer_name, layer in all_layers:
        ltype = layer.__class__.__name__

        # Extract params
        params = {}
        if hasattr(layer, 'nifm'):
            params['nifm'] = layer.nifm
        if hasattr(layer, 'nofm'):
            params['nofm'] = layer.nofm
        if hasattr(layer, 'hofm'):
            params['hofm'] = layer.hofm
        if hasattr(layer, 'wofm'):
            params['wofm'] = layer.wofm

        C = params.get('nifm', 64)
        # For InputLayer, nifm might not exist, use nofm
        if ltype == 'InputLayer':
            C = layer.nofm
            H = layer.hofm
        else:
            H = params.get('hofm', 32)

        W = H  # Simplified
        if hasattr(layer, 'wofm'):
            W = layer.wofm

        shape = {'N': 1, 'C': C, 'H': H, 'W': W}

        is_sensitive = False
        op_type = "Generic"
        if ltype in ['ConvLayer', 'FCLayer']:
            op_type = "Conv"
            is_sensitive = True
        elif ltype in ['PoolingLayer']:
            op_type = "Pool"
            is_sensitive = True
        elif ltype in ['EltwiseLayer']:
            op_type = "Eltwise"
            is_sensitive = False
        elif ltype == 'InputLayer':
            op_type = "Input"
            is_sensitive = False  # Input is source

        loop_nest = None
        preferred_layouts_to_set = []

        if mode == 'Linear':
            # Force NCHW
            l_nchw = tiler.create_layout(shape, ['N', 'C', 'H', 'W'])
            preferred_layouts_to_set.append(l_nchw)
            loop_nest = tiler.create_loop_nest(shape, ['N', 'C', 'H', 'W'])
            if not is_sensitive and ltype != 'InputLayer':
                is_sensitive = True

        elif mode == 'Optimized':
            if is_sensitive:
                # [Updated] Use Hardware-Aware Layout Generation

                # Candidate 1: Hardware-Aware (Spatial Channel)
                try:
                    l_hw_aware = tiler.generate_hardware_aware_layout(
                        shape, op_type)
                    preferred_layouts_to_set.append(l_hw_aware)

                    # Set loop nest to match the hardware layout
                    if loop_nest is None:
                        loop_order = [d.name for d in l_hw_aware.ordering]
                        loop_nest = tiler.create_loop_nest(shape, loop_order)
                except ValueError:
                    pass

                # Candidate 2: NHWC (Fallback)
                try:
                    l_nhwc = tiler.create_layout(shape, ['N', 'H', 'W', 'C'])
                    preferred_layouts_to_set.append(l_nhwc)
                except ValueError:
                    pass

                # Fallback
                if not preferred_layouts_to_set:
                    l_nchw = tiler.create_layout(shape, ['N', 'C', 'H', 'W'])
                    preferred_layouts_to_set.append(l_nchw)
                    loop_nest = tiler.create_loop_nest(
                        shape, ['N', 'C', 'H', 'W'])

            else:
                loop_nest = None

        # Add Node first
        propagator.add_node(layer_name, op_type, is_sensitive, loop_nest)

        # Then set preferences
        for layout in preferred_layouts_to_set:
            propagator.set_preferred_layout(layer_name, layout)

    # Add Edges
    for layer_name, prevs in network.prevs_dict.items():
        for p in prevs:
            if p in propagator.nodes and layer_name in propagator.nodes:
                propagator.add_edge(p, layer_name)

    try:
        decisions = propagator.run()
        return decisions['total_cost']
    except Exception as e:
        print(f"Error in {network.net_name}: {e}")
        return 0.0

# --- Plotting ---


def main():
    nns_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'nn_dataflow', 'nns')

    networks = [f for f in os.listdir(
        nns_dir) if f.endswith('.py') and f != '__init__.py']
    networks.sort()

    results = {'Network': [],
               'Linear (NCHW)': [], 'Optimized (Propagated)': []}

    print(f"Running Comparative Analysis on {len(networks)} networks...")

    for net_file in networks:
        filepath = os.path.join(nns_dir, net_file)
        if os.path.exists(filepath):
            network = load_network(filepath)
            if not network:
                continue

            cost_linear = run_analysis(network, 'Linear')
            cost_opt = run_analysis(network, 'Optimized')

            net_name = net_file.replace('.py', '').replace('_net', '')
            results['Network'].append(net_name)
            results['Linear (NCHW)'].append(cost_linear)
            results['Optimized (Propagated)'].append(cost_opt)

            print(f"{net_name}: Linear={cost_linear:.2f}, Optimized={cost_opt:.2f}")

    # Plotting
    x = np.arange(len(results['Network']))
    width = 0.35

    # Adjust figure size based on number of networks
    fig_width = max(10, len(networks) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # Use patterns (hatching) instead of colors for distinction
    # Linear: White with diagonal hatch
    rects1 = ax.bar(x - width/2, results['Linear (NCHW)'], width,
                    label='Linear (NCHW)', color='white', edgecolor='black', hatch='///')

    # Optimized: White with dot hatch (or cross hatch)
    rects2 = ax.bar(x + width/2, results['Optimized (Propagated)'], width,
                    label='Optimized (Propagated)', color='white', edgecolor='black', hatch='...')

    ax.set_ylabel('Total Layout Cost (Normalized)')
    ax.set_title('Layout Propagation Cost Comparison')
    ax.set_xticks(x)
    # Rotate labels for better readability
    ax.set_xticklabels(results['Network'], rotation=45, ha='right')
    ax.legend()

    # Add labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    output_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'experiment_results_chart.png')
    plt.savefig(output_path)
    print(f"\nChart saved to: {output_path}")


if __name__ == "__main__":
    main()
