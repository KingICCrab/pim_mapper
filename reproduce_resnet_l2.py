import sys
import os
from pathlib import Path
import json

PROJECT_ROOT = Path(".").resolve()
SRC_DIR = PROJECT_ROOT / 'src'
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / 'golden_model'))

from nn_dataflow.nns import import_network
from nn_dataflow.core import ConvLayer

from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.arch import PIMArchitecture
from pim_optimizer.workload import ConvWorkload
from golden_model.unindp_bridge import verify_ilp_with_unindp

def main():
    network = import_network("resnet50")
    layer_name = "conv2_0_a"
    layer = network[layer_name]
    
    print(f"Validating {layer_name}...")
    
    # 1. Run ILP
    print("Running ILP...")
    arch = PIMArchitecture(
        vault_count=32,
        pu_count=8,
        timing_config={
            'tRP': 15,
            'tRCDRD': 15,
        }
    )
    
    # Convert nn_dataflow layer to ConvWorkload
    # Calculate P, Q
    P = (layer.hifm - layer.hfil) // layer.htrd + 1
    Q = (layer.wifm - layer.wfil) // layer.wtrd + 1
    
    workload = ConvWorkload(
        name=layer_name,
        N=1, # Batch size
        C=layer.nifm,
        K=layer.nofm,
        P=P,
        Q=Q,
        R=layer.hfil,
        S=layer.wfil,
        stride=(layer.htrd, layer.wtrd),
    )
    
    optimizer = PIMOptimizer(arch=arch, verbose=True)
    result = optimizer.optimize([workload])
    
    print("ILP Result:", result)
    
    # Extract mapping and predictions
    mapping_obj = result.mappings[0]
    
    def get_l0_tile_size(mapping, dim):
        size = 1
        if 0 in mapping.loop_bounds:
            for loop_type in mapping.loop_bounds[0]:
                if dim in mapping.loop_bounds[0][loop_type]:
                    size *= mapping.loop_bounds[0][loop_type][dim]
        return size

    # C is dim 4, K is dim 5
    tile_C = get_l0_tile_size(mapping_obj, 4)
    tile_K = get_l0_tile_size(mapping_obj, 5)
    
    # Map Conv tile sizes to GEMM tile sizes for UniNDP
    # UniNDP GEMM: K=Input Channels, L=Output Channels
    # ILP Conv: C=Input Channels, K=Output Channels
    
    ilp_mapping_dict = {
        'tile_K': tile_C,
        'tile_L': tile_K,
        'num_channels': arch.vault_count,
        'num_pus': arch.pu_count,
    }
    
    workload_dict = {
        'M': 1,
        'K': workload.C * workload.R * workload.S,
        'L': workload.K,
        'B': 1,
    }
    
    ilp_predictions_dict = {
        'total_cycles': mapping_obj.metrics['latency'],
        'row_activations': mapping_obj.metrics['row_activations'],
    }
    
    print("Calling verify_ilp_with_unindp...")
    validation_result = verify_ilp_with_unindp(ilp_mapping_dict, workload_dict, ilp_predictions_dict)
    
    print("Validation Result:")
    print(json.dumps(validation_result, indent=2))

if __name__ == "__main__":
    main()
