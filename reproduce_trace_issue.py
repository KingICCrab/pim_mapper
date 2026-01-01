import sys
import os
from pathlib import Path
import json

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'validation' / 'dram'))

from nn_dataflow.nns import import_network
from pim_optimizer.optimizer import PIMOptimizer
from pim_optimizer.arch import PIMArchitecture
from pim_optimizer.workload import ConvWorkload

# Import TraceGenerator
import trace_generator as tg
from golden_model.dram_trace.timing_simulator import DRAMTimingParams, DRAMTimingSimulator

def main():
    network = import_network("resnet50")
    layer_name = "conv2_0_b"
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
    
    mapping_obj = result.mappings[0]
    print("\nMapping Details:")
    print(f"  Loop Bounds: {mapping_obj.loop_bounds}")
    print(f"  Permutation: {mapping_obj.permutation}")
    print("ILP Row Activations:", mapping_obj.metrics['row_activations'])
    print("ILP Input Row Activations:", mapping_obj.metrics.get('row_activations_input', 'N/A'))
    print("ILP Weight Row Activations:", mapping_obj.metrics.get('row_activations_weight', 'N/A'))
    print("ILP Output Row Activations:", mapping_obj.metrics.get('row_activations_output', 'N/A'))
    
    # 2. Generate Trace
    print("Generating Trace...")
    
    # Initialize TraceGenerator
    class MockDRAM:
        row_buffer_bytes = 1024
        num_rows = 16384
        num_banks = 4
        element_size = 1
        row_size_elements = 1024
    
    generator = tg.TraceGenerator(MockDRAM())
    
    # Generate trace
    # Signature: generate_trace(self, mapping, workload, strict_ordering=False)
    trace = generator.generate_trace(mapping_obj, workload)
    
    print(f"Generated {len(trace)} trace lines.")
    
    # 3. Simulate Trace
    print("Simulating Trace...")
    
    params = DRAMTimingParams(
        row_buffer_size=1024,
        num_banks=4,
        tRP=15,
        tRCDRD=15,
        tRCDWR=15,
    )
    
    simulator = DRAMTimingSimulator(params)
    
    # Parse trace strings to MemoryAccess objects
    # Trace format: "LD 0x..." or "ST 0x..."
    from golden_model.dram_trace.timing_simulator import MemoryAccess, AccessType, DataType
    
    # Calculate address boundaries for parser (Compact Allocation)
    row_size_bytes = 1024
    
    # Note: layer.hifm/wifm are H_in/W_in
    input_size_bytes = workload.N * workload.C * layer.hifm * layer.wifm * 1
    input_size_aligned = ((input_size_bytes + row_size_bytes - 1) // row_size_bytes) * row_size_bytes
    
    weight_size_bytes = workload.K * workload.C * workload.R * workload.S * 1
    weight_size_aligned = ((weight_size_bytes + row_size_bytes - 1) // row_size_bytes) * row_size_bytes
    
    input_end = input_size_aligned
    weight_end = input_end + weight_size_aligned
    
    parsed_trace = []
    for line in trace:
        parts = line.split()
        op = parts[0]
        addr = int(parts[1], 16)
        
        access_type = AccessType.READ if op == "LD" else AccessType.WRITE
        
        # Determine datatype based on address range
        if addr < input_end:
            dtype = DataType.INPUT
        elif addr < weight_end:
            dtype = DataType.WEIGHT
        else:
            dtype = DataType.OUTPUT
            
        parsed_trace.append(MemoryAccess(
            address=addr,
            access_type=access_type,
            size_bytes=1,
            datatype=dtype,
            row_id=(addr >> 10) & 0x3FFF, # Assuming 10 bits col
            timestamp=0,
            tensor_coords=None
        ))
        
    stats = simulator.simulate_trace(parsed_trace)
    
    print("Simulation Stats:")
    print(f"  Total Cycles: {stats.total_cycles}")
    print(f"  Input Activations: {stats.input_activations}")
    print(f"  Weight Activations: {stats.weight_activations}")
    print(f"  Output Activations: {stats.output_activations}")
    print(f"  Total Activations: {stats.row_activations}")

if __name__ == "__main__":
    main()
