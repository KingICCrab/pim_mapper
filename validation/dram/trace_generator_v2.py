"""
Simplified Trace Generator for Ramulator2 validation.

Based on verified address calculation logic from test_address_calc.py.

Key insights:
1. Only Level 3 loops generate DRAM accesses (inner levels use cached data)
2. Address strides accumulate from inner to outer loop
3. For row_aligned layout, Level 3 C stride = row_size (not cumulative)
4. Input address uses h = q + r, w = p + s (not direct P, Q values)
5. Each tensor has independent row buffer state
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
from tqdm import tqdm


@dataclass
class DRAMConfig:
    """DRAM configuration for address mapping."""
    num_channels: int = 1
    num_ranks: int = 1
    num_banks: int = 4
    num_rows: int = 16384
    row_buffer_bytes: int = 1024  # Row buffer size in bytes
    element_size: int = 1  # Bytes per element
    
    @property
    def row_size_elements(self) -> int:
        """Row size in elements."""
        return self.row_buffer_bytes // self.element_size


# Dimension indices: R, S, P, Q, C, K, N
DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6
DIM_NAMES = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']


class TraceGeneratorV2:
    """
    Simplified trace generator with verified address calculation.
    """
    
    def __init__(self, dram_config: DRAMConfig = None):
        self.dram = dram_config or DRAMConfig()
    
    def build_loop_nesting(self, mapping) -> List[Tuple[int, str, int, int]]:
        """
        Build complete loop nesting from outer to inner.
        
        Returns list of (level, key, dim, bound) tuples.
        """
        loops = []
        
        # Process levels from outer (3) to inner (0)
        for level in [3, 2, 1]:
            if level not in mapping.permutation:
                continue
            
            perm = mapping.permutation[level]
            # Sort by permutation position (outer to inner = high to low)
            sorted_perm = sorted(perm.items(), key=lambda x: x[0], reverse=True)
            
            for pos, dim in sorted_perm:
                if level not in mapping.loop_bounds:
                    continue
                bound = mapping.loop_bounds[level].get('temporal', {}).get(dim, 1)
                if bound > 1:
                    loops.append((level, 'temporal', dim, bound))
        
        # Level 0: temporal loops
        if 0 in mapping.permutation:
            perm = mapping.permutation[0]
            sorted_perm = sorted(perm.items(), key=lambda x: x[0], reverse=True)
            
            for pos, dim in sorted_perm:
                if 0 not in mapping.loop_bounds:
                    continue
                bound = mapping.loop_bounds[0].get('temporal', {}).get(dim, 1)
                if bound > 1:
                    loops.append((0, 'temporal', dim, bound))
        
        # Level 0: spatial H loops
        if 0 in mapping.loop_bounds and 'H' in mapping.loop_bounds[0]:
            for dim, bound in mapping.loop_bounds[0]['H'].items():
                if bound > 1:
                    loops.append((0, 'H', dim, bound))
        
        # Level 0: spatial W loops
        if 0 in mapping.loop_bounds and 'W' in mapping.loop_bounds[0]:
            for dim, bound in mapping.loop_bounds[0]['W'].items():
                if bound > 1:
                    loops.append((0, 'W', dim, bound))
        
        return loops
    
    def compute_strides(self, loops: List, mapping, workload) -> Dict:
        """
        Compute address strides for all tensors.
        
        Returns dict with strides for input, weight, output.
        """
        block_h = mapping.tile_info.get('block_h', 1)
        block_w = mapping.tile_info.get('block_w', 1)
        row_size = self.dram.row_size_elements
        
        input_layout = mapping.layout.get(0, "sequential")
        row_aligned = (input_layout == "row_aligned")
        
        # Find C bounds at different levels
        c_h_bound = 1
        for level, key, dim, bound in loops:
            if dim == DIM_C and level == 0 and key == 'H':
                c_h_bound = bound
        
        # Input strides (tile-wise layout)
        input_strides = {
            'h': 1,  # h coordinate stride
            'w': block_h,  # w coordinate stride
            'c_h': block_h * block_w,  # C in Level 0 H
            'c3': row_size if row_aligned else block_h * block_w * c_h_bound,  # C in Level 3
        }
        
        # Weight strides (cumulative from inner to outer)
        weight_dims = {DIM_C, DIM_K, DIM_R, DIM_S}
        weight_loops = [(l, k, d, b) for l, k, d, b in loops if d in weight_dims]
        weight_loops_inner_to_outer = list(reversed(weight_loops))
        weight_strides = {}
        stride = 1
        for level, key, dim, bound in weight_loops_inner_to_outer:
            weight_strides[(level, key, dim)] = stride
            stride *= bound
        
        # Output strides (cumulative from inner to outer)
        output_dims = {DIM_P, DIM_Q, DIM_K, DIM_N}
        output_loops = [(l, k, d, b) for l, k, d, b in loops if d in output_dims]
        output_loops_inner_to_outer = list(reversed(output_loops))
        output_strides = {}
        stride = 1
        for level, key, dim, bound in output_loops_inner_to_outer:
            output_strides[(level, key, dim)] = stride
            stride *= bound
        
        return {
            'input': input_strides,
            'weight': weight_strides,
            'output': output_strides,
            'block_h': block_h,
            'block_w': block_w,
            'row_aligned': row_aligned,
        }
    
    def generate_trace(self, mapping, workload) -> List[str]:
        """
        Generate DRAM access trace from mapping and workload.
        
        Only Level 3 loops generate DRAM accesses.
        But address calculation considers all level's loop variables.
        """
        loops = self.build_loop_nesting(mapping)
        strides = self.compute_strides(loops, mapping, workload)
        
        # Only Level 3 loops generate DRAM accesses
        level3_loops = [(l, k, d, b) for l, k, d, b in loops if l == 3]
        
        # Calculate total Level 3 iterations
        total_iters = 1
        for l, k, d, b in level3_loops:
            total_iters *= b
        
        trace = []
        loop_vars = {(l, k, d): 0 for l, k, d, b in loops}
        
        # Base addresses for each tensor (separate address spaces)
        base_addr = {
            'input': 0,
            'weight': 0x10000000,
            'output': 0x20000000,
        }
        
        pbar = tqdm(total=total_iters, desc="Generating trace", leave=False)
        
        def compute_input_addr():
            # h = q + r, w = p + s (assuming stride=1, dilation=1)
            r = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == DIM_R)
            s = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == DIM_S)
            p = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == DIM_P)
            q = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == DIM_Q)
            c_h = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == DIM_C and l == 0 and k == 'H')
            # For Input C, consider all DRAM-level C (Level 1, 2, 3)
            c_dram = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == DIM_C and l >= 1)
            
            h = q + r
            w = p + s
            
            addr = (h * strides['input']['h'] + 
                    w * strides['input']['w'] + 
                    c_h * strides['input']['c_h'] + 
                    c_dram * strides['input']['c3'])
            return addr
        
        def compute_weight_addr():
            addr = 0
            for (l, k, d), stride in strides['weight'].items():
                addr += loop_vars.get((l, k, d), 0) * stride
            return addr
        
        def compute_output_addr():
            addr = 0
            for (l, k, d), stride in strides['output'].items():
                addr += loop_vars.get((l, k, d), 0) * stride
            return addr
        
        def iterate_level3(idx):
            if idx >= len(level3_loops):
                # Generate DRAM accesses
                input_addr = base_addr['input'] + compute_input_addr() * self.dram.element_size
                weight_addr = base_addr['weight'] + compute_weight_addr() * self.dram.element_size
                output_addr = base_addr['output'] + compute_output_addr() * self.dram.element_size
                
                # Read input and weight, write output
                trace.append(f"0x{input_addr:016x} R")
                trace.append(f"0x{weight_addr:016x} R")
                trace.append(f"0x{output_addr:016x} W")
                
                pbar.update(1)
                return
            
            level, key, dim, bound = level3_loops[idx]
            for i in range(bound):
                loop_vars[(level, key, dim)] = i
                iterate_level3(idx + 1)
        
        iterate_level3(0)
        pbar.close()
        
        return trace
    
    def count_row_activations(self, mapping, workload) -> Dict[str, int]:
        """
        Count row activations without generating full trace.
        
        Returns dict with row_acts for input, weight, output.
        """
        loops = self.build_loop_nesting(mapping)
        strides = self.compute_strides(loops, mapping, workload)
        
        # Only Level 3 loops generate DRAM accesses
        level3_loops = [(l, k, d, b) for l, k, d, b in loops if l == 3]
        
        row_size = self.dram.row_size_elements
        num_banks = self.dram.num_banks
        
        loop_vars = {(l, k, d): 0 for l, k, d, b in loops}
        
        # Row buffer state per tensor per bank
        row_buffer_state = {
            'input': {},
            'weight': {},
            'output': {}
        }
        
        row_acts = {'input': 0, 'weight': 0, 'output': 0}
        
        def compute_input_addr():
            r = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == DIM_R)
            s = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == DIM_S)
            p = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == DIM_P)
            q = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == DIM_Q)
            c_h = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == DIM_C and l == 0 and k == 'H')
            # For Input C, consider all DRAM-level C (Level 1, 2, 3)
            c_dram = sum(loop_vars.get((l, k, d), 0) for l, k, d, b in loops if d == DIM_C and l >= 1)
            h = q + r
            w = p + s
            return (h * strides['input']['h'] + w * strides['input']['w'] + 
                    c_h * strides['input']['c_h'] + c_dram * strides['input']['c3'])
        
        def compute_weight_addr():
            addr = 0
            for (l, k, d), stride in strides['weight'].items():
                addr += loop_vars.get((l, k, d), 0) * stride
            return addr
        
        def compute_output_addr():
            addr = 0
            for (l, k, d), stride in strides['output'].items():
                addr += loop_vars.get((l, k, d), 0) * stride
            return addr
        
        def check_row_activation(tensor_name, addr):
            row = addr // row_size
            bank = row % num_banks
            physical_row = row // num_banks
            
            state = row_buffer_state[tensor_name]
            if bank not in state or state[bank] != physical_row:
                state[bank] = physical_row
                return True
            return False
        
        def iterate_level3(idx):
            if idx >= len(level3_loops):
                input_addr = compute_input_addr()
                weight_addr = compute_weight_addr()
                output_addr = compute_output_addr()
                
                if check_row_activation('input', input_addr):
                    row_acts['input'] += 1
                if check_row_activation('weight', weight_addr):
                    row_acts['weight'] += 1
                if check_row_activation('output', output_addr):
                    row_acts['output'] += 1
                return
            
            level, key, dim, bound = level3_loops[idx]
            for i in range(bound):
                loop_vars[(level, key, dim)] = i
                iterate_level3(idx + 1)
        
        iterate_level3(0)
        
        return row_acts


def generate_trace_for_mapping(optimizer, workload, output_path: str) -> int:
    """
    Helper function to generate trace from optimizer result.
    """
    mapping = optimizer.result.mappings[0]
    
    gen = TraceGeneratorV2()
    trace = gen.generate_trace(mapping, workload)
    
    with open(output_path, 'w') as f:
        for line in trace:
            f.write(line + '\n')
    
    return len(trace)


def count_row_activations_from_trace(trace_path: str, config: DRAMConfig = None) -> Dict:
    """
    Count row activations from trace file.
    """
    config = config or DRAMConfig()
    
    row_size = config.row_buffer_bytes
    num_banks = config.num_banks
    
    # Row buffer state per bank
    row_buffer_state = {}
    
    row_acts = 0
    total_accesses = 0
    
    with open(trace_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            addr_str = parts[0]
            addr = int(addr_str, 16)
            
            total_accesses += 1
            
            row = addr // row_size
            bank = row % num_banks
            physical_row = row // num_banks
            
            if bank not in row_buffer_state or row_buffer_state[bank] != physical_row:
                row_buffer_state[bank] = physical_row
                row_acts += 1
    
    return {
        'total_accesses': total_accesses,
        'total_row_acts': row_acts,
    }
