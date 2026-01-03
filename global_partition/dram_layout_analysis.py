"""
DRAM Layout Analysis for Neural Network Data Layout Strategies

This module provides analytical models to estimate DRAM access costs
for different data layout strategies:
1. Propagation - Keep native layout, let dataflow adapt
2. Adaptation - Transform layout to match dataflow preference  
3. Transformation - Add explicit transform operators

Based on DDR4/HBM timing and bandwidth models.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum


class Layout(Enum):
    """Data layout formats"""
    NCHW = "NCHW"  # Batch, Channel, Height, Width
    NHWC = "NHWC"  # Batch, Height, Width, Channel


class Dataflow(Enum):
    """Common dataflow strategies"""
    OUTPUT_STATIONARY = "OS"   # Minimize output writes
    WEIGHT_STATIONARY = "WS"   # Minimize weight reads
    INPUT_STATIONARY = "IS"    # Minimize input reads
    ROW_STATIONARY = "RS"      # Eyeriss-style row stationary


@dataclass
class DRAMConfig:
    """DRAM configuration parameters"""
    # DDR4-2400 typical parameters
    bandwidth_gbps: float = 19.2  # GB/s per channel
    row_buffer_size: int = 8192   # bytes (8KB)
    burst_length: int = 8
    data_width: int = 64          # bits per beat
    tRCD: float = 13.75           # row access time (ns)
    tRP: float = 13.75            # precharge time (ns)
    tCAS: float = 13.75           # CAS latency (ns)
    
    # Derived parameters
    @property
    def row_hit_latency_ns(self) -> float:
        """Latency for row buffer hit"""
        return self.tCAS
    
    @property
    def row_miss_latency_ns(self) -> float:
        """Latency for row buffer miss (same bank)"""
        return self.tRP + self.tRCD + self.tCAS
    
    @property
    def bytes_per_beat(self) -> int:
        return self.data_width // 8


@dataclass
class ConvLayer:
    """Convolution layer parameters"""
    batch: int
    in_channels: int
    out_channels: int
    in_height: int
    in_width: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    
    @property
    def out_height(self) -> int:
        return (self.in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
    
    @property
    def out_width(self) -> int:
        return (self.in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
    
    @property
    def input_size(self) -> int:
        """Total input tensor size in elements"""
        return self.batch * self.in_channels * self.in_height * self.in_width
    
    @property
    def output_size(self) -> int:
        """Total output tensor size in elements"""
        return self.batch * self.out_channels * self.out_height * self.out_width
    
    @property
    def weight_size(self) -> int:
        """Total weight tensor size in elements"""
        return self.out_channels * self.in_channels * self.kernel_size * self.kernel_size


class DRAMAccessAnalyzer:
    """Analyze DRAM access patterns for different layouts"""
    
    def __init__(self, config: DRAMConfig = None, element_size: int = 4):
        self.config = config or DRAMConfig()
        self.element_size = element_size  # 4 bytes for FP32
    
    def analyze_sequential_access(self, total_bytes: int) -> Dict:
        """Analyze cost of sequential DRAM access"""
        num_row_accesses = math.ceil(total_bytes / self.config.row_buffer_size)
        
        # First access is a row miss, subsequent are row hits within same row
        first_row_latency = self.config.row_miss_latency_ns
        bytes_per_row = self.config.row_buffer_size
        
        # Time to transfer all data
        transfer_time = total_bytes / (self.config.bandwidth_gbps * 1e9) * 1e9  # in ns
        
        # Total latency including row activations
        total_latency = first_row_latency + (num_row_accesses - 1) * self.config.tRCD + transfer_time
        
        return {
            'total_bytes': total_bytes,
            'num_row_accesses': num_row_accesses,
            'transfer_time_ns': transfer_time,
            'total_latency_ns': total_latency,
            'effective_bandwidth_gbps': total_bytes / total_latency * 1e9 / 1e9
        }
    
    def analyze_strided_access(self, total_elements: int, stride: int, 
                                access_length: int) -> Dict:
        """
        Analyze cost of strided DRAM access
        
        Args:
            total_elements: Total number of elements to access
            stride: Stride between consecutive accesses (in elements)
            access_length: Number of consecutive elements per access
        """
        stride_bytes = stride * self.element_size
        access_bytes = access_length * self.element_size
        total_bytes = total_elements * self.element_size
        
        num_accesses = total_elements // access_length
        
        # Check if stride fits within row buffer
        row_buffer_elements = self.config.row_buffer_size // self.element_size
        
        if stride <= row_buffer_elements:
            # Stride fits in row buffer - some row hits possible
            accesses_per_row = self.config.row_buffer_size // stride_bytes
            num_row_activations = math.ceil(num_accesses / accesses_per_row)
            row_hit_rate = 1 - (num_row_activations / num_accesses)
        else:
            # Each access causes row miss
            num_row_activations = num_accesses
            row_hit_rate = 0.0
        
        # Calculate latency
        row_miss_cost = num_row_activations * self.config.row_miss_latency_ns
        row_hit_cost = (num_accesses - num_row_activations) * self.config.row_hit_latency_ns
        transfer_time = total_bytes / (self.config.bandwidth_gbps * 1e9) * 1e9
        
        total_latency = row_miss_cost + row_hit_cost + transfer_time
        
        return {
            'total_bytes': total_bytes,
            'num_accesses': num_accesses,
            'num_row_activations': num_row_activations,
            'row_hit_rate': row_hit_rate,
            'transfer_time_ns': transfer_time,
            'total_latency_ns': total_latency,
            'effective_bandwidth_gbps': total_bytes / total_latency * 1e9 / 1e9
        }
    
    def estimate_layout_transform_cost(self, tensor_shape: Tuple[int, ...], 
                                        from_layout: Layout, to_layout: Layout) -> Dict:
        """
        Estimate DRAM cost of transforming tensor layout
        
        Transformation requires reading entire tensor and writing in new order.
        Read is efficient (sequential), write may have poor locality.
        """
        total_elements = math.prod(tensor_shape)
        total_bytes = total_elements * self.element_size
        
        # Read cost - sequential
        read_cost = self.analyze_sequential_access(total_bytes)
        
        # Write cost - depends on transformation
        if from_layout == Layout.NCHW and to_layout == Layout.NHWC:
            # NCHW -> NHWC: writes are strided by C
            # Reading NCHW: sequential
            # Writing NHWC: for each (n,c,h,w) -> (n,h,w,c)
            # Write stride in elements = C (channels)
            N, C, H, W = tensor_shape
            write_stride = C
            write_access_length = 1  # Single element per access
        elif from_layout == Layout.NHWC and to_layout == Layout.NCHW:
            # NHWC -> NCHW: similar analysis
            N, H, W, C = tensor_shape
            write_stride = H * W
            write_access_length = 1
        else:
            # Same layout - no transform needed
            write_stride = 1
            write_access_length = total_elements
        
        write_cost = self.analyze_strided_access(
            total_elements, write_stride, write_access_length
        )
        
        return {
            'read_cost': read_cost,
            'write_cost': write_cost,
            'total_latency_ns': read_cost['total_latency_ns'] + write_cost['total_latency_ns'],
            'transform_overhead_factor': (read_cost['total_latency_ns'] + write_cost['total_latency_ns']) / read_cost['total_latency_ns']
        }


class LayoutStrategyAnalyzer:
    """
    Compare different layout handling strategies for consecutive conv layers
    """
    
    def __init__(self, dram_config: DRAMConfig = None):
        self.dram_analyzer = DRAMAccessAnalyzer(dram_config)
    
    def analyze_propagation_strategy(self, layers: List[ConvLayer], 
                                      native_layout: Layout,
                                      preferred_layouts: List[Layout]) -> Dict:
        """
        Propagation strategy: Keep native layout, dataflow adapts
        
        Cost = sum of inefficient accesses due to layout mismatch
        """
        total_cost = 0
        layer_costs = []
        
        for i, (layer, preferred) in enumerate(zip(layers, preferred_layouts)):
            if native_layout == preferred:
                # Layout matches - sequential access
                input_cost = self.dram_analyzer.analyze_sequential_access(
                    layer.input_size * 4
                )
                mismatch_factor = 1.0
            else:
                # Layout mismatch - strided access pattern
                if native_layout == Layout.NCHW and preferred == Layout.NHWC:
                    # Dataflow wants NHWC but data is NCHW
                    # Access pattern has stride of H*W for channel iteration
                    stride = layer.in_height * layer.in_width
                else:
                    # Dataflow wants NCHW but data is NHWC
                    # Access pattern has stride of C for spatial iteration
                    stride = layer.in_channels
                
                input_cost = self.dram_analyzer.analyze_strided_access(
                    layer.input_size, stride, 1
                )
                mismatch_factor = input_cost['total_latency_ns'] / \
                    self.dram_analyzer.analyze_sequential_access(layer.input_size * 4)['total_latency_ns']
            
            layer_costs.append({
                'layer': i,
                'input_cost': input_cost,
                'mismatch_factor': mismatch_factor,
                'layout_match': native_layout == preferred
            })
            total_cost += input_cost['total_latency_ns']
        
        return {
            'strategy': 'propagation',
            'layer_costs': layer_costs,
            'total_latency_ns': total_cost,
            'num_transforms': 0
        }
    
    def analyze_adaptation_strategy(self, layers: List[ConvLayer],
                                     initial_layout: Layout,
                                     preferred_layouts: List[Layout]) -> Dict:
        """
        Adaptation strategy: Transform layout at boundaries where it changes
        
        Cost = transform costs + sequential access costs
        """
        total_cost = 0
        layer_costs = []
        num_transforms = 0
        current_layout = initial_layout
        
        for i, (layer, preferred) in enumerate(zip(layers, preferred_layouts)):
            transform_cost = None
            
            if current_layout != preferred:
                # Need to transform
                if current_layout == Layout.NCHW:
                    shape = (layer.batch, layer.in_channels, layer.in_height, layer.in_width)
                else:
                    shape = (layer.batch, layer.in_height, layer.in_width, layer.in_channels)
                
                transform_cost = self.dram_analyzer.estimate_layout_transform_cost(
                    shape, current_layout, preferred
                )
                total_cost += transform_cost['total_latency_ns']
                num_transforms += 1
                current_layout = preferred
            
            # Access with matching layout - sequential
            input_cost = self.dram_analyzer.analyze_sequential_access(
                layer.input_size * 4
            )
            total_cost += input_cost['total_latency_ns']
            
            layer_costs.append({
                'layer': i,
                'transform_cost': transform_cost,
                'input_cost': input_cost,
                'layout_after': current_layout
            })
        
        return {
            'strategy': 'adaptation',
            'layer_costs': layer_costs,
            'total_latency_ns': total_cost,
            'num_transforms': num_transforms
        }
    
    def analyze_transformation_strategy(self, layers: List[ConvLayer],
                                         transform_positions: List[int],
                                         layouts: List[Layout]) -> Dict:
        """
        Transformation strategy: Insert explicit transform ops at specified positions
        
        This allows more control over where transformations happen.
        """
        # Similar to adaptation but with explicit transform positions
        return self.analyze_adaptation_strategy(layers, layouts[0], layouts)


def example_analysis():
    """Run example analysis on VGG-like consecutive conv layers"""
    
    # Define consecutive conv layers (VGG-16 style)
    layers = [
        ConvLayer(batch=1, in_channels=64, out_channels=128, 
                  in_height=112, in_width=112, kernel_size=3, padding=1),
        ConvLayer(batch=1, in_channels=128, out_channels=128, 
                  in_height=112, in_width=112, kernel_size=3, padding=1),
        ConvLayer(batch=1, in_channels=128, out_channels=256, 
                  in_height=56, in_width=56, kernel_size=3, padding=1),
        ConvLayer(batch=1, in_channels=256, out_channels=256, 
                  in_height=56, in_width=56, kernel_size=3, padding=1),
    ]
    
    # Dataflow preferences (assume alternating preferences)
    preferred_layouts = [Layout.NCHW, Layout.NHWC, Layout.NCHW, Layout.NHWC]
    
    analyzer = LayoutStrategyAnalyzer()
    
    print("=" * 70)
    print("DRAM Layout Strategy Analysis")
    print("=" * 70)
    
    # Analyze propagation strategy (keep NCHW)
    prop_result = analyzer.analyze_propagation_strategy(
        layers, Layout.NCHW, preferred_layouts
    )
    print(f"\n1. Propagation Strategy (keep NCHW):")
    print(f"   Total latency: {prop_result['total_latency_ns']:.2f} ns")
    print(f"   Transforms: {prop_result['num_transforms']}")
    for lc in prop_result['layer_costs']:
        status = "match" if lc['layout_match'] else f"mismatch (factor: {lc['mismatch_factor']:.2f}x)"
        print(f"   Layer {lc['layer']}: {status}")
    
    # Analyze adaptation strategy
    adapt_result = analyzer.analyze_adaptation_strategy(
        layers, Layout.NCHW, preferred_layouts
    )
    print(f"\n2. Adaptation Strategy (transform at boundaries):")
    print(f"   Total latency: {adapt_result['total_latency_ns']:.2f} ns")
    print(f"   Transforms: {adapt_result['num_transforms']}")
    for lc in adapt_result['layer_costs']:
        if lc['transform_cost']:
            print(f"   Layer {lc['layer']}: transform cost = {lc['transform_cost']['total_latency_ns']:.2f} ns")
    
    # Compare
    print(f"\n3. Comparison:")
    speedup = prop_result['total_latency_ns'] / adapt_result['total_latency_ns']
    if speedup > 1:
        print(f"   Adaptation is {speedup:.2f}x faster than Propagation")
    else:
        print(f"   Propagation is {1/speedup:.2f}x faster than Adaptation")
    
    return prop_result, adapt_result


if __name__ == "__main__":
    example_analysis()
