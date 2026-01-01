#!/usr/bin/env python3
"""
配置类和常量定义
"""

from dataclasses import dataclass

# ==================== 维度常量 ====================
DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6
DIM_NAMES = ['R', 'S', 'P', 'Q', 'C', 'K', 'N']

# 默认 permutation (从内到外)
DEFAULT_PERMUTATION = (DIM_R, DIM_S, DIM_Q, DIM_P, DIM_C, DIM_K, DIM_N)


@dataclass
class ArchConfig:
    """架构配置参数"""
    # DRAM 配置
    row_buffer_bytes: int = 1024      # RowBuffer 大小 (bytes)
    num_banks: int = 4                # Bank 数量
    num_rows: int = 16384             # 每个 bank 的 row 数量
    element_size: int = 1             # 每个元素的字节数
    
    # Buffer 配置
    global_buffer_bytes: int = 256 * 4  # GlobalBuffer 大小 (256KB)
    
    # PE Array 配置
    pe_array_h: int = 16
    pe_array_w: int = 16
    
    def __post_init__(self):
        self.row_buffer_elements = self.row_buffer_bytes // self.element_size
        self.global_buffer_elements = self.global_buffer_bytes // self.element_size


@dataclass 
class WorkloadConfig:
    """Workload 配置参数
    
    卷积的 7 个维度:
    - R, S: Filter 空间维度
    - P, Q: Output 空间维度  
    - C: Input channel
    - K: Output channel
    - N: Batch size
    """
    # 必需参数
    P: int  # Output height
    Q: int  # Output width
    K: int  # Output channels
    C: int  # Input channels
    H: int  # Input height
    W: int  # Input width
    
    # 可选参数
    N: int = 1   # Batch size
    R: int = 3   # Filter height
    S: int = 3   # Filter width
    stride_h: int = 1
    stride_w: int = 1
    dilation_h: int = 1
    dilation_w: int = 1
    
    @classmethod
    def from_conv_workload(cls, workload) -> 'WorkloadConfig':
        """从 ConvWorkload 对象创建"""
        stride = getattr(workload, 'stride', (1, 1))
        if isinstance(stride, int):
            stride = (stride, stride)
        dilation = getattr(workload, 'dilation', (1, 1))
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
            
        return cls(
            P=workload.P, Q=workload.Q,
            K=workload.K, C=workload.C,
            H=getattr(workload, 'H', workload.P + workload.R - 1),
            W=getattr(workload, 'W', workload.Q + workload.S - 1),
            N=getattr(workload, 'N', 1),
            R=getattr(workload, 'R', 3),
            S=getattr(workload, 'S', 3),
            stride_h=stride[0], stride_w=stride[1],
            dilation_h=dilation[0], dilation_w=dilation[1]
        )
