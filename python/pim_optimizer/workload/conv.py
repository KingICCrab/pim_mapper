"""
Convolution workload definition for PIM optimizer.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


def get_divisors(n: int) -> list[int]:
    """Get all divisors of n in sorted order."""
    divisors = []
    large_divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i * i != n:
                large_divisors.append(n // i)
        i += 1
    divisors.extend(reversed(large_divisors))
    return divisors


@dataclass
class ConvWorkload:
    """
    Convolution workload definition.
    
    7 dimensions for standard CNN conv2d:
    - R: Kernel width
    - S: Kernel height
    - P: Output width
    - Q: Output height
    - C: Input channels
    - K: Output channels (filters)
    - N: Batch size
    
    Attributes:
        name: Optional name for the workload
        R, S, P, Q, C, K, N: Problem dimensions
        stride: (width_stride, height_stride)
        dilation: (width_dilation, height_dilation)
        path: Optional path for identification
        weight: Weight factor for multi-workload optimization
    """
    
    name: str = "conv_workload"
    R: int = 3
    S: int = 3
    P: int = 56
    Q: int = 56
    C: int = 64
    K: int = 64
    N: int = 1
    stride: tuple[int, int] = (1, 1)
    dilation: tuple[int, int] = (1, 1)
    path: str = "conv_workload"
    weight: float = 1.0
    
    def __post_init__(self):
        """Initialize derived attributes."""
        self._init_dimensions()
        self._init_relevancy_matrix()
        self._compute_input_size()
    
    def _init_dimensions(self):
        """Initialize dimension-related attributes."""
        self.num_dims = 7
        self.dim_idxs = list(range(self.num_dims))
        self.dim_idx_name_dict = {0: 'R', 1: 'S', 2: 'P', 3: 'Q', 4: 'C', 5: 'K', 6: 'N'}
        self.dim_name_idx_dict = {v: k for k, v in self.dim_idx_name_dict.items()}
        
        # Bounds for each dimension
        self.bounds = [self.R, self.S, self.P, self.Q, self.C, self.K, self.N]
        
        # Compute divisors for each dimension
        self.divisors = [get_divisors(dim) for dim in self.bounds]
        
        # Total MACs
        self.macs = int(np.prod(self.bounds))
    
    def _init_relevancy_matrix(self):
        """
        Initialize dimension-datatype relevancy matrix O.
        
        O[j][t] = 1 if dimension j is relevant to datatype t
        - t=0: Inputs (relevant: R, S, P, Q, C, N)
        - t=1: Weights (relevant: R, S, C, K)
        - t=2: Outputs (relevant: P, Q, K, N)
        """
        self.O = [
            # t:  Inputs, Weights, Outputs   j:
            [1,   1,       0],              # 0: R
            [1,   1,       0],              # 1: S
            [1,   0,       1],              # 2: P
            [1,   0,       1],              # 3: Q
            [1,   1,       0],              # 4: C
            [0,   1,       1],              # 5: K
            [1,   0,       1],              # 6: N
        ]
        
        # Dimension reduction type matrix
        # [parallel, reduction]
        self.dim_reduction_type = [
            [1, 0],  # R: parallel
            [1, 0],  # S: parallel
            [1, 0],  # P: parallel
            [1, 0],  # Q: parallel
            [0, 1],  # C: reduction
            [0, 1],  # K: reduction
            [1, 0],  # N: parallel
        ]
    
    def _compute_input_size(self):
        """Compute input tensor dimensions (H, W)."""
        # W_in = Wstride * (P - 1) + Wdilation * (R - 1) + 1
        # H_in = Hstride * (Q - 1) + Hdilation * (S - 1) + 1
        W_in = self.stride[0] * max(self.P - 1, 0) + self.dilation[0] * max(self.R - 1, 0) + 1
        H_in = self.stride[1] * max(self.Q - 1, 0) + self.dilation[1] * max(self.S - 1, 0) + 1
        
        self.input_size = {'W': W_in, 'H': H_in}
        self.hw_divisors = {
            'W': get_divisors(W_in),
            'H': get_divisors(H_in),
        }
    
    def is_reduction_dim(self, dim_idx: int) -> bool:
        """Check if dimension is a reduction dimension."""
        if 0 <= dim_idx < len(self.dim_reduction_type):
            return self.dim_reduction_type[dim_idx][1] == 1
        return False
    
    def is_parallel_dim(self, dim_idx: int) -> bool:
        """Check if dimension is a parallel dimension."""
        if 0 <= dim_idx < len(self.dim_reduction_type):
            return self.dim_reduction_type[dim_idx][0] == 1
        return False
    
    def get_parallel_dims(self) -> list[int]:
        """Get indices of all parallel dimensions."""
        return [i for i in range(self.num_dims) if self.is_parallel_dim(i)]
    
    def get_reduction_dims(self) -> list[int]:
        """Get indices of all reduction dimensions."""
        return [i for i in range(self.num_dims) if self.is_reduction_dim(i)]
    
    @classmethod
    def from_dict(cls, config: dict, path: str = "workload") -> "ConvWorkload":
        """
        Create ConvWorkload from a dictionary.
        
        Args:
            config: Dictionary with problem dimensions
            path: Identifier for this workload
            
        Returns:
            ConvWorkload instance
        """
        prob = config.get('problem', config)
        if isinstance(prob.get('shape'), str):
            # Reference to shape definition, use instance
            prob = config.get('problem', {}).get('instance', prob)
        
        return cls(
            R=prob.get('R', 1),
            S=prob.get('S', 1),
            P=prob.get('P', 1),
            Q=prob.get('Q', 1),
            C=prob.get('C', 1),
            K=prob.get('K', 1),
            N=prob.get('N', 1),
            stride=(prob.get('Wstride', 1), prob.get('Hstride', 1)),
            dilation=(prob.get('Wdilation', 1), prob.get('Hdilation', 1)),
            path=path,
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ConvWorkload":
        """
        Create ConvWorkload from a YAML file.
        
        Args:
            yaml_path: Path to YAML file
            
        Returns:
            ConvWorkload instance
        """
        path = Path(yaml_path)
        with path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        return cls.from_dict(config, path=str(path))
    
    def to_dict(self) -> dict:
        """Convert workload to dictionary."""
        return {
            'problem': {
                'R': self.R,
                'S': self.S,
                'P': self.P,
                'Q': self.Q,
                'C': self.C,
                'K': self.K,
                'N': self.N,
                'Wstride': self.stride[0],
                'Hstride': self.stride[1],
                'Wdilation': self.dilation[0],
                'Hdilation': self.dilation[1],
            }
        }
    
    def __repr__(self) -> str:
        dims = f"R={self.R}, S={self.S}, P={self.P}, Q={self.Q}, C={self.C}, K={self.K}, N={self.N}"
        return f"ConvWorkload({dims}, stride={self.stride}, dilation={self.dilation})"
    
    def summary(self) -> str:
        """Return a summary string."""
        lines = [
            f"ConvWorkload: {self.path}",
            f"  Dimensions: R={self.R}, S={self.S}, P={self.P}, Q={self.Q}, C={self.C}, K={self.K}, N={self.N}",
            f"  Input size: H={self.input_size['H']}, W={self.input_size['W']}",
            f"  Stride: {self.stride}, Dilation: {self.dilation}",
            f"  MACs: {self.macs:,}",
        ]
        return "\n".join(lines)
