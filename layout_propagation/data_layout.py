from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
import math


class LayoutConstraint(Enum):
    """Layout constraints for physical dimensions."""
    COMPACT = auto()        # Tightly packed, stride = prev_stride * prev_size
    ALIGNED = auto()        # Stride aligned to specific byte boundary (e.g. DRAM Row)
    STRIDED = auto()        # Explicit stride (e.g. for sliding window views)


class DimensionType(Enum):
    """Type of the dimension in the mapping space."""
    TEMPORAL = auto()       # Standard sequential execution (Loop)
    SPATIAL = auto()        # Spatially unrolled (Parallelism / Vectorization)


@dataclass
class PhysicalDimension:
    """
    A physical dimension in the data layout.
    Can be a full logical dimension or a tile of it.
    """
    name: str               # e.g., "C", "C_outer", "H_block"
    logical_dim: str        # The logical dimension it belongs to (e.g., "C")
    size: int               # The size of this dimension (number of elements)

    # Constraint defines how the stride is calculated relative to the inner dimension
    constraint: LayoutConstraint = LayoutConstraint.COMPACT
    alignment: int = 1      # Byte alignment if constraint is ALIGNED
    # Explicit stride if constraint is STRIDED
    explicit_stride: Optional[int] = None

    # Type of dimension (Temporal vs Spatial)
    # This expands the layout definition to cover the full Map Space (Spatial Partitioning)
    dim_type: DimensionType = DimensionType.TEMPORAL

    def __repr__(self):
        extra = ""
        if self.constraint == LayoutConstraint.ALIGNED:
            extra = f", align={self.alignment}"
        elif self.constraint == LayoutConstraint.STRIDED:
            extra = f", stride={self.explicit_stride}"

        type_str = ""
        if self.dim_type == DimensionType.SPATIAL:
            type_str = ", SPATIAL"

        return f"Dim({self.name}, size={self.size}{extra}{type_str})"


@dataclass
class DataLayout:
    """
    Comprehensive Data Layout Definition.

    Represents the mapping from Logical Tensor (N-D) to Physical Memory (1-D).
    Inspired by PIMDLS trace_generator.py which handles:
    1. Hierarchical Tiling (L3, L2, L1...)
    2. Dimension Permutation (Access Order)
    3. Alignment/Padding (Row Alignment)

    Structure:
    - Ordered list of PhysicalDimensions (Major -> Minor / Outer -> Inner)
    - Base address
    - Element size
    """

    # Logical shape of the tensor, e.g., {'N': 1, 'C': 64, 'H': 224, 'W': 224}
    logical_shape: Dict[str, int]

    # Ordered list of physical dimensions from Outermost (Major) to Innermost (Minor)
    # Example NCHW: [N, C, H, W]
    # Example NCHW16c: [N, C_outer, H, W, C_inner(16)]
    ordering: List[PhysicalDimension]

    element_size: int = 1  # Bytes per element
    base_addr: int = 0

    # Hardware constraints that this layout is optimized for
    # Size of the physical buffer unit (e.g. DRAM Row) in bytes
    buffer_size: Optional[int] = None

    def __post_init__(self):
        self._validate()
        self._strides = self._compute_strides()
        self._hierarchy = self._build_hierarchy()

    def _build_hierarchy(self):
        """Build hierarchy map: logical_dim -> list of (index_in_ordering, size, name)"""
        hierarchy = {}
        for i, dim in enumerate(self.ordering):
            if dim.logical_dim not in hierarchy:
                hierarchy[dim.logical_dim] = []
            hierarchy[dim.logical_dim].append((i, dim.size, dim.name))
        return hierarchy

    def __hash__(self):
        # Hash based on ordering and element size (ignoring base_addr)
        # We need to make ordering hashable (tuple of tuples)
        ordering_tuple = tuple(
            (d.name, d.logical_dim, d.size, d.constraint,
             d.alignment, d.explicit_stride)
            for d in self.ordering
        )
        return hash((ordering_tuple, self.element_size, self.buffer_size))

    def __eq__(self, other):
        if not isinstance(other, DataLayout):
            return False
        # Compare ordering and element size (ignoring base_addr)
        if len(self.ordering) != len(other.ordering):
            return False

        for d1, d2 in zip(self.ordering, other.ordering):
            if (d1.name != d2.name or
                d1.logical_dim != d2.logical_dim or
                d1.size != d2.size or
                d1.constraint != d2.constraint or
                d1.alignment != d2.alignment or
                    d1.explicit_stride != d2.explicit_stride):
                return False

        return (self.element_size == other.element_size and
                self.buffer_size == other.buffer_size)

    def is_compatible(self, other: 'DataLayout') -> bool:
        """Check if two layouts are physically compatible (can be used interchangeably without transform)."""
        return self == other

    @property
    def format_name(self) -> str:
        """Generate a human-readable format name (e.g. NCHW, NCHW16c)."""
        # Simple heuristic
        names = [d.name for d in self.ordering]
        # Check for blocking
        is_blocked = any("_inner" in n or "_outer" in n for n in names)

        if not is_blocked:
            return "".join(names)
        else:
            # Try to reconstruct standard blocked names
            # e.g. N, C_outer, H, W, C_inner -> NCHW[C]
            base_names = []
            suffix = ""
            for d in self.ordering:
                if "_outer" in d.name:
                    base_names.append(d.logical_dim)
                elif "_inner" in d.name:
                    suffix += f"{d.size}{d.logical_dim.lower()}"
                else:
                    base_names.append(d.name)
            return "".join(base_names) + suffix

    def _validate(self):
        """Validate that physical dimensions match logical shape."""
        # Check if product of physical sizes matches logical size for each dim
        computed_shape = {k: 1 for k in self.logical_shape}
        for dim in self.ordering:
            if dim.logical_dim not in computed_shape:
                raise ValueError(
                    f"Unknown logical dimension: {dim.logical_dim}")
            computed_shape[dim.logical_dim] *= dim.size

        for k, v in self.logical_shape.items():
            if computed_shape[k] < v:
                raise ValueError(
                    f"Dimension mismatch for {k}: Logical={v}, Physical={computed_shape[k]}")

    def _compute_strides(self) -> List[int]:
        """
        Compute physical strides (in elements) for each dimension.
        Iterates from Inner (Minor) to Outer (Major).
        """
        strides = [0] * len(self.ordering)
        current_stride = 1  # Element-wise stride

        # Iterate backwards (Minor -> Major)
        for i in range(len(self.ordering) - 1, -1, -1):
            dim = self.ordering[i]

            if dim.constraint == LayoutConstraint.STRIDED:
                if dim.explicit_stride is None:
                    raise ValueError(
                        f"Dimension {dim.name} is STRIDED but no stride provided")
                # For STRIDED, the stride is explicit.
                # But usually stride represents the step *between* elements of this dimension.
                # So we assign it and update current_stride for the *next* dimension?
                # No, explicit stride usually overrides the accumulation.
                strides[i] = dim.explicit_stride
                # Update current_stride for the next outer dimension?
                # Usually: next_stride = this_stride * this_size
                current_stride = dim.explicit_stride * dim.size

            elif dim.constraint == LayoutConstraint.ALIGNED:
                # Align the *current* accumulated size (which is the stride of this dimension)
                # Wait, stride of dimension i is the distance between element x and x+1 in this dimension.
                # This distance spans all inner dimensions.
                # So stride[i] = size(i+1) * stride(i+1) ...

                # If ALIGNED, we align the *block size* of the inner dimensions.
                # The stride of *this* dimension is the aligned size of the inner block.

                # Calculate raw size of inner block in bytes
                inner_block_bytes = current_stride * self.element_size

                # Align
                aligned_bytes = math.ceil(
                    inner_block_bytes / dim.alignment) * dim.alignment
                aligned_stride = aligned_bytes // self.element_size

                strides[i] = aligned_stride
                current_stride = aligned_stride * dim.size

            else:  # COMPACT
                strides[i] = current_stride
                current_stride *= dim.size

        return strides

    def get_physical_address(self, indices: Dict[str, int]) -> int:
        """
        Calculate physical address for a given index.
        Supports both logical indices (e.g. {'C': 35}) and physical indices (e.g. {'C_outer': 2}).
        Physical indices take precedence over logical decomposition.
        """
        physical_vals = {}  # index_in_ordering -> value

        # 1. Logical Decomposition
        # Use cached hierarchy if available, otherwise build it (for backward compatibility if __post_init__ didn't run)
        hierarchy = getattr(self, '_hierarchy', None)
        if hierarchy is None:
            # Fallback if _hierarchy is missing
            hierarchy = self._build_hierarchy()

        for log_dim, phys_dims in hierarchy.items():
            if log_dim in indices:
                # Sort by index in ordering (Major to Minor)
                sorted_phys = sorted(phys_dims, key=lambda x: x[0])

                val = indices[log_dim]

                # Calculate suffix products for decomposition
                splits = [p[1] for p in sorted_phys]
                suffix_products = [1] * len(splits)
                p = 1
                for i in range(len(splits)-1, -1, -1):
                    suffix_products[i] = p
                    p *= splits[i]

                # Decompose value
                for i, (order_idx, size, _) in enumerate(sorted_phys):
                    log_stride = suffix_products[i]
                    phys_val = (val // log_stride) % size
                    physical_vals[order_idx] = phys_val

        # 2. Direct Physical Mapping (Overrides logical decomposition)
        for i, dim in enumerate(self.ordering):
            if dim.name in indices:
                physical_vals[i] = indices[dim.name]

        # 3. Calculate offset
        offset = 0
        for i, stride in enumerate(self._strides):
            idx = physical_vals.get(i, 0)
            offset += idx * stride

        return self.base_addr + offset * self.element_size

    @property
    def total_size_bytes(self) -> int:
        """Total size in bytes, including padding."""
        if not self.ordering:
            return 0
        # The stride of the hypothetical "dimension -1" (super-major)
        # is ordering[0].stride * ordering[0].size
        return self._strides[0] * self.ordering[0].size * self.element_size

    @classmethod
    def from_nchw(cls, N, C, H, W, element_size=1, base_addr=0):
        """Create standard NCHW layout."""
        dims = [
            PhysicalDimension("N", "N", N),
            PhysicalDimension("C", "C", C),
            PhysicalDimension("H", "H", H),
            PhysicalDimension("W", "W", W),
        ]
        return cls({'N': N, 'C': C, 'H': H, 'W': W}, dims, element_size, base_addr)

    @classmethod
    def from_nhwc(cls, N, C, H, W, element_size=1, base_addr=0):
        """Create standard NHWC layout."""
        dims = [
            PhysicalDimension("N", "N", N),
            PhysicalDimension("H", "H", H),
            PhysicalDimension("W", "W", W),
            PhysicalDimension("C", "C", C),
        ]
        return cls({'N': N, 'C': C, 'H': H, 'W': W}, dims, element_size, base_addr)

    @classmethod
    def from_nchw_c(cls, N, C, H, W, c_block, element_size=1, base_addr=0):
        """Create NCHW[c] blocked layout (e.g. NCHW16c)."""
        if C % c_block != 0:
            raise ValueError(
                f"C={C} must be divisible by block size {c_block}")

        dims = [
            PhysicalDimension("N", "N", N),
            PhysicalDimension("C_outer", "C", C // c_block),
            PhysicalDimension("H", "H", H),
            PhysicalDimension("W", "W", W),
            PhysicalDimension("C_inner", "C", c_block),
        ]
        return cls({'N': N, 'C': C, 'H': H, 'W': W}, dims, element_size, base_addr)

    @classmethod
    def from_row_aligned(cls, N, C, H, W, row_size_bytes, element_size=1, base_addr=0):
        """
        Create Row-Aligned Layout (similar to trace_generator).
        Assumes N, C are outer (L3), H, W are inner (L2).
        Each (n, c) tile contains a full (H, W) plane, padded to row boundary.
        """
        # Inner block is H*W
        # We want the stride of C (which steps over H*W) to be row aligned.

        # Hierarchy: N -> C -> [H, W]
        # H, W are compact.
        # C is ALIGNED.

        dims = [
            PhysicalDimension("N", "N", N),
            PhysicalDimension(
                "C", "C", C, constraint=LayoutConstraint.ALIGNED, alignment=row_size_bytes),
            PhysicalDimension("H", "H", H),
            PhysicalDimension("W", "W", W),
        ]
        return cls({'N': N, 'C': C, 'H': H, 'W': W}, dims, element_size, base_addr, buffer_size=row_size_bytes)

    @classmethod
    def from_shape(cls, shape: Tuple[int, ...], names: Optional[List[str]] = None, element_size=1, base_addr=0):
        """Create a simple compact layout from a shape tuple."""
        if names is None:
            # Generate default names: D0, D1, ... or N, C, H, W if len=4
            if len(shape) == 4:
                names = ["N", "C", "H", "W"]
            else:
                names = [f"D{i}" for i in range(len(shape))]

        if len(names) != len(shape):
            raise ValueError("Number of names must match number of dimensions")

        logical_shape = {name: size for name, size in zip(names, shape)}
        dims = [PhysicalDimension(name, name, size)
                for name, size in zip(names, shape)]

        return cls(logical_shape, dims, element_size, base_addr)
