from typing import List, Dict, Tuple
from .data_layout import DataLayout


class LayoutCostEvaluator:
    """
    Evaluates the efficiency of a DataLayout under a specific Data Access Pattern (Loop Nest).

    Cost Model Principles (No Cache):
    1. DRAM Row Hits: Accesses should stay within the same DRAM row to minimize Precharge/Activate overhead.
    2. Burst Efficiency: DRAM transfers data in bursts (e.g., 32B or 64B chunks). 
       We measure how many requested bytes are actually useful within each burst.
       This handles "periodic continuity" (e.g. Tiling) correctly.
    """

    def __init__(self, layout: DataLayout, dram_row_size: int = 2048, burst_size: int = 32):
        self.layout = layout
        self.dram_row_size = dram_row_size
        self.burst_size = burst_size  # e.g., 32 Bytes (BL8 * 4B) or 64 Bytes

    def evaluate_loop_nest(self, loop_nest: List[Tuple[str, int]]) -> Dict[str, float]:
        """
        Evaluate the cost of a given loop nest.

        Args:
            loop_nest: List of (dimension_name, extent) from Outer to Inner.
                       Matches the loop structure defined in trace_generator.

        Returns:
            Dict containing metrics:
            - 'burst_efficiency': Useful Bytes / Transferred Bytes (0.0 - 1.0).
            - 'row_buffer_miss_rate': Estimated DRAM row buffer miss rate.
            - 'total_cost': Normalized cost per access.
        """
        if not loop_nest:
            return {"total_cost": 0.0}

        # Simulate the loop nest execution
        # We need to simulate enough accesses to capture the behavior of inner loop wrap-arounds.
        # e.g. if inner loop is 16, we need >16 accesses to see the jump to the next tile.
        SIM_LIMIT = 256

        addresses = []

        # Initialize indices (Outer -> Inner)
        dims = [d for d, _ in loop_nest]
        extents = [e for _, e in loop_nest]
        extents_map = dict(loop_nest)
        current_idx = [0] * len(loop_nest)

        for _ in range(SIM_LIMIT):
            # 1. Calculate Address
            # Map current loop counters to dimension names
            indices_dict = {dims[i]: current_idx[i] for i in range(len(dims))}

            # Handle Split Dimensions (Reconstruct Logical Indices)
            reconstructed_indices = indices_dict.copy()
            for dim, val in indices_dict.items():
                if dim.endswith('_in'):
                    base_name = dim[:-3]
                    outer_name = base_name + '_out'
                    if outer_name in indices_dict:
                        inner_extent = extents_map[dim]
                        combined_val = indices_dict[outer_name] * \
                            inner_extent + val
                        reconstructed_indices[base_name] = combined_val

            addr = self.layout.get_physical_address(reconstructed_indices)
            addresses.append(addr)

            # 2. Increment Indices (Innermost first)
            # Iterate backwards from the last dimension
            for i in range(len(dims) - 1, -1, -1):
                current_idx[i] += 1
                if current_idx[i] < extents[i]:
                    break  # Successfully incremented, no carry needed
                else:
                    # Wrap around and carry to next outer loop
                    current_idx[i] = 0
            else:
                # If the loop completes (all counters wrap around), stop simulation
                break

        if not addresses:
            return {"total_cost": 0.0}

        # --- Metric 1: Burst Efficiency ---
        # Simulate a simple memory controller that fetches 'burst_size' aligned blocks
        transferred_bytes = 0
        useful_bytes = 0

        current_burst_start = -1
        current_burst_end = -1

        burst_fetches = 0

        for addr in addresses:
            useful_bytes += self.layout.element_size

            # Check if address is within the current active burst
            if current_burst_start <= addr < current_burst_end:
                # Hit in current burst
                continue
            else:
                # New Burst needed
                # Align address to burst size
                burst_fetches += 1
                aligned_start = (addr // self.burst_size) * self.burst_size
                current_burst_start = aligned_start
                current_burst_end = aligned_start + self.burst_size

        transferred_bytes = burst_fetches * self.burst_size
        burst_efficiency = useful_bytes / transferred_bytes if transferred_bytes > 0 else 0

        # --- Metric 2: DRAM Row Misses ---
        row_misses = 0
        # Initialize with the first row
        current_row = addresses[0] // self.dram_row_size

        for i in range(1, len(addresses)):
            new_row = addresses[i] // self.dram_row_size
            if new_row != current_row:
                row_misses += 1
                current_row = new_row

        row_miss_rate = row_misses / len(addresses)

        # --- Total Cost ---
        # Base Cost = 1.0
        cost = 1.0

        # Penalty for Wasted Bandwidth
        if burst_efficiency > 0:
            bandwidth_penalty = (1.0 / burst_efficiency) - 1.0
            cost += bandwidth_penalty * 0.5

        # Penalty for Row Miss (Latency)
        ROW_MISS_PENALTY = 20.0
        cost += row_miss_rate * ROW_MISS_PENALTY

        return {
            "burst_efficiency": burst_efficiency,
            "row_buffer_miss_rate": row_miss_rate,
            "total_cost": cost
        }
