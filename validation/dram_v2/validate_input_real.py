import math
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from pim_optimizer.model.row_activation import precompute_tile_crossing_info, compute_dram_row_crossing_count
from validation.dram.trace_generator import TraceGenerator, DRAMConfig

# Define Mapping and ConvWorkload locally if not importable
class Mapping:
    def __init__(self):
        self.loop_bounds = {}
        self.permutation = {}
        self.layout = {}
        self.tile_info = {}

class ConvWorkload:
    def __init__(self, C, K, R, S, P, Q, stride=(1,1), dilation=(1,1), N=1):
        self.C = C
        self.K = K
        self.R = R
        self.S = S
        self.P = P
        self.Q = Q
        self.stride = stride
        self.dilation = dilation
        self.N = N
        
        # Derived
        self.H = (P - 1) * stride[0] + R
        self.W = (Q - 1) * stride[1] + S
        
        self.input_size = {'H': self.H, 'W': self.W}
        
        # Bounds array for generic access
        # R, S, P, Q, C, K, N
        self.bounds = [R, S, P, Q, C, K, N]
        self.O = [[0]*3 for _ in range(7)] # Not used in this script but required by some logic?
        # Actually we don't use O here.

# Dimension Constants
DIM_R, DIM_S, DIM_P, DIM_Q, DIM_C, DIM_K, DIM_N = 0, 1, 2, 3, 4, 5, 6

def count_row_activations(trace, dram_config):
    """
    Count row activations from a trace.
    """
    activations = {} # bank -> count
    open_rows = {}   # bank -> row_idx
    
    row_mask = (1 << 14) - 1 # 14 bits for row
    # Address mapping (from trace_generator.py):
    # col: 0-9 (10 bits)
    # bank: 10-12 (3 bits) - Wait, trace_generator uses 4 banks (2 bits) or 8 banks?
    # Let's check trace_generator.py _get_dram_addr
    # It uses:
    # col_bits = log2(row_size) = log2(1024) = 10
    # bank_bits = log2(num_banks) = log2(4) = 2
    # rank_bits = log2(num_ranks) = 0
    # channel_bits = log2(num_channels) = 0
    #
    # addr = row << (col + bank + rank + chan) | bank << ...
    # So Row is at top.
    
    # Let's assume standard mapping from trace_generator
    # col: 0-9
    # bank: 10-11
    # row: 12+
    
    for line in trace:
        parts = line.strip().split()
        if len(parts) < 2: continue
        
        cmd = parts[0]
        addr_str = parts[1]
        try:
            addr = int(addr_str, 16)
        except ValueError:
            continue
            
        # Decode Address
        # Assuming 1024 byte row buffer -> 10 bits col
        col_bits = 10
        bank_bits = 2
        
        bank = (addr >> col_bits) & ((1 << bank_bits) - 1)
        row = (addr >> (col_bits + bank_bits))
        
        if bank not in open_rows:
            open_rows[bank] = -1
            activations[bank] = 0
            
        if open_rows[bank] != row:
            activations[bank] += 1
            open_rows[bank] = row
            
    return activations

class InputWorkloadValidator:
    def __init__(self):
        self.dram_config = DRAMConfig(
            num_channels=1, num_ranks=1, num_banks=4, num_rows=16384,
            row_buffer_bytes=1024, element_size=1
        )
        self.trace_gen = TraceGenerator(self.dram_config)

    def calculate_ilp_reuse_params(self, workload, permutation_l2, permutation_l1, tensor_type='input'):
        """
        Calculate Reuse Penalty and Outer Irrelevant Product for Input.
        
        Input Tensor: N, C, H, W
        Relevant Dims: N, C, P, Q, R, S (H, W are derived from P, Q, R, S)
        Actually, Input depends on N, C, (P, R), (Q, S).
        Irrelevant Dims: K.
        
        Reuse Penalty: Product of bounds of Irrelevant Dims in L2.
        Outer Irrelevant Product: Product of bounds of Irrelevant Dims in L2 that are NOT reused (if any).
        
        Wait, standard logic:
        Reuse = Product of Irrelevant Dims in L2 (if they are inner to all relevant dims in L2? No).
        
        Let's use the logic from validate_weight_output_real.py:
        Iterate L2 from Outer to Inner.
        If Dim is Irrelevant:
            If NO Relevant Dim is Inner to it:
                It contributes to Outer Product (we visit it once, do all relevant stuff, never come back).
            If HAS Relevant Dim Inner to it:
                It contributes to Reuse Penalty (we visit it, do relevant stuff, then come back to it later? No).
                
        Wait, if Irrelevant Dim is OUTER to Relevant Dim:
            We fix Irrelevant, iterate Relevant.
            Then change Irrelevant, iterate Relevant again.
            So Relevant data is re-accessed.
            So Reuse Penalty *= Irrelevant_Bound.
            
        If Irrelevant Dim is INNER to Relevant Dim:
            We fix Relevant, iterate Irrelevant.
            Relevant data is constant (Stationary).
            So no penalty.
            
        Correct Logic:
        Iterate L2 from Outer to Inner.
        If Dim is Irrelevant:
            If it wraps any Relevant dim (i.e. Relevant dim is Inner):
                Reuse *= Dim_Bound
            Else (All inner dims are also irrelevant):
                Outer *= Dim_Bound
        """
        # Input Relevant Dims: N, C, P, Q, R, S
        # (Note: P, Q, R, S determine H, W. So they are all relevant)
        relevant_dims = {DIM_N, DIM_C, DIM_P, DIM_Q, DIM_R, DIM_S}
        
        # Check if any relevant dim is in L1 (SRAM)
        # If a relevant dim is in L1, it means we iterate it inside the tile.
        # This doesn't affect DRAM reuse logic directly, except that the tile size covers it.
        
        # Analyze L2 Permutation (Outer -> Inner)
        # We need to know if there are relevant dims *inside* the current dim in L2.
        
        # Precompute "has_relevant_inner" for each position in L2
        xr = [0] * len(permutation_l2)
        has_relevant_inner = False
        
        # Scan from Inner to Outer
        for i in range(len(permutation_l2) - 1, -1, -1):
            dim = permutation_l2[i]
            is_relevant = dim in relevant_dims
            
            if is_relevant:
                has_relevant_inner = True
            
            # If this dim is irrelevant, and there is a relevant dim inside it,
            # then this dim causes reuse.
            if not is_relevant and has_relevant_inner:
                xr[i] = 1
            else:
                xr[i] = 0
                
        reuse_penalty = 1
        outer_irr_product = 1
        
        for i, dim in enumerate(permutation_l2):
            dim_bound = workload.bounds[dim]
            is_relevant = dim in relevant_dims
            
            if not is_relevant:
                if xr[i] == 1:
                    reuse_penalty *= dim_bound
                else:
                    outer_irr_product *= dim_bound
                    
        return reuse_penalty, outer_irr_product

    def calculate_ilp_cost_logic(self, workload, tile_size_elements, tensor_type, reuse_penalty, outer_irr_product):
        """
        Calculate ILP predicted cost.
        """
        element_bytes = 1
        row_bytes = 1024
        
        if tensor_type == 'input':
            # Note: This is total elements in the tensor, not just the tile
            # But for crossing probability, we care about the TILE size.
            pass
            
        tile_bytes = tile_size_elements * element_bytes
        
        # Call the actual ILP logic function
        # We pass a dummy tensor_total_bytes because it only affects num_tiles calculation
        # which we don't use here (we use outer_irr_product * reuse_penalty logic).
        # Wait, precompute_tile_crossing_info returns (non_crossing_acts, crossing_counts)
        # for a SINGLE tile access?
        # No, it returns it for the WHOLE tensor if we pass tensor_total_bytes.
        # But here we want per-tile info to multiply by our own loop counts.
        
        # Let's look at precompute_tile_crossing_info again.
        # It calculates num_tiles = tensor_bytes / tile_bytes.
        # Then returns total acts.
        
        # We want the cost PER TILE access.
        # So we should pass tensor_total_bytes = tile_bytes.
        # Then num_tiles = 1.
        
        non_crossing_acts_list, crossing_counts_list = precompute_tile_crossing_info(
            [tile_size_elements], element_bytes, row_bytes, tile_bytes
        )
        
        non_crossing_acts = non_crossing_acts_list[0]
        crossing_count = crossing_counts_list[0]
        
        # Apply Reuse and Outer Product
        # Formula: Cost = (NonCrossing + Crossing * Reuse) * Outer
        # Wait, if Crossing, we pay penalty every time we reuse?
        # If we reuse the tile, do we fetch it from DRAM again?
        # Yes, Reuse Penalty means we fetch it multiple times.
        
        # If Tile fits in Row (NonCrossing):
        # We fetch it once per Reuse?
        # Yes.
        
        # So Total Cost = (NonCrossing + Crossing) * Reuse * Outer?
        # Let's check validate_weight_output_real.py logic.
        # base_row_acts = non_crossing_acts + 1 * crossing_count * reuse_penalty
        # total_cost = base_row_acts * outer_irr_product
        
        # This implies NonCrossing part is NOT multiplied by reuse_penalty?
        # That seems wrong. If we reuse the tile, we fetch it again, regardless of crossing.
        # Unless "NonCrossing" means "Row Buffer Hit"?
        # No, Row Activation means we OPEN the row.
        # If we access the same tile again later (after accessing other stuff), we must OPEN the row again.
        # So Reuse Penalty applies to EVERYTHING.
        
        # Why did validate_weight_output_real.py use that formula?
        # "base_row_acts = non_crossing_acts + 1 * crossing_count * reuse_penalty"
        # If non_crossing_acts > 0, it should also be multiplied.
        
        # Let's re-read validate_weight_output_real.py carefully.
        # It says:
        # if tile_size_elements > 1024: ...
        # else: base_row_acts = non_crossing_acts + 1 * crossing_count * reuse_penalty
        
        # If tile fits in row (crossing=0, non_crossing=1):
        # base = 1 + 0 = 1.
        # total = 1 * outer.
        # Where is reuse?
        # Ah, in validate_weight_output_real.py, for Weight Stationary, Reuse=1.
        # For Output Stationary, Reuse=4.
        # Let's check OS result: ILP=640, GT=624.
        # Formula: max(nc, c) * reuse?
        # In code:
        # if tile > row: cost = 2 * crossing * reuse
        # if tile <= row: cost = max(nc, c) * reuse (This was the comment)
        
        # But the code implemented:
        # base_row_acts = non_crossing_acts + 1 * crossing_count * reuse_penalty
        # This means non_crossing_acts is NOT multiplied by reuse_penalty.
        # This implies that if it's non-crossing, we assume it stays in Row Buffer?
        # That's only true if the reuse happens IMMEDIATELY (Spatial Reuse).
        # But Reuse Penalty usually implies Temporal Reuse (Looping back).
        # If we loop back after accessing other things, Row Buffer is likely closed/evicted.
        
        # So correct logic should be:
        # Total Acts = (NonCrossing + Crossing) * Reuse * Outer
        
        # Let's try this logic for Input.
        
        total_acts_per_tile = non_crossing_acts + crossing_count
        total_cost = total_acts_per_tile * reuse_penalty * outer_irr_product
        
        return total_cost, non_crossing_acts, crossing_count

    def calculate_gt_cost(self, workload, tile_sizes, permutation_l2, permutation_l1, layout='sequential'):
        mapping = Mapping()
        
        # Tile Sizes: C, P, Q (Input specific)
        # We assume R, S are full in L1 (or tiled, but usually small)
        c_tile, p_tile, q_tile = tile_sizes
        
        # Input Tile Size depends on R, S too
        # But for tiling definition, we define bounds of loops.
        
        tiled_dims = {DIM_C: c_tile, DIM_P: p_tile, DIM_Q: q_tile}
        target_bank = 0 # Input is usually Bank 0
        
        l2_bounds = {}
        l1_bounds = {}
        
        for dim in range(7):
            l2_bounds[dim] = 1
            l1_bounds[dim] = 1
            
        for dim, bound in enumerate(workload.bounds):
            if dim in tiled_dims:
                l1_size = tiled_dims[dim]
                l2_size = (bound + l1_size - 1) // l1_size
                l1_bounds[dim] = l1_size
                if dim in permutation_l2:
                    l2_bounds[dim] = l2_size
            else:
                if dim in permutation_l2:
                    l2_bounds[dim] = bound
                elif dim in permutation_l1:
                    l1_bounds[dim] = bound
                    
        mapping.loop_bounds = {
            2: {'spatial': l2_bounds}, # DRAM
            1: {'spatial': l1_bounds}  # SRAM
        }
        
        mapping.permutation = {
            2: {i: dim for i, dim in enumerate(permutation_l2)},
            1: {i: dim for i, dim in enumerate(permutation_l1)}
        }
        
        mapping.layout = {0: layout, 1: 'sequential', 2: 'sequential'}
        
        trace = self.trace_gen.generate_trace(mapping, workload)
        
        # DEBUG: Analyze Trace for Weight Stationary (check for sequentiality)
        # We identify WS case by permutation (K is outer)
        is_ws = (permutation_l2[0] == DIM_K)
        if is_ws:
            print("\nDEBUG: Analyzing Weight Stationary Trace (First 20 accesses)")
            print(f"Total Trace Length: {len(trace)}")
            prev_row = -1
            row_switches = 0
            for i, line in enumerate(trace[:20]):
                parts = line.split()
                if len(parts) >= 2:
                    addr = int(parts[1], 16)
                    # Decode (assuming standard mapping)
                    # col: 10 bits, bank: 2 bits
                    row = addr >> 12
                    bank = (addr >> 10) & 0x3
                    print(f"  {i}: {line.strip()} -> Row: {row}, Bank: {bank}")
                    
                    if prev_row != -1 and row != prev_row:
                        row_switches += 1
                    prev_row = row
            print(f"DEBUG: Row Switches in first 20: {row_switches}")
            
            # Count total unique rows accessed
            unique_rows = set()
            for line in trace:
                parts = line.split()
                if len(parts) >= 2:
                    addr = int(parts[1], 16)
                    row = addr >> 12
                    unique_rows.add(row)
            print(f"DEBUG: Total Unique Rows Accessed: {len(unique_rows)}")
            print(f"DEBUG: Total Accesses: {len(trace)}")
            print(f"DEBUG: Avg Accesses per Row: {len(trace)/len(unique_rows):.2f}")

        activations_dict = count_row_activations(trace, self.dram_config)
        return activations_dict.get(target_bank, 0)

    def run_validation(self):
        print("Starting INPUT Workload Validation...")
        print(f"{'Case':<20} | {'Tile':<15} | {'Reuse':<5} | {'Outer':<5} | {'ILP Cost':<10} | {'GT Cost':<10} | {'Error %':<10}")
        print("-" * 100)
        
        # Workload: Conv3_x (C=128, K=128, R=3, S=3, P=4, Q=4)
        # Input Size: H=6, W=6.
        wl = ConvWorkload(C=128, K=128, R=3, S=3, P=4, Q=4, stride=(1,1), dilation=(1,1), N=1)
        
        # Tile Size: C=4, P=2, Q=2
        # Input Tile (H, W) = (2-1+3, 2-1+3) = (4, 4)
        # Elements = 4 * 4 * 4 = 64 elements.
        c_tile, p_tile, q_tile = 4, 2, 2
        
        # Calculate Input Tile Size in Elements
        # H_tile = (p_tile - 1)*stride + R
        h_tile = (p_tile - 1) * 1 + 3
        w_tile = (q_tile - 1) * 1 + 3
        tile_size_elements = c_tile * h_tile * w_tile # 4 * 4 * 4 = 64
        
        # ---------------------------------------------------------
        # Scenario 1: Input Stationary (Reuse=1)
        # L2 (DRAM): P, Q, C
        # L1 (SRAM): K, R, S
        # ---------------------------------------------------------
        perm_l2_is = [DIM_P, DIM_Q, DIM_C]
        perm_l1_is = [DIM_K, DIM_R, DIM_S]
        
        reuse_is, outer_is = self.calculate_ilp_reuse_params(wl, perm_l2_is, perm_l1_is, 'input')
        
        # Calculate Total Tiles for Input
        # Input depends on C, P, Q (and R, S which are in L1/Tile)
        # We need to count how many (C, P, Q) tiles we visit.
        # L2 bounds for C, P, Q:
        # C_tiles = ceil(128/4) = 32
        # P_tiles = ceil(4/2) = 2
        # Q_tiles = ceil(4/2) = 2
        # Total Input Tiles = 32 * 2 * 2 = 128.
        
        total_input_tiles = 128
        
        # Manual ILP Prediction:
        # Input Tile (4x4x4 = 64 bytes) is small but strided (spans ~512 bytes).
        # Fits in 1 Row (1024 bytes).
        # So Cost per Tile ~ 1 ACT.
        # Total Cost = Total Tiles * 1 = 128.
        
        ilp_is = total_input_tiles * 1.0
        
        gt_is = self.calculate_gt_cost(wl, (c_tile, p_tile, q_tile), perm_l2_is, perm_l1_is + perm_l2_is)
        
        err_is = abs(ilp_is - gt_is) / gt_is * 100 if gt_is > 0 else 0
        print(f"{'Input Stationary':<20} | {'4x4x4':<15} | {reuse_is:<5} | {outer_is:<5} | {ilp_is:<10.1f} | {gt_is:<10} | {err_is:<10.2f}")

        # ---------------------------------------------------------
        # Scenario 2: Weight Stationary (Reuse=K)
        # L2 (DRAM): K, C, P, Q
        # L1 (SRAM): R, S
        # ---------------------------------------------------------
        perm_l2_ws = [DIM_K, DIM_C, DIM_P, DIM_Q]
        perm_l1_ws = [DIM_R, DIM_S]
        
        reuse_ws, outer_ws = self.calculate_ilp_reuse_params(wl, perm_l2_ws, perm_l1_ws, 'input')
        
        # ILP Model Logic (Replicated):
        # The ILP model uses tensor_total_bytes to estimate the number of tiles.
        # This implicitly assumes that overlapping accesses hit in the Row Buffer,
        # so we only pay for "Unique Data" loading.
        
        tensor_total_bytes = wl.C * wl.input_size['H'] * wl.input_size['W'] # 128 * 6 * 6 = 4608
        tile_bytes = tile_size_elements # 64
        
        # Base Sequential Acts (from precompute_tile_crossing_info)
        # num_tiles = tensor_total_bytes / tile_bytes = 4608 / 64 = 72
        # non_crossing_acts = 72 (since tile < row)
        base_acts = 72
        
        # Block Crossing Penalty (Approximate)
        # From analysis: H-crossing=1, W-crossing=1 per pass.
        # But we need to scale this.
        # Let's just use the Base Cost for now to see how close it is.
        
        ilp_ws = base_acts * reuse_ws # 72 * 128 = 9216
        
        # Add Block Crossing Estimate (Optional, if we want to be precise)
        # The gap (9561 - 9216 = 345) is likely block crossing.
        
        gt_ws = self.calculate_gt_cost(wl, (c_tile, p_tile, q_tile), perm_l2_ws, perm_l1_ws + perm_l2_ws)
        
        err_ws = abs(ilp_ws - gt_ws) / gt_ws * 100 if gt_ws > 0 else 0
        print(f"{'Weight Stationary':<20} | {'4x4x4':<15} | {reuse_ws:<5} | {outer_ws:<5} | {ilp_ws:<10.1f} | {gt_ws:<10} | {err_ws:<10.2f}")

if __name__ == "__main__":
    validator = InputWorkloadValidator()
    validator.run_validation()
