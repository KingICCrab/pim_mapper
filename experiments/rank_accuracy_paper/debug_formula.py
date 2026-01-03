
import math

def compute_dram_row_crossing_count(tile_bytes: float, row_bytes: float, num_tiles: int) -> int:
    if tile_bytes <= 0 or num_tiles <= 0:
        return 0
    if tile_bytes > row_bytes:
        return num_tiles
    if tile_bytes == row_bytes:
        return 0
    
    tile_bytes_int = int(tile_bytes)
    row_bytes_int = int(row_bytes)
    
    g = math.gcd(tile_bytes_int, row_bytes_int)
    period = row_bytes_int // g
    
    threshold = row_bytes_int - tile_bytes_int + 1
    cross_count_per_period = period - math.ceil(threshold / g)
    cross_count_per_period = max(0, cross_count_per_period)
    
    num_complete_periods = num_tiles // period
    remainder_tiles = num_tiles % period
    
    crossings_in_remainder = 0
    for i in range(remainder_tiles):
        start_offset = i * tile_bytes_int
        start_row = start_offset // row_bytes_int
        end_row = (start_offset + tile_bytes_int - 1) // row_bytes_int
        if end_row > start_row:
            crossings_in_remainder += 1
            
    return num_complete_periods * cross_count_per_period + crossings_in_remainder

def debug_formula():
    tile_bytes = 144
    row_bytes = 2048
    num_tiles = 8192
    
    crossing_count = compute_dram_row_crossing_count(tile_bytes, row_bytes, num_tiles)
    
    tiles_per_row = max(1, int(row_bytes / tile_bytes))
    non_crossing_count = num_tiles - crossing_count
    non_crossing_acts = math.ceil(non_crossing_count / tiles_per_row) if non_crossing_count > 0 else 0
    
    print(f"Tile Bytes: {tile_bytes}")
    print(f"Row Bytes: {row_bytes}")
    print(f"Num Tiles: {num_tiles}")
    print(f"Tiles Per Row: {tiles_per_row}")
    print(f"Crossing Count: {crossing_count}")
    print(f"Non-Crossing Count: {non_crossing_count}")
    print(f"Non-Crossing Acts: {non_crossing_acts}")
    
    # Formula A: crossing * 2 + non_crossing_acts
    total_A = crossing_count * 2 + non_crossing_acts
    print(f"Total A (crossing*2 + grouped): {total_A}")
    
    # Formula B: crossing * 1 + non_crossing_acts (Maybe we only need 1 act for crossing if sequential?)
    # If sequential, a crossing just means we move to the next row. It's 1 switch.
    # But wait, if we cross, we access Row K and Row K+1.
    # If we are strictly sequential, we access Row K (already open), then Row K+1 (new open).
    # So it adds 1 new activation.
    
    # Let's see what the ILP code actually does.

if __name__ == "__main__":
    debug_formula()
