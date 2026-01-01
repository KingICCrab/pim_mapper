
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

tile = 896
row = 1024
num = 28

cross = compute_dram_row_crossing_count(tile, row, num)
print(f"Tile={tile}, Row={row}, Num={num}")
print(f"Crossings={cross}")
print(f"Ratio={cross/num}")

tile = 128
num = 196
cross = compute_dram_row_crossing_count(tile, row, num)
print(f"Tile={tile}, Row={row}, Num={num}")
print(f"Crossings={cross}")
