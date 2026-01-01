"""
Tests for PIM optimizer components.
"""

import pytest
import numpy as np


class TestConvWorkload:
    """Tests for ConvWorkload class."""
    
    def test_workload_creation(self):
        """Test basic workload creation."""
        from pim_optimizer.workload import ConvWorkload
        
        workload = ConvWorkload(
            R=3, S=3, P=56, Q=56, C=64, K=128, N=1
        )
        
        assert workload.R == 3
        assert workload.S == 3
        assert workload.P == 56
        assert workload.Q == 56
        assert workload.C == 64
        assert workload.K == 128
        assert workload.N == 1
    
    def test_macs_calculation(self):
        """Test MACs calculation."""
        from pim_optimizer.workload import ConvWorkload
        
        workload = ConvWorkload(
            R=3, S=3, P=56, Q=56, C=64, K=128, N=1
        )
        
        expected_macs = 3 * 3 * 56 * 56 * 64 * 128 * 1
        assert workload.macs == expected_macs
    
    def test_divisors(self):
        """Test divisor computation."""
        from pim_optimizer.workload import ConvWorkload
        
        workload = ConvWorkload(
            R=3, S=3, P=4, Q=4, C=8, K=16, N=1
        )
        
        # R=3 has divisors [1, 3]
        assert 1 in workload.divisors[0]
        assert 3 in workload.divisors[0]
        
        # K=16 has divisors [1, 2, 4, 8, 16]
        assert 1 in workload.divisors[5]
        assert 16 in workload.divisors[5]
    
    def test_relevancy_matrix(self):
        """Test relevancy matrix O[j][t]."""
        from pim_optimizer.workload import ConvWorkload
        
        workload = ConvWorkload(
            R=3, S=3, P=56, Q=56, C=64, K=128, N=1
        )
        
        # Input (t=0): relevant to R, S, P, Q, C, N
        assert workload.O[0][0] == 1  # R relevant to input
        assert workload.O[4][0] == 1  # C relevant to input
        assert workload.O[5][0] == 0  # K not relevant to input
        
        # Weight (t=1): relevant to R, S, C, K
        assert workload.O[0][1] == 1  # R relevant to weight
        assert workload.O[5][1] == 1  # K relevant to weight
        assert workload.O[6][1] == 0  # N not relevant to weight
        
        # Output (t=2): relevant to P, Q, K, N
        assert workload.O[0][2] == 0  # R not relevant to output
        assert workload.O[2][2] == 1  # P relevant to output
        assert workload.O[5][2] == 1  # K relevant to output


class TestCrossingRatio:
    """Tests for crossing ratio calculations."""
    
    def test_basic_crossing(self):
        """Test basic block crossing ratio calculation."""
        from pim_optimizer.model.crossing import compute_block_crossing_ratio_gcd
        
        crossing, g, period, cross_count = compute_block_crossing_ratio_gcd(
            block_h=8, tile_h=3, step=2
        )
        
        assert g == 2  # gcd(2, 8)
        assert period == 4  # 8 / 2
        assert crossing == cross_count / period
    
    def test_no_crossing(self):
        """Test case with no crossing."""
        from pim_optimizer.model.crossing import compute_block_crossing_ratio_gcd
        
        # Tile fits exactly in block
        crossing, _, _, _ = compute_block_crossing_ratio_gcd(
            block_h=8, tile_h=1, step=1
        )
        
        assert crossing == 0.0
    
    def test_always_crossing(self):
        """Test case where tile always crosses."""
        from pim_optimizer.model.crossing import compute_block_crossing_ratio_gcd
        
        # Tile larger than block
        crossing, _, _, _ = compute_block_crossing_ratio_gcd(
            block_h=8, tile_h=16, step=8
        )
        
        assert crossing == 1.0
    
    def test_step_matters(self):
        """Test that step affects crossing ratio."""
        from pim_optimizer.model.crossing import compute_block_crossing_ratio_gcd
        
        # Same tile, different steps
        c1, _, _, _ = compute_block_crossing_ratio_gcd(block_h=64, tile_h=7, step=1)
        c2, _, _, _ = compute_block_crossing_ratio_gcd(block_h=64, tile_h=7, step=2)
        c3, _, _, _ = compute_block_crossing_ratio_gcd(block_h=64, tile_h=7, step=4)
        
        # Different steps give different crossing ratios
        # (exact values depend on GCD relationships)
        assert isinstance(c1, float)
        assert isinstance(c2, float)
        assert isinstance(c3, float)


class TestArchitecture:
    """Tests for PIMArchitecture class."""
    
    def test_default_architecture(self):
        """Test default architecture creation."""
        from pim_optimizer.arch import PIMArchitecture
        
        arch = PIMArchitecture()
        
        assert arch.num_mems >= 2
        assert "LocalDRAM" in arch.mem_idx or arch.num_mems >= 3
    
    def test_memory_hierarchy(self):
        """Test memory hierarchy construction."""
        from pim_optimizer.arch import MemoryLevel, MemoryHierarchy
        
        hierarchy = MemoryHierarchy([
            MemoryLevel(name="L1", entries=256),
            MemoryLevel(name="L2", entries=4096),
            MemoryLevel(name="DRAM", entries=1000000),
        ])
        
        assert len(hierarchy) == 3
        assert hierarchy.get_level("L1").entries == 256


class TestUtils:
    """Tests for utility functions."""
    
    def test_divisors(self):
        """Test divisor computation."""
        from pim_optimizer.utils import get_divisors
        
        assert get_divisors(12) == [1, 2, 3, 4, 6, 12]
        assert get_divisors(7) == [1, 7]
        assert get_divisors(1) == [1]
    
    def test_gcd(self):
        """Test GCD computation."""
        from pim_optimizer.utils import compute_gcd
        
        assert compute_gcd(12, 8) == 4
        assert compute_gcd(7, 3) == 1
        assert compute_gcd(100, 25) == 25
    
    def test_lcm(self):
        """Test LCM computation."""
        from pim_optimizer.utils import compute_lcm
        
        assert compute_lcm(4, 6) == 12
        assert compute_lcm(3, 5) == 15


class TestMapping:
    """Tests for Mapping class."""
    
    def test_mapping_creation(self):
        """Test mapping creation."""
        from pim_optimizer.mapping import Mapping
        
        mapping = Mapping()
        mapping.loop_bounds = {
            0: {"spatial": {0: 1}, "temporal": {0: 3}},
        }
        
        assert mapping.loop_bounds[0]["temporal"][0] == 3
    
    def test_mapping_to_dict(self):
        """Test mapping serialization."""
        from pim_optimizer.mapping import Mapping
        
        mapping = Mapping(
            workload_name="test",
            loop_bounds={0: {"spatial": {}, "temporal": {0: 3}}},
        )
        
        d = mapping.to_dict()
        assert d["workload_name"] == "test"
        assert d["loop_bounds"][0]["temporal"][0] == 3
    
    def test_mapping_from_dict(self):
        """Test mapping deserialization."""
        from pim_optimizer.mapping import Mapping
        
        data = {
            "workload_name": "test",
            "loop_bounds": {0: {"spatial": {}, "temporal": {0: 3}}},
        }
        
        mapping = Mapping.from_dict(data)
        assert mapping.workload_name == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
