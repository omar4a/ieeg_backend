import pytest
import numpy as np
from src.core.data_manager import DataManager
from src.core.sliding_window import SlidingWindowGenerator

def test_initialization_invalid():
    """Tests rigorous parameter validations for the SlidingWindow geometry."""
    # We can't have an overlap larger than or equal to window size
    with pytest.raises(ValueError):
        SlidingWindowGenerator(None, window_size_sec=10, overlap_sec=10)
    
    with pytest.raises(ValueError):
        SlidingWindowGenerator(None, window_size_sec=10, overlap_sec=15)
        
    with pytest.raises(ValueError):
        SlidingWindowGenerator(None, window_size_sec=0, overlap_sec=-5)

def test_generate_loop(dummy_edf_path):
    """
    Tests the mathematical correctness of the windowing iterator on actual data chunks.
    """
    with DataManager(dummy_edf_path) as dm:
        # A 60-second window with 10-second overlap implies a 50 second linear jump.
        gen = SlidingWindowGenerator(dm, window_size_sec=60.0, overlap_sec=10.0)
        chunks = list(gen.generate())
        
        # Math verification on 600 sec file: 
        # 0->60, 50->110, 100->160, 150->210, 200->260, 250->310, 300->360, 350->410, 
        # 400->460, 450->510, 500->560, 550->600 (Truncated).
        assert len(chunks) == 12 
        
        # Verify First Block Logic
        start, end, data, times = chunks[0]
        assert start == 0.0
        assert end == 60.0
        assert data.shape == (3, 15000) # 3 channels * 250Hz * 60s
        
        # Verify Truncated Remainder Base Logic (Last Block)
        start, end, data, times = chunks[-1]
        assert start == 550.0 # From previous 500 + 50
        assert np.isclose(end, 600.0, atol=0.1)
        assert data.shape[0] == 3
        # allow rounding variations from exact 12500
        assert 12000 < data.shape[1] <= 12500
