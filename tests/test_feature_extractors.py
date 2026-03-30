import numpy as np
import pytest
from plugins.features.epileptogenicity_index import EpileptogenicityIndex

def test_epileptogenicity_index_extraction():
    """Validates the EI algorithm executes without crashing and adheres strictly 
    to the BaseFeatureExtractor JSON-serializable dictionary output contract."""
    
    # Simulate a massively concurrent sliding window (100 channels, 500 timepoints/samples @ 200 Hz)
    window_data = np.random.randn(100, 500)
    
    extractor = EpileptogenicityIndex(sfreq=200.0)
    result = extractor.extract(window_data)
    
    # Verify the structure strictly matches the ABC contract
    assert isinstance(result, dict)
    assert "epileptogenicity_index" in result
    assert "channels_processed" in result
    
    # 99 channels because the naive bipolar montage sequentially reduced N channels by 1
    assert result["channels_processed"] == 99
    
    # Ensure it returns native Python lists, not Numpy Arrays, proving IPC safety for serialization
    assert isinstance(result["epileptogenicity_index"], list)
    
    # mathematically demonstrate normalization (All values must be strictly [0.0, 1.0])
    scores = np.array(result["epileptogenicity_index"])
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
    
    # To demonstrate the dynamic parameter fix works, pass an ultra-short window 
    # (e.g., 100 samples when sfreq is 200). Older welch() implementations would instantly crash here.
    short_window_data = np.random.randn(5, 100)
    short_result = extractor.extract(short_window_data)
    assert short_result["channels_processed"] == 4
