import pytest
from src.core.data_manager import DataManager

def test_initialization(dummy_edf_path):
    """Validates that DataManager properly infers headers without loading bulk data."""
    # Implicitly tests the channel picking logic as well
    dm = DataManager(dummy_edf_path, pick_types=['eeg'])
    assert dm.sfreq == 250
    assert dm.n_channels == 3 # MNE coerces 'Trigger' to eeg internally
    assert dm.total_duration_sec > 599.0 # ~600 seconds
    dm.close() # Test manual close

def test_context_manager(dummy_edf_path):
    """Validates the Context Manager behavior prevents dangling file locks."""
    with DataManager(dummy_edf_path) as dm:
        assert dm.raw is not None
        assert dm.n_channels == 3 # eeg, seeg, ecog default
    # If the block exits safely without exception, the file is successfully closed by __exit__

def test_get_window_valid(dummy_edf_path):
    """Tests fetching a standard 10-second window."""
    with DataManager(dummy_edf_path) as dm:
        data, times = dm.get_window(start_sec=0.0, duration_sec=10.0)
        # 3 channels x 250 Hz * 10 Sec = (3, 2500)
        assert data.shape == (3, 2500)
        assert len(times) == 2500

def test_get_window_exceeds_duration(dummy_edf_path):
    """Ensures that asking for time beyond the file trims gracefully with a warning."""
    with DataManager(dummy_edf_path) as dm:
        # Ask for 20 seconds, but start at 595 (only 5 seconds left in the 600s file)
        data, times = dm.get_window(start_sec=595.0, duration_sec=20.0) 
        # Should gracefully truncate to around 5 seconds
        assert data.shape[0] == 3
        assert data.shape[1] < 2500

def test_get_window_invalid(dummy_edf_path):
    """Ensures that completely invalid requests raise exceptions early."""
    with DataManager(dummy_edf_path) as dm:
        with pytest.raises(ValueError):
            dm.get_window(start_sec=601.0, duration_sec=10.0) # start time beyond total duration
        with pytest.raises(ValueError):
            dm.get_window(start_sec=0.0, duration_sec=-5.0) # negative time
