import pytest
import mne
import numpy as np

@pytest.fixture(scope="session")
def dummy_edf_path(tmp_path_factory):
    """
    Creates a temporary 10-minute, 4-channel .edf file for robust testing 
    across all IO components without needing real clinical data.
    """
    sfreq = 250
    duration_sec = 600 # 10 minutes
    n_samples = sfreq * duration_sec
    
    # We include 2 EEG channels and 2 Non-EEG (ECG, Trigger) 
    # to perfectly test the `pick_types` filter logic.
    ch_names = ['EEG1', 'EEG2', 'ECG', 'Trigger']
    ch_types = ['eeg', 'eeg', 'ecg', 'stim']

    # Generate synthetic random noise
    data = np.random.randn(len(ch_names), n_samples)
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    
    # Save the file to Pytest's temporary session-scoped directory
    fn = tmp_path_factory.mktemp("data") / "test_recording.edf"
    mne.export.export_raw(str(fn), raw, fmt='edf', overwrite=True)
    
    return str(fn)
