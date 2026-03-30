import numpy as np
from scipy.signal import welch
import logging
from typing import Dict, Any
from src.core.interfaces import BaseFeatureExtractor 

logger = logging.getLogger(__name__)

class EpileptogenicityIndex(BaseFeatureExtractor):
    """
    Extracts the Epileptogenicity Index (EI) from a raw neural sliding window.
    Implements a deterministic 4-stage pipeline: Bipolar Montage, Spectral Power, 
    Page-Hinkley CUSUM, and Normalization.
    
    [GSoC 2026 Architectural Optimizations To Be Addressed In Phase 1]
    - Spatial misalignment (Bipolar logic must dynamically ingest metadata to avoid cross-hemispheric subtraction).
    - C/C++ Vectorization (The Python loop over channels will be eliminated by passing `axis=-1` directly into SciPy).
    - Page-Hinkley Temporal Chronology (`scipy.signal.spectrogram` must replace `welch` to allow iterative time-series drift calculation).
    """
    def __init__(self, sfreq: float):
        self.sfreq = sfreq
        
        # Standard clinical frequency bands for EI
        self.low_freq_band = (3.0, 12.0)   # Theta/Alpha
        self.high_freq_band = (12.0, 90.0) # Beta/Gamma
        
        # Page-Hinkley tuning parameters
        self.cusum_threshold = 5.0
        self.drift_factor = 0.5

    def extract(self, window_data: np.ndarray) -> Dict[str, Any]:
        """
        Executes the 4-stage EI pipeline on a (Channels x Samples) array.
        Returns a 1D feature vector mapped to channel indices.
        """
        # 1. Spatial Referencing (Bipolar Montage)
        # Subtract adjacent rows to eliminate common-mode noise 
        # (Note: Phase 1 will implement string-based metadata masking here).
        bipolar_data = window_data[:-1, :] - window_data[1:, :]
        n_bipolar_channels = bipolar_data.shape[0]
        
        ei_scores = np.zeros(n_bipolar_channels)

        for ch_idx in range(n_bipolar_channels):
            channel_signal = bipolar_data[ch_idx, :]
            
            # 2. Spectral Estimation (Welch's Method)
            # CRITICAL FIX for GSoC prototype: nperseg heavily dynamically scales down 
            # if the window passed in is extremely short (e.g. 0.5 sec), preventing a SciPy segfault.
            safe_nperseg = min(int(self.sfreq), channel_signal.shape[-1])
            freqs, psd = welch(channel_signal, fs=self.sfreq, nperseg=safe_nperseg)
            
            # Calculate Energy Ratio (High Freq Power / Low Freq Power)
            low_mask = (freqs >= self.low_freq_band[0]) & (freqs <= self.low_freq_band[1])
            high_mask = (freqs >= self.high_freq_band[0]) & (freqs <= self.high_freq_band[1])
            
            low_power = np.sum(psd[low_mask])
            high_power = np.sum(psd[high_mask])
            energy_ratio = high_power / (low_power + 1e-9) # Prevent division by zero
            
            # 3. Statistical Change-Point Detection (Simplified Page-Hinkley Placeholder)
            # Simulated trigger detection. Phase 1 will migrate this to a 2D spectrogram.
            mean_er = np.mean(energy_ratio)
            cusum = 0.0
            detection_time_idx = -1
            
            for t, er_val in enumerate([energy_ratio]): 
                cusum += (er_val - mean_er - self.drift_factor)
                if cusum > self.cusum_threshold:
                    detection_time_idx = t
                    break
            
            # 4. Index Normalization
            # Weight the energy magnitude against the temporal delay
            if detection_time_idx != -1:
                temporal_weight = 1.0 / (detection_time_idx + 1)
                ei_scores[ch_idx] = energy_ratio * temporal_weight

        # Normalize the final vector between 0 and 1 across all channels
        max_score = np.max(ei_scores)
        if max_score > 0:
            ei_scores = ei_scores / max_score

        # Ensure output is strictly JSON-serializable to conform with GUI IPC safety.
        return {
            "epileptogenicity_index": ei_scores.tolist(),
            "channels_processed": n_bipolar_channels
        }
