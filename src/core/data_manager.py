import logging
from typing import Tuple, List, Optional
import numpy as np
import mne

# Set up a logger for this module to track what it's doing
logger = logging.getLogger(__name__)

class DataManager:
    """
    A robust wrapper around MNE-Python for handling massive iEEG files.
    
    The primary job of this class is to open the file, read the metadata (headers),
    and allow us to fetch specific time chunks cleanly without putting the entire 
    file into the computer's RAM.
    
    It supports Context Managers (`with DataManager(...) as dm:`) to guarantee
    that the file is closed properly when we're done, preventing Windows File Locks.
    """
    
    def __init__(self, file_path: str, pick_types: Optional[List[str]] = None):
        """
        Initializes the DataManager and reads the file headers.
        
        Args:
            file_path (str): The absolute path to the data file (.edf, .vhdr, .mef3, etc.)
            pick_types (List[str], optional): If provided, MNE will ONLY look at these 
                types of channels (e.g., ["eeg", "seeg", "ecog"]). This saves RAM by 
                ignoring useless trigger or empty channels on the hard drive.
        """
        self.file_path = file_path
        
        # We default to picking ieeg channels if nothing is specified.
        # This prevents bloating the memory with non-neural channels.
        if pick_types is None:
            pick_types = ['eeg', 'seeg', 'ecog']
            
        logger.info(f"Opening file: {self.file_path}")

        # By setting preload=False, MNE reads the header information (like channel names 
        # and sampling frequency) but leaves the heavy binary data sitting on the hard drive.
        # It automatically infers the format (EDF, BrainVision, etc) from the file extension.
        self.raw = mne.io.read_raw(self.file_path, preload=False, verbose='ERROR')
        
        # Filter down to the specific channels we actually care about
        mne_pick_kwargs = {ch_type: True for ch_type in pick_types}
        valid_indices = mne.pick_types(self.raw.info, **mne_pick_kwargs)
        self.raw.pick(picks=valid_indices)
        
        # Extract basic metadata
        self.sfreq = self.raw.info['sfreq']                  # Sampling frequency (Hz)
        self.ch_names = self.raw.ch_names                    # List of channel names
        self.n_channels = len(self.ch_names)                 # Total number of valid channels
        self.total_duration_sec = self.raw.times[-1]         # How long the recording is in seconds
        
        logger.info(
            f"DataManager Ready: {self.n_channels} channels, "
            f"{self.sfreq}Hz, {self.total_duration_sec:.2f}s total duration."
        )

    def __enter__(self):
        """
        This allows other developers to use this class cleanly like this:
            with DataManager("file.edf") as manager:
                manager.get_window(...)
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        When the 'with' block ends or crashes, this is automatically called.
        It guarantees the file handle is given back to the Operating System.
        """
        self.close()

    def close(self):
        """
        Explicitly closes the MNE raw object, releasing the OS file lock.
        """
        if self.raw is not None:
            self.raw.close()
            logger.debug(f"Successfully closed file handle for {self.file_path}")

    def get_window(self, start_sec: float, duration_sec: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fetches a specific chunk of time from the hard drive into RAM.
        
        Args:
            start_sec (float): When to start reading (in seconds).
            duration_sec (float): How much time to read (in seconds).
            
        Returns:
            data (np.ndarray): The raw neural data. Shape: (Channels, Samples)
            times (np.ndarray): The corresponding timestamps for each sample.
        """
        # Safety Check: Did the user ask for time that doesn't exist?
        if start_sec + duration_sec > self.total_duration_sec:
            logger.warning("Requested window exceeds the file limit. Truncating to the end of the file.")
            duration_sec = self.total_duration_sec - start_sec
            
        if duration_sec <= 0:
            raise ValueError("Invalid duration or start time beyond the end of the file.")

        # Convert time in seconds to exact index integers based on the Sampling Frequency
        start_idx = int(start_sec * self.sfreq)
        duration_idx = int(duration_sec * self.sfreq)
        end_idx = start_idx + duration_idx

        # ---------------------------------------------------------
        # THIS IS WHERE RAM IS FINALLY USED
        # ---------------------------------------------------------
        # MNE automatically calculates the exact byte offset on the hard drive
        # and loads ONLY this specific (Channels x Samples) matrix into memory.
        data, times = self.raw[:, start_idx:end_idx]
        
        return data, times
