import logging
from typing import Generator, Tuple
import numpy as np
from src.core.data_manager import DataManager

# Set up logger for tracking chunking progress
logger = logging.getLogger(__name__)

class SlidingWindowGenerator:
    """
    A generator class that chunks a continuous neural recording into 
    manageable overlapping windows (epochs). 
    
    It relies entirely on the 'DataManager' to do the heavy lifting of 
    reading from the hard drive. This perfectly separates the "IO logic" 
    from the "Math logic".
    """

    def __init__(self, data_manager: DataManager, window_size_sec: float, overlap_sec: float):
        """
        Initializes the generator logic.
        
        Args:
            data_manager (DataManager): An instantiated DataManager object.
            window_size_sec (float): How long each chunk should be (e.g., 60 seconds).
            overlap_sec (float): How much consecutive chunks should overlap (e.g., 10 seconds).
        """
        self.data_manager = data_manager
        self.window_size_sec = window_size_sec
        self.overlap_sec = overlap_sec
        
        # Validation checks to prevent math errors down the line
        if self.overlap_sec >= self.window_size_sec:
            raise ValueError("Overlap duration cannot be rigorously greater than or equal to window size.")
        
        if self.window_size_sec <= 0:
            raise ValueError("Window size must be greater than zero.")

        # Step size is how much we move forward after each window.
        # Example: 60s window with 10s overlap means we step forward 50 seconds each time.
        self.step_size_sec = self.window_size_sec - self.overlap_sec

    def generate(self) -> Generator[Tuple[float, float, np.ndarray, np.ndarray], None, None]:
        """
        A Python Generator that yields chunks iteratively.
        Using 'yield' instead of returning a massive list of chunks ensures
        we only keep exactly 1 chunk in the RAM at any given millisecond.
        
        Yields:
            start_sec (float): The actual starting timestamp of the chunk.
            end_sec (float): The actual ending timestamp of the chunk.
            data (np.ndarray): The 2D neural array (Channels x Samples).
            times (np.ndarray): The 1D array of exact timestamps.
        """
        current_start_sec = 0.0
        total_duration = self.data_manager.total_duration_sec
        
        logger.info(f"Starting sliding window: {self.window_size_sec}s size, {self.overlap_sec}s overlap.")
        
        # Loop until the start of our window passes the end of the recording
        while current_start_sec < total_duration:
            
            # Determine the actual duration for this chunk (handles truncation at end of file)
            time_remaining = total_duration - current_start_sec
            actual_duration = min(self.window_size_sec, time_remaining)
            current_end_sec = current_start_sec + actual_duration
            
            # Fetch a regular clean chunk using the DataManager
            data, times = self.data_manager.get_window(
                start_sec=current_start_sec, 
                duration_sec=actual_duration
            )
            
            # Hand the chunk back to whoever called this function,
            # pausing execution here until they ask for the next chunk.
            yield current_start_sec, current_end_sec, data, times
            
            # Step forward in time (Window Size - Overlap)
            current_start_sec += self.step_size_sec
        
        logger.info("Finished yielding all sliding windows.")
