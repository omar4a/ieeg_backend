from memory_profiler import profile
from src.core.data_manager import DataManager
import time
import gc

# A small delay to make the steps clearly visible on the final line graph
DELAY = 1.0 

@profile
def run_lazy_load_test(file_path: str):
    print("--- Starting 1.2GB Stress Baseline ---")
    time.sleep(DELAY) 
    
    # Using the context manager ensures the file handle is locked only when needed
    with DataManager(file_path) as manager:
        print("\n1. Initialized DataManager (Headers only). Memory should remain strictly flat.")
        time.sleep(DELAY)
        
        # Simulate a sliding window moving across the huge file for 10 chunks
        # This will prove the RAM usage doesn't staircase up the longer the file runs
        for i in range(10):
            start_time = 0 + (i * 60)
            print(f"\n-> Fetching 60-second chunk at {start_time}s...")
            
            # This safely pulls directly from the hard drive
            chunk, _ = manager.get_window(start_sec=start_time, duration_sec=60)
            print(f"   Loaded array: {chunk.shape}")
            
            # Simulate heavy processing time
            time.sleep(0.5) 
            
            # Explicitly delete the variable and force garbage collection
            del chunk
            gc.collect()
            time.sleep(DELAY)

    print("\n--- Test Complete (File Handle Released) ---")

if __name__ == "__main__":
    # Point directly to the user's massive 1.2GB test file
    TEST_FILE = r"C:\Omar\Work\GSoC_2026\iEEG\synthetic_stress_iEEG.edf"
    run_lazy_load_test(TEST_FILE)
