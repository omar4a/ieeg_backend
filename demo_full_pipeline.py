import os
import time
import logging
import psutil
from src.core.data_manager import DataManager
from src.core.sliding_window import SlidingWindowGenerator
from plugins.features.epileptogenicity_index import EpileptogenicityIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

def get_ram_mb():
    process = psutil.Process()
    # RSS: Resident Set Size (physical memory)
    return process.memory_info().rss / (1024 * 1024)

def run_demonstration():
    edf_path = r"C:\Omar\Work\GSoC_2026\iEEG\synthetic_stress_iEEG.edf"
    
    if not os.path.exists(edf_path):
        logging.error(f"Could not find 1.2GB Stress Test File at: {edf_path}")
        return

    logging.info("="*60)
    logging.info("🚀 INITIALIZING GSoC END-TO-END PIPELINE DEMONSTRATION")
    logging.info("="*60)
    
    # 1. Initialize the massive file safely without loading to RAM
    with DataManager(edf_path) as dm:
        sfreq = dm.raw.info['sfreq']
        channels = len(dm.raw.info['ch_names'])
        
        logging.info(f"📂 [Ingestion Matrix] Read Headers: {channels} Channels @ {sfreq}Hz")
        logging.info(f"🛡️ [Memory Guardian] Initial RAM Footprint: {get_ram_mb():.1f} MB")
        
        # 2. Instantiate the pure math Feature Extractor
        logging.info("🔌 [Registry Integration] Booting `EpileptogenicityIndex` Plugin...")
        extractor = EpileptogenicityIndex(sfreq=sfreq)
        
        # 3. Create the lazy Sliding Window Generator (10-second chunks, 0 overlap)
        window_size = 10.0
        generator = SlidingWindowGenerator(dm, window_size_sec=window_size, overlap_sec=0.0)
        
        logging.info("-" * 60)
        logging.info(f"⏳ STREAMING DATA (Window: {window_size}s). Executing AI Math Pipeline on first 5 chunks...")
        logging.info("-" * 60)
        
        # 4. The Runtime Execution Loop
        for i, (start_sec, end_sec, window_data, times) in enumerate(generator.generate()):
            if i >= 5:  # Only process the first 5 for the visual demo screenshot
                break
                
            start_time = time.perf_counter()
            
            # The exact injection point where massive raw arrays hit the math plugin
            payload = extractor.extract(window_data)
            
            compute_time = (time.perf_counter() - start_time) * 1000 # to ms
            
            # Extract JSON metric strictly for terminal visual proof
            ei_scores = payload["epileptogenicity_index"]
            ch_processed = payload["channels_processed"]
            max_score = max(ei_scores) if ei_scores else 0.0
            
            logging.info(f"⚡ Chunk {i+1:02d} | Compute: {compute_time:6.1f}ms | Out: {ch_processed:3d} JSON nodes (Max EI: {max_score:.3f}) | RAM: {get_ram_mb():.1f} MB")
            
    logging.info("="*60)
    logging.info("✅ PIPELINE COMPLETED: Safely released OS File Handles.")

if __name__ == "__main__":
    run_demonstration()
