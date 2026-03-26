# iEEG Backend - INCF GSoC 2026

Welcome! This repository houses the core backend for the iEEG management suite. 

## 📂 Project Architecture

The codebase strictly adheres to OOP and SOLID principles to guarantee maintainability and seamless GUI integration:

```text
ieeg_backend/
├── README.md                   # Project overview, installation, and plugin guide
├── pyproject.toml              # Python build system configuration
├── requirements.txt            # Explicit dependency locking (e.g., torch, mne)
├── .gitignore                  # Excludes heavy clinical workspace data
│
├── src/                        # Immutable source code (Python package root)
│   ├── core/                   # Core pipeline engine (Closed for modification)
│   │   ├── __init__.py
│   │   ├── interfaces.py       # Pure ABC contracts (BasePreProcessor, BaseModel)
│   │   ├── base_models/        # Framework adapters (PyTorch/Sklearn wrappers)
│   │   │   ├── base_pytorch.py
│   │   │   └── base_classical.py
│   │   ├── data_manager.py     # MNE lazy-loading and BIDS parsing
│   │   ├── sliding_window.py   # Continuous data chunking generator
│   │   ├── dynamic_loader.py   # Importlib scanner for the plugins directory
│   │   └── async_workers.py    # Multithreading logic for non-blocking execution
│   │
│   ├── bridge/                 # GUI Integration Layer (IPC)
│   │   ├── __init__.py
│   │   ├── event_handlers.py   # Receives execution triggers from the frontend
│   │   └── schemas.py          # Pydantic models enforcing JSON payload structure
│   │
│   └── benchmarking/           # Clinical Evaluation Suite
│       ├── __init__.py
│       ├── cross_validator.py  # Grouped patient-level cross-validation
│       └── metric_calculator.py# Metrics robust to class imbalance (PR-AUC, F1)
│
├── plugins/                    # Dynamic Registry (Open for extension)
│   ├── preprocessors/          # Drop-in cleaning algorithms (Notch, Bipolar)
│   ├── features/               # Drop-in extraction math (HFOs, DWT)
│   └── models/                 # Drop-in inference scripts (EEGSurvNet, XGBoost)
│
├── workspace/                  # Local clinical data outputs (Git-ignored)
│   ├── derivatives/            # Standardized BIDS events.tsv (HITL annotations)
│   └── reports/                # Exported benchmarking JSONs and clinical PDFs
│
├── weights/                    # Heavy serialized model weights (Git-ignored)
│   └── eegsurvnet_v1.pt
│
├── tests/                      # Pytest Suite
│   ├── test_sliding_window.py
│   ├── test_dynamic_loader.py
│   └── test_contracts.py       # Enforces plugin compliance with ABC rules
│
└── main.py                     # Entry point that initializes the engine and UI bridge
```

## Current Implementation Status

At this initial phase, we have successfully implemented and verified the foundational **Data Ingestion Engine**.

- **`src/core/data_manager.py`**: A robust OOP wrapper natively reading massive multi-GB iEEG files using memory pointers (via `mne.io.read_raw(preload=False)`). Includes rigorous validation mapping exact neural channels (`eeg`, `seeg`, `ecog`) safely using `mne.pick_types`, preventing runtime failures from inconsistent hospital naming schemes. It leverages a Python Context Manager to lock/unlock OS file handles.
- **`src/core/sliding_window.py`**: A pure Python Generator that ingests the `DataManager` stream and slices it into $O(1)$ memory chunks using configurable offsets. It eliminates arbitrary end-of-file truncation branching by dynamically calculating duration limits.
- **The Pipeline**: Together, the `DataManager` pulls metadata and passes it to the `SlidingWindowGenerator`, which invokes `get_window()` to stream isolated arrays from the disk *only* when the memory loop yields.

*(Sections detailing the ABC plugin contracts, PyTorch models, and async bridge will be populated as the registry and UI integration layers are finalized.)*

## 🧪 Comprehensive Testing Results

The data ingestion pipeline has been subjected to comprehensive testing, edge-case validation, and memory profiling. 

- **Unit Tests (100% Coverage)**: The modules pass `pytest` with **100% coverage** over the core modules. The dynamic assertions lock down zero-length windows, negative overlap parameters, out-of-bounds duration requests, and fractional end-of-file truncation rounding.
- **1.2GB Clinical Stress Test**: The pipeline executed a continuous MProf heartbeat trace over a 1.2GB `.edf` file, simulating chunks scaling up to 172 simultaneous data channels.
- **RAM Resilience**: Memory usage proved entirely resilient. The RAM consumption idled strictly at a `~147 MB` baseline. During disk-read bursts, memory spiked but immediately compressed back down to exactly `147 MB` via aggressive garbage collection (`gc.collect()`), avoiding the risk of memory overflow regardless of recording length.
