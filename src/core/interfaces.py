from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any

class BaseModel(ABC):
    """
    The absolute contract for all AI inference models in the registry.
    Ensures the GUI receives a predictable JSON-serializable dictionary
    regardless of whether the underlying math is PyTorch, ONNX, XGBoost, or other.
    """
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Executes the inference logic.
        
        Args:
            features (np.ndarray): The 1D or 2D array outputted by the FeatureExtractor.
            
        Returns:
            Dict[str, Any]: A JSON-safe dictionary (e.g., {"soz_probability": 0.85})
                            to be streamed back to the Main GUI Thread via IPC.
        """
        pass

class BaseFeatureExtractor(ABC):
    """
    Contract for all deterministic mathematical feature extraction plugins.
    Ensures that any signal processing pipeline yields standardized dictionaries 
    parseable by both Models (like EEGSurvNet) and the GUI.
    """
    @abstractmethod
    def extract(self, window_data: np.ndarray) -> Dict[str, Any]:
        """
        Extracts mathematical features from a sliding window array.

        Args:
            window_data: A (Channels x Samples) numpy array.

        Returns:
            Dict[str, Any]: A JSON-serializable dictionary with extracted feature arrays.
        """
        pass
