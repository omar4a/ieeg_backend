from abc import abstractmethod
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any
from src.core.interfaces import BaseModel

logger = logging.getLogger(__name__)

class BasePyTorchModel(BaseModel):
    """
    A protected framework wrapper that implements the BaseModel contract.
    It sandboxes PyTorch execution, handling device management (CUDA/MPS/CPU),
    Tensor casting, batch dimension safety, and VRAM protection.
    """
    
    def __init__(self, weights_path: str):
        # Hardware Awareness: Automatically detect the best available hardware.
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        logger.info(f"Initializing PyTorch Model on device: {self.device}")
        
        # OOP State Safety: Force child classes to define the network structure
        # before this wrapper attempts to load heavy binaries.
        self.model = self._build_model()
        if not isinstance(self.model, nn.Module):
            raise TypeError("_build_model must return a valid torch.nn.Module.")
            
        self.weights_path = weights_path
        self._load_weights()

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """
        Child classes (e.g., EEGSurvNet) MUST override this method to instantiate 
        and return their specific PyTorch neural network architecture.
        """
        pass

    def _load_weights(self):
        """Safely loads heavy binary weights from the /weights/ directory."""
        # map_location ensures a model trained on a GPU cluster can run on a hospital CPU or Mac
        self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval() # Strictly enforce evaluation mode (disables dropout/batchnorm)

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Implements the ABC contract. Translates standard NumPy arrays into 
        PyTorch Tensors, executes inference safely, and translates back to Python.
        """
        # Dimensionality: PyTorch requires [Batch, Channels, Time]. 
        tensor_features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # VRAM Protection: Disable gradient tracking to prevent memory leaks.
        with torch.no_grad():
            raw_output = self.model(tensor_features)
            
        # Translate the PyTorch output back into a standard JSON dictionary for the GUI
        return self._format_output(raw_output)

    @abstractmethod
    def _format_output(self, raw_output: torch.Tensor) -> Dict[str, Any]:
        """
        Child classes (like EEGSurvNet) MUST implement this to convert their 
        specific Tensor output into the standard JSON dictionary.
        """
        pass
