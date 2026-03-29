import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
from src.core.base_models.base_pytorch import BasePyTorchModel

# ---------------------------------------------------------
# Test 1: Contract Validation (OOP Strictness)
# ---------------------------------------------------------
def test_contract_enforcement():
    """
    Proves that a model failing to implement the required formatting 
    method will instantly trigger an exception upon instantiation,
    protecting the GUI from receiving unpredictable data.
    """
    # BadModel inherits but forgets to implement '_format_output'
    class BadModel(BasePyTorchModel):
        def _build_model(self):
            return nn.Linear(10, 2)
        
        # Missing _format_output implementation!
        
    with pytest.raises(TypeError) as excinfo:
        BadModel("dummy_path.pt")
        
    assert "Can't instantiate abstract class" in str(excinfo.value)
    assert "_format_output" in str(excinfo.value)

# ---------------------------------------------------------
# Valid Mock Setup for Tests 2 & 3
# ---------------------------------------------------------
class MockEEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Simulates a network reducing 256 channel * 10 timepoints down to 2 classes
        self.fc = nn.Linear(256 * 10, 2) 

    def forward(self, x):
        # Flatten the incoming batch tensor for the Linear layer
        x = x.view(x.size(0), -1) 
        return self.fc(x)

class ValidModel(BasePyTorchModel):
    """A valid, compliant wrapper around MockEEGNet."""
    def _build_model(self) -> nn.Module:
        return MockEEGNet()
        
    def _format_output(self, raw_output: torch.Tensor) -> dict:
        # Convert the GPU tensor back to a standard Python float for the GUI
        probs = torch.softmax(raw_output, dim=1).squeeze(0).tolist()
        return {
            "normal_probability": probs[0],
            "spike_probability": probs[1]
        }

@pytest.fixture
def dummy_weights_path():
    """Generates a temporary dummy weights file to load during tests."""
    mock_net = MockEEGNet()
    path = os.path.join(tempfile.gettempdir(), "mock_weights.pt")
    torch.save(mock_net.state_dict(), path)
    return path

# ---------------------------------------------------------
# Test 2: VRAM Leak Prevention (Hardware Sandbox)
# ---------------------------------------------------------
def test_vram_leak_prevention(dummy_weights_path):
    """
    Mathematically proves that torch.no_grad() successfully disabled the
    computational graph during inference, preventing VRAM overflow in hospitals.
    """
    model = ValidModel(dummy_weights_path)
    # Simulate a single sliding window matrix (256 channels, 10 timepoints)
    fake_window = np.random.randn(256, 10)
    
    # To test the internal VRAM state, we temporarily override the _format_output
    # to just return the raw tensor before it gets stripped of its PyTorch attributes.
    model._format_output = lambda raw_out: raw_out 
    
    raw_tensor = model.predict(fake_window)
    
    # Assert the computational graph is entirely detached (no gradients tracked)
    assert raw_tensor.requires_grad is False

# ---------------------------------------------------------
# Test 3: GUI Serialization Safety (Pure JSON Translation)
# ---------------------------------------------------------
def test_gui_serialization_safety(dummy_weights_path):
    """
    Verifies that the final output exiting the wrapper contains absolutely
    zero PyTorch Tensors and is composed entirely of native Python primitives.
    """
    model = ValidModel(dummy_weights_path)
    fake_window = np.random.randn(256, 10)
    
    # Normal execution (uses the actual _format_output)
    result = model.predict(fake_window)
    
    # Assert it returns a pure Python dictionary
    assert isinstance(result, dict)
    
    # Assert the specific keys conform strictly to what the GUI expects
    assert "spike_probability" in result
    assert "normal_probability" in result
    
    # Assert the values are native Python floats, NOT torch.Tensors
    assert isinstance(result["spike_probability"], float)
    assert isinstance(result["normal_probability"], float)
    assert not isinstance(result["spike_probability"], torch.Tensor)
