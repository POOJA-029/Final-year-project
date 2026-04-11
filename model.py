import torch
import torch.nn as nn
import os

class SimpleClassifier(nn.Module):
    """PyTorch Neural Network for Tabular Classification"""
    def __init__(self, input_dim):
        super(SimpleClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

def quantize_model(model):
    """Applies Post-Training Dynamic Quantization (Green AI) to shrink model to INT8."""
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    return quantized_model

def get_model_size(model):
    """Calculates model size in KB"""
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1024
    os.remove("temp.p")
    return size
