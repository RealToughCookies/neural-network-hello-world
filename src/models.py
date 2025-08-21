import torch.nn as nn


class TinyLinear(nn.Module):
    """Minimal linear classifier for FashionMNIST.
    
    Flattens 28x28 images and applies a linear layer to 10 classes.
    """
    
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(28 * 28, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)