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


class TinyCNN(nn.Module):
    """Minimal CNN for FashionMNIST.
    
    Standard Conv→ReLU→Pool architecture: 1x28x28 → 32→64 features → 10 classes.
    """
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
            nn.Flatten(start_dim=1),
            nn.Linear(64*7*7, 10),
        )
    
    def forward(self, x):
        return self.net(x)