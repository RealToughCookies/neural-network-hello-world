import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def set_deterministic_seed(seed, deterministic=False):
    """Set deterministic random seed for reproducible results.
    
    Args:
        seed: Random seed value
        deterministic: If True, enables full deterministic mode for maximum reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def generate_toy_data(n_samples, seed=0):
    """Generate toy data: x ~ N(0,1), y = 3x + 2 + noise."""
    torch.manual_seed(seed)
    x = torch.randn(n_samples, 1)
    noise = 0.1 * torch.randn(n_samples, 1)
    y = 3 * x + 2 + noise
    return x, y


def train_model(x, y, epochs=200, lr=0.1):
    """Train a linear model to learn y = wx + b."""
    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    # Extract learned parameters
    w = model.weight.item()
    b = model.bias.item()
    final_loss = loss.item()
    
    return w, b, final_loss


def smoke_test(deterministic=False):
    """Smoke test: train quickly and verify parameters are close to target."""
    set_deterministic_seed(0, deterministic=deterministic)
    x, y = generate_toy_data(256, seed=0)
    w, b, loss = train_model(x, y, epochs=200, lr=0.1)
    
    # Check if learned parameters are close to target (w=3, b=2)
    w_ok = abs(w - 3) < 0.2
    b_ok = abs(b - 2) < 0.2
    loss_ok = loss < 0.05
    
    return w_ok and b_ok and loss_ok


def main():
    parser = argparse.ArgumentParser(description='Train a simple neural network on toy data')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--n', type=int, default=256, help='Number of data samples')
    parser.add_argument('--deterministic', action='store_true', help='Enable full deterministic mode for reproducible results')
    args = parser.parse_args()
    
    set_deterministic_seed(args.seed, deterministic=args.deterministic)
    x, y = generate_toy_data(args.n, seed=args.seed)
    w, b, loss = train_model(x, y, epochs=args.epochs, lr=args.lr)
    
    print(f"Final parameters: w={w:.4f}, b={b:.4f}, loss={loss:.6f}")


if __name__ == "__main__":
    main()