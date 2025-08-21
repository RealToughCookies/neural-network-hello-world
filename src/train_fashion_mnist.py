import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data import get_fashion_mnist_dataloaders
from models import TinyLinear
from train_loop import train_one_epoch, evaluate


def set_seeds(seed):
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(device_arg):
    """Get device based on argument and availability."""
    if device_arg == "cpu":
        return torch.device("cpu")
    elif device_arg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_arg)


def smoke_test():
    """Smoke test: quick training to verify setup works correctly.
    
    Returns:
        bool: True if training reduces loss and achieves reasonable accuracy
    """
    set_seeds(0)
    device = torch.device("cpu")  # Use CPU for consistent smoke test
    
    # Get small subset for fast testing
    train_dl, test_dl = get_fashion_mnist_dataloaders(
        batch_size=128, subset=2000, seed=0
    )
    
    # Setup model, loss, optimizer
    model = TinyLinear().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Get initial loss on first batch
    model.eval()
    with torch.no_grad():
        first_batch = next(iter(train_dl))
        data, target = first_batch[0].to(device), first_batch[1].to(device)
        initial_loss = loss_fn(model(data), target).item()
    
    # Train for 1 epoch
    final_train_loss = train_one_epoch(model, train_dl, optimizer, loss_fn, device)
    val_loss, val_acc = evaluate(model, test_dl, loss_fn, device)
    
    # Check if training worked: loss decreased and accuracy is reasonable
    loss_decreased = final_train_loss <= 0.9 * initial_loss
    accuracy_ok = val_acc >= 0.60
    
    return loss_decreased and accuracy_ok


def main():
    parser = argparse.ArgumentParser(description='Train TinyLinear on FashionMNIST')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--subset', type=int, default=2000, help='Subset size (None for full dataset)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', choices=['cpu', 'auto'], default='cpu', 
                       help='Device: cpu or auto (tries MPS/CUDA if available)')
    args = parser.parse_args()
    
    set_seeds(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Get dataloaders
    train_dl, test_dl = get_fashion_mnist_dataloaders(
        batch_size=args.batch_size, subset=args.subset, seed=args.seed
    )
    
    # Setup model, loss, optimizer
    model = TinyLinear().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Training loop
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_dl, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, test_dl, loss_fn, device)
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")


if __name__ == "__main__":
    main()