import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.data import get_fashion_mnist_dataloaders
from src.models import TinyLinear
from src.train_loop import train_one_epoch, evaluate


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


def run_once(*, epochs=1, batch_size=128, subset=2000, seed=0, device="cpu"):
    """Shared training/evaluation pipeline for both CLI and smoke test."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    from src.data import get_fashion_mnist_dataloaders
    from src.models import TinyLinear
    from src.train_loop import train_one_epoch, evaluate
    dl_train, dl_test = get_fashion_mnist_dataloaders(batch_size=batch_size, subset=subset, seed=seed, data_dir=".data")
    model = TinyLinear().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    # initial loss on train (eval/no_grad)
    model.eval()
    initial_loss, _ = evaluate(model, dl_train, loss_fn, device)
    # train epochs
    final_train_loss = None
    for _ in range(epochs):
        final_train_loss = train_one_epoch(model, dl_train, opt, loss_fn, device)
    # eval on test
    val_loss, val_acc = evaluate(model, dl_test, loss_fn, device)
    return initial_loss, final_train_loss, val_loss, val_acc


def smoke_test():
    """Smoke test: quick training to verify setup works correctly.
    
    Returns:
        bool: True if training reduces loss and achieves reasonable accuracy
    """
    initial, final_train, val_loss, val_acc = run_once(epochs=1, batch_size=128, subset=2000, seed=0, device="cpu")
    print(f"smoke: initial_loss={initial:.4f} final_train_loss={final_train:.4f} val_acc={val_acc:.4f}")
    return (final_train <= 0.9 * initial) and (val_acc >= 0.60)


def main():
    parser = argparse.ArgumentParser(description='Train TinyLinear on FashionMNIST')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--subset', type=int, default=2000, help='Subset size (None for full dataset)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', choices=['cpu', 'auto'], default='cpu', 
                       help='Device: cpu or auto (tries MPS/CUDA if available)')
    args = parser.parse_args()
    
    device = get_device(args.device) 
    print(f"Using device: {device}")
    
    initial, final_train, val_loss, val_acc = run_once(
        epochs=args.epochs, batch_size=args.batch_size, subset=args.subset, seed=args.seed, device=str(device)
    )
    print(f"Epoch {args.epochs}: train_loss={final_train:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")


if __name__ == "__main__":
    main()