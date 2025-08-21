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


def run_once(*, epochs=1, batch_size=128, subset_train=None, seed=0, device="cpu"):
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
    dl_train, dl_test = get_fashion_mnist_dataloaders(batch_size=batch_size, subset_train=subset_train, subset_test=None, seed=seed, data_dir=".data")
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
    return initial_loss, final_train_loss, val_loss, val_acc, model, opt


def smoke_test():
    """Smoke test: quick training to verify setup works correctly.
    
    Returns:
        bool: True if training reduces loss and achieves reasonable accuracy
    """
    initial, final_train, val_loss, val_acc, _, _ = run_once(epochs=2, batch_size=128, subset_train=2000, seed=0, device="cpu")
    print(f"smoke: initial_loss={initial:.4f} final_train_loss={final_train:.4f} val_acc={val_acc:.4f}")
    return (final_train <= 0.9 * initial) and (val_acc >= 0.60)


def main():
    parser = argparse.ArgumentParser(description='Train TinyLinear on FashionMNIST')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--subset-train', type=int, default=None, help='Training subset size (None for full dataset)')
    parser.add_argument('--subset', type=int, default=None, help='Deprecated: use --subset-train instead')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', choices=['cpu', 'auto'], default='cpu', 
                       help='Device: cpu or auto (tries MPS/CUDA if available)')
    parser.add_argument('--outdir', type=str, default='artifacts', help='Output directory for checkpoints')
    args = parser.parse_args()
    
    # Handle deprecated --subset argument
    subset_train = args.subset_train if args.subset_train is not None else args.subset
    
    device = get_device(args.device) 
    print(f"Using device: {device}")
    
    # Set up training with checkpointing
    import random
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    from src.data import get_fashion_mnist_dataloaders
    from src.models import TinyLinear
    from src.train_loop import train_one_epoch, evaluate
    from src.checkpoint import save_checkpoint
    
    dl_train, dl_test = get_fashion_mnist_dataloaders(batch_size=args.batch_size, subset_train=subset_train, subset_test=None, seed=args.seed, data_dir=".data")
    model = TinyLinear().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Track best validation accuracy
    best_acc = -1.0
    
    # Training loop with checkpointing
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, dl_train, opt, loss_fn, device)
        val_loss, val_acc = evaluate(model, dl_test, loss_fn, device)
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        # Save last checkpoint
        save_checkpoint(model, opt, epoch+1, args.outdir, tag="last",
                       subset_train=subset_train, seed=args.seed, lr=0.1)
        
        # Save best checkpoint if validation accuracy improved
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, opt, epoch+1, args.outdir, tag="best",
                           subset_train=subset_train, seed=args.seed, lr=0.1)


if __name__ == "__main__":
    main()