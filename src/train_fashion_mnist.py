import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from dataclasses import asdict

from src.config import TrainConfig
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
    parser.add_argument('--config', type=str, help='Load config from JSON file')
    parser.add_argument('--save-config', type=str, help='Save effective config to JSON file')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint (best/last or file path)')
    
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--subset-train', type=int, default=None, help='Training subset size (None for full dataset)')
    parser.add_argument('--subset', type=int, default=None, help='Deprecated: use --subset-train instead')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', choices=['cpu', 'auto'], default='cpu', 
                       help='Device: cpu or auto (tries MPS/CUDA if available)')
    parser.add_argument('--outdir', type=str, default='artifacts', help='Output directory for checkpoints')
    args = parser.parse_args()
    
    # Build config from CLI arguments
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        subset_train=args.subset_train if args.subset_train is not None else args.subset,  # Handle deprecated --subset
        seed=args.seed,
        device=args.device,
        outdir=args.outdir
    )
    
    # Override with config file if provided
    if args.config:
        cfg = TrainConfig.from_json(args.config)
        # CLI args override config file
        if args.epochs != 1: cfg.epochs = args.epochs
        if args.batch_size != 128: cfg.batch_size = args.batch_size
        if args.subset_train is not None: cfg.subset_train = args.subset_train
        elif args.subset is not None: cfg.subset_train = args.subset
        if args.seed != 0: cfg.seed = args.seed
        if args.device != 'cpu': cfg.device = args.device
        if args.outdir != 'artifacts': cfg.outdir = args.outdir
    
    # Save config if requested
    if args.save_config:
        cfg.to_json(args.save_config)
        print(f"Saved config to {args.save_config}")
    
    device = get_device(cfg.device) 
    print(f"Using device: {device}")
    
    # Set up training with checkpointing
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    from src.checkpoint import save_checkpoint, load_checkpoint
    
    dl_train, dl_test = get_fashion_mnist_dataloaders(batch_size=cfg.batch_size, subset_train=cfg.subset_train, subset_test=None, seed=cfg.seed, data_dir=".data")
    model = TinyLinear().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    
    # Track best validation accuracy
    best_acc = -1.0
    start_epoch = 0
    
    # Resume from checkpoint if requested
    if args.resume:
        if args.resume == "best":
            resume_path = Path(cfg.outdir) / "checkpoints/best.pt"
        elif args.resume == "last":
            resume_path = Path(cfg.outdir) / "checkpoints/last.pt"
        else:
            resume_path = Path(args.resume)
        
        epoch, extra = load_checkpoint(resume_path, model, opt, map_location="cpu")
        start_epoch = epoch
        if 'config' in extra:
            print(f"Resumed from epoch {epoch} with config: {extra['config']}")
        else:
            print(f"Resumed from epoch {epoch}")
    
    # Training loop with checkpointing
    for epoch in range(start_epoch, cfg.epochs):
        train_loss = train_one_epoch(model, dl_train, opt, loss_fn, device)
        val_loss, val_acc = evaluate(model, dl_test, loss_fn, device)
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        # Save last checkpoint
        save_checkpoint(model, opt, epoch+1, cfg.outdir, tag="last",
                       config=asdict(cfg))
        
        # Save best checkpoint if validation accuracy improved
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, opt, epoch+1, cfg.outdir, tag="best",
                           config=asdict(cfg))


if __name__ == "__main__":
    main()