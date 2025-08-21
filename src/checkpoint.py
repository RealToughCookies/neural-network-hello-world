from pathlib import Path
import torch
import json


def save_checkpoint(model, optimizer, epoch, outdir, tag, **extra):
    """Save model and optimizer state to checkpoint file.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer (can be None)
        epoch: Current epoch number
        outdir: Output directory for checkpoints
        tag: Checkpoint tag (e.g., 'best', 'last')
        **extra: Additional metadata to save
    
    Returns:
        str: Path to saved checkpoint
    """
    outdir = Path(outdir) / "checkpoints"
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "extra": extra or {},
        "pytorch_version": torch.__version__,
    }
    path = outdir / f"{tag}.pt"
    torch.save(payload, path)
    return str(path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    """Load model and optimizer state from checkpoint file.
    
    Args:
        path: Path to checkpoint file
        model: PyTorch model to load state into
        optimizer: PyTorch optimizer to load state into (optional)
        map_location: Device to load checkpoint on
    
    Returns:
        tuple: (epoch, extra_metadata)
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("extra", {})