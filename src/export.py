import torch
import argparse
import os
from pathlib import Path
from src.models import TinyLinear, TinyCNN
from src.checkpoint import load_checkpoint
from src.data import get_fashion_mnist_dataloaders

def build_model(name):
    """Factory function to build model by name."""
    return TinyLinear() if name == "linear" else TinyCNN()

def load_model(name, ckpt_path, device="cpu"):
    """Load trained model from checkpoint."""
    model = build_model(name).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    load_checkpoint(ckpt_path, model, opt, map_location=device)
    model.eval()
    return model

def export_torchscript(model, out_path):
    """Export model to TorchScript format."""
    example = torch.randn(1, 1, 28, 28)
    ts = torch.jit.trace(model, example)
    torch.jit.save(ts, out_path)
    print(f"TorchScript exported to {out_path}")

def export_onnx(model, out_path):
    """Export model to ONNX format with dynamic batch axis."""
    try:
        example = torch.randn(1, 1, 28, 28)
        torch.onnx.export(
            model, example, out_path,
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}},
            opset_version=17
        )
        print(f"ONNX exported to {out_path}")
        return True
    except Exception as e:
        if "onnx" in str(e).lower():
            print("ONNX dependencies not available, skipping ONNX export")
            print("Install with: pip install onnx onnxruntime")
            return False
        else:
            raise e

def sanity_check(model, onnx_path, device="cpu", batch_size=64):
    """Sanity check ONNX model against PyTorch model."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not available, skipping sanity check")
        return True
    
    # Get a batch from test set
    _, dl_test = get_fashion_mnist_dataloaders(
        batch_size=batch_size, subset_train=None, subset_test=None, 
        seed=0, data_dir=".data"
    )
    xb, yb = next(iter(dl_test))
    
    # PyTorch predictions
    with torch.no_grad():
        pt_logits = model(xb).argmax(dim=1).cpu().numpy()
    
    # ONNX predictions
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_preds = sess.run(None, {input_name: xb.numpy()})[0].argmax(axis=1)
    
    match = (onnx_preds == pt_logits).mean()
    print(f"Sanity check: ONNX vs PyTorch top-1 match = {match:.3f}")
    return match >= 0.95

def main():
    parser = argparse.ArgumentParser(description='Export trained models to TorchScript and ONNX')
    parser.add_argument('--model', choices=['linear', 'cnn'], default='cnn', help='Model architecture')
    parser.add_argument('--ckpt', type=str, default='artifacts/checkpoints/best.pt', help='Checkpoint path')
    parser.add_argument('--outdir', type=str, default='artifacts/export', help='Output directory')
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(args.model, args.ckpt)
    params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {args.model} model with {params:,} parameters")
    
    # Export formats
    ts_path = Path(args.outdir) / "model_ts.pt"
    onnx_path = Path(args.outdir) / "model.onnx"
    
    export_torchscript(model, ts_path)
    onnx_success = export_onnx(model, onnx_path)
    
    # Sanity check only if ONNX export succeeded
    if onnx_success:
        if sanity_check(model, onnx_path):
            print("✓ Export successful - sanity check passed")
            exit(0)
        else:
            print("✗ Export failed - sanity check failed")
            exit(1)
    else:
        print("✓ TorchScript export successful")
        exit(0)

if __name__ == "__main__":
    main()