import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from src.metrics import FASHION_LABELS
from src.data import get_fashion_mnist_dataloaders

# Preprocessing transform (matches training)
tx = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def load_image(image_path):
    """Load and preprocess image for inference."""
    img = Image.open(image_path).convert('L')  # Grayscale
    img = img.resize((28, 28))  # FashionMNIST size
    tensor = tx(img).unsqueeze(0)  # [1, 1, 28, 28]
    return tensor

def load_sample(sample_idx):
    """Load sample from FashionMNIST test set."""
    _, dl_test = get_fashion_mnist_dataloaders(
        batch_size=10000, subset_train=None, subset_test=None,
        seed=0, data_dir=".data"
    )
    xb, yb = next(iter(dl_test))  # Get all test data
    if sample_idx >= len(xb):
        raise ValueError(f"Sample index {sample_idx} >= test set size {len(xb)}")
    
    sample = xb[sample_idx:sample_idx+1]  # [1, 1, 28, 28]
    true_label = int(yb[sample_idx])
    return sample, true_label

def predict_torchscript(model_file, xb):
    """Run inference with TorchScript backend."""
    model = torch.jit.load(model_file, map_location="cpu")
    model.eval()
    with torch.no_grad():
        logits = model(xb)
        prob = torch.softmax(logits, dim=1)
    return prob

def predict_onnx(model_file, xb):
    """Run inference with ONNX Runtime backend."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime not available. Install with: pip install onnxruntime")
    
    sess = ort.InferenceSession(model_file, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    logits = sess.run(None, {input_name: xb.numpy()})[0]
    prob = torch.from_numpy(logits).softmax(1)
    return prob

def main():
    parser = argparse.ArgumentParser(description='Run inference on exported models')
    parser.add_argument('--backend', choices=['ts', 'onnx'], default='ts', help='Inference backend')
    parser.add_argument('--model-file', type=str, help='Model file path (auto-detected if not specified)')
    parser.add_argument('--image', type=str, help='Input image path (PNG/JPG)')
    parser.add_argument('--sample', type=int, help='Test set sample index')
    parser.add_argument('--compare', action='store_true', help='Compare TorchScript vs ONNX predictions')
    args = parser.parse_args()
    
    # Auto-detect model file if not specified
    if not args.model_file:
        if args.backend == 'ts':
            args.model_file = 'artifacts/export/model_ts.pt'
        else:
            args.model_file = 'artifacts/export/model.onnx'
    
    # Load input
    if args.image:
        xb = load_image(args.image)
        true_label = None
        print(f"Input: {args.image}")
    elif args.sample is not None:
        xb, true_label = load_sample(args.sample)
        print(f"Input: test sample {args.sample} (true={FASHION_LABELS[true_label]})")
    else:
        raise ValueError("Specify either --image or --sample")
    
    # Run inference
    if args.compare:
        prob_ts = predict_torchscript('artifacts/export/model_ts.pt', xb)
        prob_onnx = predict_onnx('artifacts/export/model.onnx', xb)
        pred_ts = int(prob_ts.argmax(1))
        pred_onnx = int(prob_onnx.argmax(1))
        print(f"TorchScript: {FASHION_LABELS[pred_ts]} (prob={prob_ts[0,pred_ts]:.3f})")
        print(f"ONNX: {FASHION_LABELS[pred_onnx]} (prob={prob_onnx[0,pred_onnx]:.3f})")
        print(f"Match: {pred_ts == pred_onnx}")
    else:
        if args.backend == 'ts':
            prob = predict_torchscript(args.model_file, xb)
        else:
            prob = predict_onnx(args.model_file, xb)
        
        pred = int(prob.argmax(1))
        conf = float(prob[0, pred])
        print(f"Prediction: {FASHION_LABELS[pred]} (prob={conf:.3f})")

if __name__ == "__main__":
    main()