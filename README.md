# Neural Network Hello World

A minimal PyTorch implementation featuring both synthetic data regression and FashionMNIST classification. Includes linear regression learning y ≈ 3x + 2 on toy data, and a TinyLinear classifier for FashionMNIST with comprehensive training/evaluation loops and smoke tests.

## Usage

Train the model:
```bash
python src/hello_nn.py --seed 0
```

Run smoke test:
```bash
python -c "import src.hello_nn as m; print(m.smoke_test())"
```

## Deterministic Runs

For fully reproducible results across different runs and platforms, use the `--deterministic` flag:

```bash
python src/hello_nn.py --seed 0 --deterministic
```

The deterministic mode enables PyTorch's deterministic algorithms and sets seeds for Python, NumPy, and PyTorch to ensure identical results on repeated runs.

## FashionMNIST Classification

The FashionMNIST dataset will auto-download to `.data/` on first run. Run commands use `python -m` to execute within the package system.

Train TinyLinear classifier (default: CPU, full dataset):
```bash
python -m src.train_fashion_mnist --epochs 1 --subset-train 2000
```

Use GPU/MPS acceleration with subset training:
```bash
python -m src.train_fashion_mnist --epochs 5 --device auto --subset-train 5000
```

Run FashionMNIST smoke test:
```bash
python -c "import src.train_fashion_mnist as t; print(t.smoke_test())"
```

The smoke test trains for 2 epochs on 2000 training samples, evaluates on the full 10k test set, and verifies that training reduces loss by ≥10% and validation accuracy ≥60%. Evaluation uses `model.eval()` and `torch.no_grad()` for proper inference mode. The deprecated `--subset` flag is mapped to `--subset-train`.

## Model Architectures

Choose between linear classifier and CNN:

```bash
# Train linear classifier (fast, ~7k params)
python -m src.train_fashion_mnist --epochs 1 --subset-train 2000 --model linear

# Train CNN (stronger, ~3M params)
python -m src.train_fashion_mnist --epochs 2 --subset-train 2000 --model cnn
```

**Verification commands:**
```bash
python -m src.train_fashion_mnist --epochs 1
# Test CNN model (targets ≥70% accuracy)
python -c "import src.train_fashion_mnist as t; print(t.smoke_test('cnn'))"
```

## Checkpoints

Train with checkpoint saving (saves best and last models):
```bash
python -m src.train_fashion_mnist --epochs 2 --batch-size 128 --subset-train 2000 --seed 0 --device cpu --outdir artifacts
```

Checkpoints are saved to `{outdir}/checkpoints/` as `best.pt` and `last.pt`. Quick verify checkpoint loading:
```bash
python -c "
from src.checkpoint import load_checkpoint; from src.models import TinyLinear
model = TinyLinear(); epoch, extra = load_checkpoint('artifacts/checkpoints/best.pt', model)
print(f'Loaded epoch {epoch}, extra: {extra}')
"
```

**Note**: PyTorch 2.6+ sets `torch.load` default `weights_only=True` for security. We use `weights_only=False` for trusted local checkpoint files to avoid unpickling errors. See [PyTorch docs](https://pytorch.org/docs/stable/generated/torch.load.html) for details.

## Config & Resume

Save training config to JSON and resume from checkpoints:

```bash
# Save config while training
python -m src.train_fashion_mnist --epochs 2 --subset-train 2000 --device cpu --outdir artifacts --save-config artifacts/run.json

# Resume from best checkpoint using saved config
python -m src.train_fashion_mnist --config artifacts/run.json --resume best

# Resume from specific checkpoint file
python -m src.train_fashion_mnist --config artifacts/run.json --resume artifacts/checkpoints/last.pt
```

## Logging & Plots

Track training metrics with CSV logging and generate plots:

```bash
# Train with CSV logging and deterministic mode
python -m src.train_fashion_mnist --epochs 5 --subset-train 2000 --deterministic --log-csv artifacts/metrics.csv

# Train with plotting enabled
python -m src.train_fashion_mnist --epochs 5 --subset-train 2000 --plot
```

CSV metrics are logged using stdlib `csv` module. Plots are generated with `matplotlib.pyplot.savefig()` for headless operation. The `--deterministic` flag enables PyTorch's deterministic algorithms for maximum reproducibility.

## Confusion Matrix & Per-Class Metrics

Generate confusion matrix and per-class accuracy analysis:

```bash
# Train with confusion matrix analysis
python -m src.train_fashion_mnist --epochs 2 --subset-train 2000 --device cpu --confusion-matrix

# Combined with other features
python -m src.train_fashion_mnist --epochs 5 --subset-train 5000 --deterministic --log-csv artifacts/metrics.csv --plot --confusion-matrix
```

Outputs:
- `artifacts/metrics_confusion.csv`: Confusion matrix with FashionMNIST class labels
- `artifacts/plots/confusion_matrix.png`: Heatmap visualization
- Console: Per-class accuracy breakdown

Uses proper evaluation mode (`model.eval()` + `torch.no_grad()`) for accurate metrics computation.

## Hyperparameter Tuning

Run grid search for TinyCNN hyperparameters:

```bash
# Default grid search (small and fast)
python -m src.tune

# Quick smoke test
python -c "import src.tune as t; print(t.smoke_test())"
```

Outputs:
- `artifacts/tuning.csv`: Grid search results with hyperparameters and metrics
- Console: Top-3 configurations sorted by validation accuracy

Uses `itertools.product` for grid combinations and stdlib `csv` for logging. Focuses on SGD hyperparameters: learning rate, momentum, weight decay, and batch size.

## Export

Export trained models to TorchScript and ONNX formats:

```bash
# Install export dependencies (one-time)
pip install onnx onnxruntime

# Export best CNN checkpoint
python -m src.export --model cnn --ckpt artifacts/checkpoints/best.pt --outdir artifacts/export

# Export linear model
python -m src.export --model linear --ckpt artifacts/checkpoints/best.pt --outdir artifacts/export
```

Outputs:
- `artifacts/export/model_ts.pt`: TorchScript format for C++ deployment
- `artifacts/export/model.onnx`: ONNX format for cross-framework inference
- Console: ONNX Runtime sanity check comparing predictions with PyTorch

TorchScript enables deployment without Python in C++ environments. ONNX provides cross-framework compatibility with ONNX Runtime for efficient CPU/GPU inference.

## Run as Module

All FashionMNIST commands use `python -m` to run within the package import system, which resolves the package-absolute imports correctly. This avoids ModuleNotFoundError issues when importing between src modules. Both the CLI and smoke test use the same `run_once()` pipeline for identical training/evaluation behavior.

## Editor AI

This project includes AI coding guidelines:
- See `CLAUDE.md` for house rules and conventions
- Cursor rules in `.cursor/rules/` define coding standards and policies