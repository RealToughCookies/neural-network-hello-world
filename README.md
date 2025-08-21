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

**Verification commands:**
```bash
python -m src.train_fashion_mnist --epochs 1
python -c "import src.train_fashion_mnist as t; print(t.smoke_test())"
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

## Run as Module

All FashionMNIST commands use `python -m` to run within the package import system, which resolves the package-absolute imports correctly. This avoids ModuleNotFoundError issues when importing between src modules. Both the CLI and smoke test use the same `run_once()` pipeline for identical training/evaluation behavior.