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

The FashionMNIST dataset will auto-download to `.data/` on first run.

Train TinyLinear classifier (default: CPU, subset of 2000 samples):
```bash
python src/train_fashion_mnist.py --epochs 1
```

Use GPU/MPS acceleration if available:
```bash
python src/train_fashion_mnist.py --epochs 5 --device auto
```

Run FashionMNIST smoke test:
```bash
python -c "import src.train_fashion_mnist as m; print(m.smoke_test())"
```

The smoke test trains for 1 epoch on 2000 samples and verifies that training reduces loss and achieves at least 60% validation accuracy.