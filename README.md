# Neural Network Hello World

A minimal PyTorch implementation that demonstrates a simple neural network learning the linear relationship y ≈ 3x + 2 on synthetic data. The project includes a single linear layer trained with SGD on noisy data, along with a smoke test to verify the model learns the target parameters within acceptable tolerance.

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