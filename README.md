# Neural Network Hello World

A minimal PyTorch implementation featuring both synthetic data regression and FashionMNIST classification. Includes linear regression learning y ≈ 3x + 2 on toy data, and a TinyLinear classifier for FashionMNIST with comprehensive training/evaluation loops and smoke tests.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make train-cnn && make export && make compare
```

This trains a CNN on FashionMNIST, exports to TorchScript/ONNX, and validates both backends produce identical predictions. See `Makefile` for individual targets: `train-cnn`, `export`, `predict`, `compare`, `tune`, `clean`.

GitHub Actions runs automated smoke tests on push/PR to validate the training pipeline, model export, and inference backends.

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

# Export best checkpoint (auto-detect model type)
python -m src.export --model auto --ckpt artifacts/checkpoints/best.pt --outdir artifacts/export

# Force specific model type if needed
python -m src.export --model linear --ckpt artifacts/checkpoints/best.pt --outdir artifacts/export
```

Outputs:
- `artifacts/export/model_ts.pt`: TorchScript format for C++ deployment
- `artifacts/export/model.onnx`: ONNX format for cross-framework inference
- Console: ONNX Runtime sanity check comparing predictions with PyTorch

Auto-detection inspects checkpoint `state_dict` keys to determine model architecture. TorchScript enables deployment without Python in C++ environments. ONNX provides cross-framework compatibility with ONNX Runtime for efficient CPU/GPU inference.

## Inference

Run inference on exported models:

```bash
# TorchScript inference on test sample
python -m src.predict --backend ts --sample 0

# ONNX Runtime inference on test sample
python -m src.predict --backend onnx --sample 0

# Compare both backends
python -m src.predict --compare --sample 42

# Inference on custom image (28x28 grayscale recommended)
python -m src.predict --backend ts --image my_fashion_item.png
```

The inference tool automatically preprocesses inputs with the same transforms used during training. Compare mode validates that both TorchScript and ONNX backends produce identical predictions.

## Dota-Class Agent Roadmap

Research plan for scaling reinforcement learning to complex, multi-agent strategy games inspired by OpenAI Five. See [`docs/dota-rl-plan.md`](docs/dota-rl-plan.md) for detailed technical approach and timeline.

**Quick Start:**
```bash
# Install MPE2 (primary environment, headless)
pip install mpe2  # Note: On Python 3.13, fallback uses pettingzoo.mpe
python -m src.rl.ppo_selfplay_skeleton --dry-run
```

Alternatives: `--env grf` (Google Research Football) or `--env pistonball` ([PettingZoo](https://pettingzoo.farama.org/)) if MPE2 unavailable.

## PPO Core

Minimal Proximal Policy Optimization implementation with GAE (Generalized Advantage Estimation) and clipped policy loss:

```bash
# Run PPO smoke training
python -m src.rl.ppo_selfplay_skeleton --train

# Run self-play training (learner vs frozen opponent) with MPE2
python -m src.rl.ppo_selfplay_skeleton --selfplay --env mpe_adversary --steps 512

# Test with different rollout length
python -m src.rl.ppo_selfplay_skeleton --train --steps 512
```

Features: clipped policy gradients, MSE value loss, entropy regularization, and KL divergence early stopping. Based on Schulman et al. 2017 (PPO) and 2015 (GAE).

Self-play uses frozen snapshots with a small opponent pool; PettingZoo MPE simple_adversary_v3 runs via Parallel API. MPE spec: obs (8)/(10), discrete(5).

The `smoke_train()` function is fully self-contained, creating its own environment and running a complete training cycle without external dependencies.

## RL Checkpoint & Eval

Training automatically saves best/last checkpoints with complete state (policies, optimizers, opponent pools):

```bash
# Run self-play training with checkpoints
python -m src.rl.ppo_selfplay_skeleton --selfplay --env mpe_adversary --steps 512 --eval-every 1

# Resume from checkpoint
python -m src.rl.ppo_selfplay_skeleton --selfplay --resume last

# Evaluate saved checkpoint
python -m src.rl.eval_cli --ckpt artifacts/rl_ckpts/best.pt --episodes 4 --env mpe_adversary
```

Checkpoints include policy/value networks for both roles, optimizer state, opponent pool snapshots, and training configuration for complete resumability.

## Run as Module

All FashionMNIST commands use `python -m` to run within the package import system, which resolves the package-absolute imports correctly. This avoids ModuleNotFoundError issues when importing between src modules. Both the CLI and smoke test use the same `run_once()` pipeline for identical training/evaluation behavior.

## Editor AI

This project includes AI coding guidelines:
- See `CLAUDE.md` for house rules and conventions
- Cursor rules in `.cursor/rules/` define coding standards and policies