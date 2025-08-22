# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-08-22

### Added
- **Training Pipeline**: Linear regression (hello_nn.py) and FashionMNIST classification with TinyLinear/TinyCNN models
- **Model Architectures**: Switchable linear classifier (~7k params) and CNN (~50k params) with automatic smoke tests
- **Checkpointing**: Save/resume training with best/last model checkpoints using PyTorch state_dict
- **Configuration**: JSON-based config management with CLI override support
- **Logging & Visualization**: CSV metrics logging, matplotlib plots, and confusion matrix analysis
- **Hyperparameter Tuning**: Grid search over learning rate, momentum, weight decay, and batch size
- **Model Export**: TorchScript and ONNX export with auto-detection from checkpoint state_dict
- **Inference**: Standalone prediction CLI supporting both TorchScript and ONNX backends
- **Backend Validation**: Compare mode ensures TorchScript/ONNX prediction consistency
- **Development Tools**: Makefile with common targets, requirements.txt, quickstart documentation
- **CI/CD**: GitHub Actions workflow with automated smoke tests on push/PR

### Features
- Deterministic training with comprehensive seeding across Python/NumPy/PyTorch
- Per-class accuracy analysis with FashionMNIST class labels
- Graceful dependency handling for optional components (matplotlib, onnx, onnxruntime)
- Cross-platform model deployment support (CPU/GPU inference)
- Complete pipeline: train → export → inference with validation