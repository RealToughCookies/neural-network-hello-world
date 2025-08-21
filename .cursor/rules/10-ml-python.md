## PyTorch & Python

- Use `python -m src.module` for entrypoints.
- DataLoaders: `shuffle=True` (train), `False` (test); seed subsets with `torch.Generator`.
- Evaluation: `model.eval()` + `torch.no_grad()`.
- Loss: CE expects logits + class indices; no softmax pre-CE.
- Checkpoints: save/load `state_dict`s; load with `map_location='cpu'`.