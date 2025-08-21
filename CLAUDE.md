# Neural Network From Scratch — House Rules

- You are **Claude Code inside Cursor** working in this repo.
- **Contract:** One task at a time; print plan → show unified diffs → wait for **APPROVE** → apply; always include rollback notes.
- **Defaults:** PyTorch; run modules via `python -m src.<module>`; datasets in `.data/`; artifacts in `artifacts/` (never commit).
- **Training/Eval:** Use seeds; eval with `model.eval()` + `torch.no_grad()`; CrossEntropyLoss takes **raw logits + int labels** (no softmax). Checkpoint via `state_dict` to `artifacts/checkpoints/{last,best}.pt`.
- **Code style:** Small, composable functions; minimal deps; readable prints over fancy frameworks.
- **Commits:** `L<level>-T<task>: <concise message>`; keep diffs tight.