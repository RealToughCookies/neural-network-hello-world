# Neural Network From Scratch — House Rules

- You are **Claude Code inside Cursor** working in this repo.
- **Contract:** One task at a time; print plan → show unified diffs → wait for **APPROVE** → apply; always include rollback notes.
- **Defaults:** PyTorch; run modules via `python -m src.<module>`; datasets in `.data/`; artifacts in `artifacts/` (never commit).
- **Training/Eval:** Use seeds; eval with `model.eval()` + `torch.no_grad()`; CrossEntropyLoss takes **raw logits + int labels** (no softmax). Checkpoint via `state_dict` to `artifacts/checkpoints/{last,best}.pt`.
- **Code style:** Small, composable functions; minimal deps; readable prints over fancy frameworks.
- **Commits:** `L<level>-T<task>: <concise message>`; keep diffs tight.

## Performance Gating System

The trainer includes Wilson confidence interval-based gating for reliable model promotion:

### Basic Usage
```bash
python -m src.rl.ppo_selfplay_skeleton --train --updates 100 \
  --gate-best-min-wr 0.55 \
  --gate-pool-min-wr 0.52 \
  --pool-cap 50
```

### Key Features
- **Wilson CI Gating**: Uses lower bound of 95% confidence interval instead of point estimates
- **Best Model Promotion**: Automatically promotes `last.pt` → `best.pt` when gate passes  
- **Opponent Pool Management**: Registers strong models in league pool with automatic pruning
- **PFSP Ramping**: Optional curriculum from uniform → PFSP-even → PFSP-hard sampling

### Configuration
- `--gate-best-min-wr 0.55`: Promote to best.pt when LB ≥ 55% win rate
- `--gate-pool-min-wr 0.52`: Add to opponent pool when LB ≥ 52% win rate  
- `--pool-cap 50`: Keep 50 most recent opponents in pool
- `--pfsp-ramp auto`: Enable epoch-based PFSP progression

### Why Wilson CI?
Wilson confidence intervals are more reliable than normal approximation for small samples and extreme win rates (0% or 100%). The lower bound provides conservative estimates that reduce false positives in model promotion.