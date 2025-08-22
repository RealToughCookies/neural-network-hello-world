from itertools import product
import csv
import os
from pathlib import Path
from src.train_fashion_mnist import run_once

def run_grid(lrs=(0.05, 0.1), momentums=(0.0, 0.9), wds=(0.0, 5e-4), batch_sizes=(128, 256),
             epochs=2, subset_train=2000, seed=0, device="cpu", out_csv="artifacts/tuning.csv"):
    """Run hyperparameter grid search for TinyCNN."""
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    header = ["lr", "momentum", "weight_decay", "batch_size", "epochs", "subset_train", "seed",
              "train_loss", "val_loss", "val_acc", "params"]
    new_file = not Path(out_csv).exists()
    
    with open(out_csv, "a", newline="") as f:
        wr = csv.writer(f)
        if new_file:
            wr.writerow(header)
        
        for lr, momentum, wd, bs in product(lrs, momentums, wds, batch_sizes):
            print(f"Testing: lr={lr}, momentum={momentum}, wd={wd}, bs={bs}")
            initial, train_loss, val_loss, val_acc, model, _ = run_once(
                epochs=epochs, batch_size=bs, subset_train=subset_train, seed=seed,
                device=device, model_name="cnn", lr=lr, momentum=momentum, weight_decay=wd
            )
            params = sum(p.numel() for p in model.parameters())
            wr.writerow([lr, momentum, wd, bs, epochs, subset_train, seed, 
                        train_loss, val_loss, val_acc, params])
    
    # Print top-3 results
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    rows_sorted = sorted(rows, key=lambda r: float(r["val_acc"]), reverse=True)[:3]
    print("\nTOP-3 Results:")
    for i, r in enumerate(rows_sorted, 1):
        print(f"{i}. val_acc={r['val_acc']} -> lr={r['lr']}, momentum={r['momentum']}, wd={r['weight_decay']}, bs={r['batch_size']}")

def smoke_test():
    """Quick smoke test for tuning pipeline."""
    out_csv = "artifacts/tuning_test.csv"
    run_grid(lrs=(0.05, 0.1), momentums=(0.0,), wds=(0.0,), batch_sizes=(128,), 
             epochs=1, subset_train=1000, out_csv=out_csv)
    return os.path.exists(out_csv) and sum(1 for _ in open(out_csv)) >= 3  # header + 2 runs

if __name__ == "__main__":
    run_grid()