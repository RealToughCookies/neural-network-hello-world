from pathlib import Path
import csv


class CSVLogger:
    """Simple CSV logger for training metrics."""
    
    def __init__(self, path: str, header=("epoch", "train_loss", "val_loss", "val_acc")):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.header = list(header)
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(self.header)
    
    def append(self, **metrics):
        """Append metrics as a new row to the CSV file."""
        row = [metrics.get(k) for k in self.header]
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow(row)