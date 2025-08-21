from pathlib import Path
import csv
import torch

FASHION_LABELS = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

@torch.no_grad()
def compute_confusion_matrix(model, dl, device, num_classes=10):
    """Compute confusion matrix using proper evaluation mode."""
    model.eval()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        idx = yb * num_classes + pred
        conf += torch.bincount(idx, minlength=num_classes*num_classes).reshape(num_classes, num_classes)
    return conf

def per_class_accuracy(conf):
    """Compute per-class accuracy from confusion matrix."""
    acc = []
    for i in range(conf.size(0)):
        denom = conf[i].sum().item()
        acc.append((conf[i,i].item()/denom) if denom else 0.0)
    return acc

def save_confusion_csv(conf, path, labels=None):
    """Save confusion matrix to CSV file."""
    labels = labels or [str(i) for i in range(conf.size(0))]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([""] + labels)
        for i, row in enumerate(conf.tolist()):
            wr.writerow([labels[i]] + row)

def plot_confusion_png(conf, path, labels=None):
    """Plot confusion matrix and save as PNG."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping confusion matrix plot")
        return
    
    labels = labels or [str(i) for i in range(conf.size(0))]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(conf, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()