import torch


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    """Train model for one epoch and return average loss.
    
    Args:
        model: PyTorch model
        dataloader: Training dataloader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to run on
    
    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, dataloader, loss_fn, device):
    """Evaluate model and return average loss and accuracy.
    
    Args:
        model: PyTorch model
        dataloader: Evaluation dataloader
        loss_fn: Loss function
        device: Device to run on
    
    Returns:
        tuple: (average_loss, accuracy_float)
    """
    model.eval()
    loss_sum = 0.0
    correct = 0
    n = 0
    
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            
            logits = model(xb)
            loss = loss_fn(logits, yb)
            
            loss_sum += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            n += xb.size(0)
    
    avg_loss = loss_sum / n
    accuracy = correct / n
    
    return avg_loss, accuracy