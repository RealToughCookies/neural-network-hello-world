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
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = loss_fn(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy