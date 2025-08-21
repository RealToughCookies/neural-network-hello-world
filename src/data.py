import torch
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms


def get_fashion_mnist_dataloaders(batch_size=128, subset=None, seed=0, data_dir=".data"):
    """Get FashionMNIST train and test dataloaders.
    
    Args:
        batch_size: Batch size for dataloaders
        subset: If not None, use only this many samples from each split
        seed: Random seed for deterministic subset selection
        data_dir: Directory to store/load dataset
    
    Returns:
        tuple: (train_dataloader, test_dataloader)
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    # Apply subset if specified
    if subset is not None:
        # Use separate generators for deterministic subset selection
        train_gen = torch.Generator().manual_seed(seed)
        test_gen = torch.Generator().manual_seed(seed)
        train_indices = torch.randperm(len(train_dataset), generator=train_gen)[:subset]
        test_indices = torch.randperm(len(test_dataset), generator=test_gen)[:min(subset, len(test_dataset))]
        
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader