"""
Data Loading Utilities
=======================
Provides a simple function to download and load the MNIST dataset
using torchvision. Returns standard PyTorch DataLoader objects.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple


def get_mnist_loaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Download MNIST and return train/test DataLoaders.

    The images are normalized to [0, 1] range (required for BCE loss
    and Sigmoid output layers).

    Args:
        batch_size:  Number of images per batch (default 128)
        data_dir:    Directory to store/download the dataset
        num_workers: Number of parallel data-loading workers

    Returns:
        train_loader: DataLoader for the 60,000 training images
        test_loader:  DataLoader for the 10,000 test images
    """
    # Transform: convert PIL images to tensors (automatically scales to [0, 1])
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Download and load training data
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    # Download and load test data
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader
