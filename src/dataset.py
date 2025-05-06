"""CIFAR-10 dataset loading for architecture comparison."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_loaders(batch_size=32, num_workers=4, image_size=224):
    """Create CIFAR-10 data loaders with ImageNet-style preprocessing."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
