"""Transfer learning fine-tuning utilities."""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class TransferLearner:
    """Fine-tune pretrained models on new datasets."""

    def __init__(self, model, device, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable, lr=learning_rate)

    def train_epoch(self, loader):
        """Train for one epoch."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(loader, leave=False, desc="Train"):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        return total_loss / total, 100.0 * correct / total

    def evaluate(self, loader):
        """Evaluate model accuracy."""
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return 100.0 * correct / total
