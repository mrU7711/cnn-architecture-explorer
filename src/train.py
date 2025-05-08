"""Training pipeline for architecture comparison."""

import yaml
import torch
from src.architectures import load_architecture, freeze_backbone
from src.transfer_learning import TransferLearner
from src.dataset import get_cifar10_loaders
from src.analyzer import compare_architectures, print_comparison


def train_all(config_path="config/config.yaml"):
    """Train and compare all configured architectures."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
    )

    results = {}
    for arch_config in config["architectures"]:
        name = arch_config["name"]
        print(f"\nTraining {name}...")
        model = load_architecture(name, arch_config["pretrained"], arch_config["num_classes"])
        model = freeze_backbone(model, name)
        learner = TransferLearner(model, device, config["training"]["learning_rate"])

        for epoch in range(config["training"]["epochs"]):
            loss, train_acc = learner.train_epoch(train_loader)
            if (epoch + 1) % 5 == 0:
                test_acc = learner.evaluate(test_loader)
                print(f"  Epoch {epoch+1}: Loss={loss:.4f} Train={train_acc:.1f}% Test={test_acc:.1f}%")
        results[name] = learner.evaluate(test_loader)

    comparison = compare_architectures(config["architectures"], device=str(device))
    print_comparison(comparison)
    return results


if __name__ == "__main__":
    train_all()
