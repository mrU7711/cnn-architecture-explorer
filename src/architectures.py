"""CNN architecture loading and modification for transfer learning."""

import torch
import torch.nn as nn
import torchvision.models as models


ARCHITECTURE_REGISTRY = {
    "vgg16": (models.vgg16, "classifier.6"),
    "resnet18": (models.resnet18, "fc"),
    "resnet34": (models.resnet34, "fc"),
    "resnet50": (models.resnet50, "fc"),
    "densenet121": (models.densenet121, "classifier"),
}


def load_architecture(name, pretrained=True, num_classes=10):
    """Load a pretrained architecture and modify the final layer."""
    if name not in ARCHITECTURE_REGISTRY:
        raise ValueError(f"Unknown architecture: {name}")

    factory, fc_name = ARCHITECTURE_REGISTRY[name]
    weights = "DEFAULT" if pretrained else None
    model = factory(weights=weights)

    if name.startswith("vgg"):
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    elif name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name.startswith("densenet"):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    return model


def freeze_backbone(model, architecture_name):
    """Freeze all layers except the final classifier."""
    for param in model.parameters():
        param.requires_grad = False

    if architecture_name.startswith("vgg"):
        for param in model.classifier[6].parameters():
            param.requires_grad = True
    elif architecture_name.startswith("resnet"):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif architecture_name.startswith("densenet"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    return model


def count_parameters(model, trainable_only=True):
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
