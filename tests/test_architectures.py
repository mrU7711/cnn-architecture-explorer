"""Unit tests for architecture loading."""

import pytest
import torch
from src.architectures import load_architecture, freeze_backbone, count_parameters, ARCHITECTURE_REGISTRY


class TestLoadArchitecture:
    def test_resnet18(self):
        model = load_architecture("resnet18", pretrained=False, num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 10)

    def test_unknown_arch(self):
        with pytest.raises(ValueError):
            load_architecture("unknown_model")

    def test_all_architectures_load(self):
        for name in ARCHITECTURE_REGISTRY:
            model = load_architecture(name, pretrained=False, num_classes=5)
            assert model is not None

    def test_freeze_backbone(self):
        model = load_architecture("resnet18", pretrained=False, num_classes=10)
        model = freeze_backbone(model, "resnet18")
        trainable = count_parameters(model, trainable_only=True)
        total = count_parameters(model, trainable_only=False)
        assert trainable < total
