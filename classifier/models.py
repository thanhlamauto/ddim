"""
ResNet-18 model for PlantVillage classification.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


def create_resnet18(num_classes=15, pretrained=False):
    """
    Create ResNet-18 model for plant disease classification.

    Args:
        num_classes (int): Number of output classes (default: 15 for PlantVillage)
        pretrained (bool): Whether to use ImageNet pretrained weights

    Returns:
        nn.Module: ResNet-18 model
    """
    model = resnet18(pretrained=pretrained)

    # Replace final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


if __name__ == "__main__":
    # Test model creation
    model = create_resnet18(num_classes=15)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass
    x = torch.randn(4, 3, 256, 256)
    y = model(x)
    print(f"Output shape: {y.shape}")
