"""Transfer-learning model builders backed by torchvision weights."""

import torch.nn as nn
from torchvision import models as torchvision_models


PRETRAINED_MODEL_NAMES = [
    "resnet18-pretrained",
    "resnet50-pretrained",
    "efficientnet-b0-pretrained",
    "convnext-tiny-pretrained",
]


def build_pretrained_model(name, num_classes):
    if name == "resnet18-pretrained":
        model = torchvision_models.resnet18(
            weights=torchvision_models.ResNet18_Weights.IMAGENET1K_V1
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "resnet50-pretrained":
        model = torchvision_models.resnet50(
            weights=torchvision_models.ResNet50_Weights.IMAGENET1K_V2
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "efficientnet-b0-pretrained":
        model = torchvision_models.efficientnet_b0(
            weights=torchvision_models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if name == "convnext-tiny-pretrained":
        model = torchvision_models.convnext_tiny(
            weights=torchvision_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model
    raise ValueError(f"Unknown pretrained model: {name}")
