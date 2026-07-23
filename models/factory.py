from torchvision import models as torchvision_models

from models.pretrained import PRETRAINED_MODEL_NAMES, build_pretrained_model
from models.simple_cnn import SimpleCNN


MODEL_NAMES = [
    "simple-cnn",
    "resnet18-scratch",
    *PRETRAINED_MODEL_NAMES,
]


def build_model(model_name, num_classes):
    if model_name == "simple-cnn":
        return SimpleCNN(num_classes), "random"
    if model_name == "resnet18-scratch":
        return torchvision_models.resnet18(weights=None, num_classes=num_classes), "random"
    if model_name in PRETRAINED_MODEL_NAMES:
        return build_pretrained_model(model_name, num_classes), "imagenet"
    raise ValueError(f"Unknown model: {model_name}")


def classifier_parameters(model, model_name):
    if model_name.startswith("resnet"):
        return list(model.fc.parameters())
    return list(model.classifier.parameters())
