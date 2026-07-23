"""Scratch and transfer-learning model definitions."""

from models.factory import MODEL_NAMES, build_model
from models.simple_cnn import SimpleCNN

__all__ = ["MODEL_NAMES", "SimpleCNN", "build_model"]
