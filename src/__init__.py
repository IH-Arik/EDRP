"""
EDRP-DiaBD: Explainable Ensemble Modeling for Diabetes Risk Prediction.

This package provides tools for diabetes risk prediction using ensemble methods
with interpretability techniques.
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "research@example.com"

from .models.cnn_model import create_1d_cnn_model, create_model_callbacks
from .data.preprocessing import DataPreprocessor
from .evaluation.metrics import ModelEvaluator
from .utils.config import config
from .utils.logger import logger

__all__ = [
    "create_1d_cnn_model",
    "create_model_callbacks", 
    "DataPreprocessor",
    "ModelEvaluator",
    "config",
    "logger"
]
