"""
Sudoku Digit Recognition Training Pipeline
A modular and configurable training system for digit classification
"""

__version__ = "0.1.0"

from .config import TrainingConfig
from .training.pipeline import TrainingPipeline

__all__ = ["TrainingConfig", "TrainingPipeline"]
