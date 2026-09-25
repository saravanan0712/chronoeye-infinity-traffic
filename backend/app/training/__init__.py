"""
ChronoEye Infinity - Training, Baselines, and Evaluation Framework.
"""

from app.training.baselines import (
    PersistenceBaseline,
    MovingAverageBaseline,
    LinearRegressionBaseline,
)
from app.training.trainer import STGNNTrainer, TrainingResult

__all__ = [
    "PersistenceBaseline",
    "MovingAverageBaseline",
    "LinearRegressionBaseline",
    "STGNNTrainer",
    "TrainingResult",
]
