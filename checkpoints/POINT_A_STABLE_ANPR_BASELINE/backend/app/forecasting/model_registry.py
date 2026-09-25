"""
ChronoEye Infinity - Phase 8: Model Registry & Checkpoint Manager
Manages forecasting model registration, saving/loading model weights, and versioning.
"""

import json
from typing import Dict, Any, Optional


class ModelRegistry:
    """
    Forecasting Model Registry and Checkpoint Manager.
    """

    def __init__(self):
        self.registered_models: Dict[str, Any] = {}

    def register_model(self, model_name: str, model_instance: Any):
        """Registers active forecaster model instance."""
        self.registered_models[model_name] = model_instance

    def get_model(self, model_name: str) -> Optional[Any]:
        """Retrieves registered forecaster model instance."""
        return self.registered_models.get(model_name)

    @staticmethod
    def save_checkpoint(model_name: str, model_instance: Any, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Serializes model parameters into checkpoint dictionary or saves to JSON file.
        """
        if hasattr(model_instance, "to_dict"):
            ckpt = {
                "model_name": model_name,
                "model_type": model_instance.__class__.__name__,
                "state_dict": model_instance.to_dict(),
            }
        else:
            ckpt = {
                "model_name": model_name,
                "model_type": model_instance.__class__.__name__,
                "state_dict": {},
            }

        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(ckpt, f, indent=2)

        return ckpt

    @staticmethod
    def load_checkpoint(file_path_or_dict: Any) -> Dict[str, Any]:
        """
        Loads model checkpoint dictionary from file or dict.
        """
        if isinstance(file_path_or_dict, str):
            with open(file_path_or_dict, "r", encoding="utf-8") as f:
                return json.load(f)
        elif isinstance(file_path_or_dict, dict):
            return file_path_or_dict
        else:
            raise ValueError("Invalid checkpoint format")
