from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import os
import mlflow 
import mlflow.sklearn
from mlflow.tracking import MlflowClient
# TODO: import os, mlflow, mlflow.sklearn, MlflowClient when implementing.
from . import config


@dataclass
class LoadedModelState:
    model: Any = None
    loaded: bool = False
    error: Optional[str] = None
    model_uri: Optional[str] = None
    run_id: Optional[str] = None
    run_name: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, Any] = field(default_factory=dict)


class ModelService:
    """TODO: Load the HW02 model from MLflow.

    Required behavior:
    - Use MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, and MLFLOW_TRACKING_PASSWORD.
    - If MLFLOW_RUN_ID is set, load runs:/<run_id>/model.
    - Otherwise auto-select a clean/selected run from MLFLOW_EXPERIMENT_NAME.
    - Do not crash the API on startup. Store the error in self.state.error.
    """
   
    
    def __init__(self) -> None:
        self.state = LoadedModelState()

    def load(self) -> None:
        # TODO: Replace this placeholder with real MLflow loading.
        # Hint:
        os.environ["MLFLOW_TRACKING_USERNAME"] = config.MLFLOW_TRACKING_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = config.MLFLOW_TRACKING_PASSWORD
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
         
        try:
            print("Loading model:", f"runs:/{config.MLFLOW_RUN_ID}/model")
            model = mlflow.sklearn.load_model(f"runs:/{config.MLFLOW_RUN_ID}/model")
            print("Model loaded:", type(model))
            self.state.model = model
            print(model.named_steps["preprocess"].transformers_)
        except Exception as e:
            self.state.loaded = False
            self.state.model = None
            self.state.error = str(e)
        
        if model is not None:
            self.state.loaded = True
            self.state.error = None
            return self.model_info()
        else:
            self.state.loaded = False
            self.state.error = "TODO: load model from MLflow."

    def require_model(self):
        if not self.state.loaded or self.state.model is None:
            raise RuntimeError(self.state.error or "Model is not loaded.")
        return self.state.model

    def model_info(self) -> dict:
        return {
            "model_loaded": self.state.loaded,
            "tracking_uri": config.MLFLOW_TRACKING_URI,
            "experiment_name": config.MLFLOW_EXPERIMENT_NAME,
            "model_uri": self.state.model_uri,
            "run_id": self.state.run_id,
            "run_name": self.state.run_name,
            "dataset_version": config.DATASET_VERSION,
            "target": config.TARGET_NAME,
            "threshold": config.PREDICTION_THRESHOLD,
            "metrics": self.state.metrics,
            "params": self.state.params,
            "tags": self.state.tags,
            "error": self.state.error,
        }
        



