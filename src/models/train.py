"""
Model training and pipeline construction for PathSnap candidate matching classifiers.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from src.features.engineering import FeatureExtractor


def get_model_pipeline(
    model_name: str = "random_forest",
    **hyperparams: Any,
) -> Pipeline:
    """
    Factory creating a standardized sklearn pipeline (scaler + classifier).
    
    Supported models:
      - 'logistic_regression'
      - 'knn'
      - 'decision_tree'
      - 'random_forest'
      - 'gradient_boosting'
      - 'mlp'
    """
    name = model_name.lower()

    if name in ["logistic_regression", "lr"]:
        clf = LogisticRegression(
            C=hyperparams.get("C", 1.0),
            max_iter=hyperparams.get("max_iter", 1000),
            class_weight="balanced",
            random_state=42,
        )
    elif name in ["knn", "k_nearest_neighbors"]:
        clf = KNeighborsClassifier(
            n_neighbors=hyperparams.get("n_neighbors", 7),
            weights=hyperparams.get("weights", "distance"),
        )
    elif name in ["decision_tree", "dt"]:
        clf = DecisionTreeClassifier(
            max_depth=hyperparams.get("max_depth", 8),
            min_samples_split=hyperparams.get("min_samples_split", 10),
            class_weight="balanced",
            random_state=42,
        )
    elif name in ["random_forest", "rf"]:
        clf = RandomForestClassifier(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", 11),
            min_samples_split=hyperparams.get("min_samples_split", 8),
            min_samples_leaf=hyperparams.get("min_samples_leaf", 2),
            max_features=hyperparams.get("max_features", "sqrt"),
            n_jobs=-1,
            class_weight=hyperparams.get("class_weight", "balanced_subsample"),
            random_state=42,
        )
    elif name in ["gradient_boosting", "gb", "gbm"]:
        clf = GradientBoostingClassifier(
            n_estimators=hyperparams.get("n_estimators", 100),
            learning_rate=hyperparams.get("learning_rate", 0.1),
            max_depth=hyperparams.get("max_depth", 5),
            random_state=42,
        )
    elif name in ["mlp", "neural_network"]:
        clf = MLPClassifier(
            hidden_layer_sizes=hyperparams.get("hidden_layer_sizes", (64, 32)),
            max_iter=hyperparams.get("max_iter", 300),
            early_stopping=True,
            n_iter_no_change=10,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])
    return pipeline


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "random_forest",
    **hyperparams: Any,
) -> Pipeline:
    """
    Train a candidate match classifier on extracted feature matrices.
    """
    pipeline = get_model_pipeline(model_name=model_name, **hyperparams)
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(pipeline: Any, filepath: Union[str, Path]) -> None:
    """Save trained pipeline and feature metadata to joblib artifact."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": pipeline,
        "feature_names": FeatureExtractor.FEATURE_NAMES,
    }
    joblib.dump(payload, path)


def load_model(filepath: Union[str, Path]) -> Any:
    """Load model artifact from disk."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    data = joblib.load(path)
    if isinstance(data, dict) and "pipeline" in data:
        return data["pipeline"]
    return data

