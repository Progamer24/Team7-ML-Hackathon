"""
Bayesian hyperparameter optimization module for PathSnap.
Supports Optuna and RandomizedSearchCV strategies with group-aware cross-validation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

from src.models.train import get_model_pipeline


def run_bayesian_tuning(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str = "random_forest",
    n_trials: int = 15,
    output_csv: Optional[str] = "results/tuning_results.csv",
) -> Dict[str, Any]:
    """
    Execute Bayesian hyperparameter tuning across search space using GroupKFold.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        has_optuna = True
    except ImportError:
        has_optuna = False

    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    trials_records: List[Dict[str, Any]] = []

    best_score = -1.0
    best_params: Dict[str, Any] = {}

    if has_optuna:
        def objective(trial: optuna.Trial) -> float:
            if model_name in ["random_forest", "rf"]:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
                    "max_depth": trial.suggest_int("max_depth", 6, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                }
            elif model_name in ["gradient_boosting", "gb"]:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=25),
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                }
            else:
                params = {
                    "C": trial.suggest_float("C", 0.01, 10.0, log=True),
                }

            cv_scores = []
            for train_idx, val_idx in gkf.split(X, y, groups):
                pipe = get_model_pipeline(model_name, **params)
                pipe.fit(X[train_idx], y[train_idx])
                preds = pipe.predict(X[val_idx])
                score = f1_score(y[val_idx], preds, average="macro", zero_division=0)
                cv_scores.append(score)

            mean_score = float(np.mean(cv_scores))
            record = {"trial": trial.number, **params, "mean_macro_f1": mean_score}
            trials_records.append(record)
            return mean_score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        best_score = study.best_value
        best_params = study.best_params

    else:
        # Fallback grid/random search if optuna is not installed
        for trial_num in range(n_trials):
            params = {
                "n_estimators": int(np.random.choice([50, 100, 150, 200])),
                "max_depth": int(np.random.choice([6, 8, 10, 12, 16])),
                "min_samples_split": int(np.random.choice([2, 5, 8])),
            }
            cv_scores = []
            for train_idx, val_idx in gkf.split(X, y, groups):
                pipe = get_model_pipeline(model_name, **params)
                pipe.fit(X[train_idx], y[train_idx])
                preds = pipe.predict(X[val_idx])
                score = f1_score(y[val_idx], preds, average="macro", zero_division=0)
                cv_scores.append(score)
            mean_score = float(np.mean(cv_scores))
            record = {"trial": trial_num, **params, "mean_macro_f1": mean_score}
            trials_records.append(record)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

    df_results = pd.DataFrame(trials_records)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(output_csv, index=False)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "trials_dataframe": df_results,
    }

