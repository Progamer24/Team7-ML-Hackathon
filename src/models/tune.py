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
    n_trials: int = 25,
    output_csv: Optional[str] = "results/tuning_results.csv",
    plot_convergence: bool = True,
) -> Dict[str, Any]:
    """
    Execute Bayesian hyperparameter tuning using Optuna TPESampler with GroupKFold.
    Optimizes a blended objective: 0.6 * Macro F1 + 0.4 * Classification Accuracy.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        has_optuna = True
    except ImportError:
        has_optuna = False

    n_groups = len(np.unique(groups))
    n_splits = min(5, max(2, n_groups))
    gkf = GroupKFold(n_splits=n_splits)
    trials_records: List[Dict[str, Any]] = []

    best_score = -1.0
    best_params: Dict[str, Any] = {}

    if has_optuna:
        def objective(trial: optuna.Trial) -> float:
            if model_name in ["random_forest", "rf"]:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
                    "max_depth": trial.suggest_int("max_depth", 6, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 3),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                    "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
                }
            elif model_name in ["gradient_boosting", "gb", "gbm"]:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=25),
                    "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.25, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                }
            elif model_name in ["mlp", "neural_network"]:
                hidden_choice = trial.suggest_categorical("hidden_architecture", ["64x32", "128x64", "64x64"])
                arch_map = {
                    "64x32": (64, 32),
                    "128x64": (128, 64),
                    "64x64": (64, 64),
                }
                params = {
                    "hidden_layer_sizes": arch_map[hidden_choice],
                    "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
                    "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
                    "max_iter": 250,
                }
            else:
                params = {
                    "C": trial.suggest_float("C", 0.01, 10.0, log=True),
                }

            f1_scores = []
            acc_scores = []

            for train_idx, val_idx in gkf.split(X, y, groups):
                pipe = get_model_pipeline(model_name, **params)
                pipe.fit(X[train_idx], y[train_idx])
                preds = pipe.predict(X[val_idx])

                f1 = f1_score(y[val_idx], preds, average="macro", zero_division=0)
                acc = float(np.mean(preds == y[val_idx]))
                f1_scores.append(f1)
                acc_scores.append(acc)

            mean_f1 = float(np.mean(f1_scores))
            mean_acc = float(np.mean(acc_scores))
            # Blended objective favoring high F1 while sustaining strong accuracy
            blended_score = 0.6 * mean_f1 + 0.4 * mean_acc

            record = {
                "trial": trial.number,
                **{k: (str(v) if isinstance(v, tuple) else v) for k, v in params.items()},
                "mean_macro_f1": round(mean_f1, 4),
                "mean_accuracy": round(mean_acc, 4),
                "blended_objective": round(blended_score, 4),
            }
            trials_records.append(record)
            return blended_score

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials)
        best_score = study.best_value
        best_params = study.best_params

    else:
        # Fallback randomized search
        for trial_num in range(n_trials):
            params = {
                "n_estimators": int(np.random.choice([50, 100, 150, 200])),
                "max_depth": int(np.random.choice([8, 12, 16])),
                "min_samples_split": int(np.random.choice([2, 5])),
            }
            f1_scores = []
            acc_scores = []
            for train_idx, val_idx in gkf.split(X, y, groups):
                pipe = get_model_pipeline(model_name, **params)
                pipe.fit(X[train_idx], y[train_idx])
                preds = pipe.predict(X[val_idx])
                f1_scores.append(f1_score(y[val_idx], preds, average="macro", zero_division=0))
                acc_scores.append(float(np.mean(preds == y[val_idx])))

            mean_f1 = float(np.mean(f1_scores))
            mean_acc = float(np.mean(acc_scores))
            blended_score = 0.6 * mean_f1 + 0.4 * mean_acc
            record = {"trial": trial_num, **params, "mean_macro_f1": mean_f1, "mean_accuracy": mean_acc, "blended_objective": blended_score}
            trials_records.append(record)
            if blended_score > best_score:
                best_score = blended_score
                best_params = params

    df_results = pd.DataFrame(trials_records)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(output_csv, index=False)

    if plot_convergence and len(df_results) > 1:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(df_results["trial"], df_results["blended_objective"], marker="o", linestyle="-", color="#3B82F6", alpha=0.6, label="Trial Score")
        plt.plot(df_results["trial"], df_results["blended_objective"].cummax(), marker="^", linestyle="--", color="#10B981", linewidth=2, label="Current Best")
        plt.title(f"PathSnap: Bayesian Hyperparameter Optimization Convergence ({model_name})")
        plt.xlabel("Iteration / Trial")
        plt.ylabel("Blended Objective (0.6 F1 + 0.4 Acc)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path("results/plots/tuning_convergence.png"), dpi=150)
        plt.close()

    return {
        "best_params": best_params,
        "best_score": best_score,
        "trials_dataframe": df_results,
    }

