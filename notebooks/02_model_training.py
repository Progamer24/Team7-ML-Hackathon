"""
Reproducible Model Training and Hyperparameter Tuning Script for PathSnap.
Trains models across 5 algorithm families, performs Bayesian tuning, and saves artifacts.
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.training.pipeline import TrainingPipeline
from src.models.tune import run_bayesian_tuning


def run_training():
    print("=" * 60)
    print("PATHSNAP: MODEL TRAINING & BENCHMARKING PIPELINE")
    print("=" * 60)

    pipeline = TrainingPipeline()
    results = pipeline.run_training_and_comparison()

    print("\nModel Comparison Table:")
    print(results["comparison_df"][["Model", "Algorithm Family", "Top-1 Accuracy", "Student-Test Macro F1", "Route Continuity"]])

    print(f"\nSelected Best Model: {results['best_model_name']}")
    print("Model artifacts and configurations saved to artifacts/")


if __name__ == "__main__":
    run_training()
