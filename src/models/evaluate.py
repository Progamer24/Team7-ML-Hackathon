"""
Evaluation metrics for PathSnap candidate classifiers and route reconstruction.
Computes Observation Top-1 Accuracy, Candidate Miss Rate, Macro F1, Log Loss, and Route Continuity.
"""

from typing import Dict, List, Set, Union
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score


def compute_route_continuity(
    predicted_arcs: List[int],
    arcs: np.ndarray,
) -> float:
    """
    Compute route continuity metric: the proportion of consecutive predicted arcs
    that are either identical or topologically adjacent (share an intersection node).
    
    Args:
        predicted_arcs: List of arc_id integers in chronological order.
        arcs: Road network arcs array of shape (num_arcs, 2).
        
    Returns:
        Float value between 0.0 and 1.0.
    """
    if len(predicted_arcs) <= 1:
        return 1.0

    connected_count = 0
    total_transitions = len(predicted_arcs) - 1

    for i in range(total_transitions):
        prev_arc = predicted_arcs[i]
        curr_arc = predicted_arcs[i + 1]

        if prev_arc == curr_arc:
            connected_count += 1
            continue

        if prev_arc < 0 or prev_arc >= len(arcs) or curr_arc < 0 or curr_arc >= len(arcs):
            continue

        prev_from, prev_to = arcs[prev_arc, 0], arcs[prev_arc, 1]
        curr_from, curr_to = arcs[curr_arc, 0], arcs[curr_arc, 1]

        # Check direct topological link (forward transition or common node)
        if prev_to == curr_from or prev_to == curr_to or prev_from == curr_from:
            connected_count += 1

    return float(connected_count / total_transitions) if total_transitions > 0 else 1.0


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Standard binary classification metrics on candidate pairs.
    """
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Compute log loss safely
    eps = 1e-7
    clipped_prob = np.clip(y_prob, eps, 1.0 - eps)
    try:
        loss = log_loss(y_true, clipped_prob)
    except Exception:
        loss = float("nan")

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "log_loss": float(loss),
    }


def evaluate_observation_level(
    df_features,
    predicted_probs: np.ndarray,
    ground_truth_route: np.ndarray,
) -> Dict[str, float]:
    """
    Compute observation-level Top-1 accuracy and candidate miss rate across GPS points.
    """
    df = df_features.copy()
    df["pred_prob"] = predicted_probs
    route_set = set(ground_truth_route) if ground_truth_route is not None else set()

    total_points = df["point_idx"].nunique()
    if total_points == 0:
        return {"top1_accuracy": 0.0, "candidate_miss_rate": 0.0}

    correct_top1 = 0
    missed_points = 0

    for pt_idx, group in df.groupby("point_idx"):
        # Did candidate set contain any route segment?
        cand_arcs = set(group["arc_id"].values)
        has_route_cand = bool(cand_arcs.intersection(route_set))
        if not has_route_cand:
            missed_points += 1

        # Check if the highest probability candidate is in the route
        best_cand = group.loc[group["pred_prob"].idxmax()]["arc_id"]
        if best_cand in route_set:
            correct_top1 += 1

    return {
        "top1_accuracy": float(correct_top1 / total_points),
        "candidate_miss_rate": float(missed_points / total_points),
    }

