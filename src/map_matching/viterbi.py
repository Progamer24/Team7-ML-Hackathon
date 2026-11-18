"""
Viterbi trellis decoding algorithm for PathSnap sequential route reconstruction.
Computes the most likely sequence of road segments in log-probability space.

Coordinate convention:
- track: [longitude, latitude, timestamp]
- nodes: [longitude, latitude]
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from src.data.preprocessing import haversine_distance
from src.map_matching.hmm import MapMatchingHMM


def viterbi_decode(
    track: np.ndarray,
    candidates_by_point: List[List[Dict]],
    arcs: np.ndarray,
    nodes: np.ndarray,
    hmm: Optional[MapMatchingHMM] = None,
    classifier_probs_by_point: Optional[List[List[float]]] = None,
) -> Tuple[List[int], List[Tuple[float, float]]]:
    """
    Perform Viterbi log-space decoding to find the optimal sequence of road segments.
    
    Args:
        track: Array of shape (N, 3) with [lon, lat, timestamp].
        candidates_by_point: List of candidate arc dictionaries for each GPS point.
        arcs: Array of shape (M, 2) [from_node, to_node].
        nodes: Array of shape (K, 2) [lon, lat].
        hmm: MapMatchingHMM instance.
        classifier_probs_by_point: Optional list of predicted match probabilities per candidate.
        
    Returns:
        matched_arcs: List of arc_id values for each GPS observation.
        projected_coords: List of (lat, lon) projected coordinates for map rendering.
    """
    if hmm is None:
        hmm = MapMatchingHMM()

    n_points = len(track)
    if n_points == 0:
        return [], []

    processed_candidates: List[List[Dict]] = []
    processed_probs: List[List[float]] = []

    for i in range(n_points):
        cands = candidates_by_point[i]
        probs = (
            classifier_probs_by_point[i]
            if classifier_probs_by_point and i < len(classifier_probs_by_point)
            else [1.0] * len(cands)
        )

        if len(cands) == 0:
            placeholder = {
                "arc_id": -1,
                "from_node": 0,
                "to_node": 0,
                "perp_dist": 200.0,
                "proj_ratio": 0.5,
                "proj_lon": float(track[i, 0]),
                "proj_lat": float(track[i, 1]),
                "arc_len": 50.0,
            }
            processed_candidates.append([placeholder])
            processed_probs.append([1e-4])
        else:
            processed_candidates.append(cands)
            processed_probs.append(probs)

    trellis: List[np.ndarray] = []
    backpointers: List[np.ndarray] = []

    # t = 0 (Initial emission probabilities)
    init_cands = processed_candidates[0]
    init_probs = processed_probs[0]
    v_0 = np.zeros(len(init_cands))

    for j, cand in enumerate(init_cands):
        c_prob = init_probs[j] if j < len(init_probs) else None
        p_emit = hmm.compute_emission_prob(cand, c_prob)
        v_0[j] = np.log(max(1e-12, p_emit))

    trellis.append(v_0)

    # Forward recurrence: t = 1 ... n_points - 1
    for t in range(1, n_points):
        prev_cands = processed_candidates[t - 1]
        curr_cands = processed_candidates[t]
        curr_probs = processed_probs[t]

        # track is [lon, lat, timestamp]
        p_prev_lon, p_prev_lat = track[t - 1, 0], track[t - 1, 1]
        p_curr_lon, p_curr_lat = track[t, 0], track[t, 1]
        dist_gps = haversine_distance(p_prev_lat, p_prev_lon, p_curr_lat, p_curr_lon)

        v_t = np.full(len(curr_cands), -np.inf)
        bp_t = np.zeros(len(curr_cands), dtype=int)

        for j, curr_cand in enumerate(curr_cands):
            c_prob = curr_probs[j] if j < len(curr_probs) else None
            p_emit = hmm.compute_emission_prob(curr_cand, c_prob)
            log_emit = np.log(max(1e-12, p_emit))

            best_val = -np.inf
            best_prev = 0

            for k, prev_cand in enumerate(prev_cands):
                p_trans = hmm.compute_transition_prob(
                    prev_cand, curr_cand, dist_gps, arcs, nodes
                )
                log_trans = np.log(max(1e-12, p_trans))

                score = trellis[t - 1][k] + log_trans + log_emit
                if score > best_val:
                    best_val = score
                    best_prev = k

            v_t[j] = best_val
            bp_t[j] = best_prev

        trellis.append(v_t)
        backpointers.append(bp_t)

    # Backtracking best path
    matched_arcs = [0] * n_points
    projected_coords = [(0.0, 0.0)] * n_points

    best_last_idx = int(np.argmax(trellis[-1]))
    matched_arcs[-1] = processed_candidates[-1][best_last_idx]["arc_id"]
    projected_coords[-1] = (
        processed_candidates[-1][best_last_idx]["proj_lat"],
        processed_candidates[-1][best_last_idx]["proj_lon"],
    )

    curr_idx = best_last_idx
    for t in range(n_points - 1, 0, -1):
        prev_idx = backpointers[t - 1][curr_idx]
        matched_arcs[t - 1] = processed_candidates[t - 1][prev_idx]["arc_id"]
        projected_coords[t - 1] = (
            processed_candidates[t - 1][prev_idx]["proj_lat"],
            processed_candidates[t - 1][prev_idx]["proj_lon"],
        )
        curr_idx = prev_idx

    return matched_arcs, projected_coords
