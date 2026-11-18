"""
Feature engineering module for PathSnap.
Extracts discriminating spatial and kinematic features for candidate arc classification.

Features:
- perp_dist: Orthogonal distance from GPS point to road segment (meters)
- heading_diff: Angular difference between vehicle heading and road arc orientation (degrees [0, 180])
- proj_ratio: Normalized position of projection along the segment [0.0, 1.0]
- dist_to_start_node: Great-circle distance to from_node (meters)
- dist_to_end_node: Great-circle distance to to_node (meters)
- arc_length: Physical length of road segment (meters)
- gps_speed: Estimated instantaneous speed of vehicle (m/s)
- distance_rank: Spatial proximity rank among candidates for this epoch (1 = closest)
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.data.preprocessing import (
    haversine_distance,
    compute_bearing,
    bearing_difference,
)


class FeatureExtractor:
    """
    Constructs feature matrices for (GPS observation, candidate arc) pairs.
    """

    FEATURE_NAMES = [
        "perp_dist",
        "heading_diff",
        "proj_ratio",
        "dist_to_start_node",
        "dist_to_end_node",
        "arc_length",
        "gps_speed",
        "distance_rank",
    ]

    def __init__(self):
        pass

    def extract_features_for_trip(
        self,
        track: np.ndarray,
        arcs: np.ndarray,
        nodes: np.ndarray,
        candidates_by_point: List[List[Dict]],
        route: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Build feature records for all observations and their candidate arcs.
        
        Args:
            track: [lon, lat, timestamp]
            arcs: [from_node, to_node]
            nodes: [lon, lat]
            candidates_by_point: List of candidate sets
            route: Optional ground-truth arc sequence
        """
        records = []
        n_points = len(track)

        # Precompute GPS trajectory bearings and speeds
        gps_bearings = np.zeros(n_points)
        gps_speeds = np.zeros(n_points)

        for i in range(n_points):
            p_lon, p_lat = track[i, 0], track[i, 1]
            if i < n_points - 1:
                next_lon, next_lat = track[i + 1, 0], track[i + 1, 1]
                dt = max(1.0, track[i + 1, 2] - track[i, 2])
                dist = haversine_distance(p_lat, p_lon, next_lat, next_lon)
                gps_speeds[i] = dist / dt
                gps_bearings[i] = compute_bearing(p_lat, p_lon, next_lat, next_lon)
            elif n_points > 1:
                gps_speeds[i] = gps_speeds[i - 1]
                gps_bearings[i] = gps_bearings[i - 1]

        # Determine ground-truth matching arc for each point if route is provided
        gt_arc_for_point = {}
        if route is not None and len(route) > 0:
            route_set = set(route)
            for pt_idx, cands in enumerate(candidates_by_point):
                route_cands = [c for c in cands if c["arc_id"] in route_set]
                if route_cands:
                    best_cand = min(route_cands, key=lambda c: c["perp_dist"])
                    gt_arc_for_point[pt_idx] = best_cand["arc_id"]
                else:
                    gt_arc_for_point[pt_idx] = -1

        for pt_idx in range(n_points):
            p_lon, p_lat = track[pt_idx, 0], track[pt_idx, 1]
            cands = candidates_by_point[pt_idx]
            speed = float(gps_speeds[pt_idx])
            heading = float(gps_bearings[pt_idx])
            true_arc = gt_arc_for_point.get(pt_idx, -1)

            for rank, cand in enumerate(cands, start=1):
                from_nid = cand["from_node"]
                to_nid = cand["to_node"]
                a_lon, a_lat = nodes[from_nid, 0], nodes[from_nid, 1]
                b_lon, b_lat = nodes[to_nid, 0], nodes[to_nid, 1]

                arc_bearing = compute_bearing(a_lat, a_lon, b_lat, b_lon)
                heading_diff = bearing_difference(heading, arc_bearing)

                d_start = haversine_distance(p_lat, p_lon, a_lat, a_lon)
                d_end = haversine_distance(p_lat, p_lon, b_lat, b_lon)

                is_match = 1 if (true_arc != -1 and cand["arc_id"] == true_arc) else 0

                records.append({
                    "point_idx": pt_idx,
                    "arc_id": cand["arc_id"],
                    "perp_dist": cand["perp_dist"],
                    "heading_diff": float(heading_diff),
                    "proj_ratio": cand["proj_ratio"],
                    "dist_to_start_node": float(d_start),
                    "dist_to_end_node": float(d_end),
                    "arc_length": cand["arc_len"],
                    "gps_speed": speed,
                    "distance_rank": rank,
                    "is_match": is_match,
                })

        df = pd.DataFrame(records)
        return df
