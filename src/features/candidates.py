"""
Spatial candidate arc generation module for PathSnap.
Uses spatial indexing (KDTree) to retrieve road segment candidates for each GPS point.

Coordinate convention:
- nodes: [longitude, latitude]
- track: [longitude, latitude, timestamp]
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.spatial import cKDTree

from src.data.preprocessing import point_to_segment_distance, haversine_distance


class CandidateGenerator:
    """
    Identifies candidate road segments (arcs) within a spatial radius for each GPS point.
    """

    def __init__(self, search_radius_m: float = 120.0, top_k: int = 8):
        self.search_radius_m = search_radius_m
        self.top_k = top_k
        self.lat_scale = 111139.0

    def build_spatial_index(
        self, arcs: np.ndarray, nodes: np.ndarray
    ) -> Tuple[cKDTree, float, float]:
        """
        Build a metric KDTree indexing arc midpoints in Cartesian coordinates (meters).
        """
        from_nodes = arcs[:, 0]
        to_nodes = arcs[:, 1]

        p1 = nodes[from_nodes]  # [lon, lat]
        p2 = nodes[to_nodes]    # [lon, lat]

        midpoints = (p1 + p2) / 2.0  # [mid_lon, mid_lat]
        center_lat = np.mean(nodes[:, 1])
        lon_scale = self.lat_scale * np.cos(np.radians(center_lat))

        # Metric Cartesian coordinates: x = lon * lon_scale, y = lat * lat_scale
        metric_midpoints = np.empty_like(midpoints)
        metric_midpoints[:, 0] = midpoints[:, 0] * lon_scale
        metric_midpoints[:, 1] = midpoints[:, 1] * self.lat_scale

        kdtree = cKDTree(metric_midpoints)
        return kdtree, lon_scale, self.lat_scale

    def generate_candidates(
        self,
        track: np.ndarray,
        arcs: np.ndarray,
        nodes: np.ndarray,
    ) -> List[List[Dict]]:
        """
        Generate candidate arc sets for each GPS point in the trajectory.
        
        Args:
            track: np.ndarray of shape (N, 3) [lon, lat, timestamp]
            arcs: np.ndarray of shape (M, 2) [from_node, to_node]
            nodes: np.ndarray of shape (K, 2) [lon, lat]
            
        Returns:
            List of length len(track), each containing top-K candidate dictionaries.
        """
        kdtree, lon_scale, lat_scale = self.build_spatial_index(arcs, nodes)

        all_candidates = []
        query_radius = max(self.search_radius_m * 2.0, 250.0)

        for i in range(len(track)):
            p_lon = track[i, 0]
            p_lat = track[i, 1]
            p_metric = [p_lon * lon_scale, p_lat * lat_scale]

            indices = kdtree.query_ball_point(p_metric, r=query_radius)

            if len(indices) == 0:
                _, fallback_idx = kdtree.query(p_metric, k=1)
                indices = [fallback_idx]

            candidates_for_point = []
            for arc_id in indices:
                from_nid = arcs[arc_id, 0]
                to_nid = arcs[arc_id, 1]
                a_lon, a_lat = nodes[from_nid, 0], nodes[from_nid, 1]
                b_lon, b_lat = nodes[to_nid, 0], nodes[to_nid, 1]

                perp_dist, proj_ratio, p_lon_proj, p_lat_proj = point_to_segment_distance(
                    p_lon, p_lat, a_lon, a_lat, b_lon, b_lat
                )

                if perp_dist <= self.search_radius_m * 1.5:
                    arc_len = haversine_distance(a_lat, a_lon, b_lat, b_lon)
                    candidates_for_point.append({
                        "arc_id": int(arc_id),
                        "from_node": int(from_nid),
                        "to_node": int(to_nid),
                        "perp_dist": float(perp_dist),
                        "proj_ratio": float(proj_ratio),
                        "proj_lon": float(p_lon_proj),
                        "proj_lat": float(p_lat_proj),
                        "arc_len": float(arc_len),
                    })

            candidates_for_point.sort(key=lambda c: c["perp_dist"])
            if len(candidates_for_point) > self.top_k:
                candidates_for_point = candidates_for_point[: self.top_k]

            all_candidates.append(candidates_for_point)

        return all_candidates
