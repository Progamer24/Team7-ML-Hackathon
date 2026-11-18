"""
Data loader and validation module for PathSnap.
Handles reading and validation of .arcs, .nodes, .track, and .route files.

Dataset Coordinate Specification:
- .nodes: [longitude, latitude] (0-indexed row = node_id)
- .track: [longitude, latitude, timestamp] (nominally 1 Hz)
- .arcs:  [from_node_id, to_node_id] (0-indexed row = arc_id)
- .route: [arc_id] (chronological ground-truth sequence)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class TripLoader:
    """
    Loader for PathSnap trip directories containing road networks and GPS tracks.
    """

    def __init__(self, data_dir: Union[str, Path] = "data"):
        self.data_dir = Path(data_dir)

    def list_trips(self, split: str = "train") -> List[str]:
        """
        List all available trip identifiers in a given partition.
        """
        split_dir = self.data_dir / split
        if not split_dir.exists():
            return []
        trips = [p.name for p in split_dir.iterdir() if p.is_dir()]
        return sorted(trips)

    def load_trip(
        self, trip_id: str, split: Optional[str] = None
    ) -> Dict[str, Union[str, np.ndarray]]:
        """
        Load all data files for a specified trip.
        
        Returns:
            Dictionary containing:
                - 'trip_id': string identifier
                - 'split': partition name
                - 'arcs': np.ndarray of shape (num_arcs, 2) [from_node, to_node]
                - 'nodes': np.ndarray of shape (num_nodes, 2) [longitude, latitude]
                - 'track': np.ndarray of shape (num_points, 3) [longitude, latitude, timestamp]
                - 'route': Optional np.ndarray of shape (num_route_arcs,) ground-truth arc sequence
        """
        trip_path = None
        if split:
            candidate = self.data_dir / split / trip_id
            if candidate.exists():
                trip_path = candidate
        else:
            for s in ["evaluator_test", "student_test", "train"]:
                candidate = self.data_dir / s / trip_id
                if candidate.exists():
                    trip_path = candidate
                    split = s
                    break

        if trip_path is None or not trip_path.exists():
            raise FileNotFoundError(f"Trip '{trip_id}' not found in data directory: {self.data_dir}")

        arcs_file = trip_path / f"{trip_id}.arcs"
        nodes_file = trip_path / f"{trip_id}.nodes"
        track_file = trip_path / f"{trip_id}.track"
        route_file = trip_path / f"{trip_id}.route"

        if not arcs_file.exists() or not nodes_file.exists() or not track_file.exists():
            raise FileNotFoundError(f"Incomplete data files for trip '{trip_id}' in {trip_path}")

        arcs = np.loadtxt(arcs_file, dtype=np.int64)
        if arcs.ndim == 1 and arcs.size == 2:
            arcs = arcs.reshape(1, 2)

        nodes = np.loadtxt(nodes_file, dtype=np.float64)
        if nodes.ndim == 1 and nodes.size == 2:
            nodes = nodes.reshape(1, 2)

        track = np.loadtxt(track_file, dtype=np.float64)
        if track.ndim == 1 and track.size == 3:
            track = track.reshape(1, 3)

        route = None
        if route_file.exists():
            try:
                route_data = np.loadtxt(route_file, dtype=np.int64)
                route = np.atleast_1d(route_data)
            except Exception:
                route = np.array([], dtype=np.int64)

        return {
            "trip_id": trip_id,
            "split": split,
            "arcs": arcs,
            "nodes": nodes,
            "track": track,
            "route": route,
        }

    def validate_trip(self, trip: Dict[str, Union[str, np.ndarray]]) -> Dict[str, Union[bool, str, int, float]]:
        """
        Comprehensive data validation of coordinates, timestamps, and graph references.
        """
        track = trip["track"]
        nodes = trip["nodes"]
        arcs = trip["arcs"]
        route = trip.get("route", None)

        num_obs = len(track)
        num_nodes = len(nodes)
        num_arcs = len(arcs)

        # Check NaN or Inf
        has_nan = np.isnan(track).any() or np.isnan(nodes).any() or np.isnan(arcs).any()

        # Check coordinate bounds
        lons = track[:, 0]
        lats = track[:, 1]
        valid_coords = (-180.0 <= lons.min() and lons.max() <= 180.0 and
                        -90.0 <= lats.min() and lats.max() <= 90.0)

        # Check timestamp ordering and monotonicity
        timestamps = track[:, 2]
        dts = np.diff(timestamps)
        is_monotonic = bool(np.all(dts >= 0))
        pct_1hz = float(np.mean(np.isclose(dts, 1.0, atol=0.1))) if len(dts) > 0 else 1.0

        # Check graph index bounds
        max_node_ref = arcs.max() if len(arcs) > 0 else 0
        min_node_ref = arcs.min() if len(arcs) > 0 else 0
        valid_node_refs = (min_node_ref >= 0) and (max_node_ref < num_nodes)

        # Route validation if provided
        valid_route_refs = True
        route_continuity = 1.0
        if route is not None and len(route) > 0:
            valid_route_refs = bool((route.min() >= 0) and (route.max() < num_arcs))
            # compute route transitions
            consecutive_matches = 0
            for i in range(len(route) - 1):
                r_prev = route[i]
                r_curr = route[i + 1]
                if r_prev == r_curr:
                    consecutive_matches += 1
                elif arcs[r_prev, 1] == arcs[r_curr, 0]:
                    consecutive_matches += 1
            route_continuity = float(consecutive_matches / max(1, len(route) - 1))

        return {
            "valid": bool(not has_nan and valid_coords and valid_node_refs and valid_route_refs),
            "num_observations": num_obs,
            "num_nodes": num_nodes,
            "num_arcs": num_arcs,
            "has_nan": bool(has_nan),
            "valid_coords": bool(valid_coords),
            "timestamp_monotonic": is_monotonic,
            "nominal_1hz_pct": round(pct_1hz * 100, 2),
            "valid_node_references": bool(valid_node_refs),
            "valid_route_references": bool(valid_route_refs),
            "route_continuity": round(route_continuity, 4),
        }
