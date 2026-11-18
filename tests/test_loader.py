import pytest
from pathlib import Path
import numpy as np

from src.data.loader import TripLoader
from src.data.preprocessing import haversine_distance, compute_bearing, point_to_segment_distance


def test_trip_loader():
    loader = TripLoader("data")
    trips = loader.list_trips("student_test")
    assert len(trips) > 0, "No student test trips found"

    sample_id = trips[0]
    trip = loader.load_trip(sample_id, split="student_test")
    assert trip["trip_id"] == sample_id
    assert trip["arcs"].shape[1] == 2
    assert trip["nodes"].shape[1] == 2
    assert trip["track"].shape[1] == 3
    assert trip["route"] is not None


def test_haversine():
    # Distance between two identical points should be 0
    d0 = haversine_distance(12.9716, 77.5946, 12.9716, 77.5946)
    assert np.isclose(d0, 0.0, atol=1e-3)

    # Known distance approx: 1 degree latitude ~ 111 km
    d1 = haversine_distance(12.0, 77.0, 13.0, 77.0)
    assert 110000 < d1 < 112000


def test_point_to_segment_projection():
    # Point exactly on segment
    p_lat, p_lon = 10.0, 50.0
    a_lat, a_lon = 10.0, 49.0
    b_lat, b_lon = 10.0, 51.0
    dist, t, proj_lat, proj_lon = point_to_segment_distance(
        p_lat, p_lon, a_lat, a_lon, b_lat, b_lon
    )
    assert np.isclose(dist, 0.0, atol=10.0)
    assert np.isclose(t, 0.5, atol=0.05)

