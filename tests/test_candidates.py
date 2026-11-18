import pytest
import numpy as np

from src.data.loader import TripLoader
from src.features.candidates import CandidateGenerator
from src.features.engineering import FeatureExtractor


def test_candidate_generation():
    loader = TripLoader("data")
    trips = loader.list_trips("student_test")
    trip = loader.load_trip(trips[0], split="student_test")

    # Use first 15 GPS points for fast test
    sample_track = trip["track"][:15]
    cg = CandidateGenerator(search_radius_m=150.0, top_k=5)
    candidates = cg.generate_candidates(sample_track, trip["arcs"], trip["nodes"])

    assert len(candidates) == len(sample_track)
    for cands in candidates:
        assert len(cands) > 0
        assert "arc_id" in cands[0]
        assert "perp_dist" in cands[0]


def test_feature_extraction():
    loader = TripLoader("data")
    trips = loader.list_trips("student_test")
    trip = loader.load_trip(trips[0], split="student_test")

    sample_track = trip["track"][:10]
    cg = CandidateGenerator(search_radius_m=150.0, top_k=5)
    candidates = cg.generate_candidates(sample_track, trip["arcs"], trip["nodes"])

    fe = FeatureExtractor()
    df = fe.extract_features_for_trip(
        sample_track, trip["arcs"], trip["nodes"], candidates, route=trip["route"]
    )

    assert len(df) > 0
    for col in fe.FEATURE_NAMES:
        assert col in df.columns
    assert "is_match" in df.columns

