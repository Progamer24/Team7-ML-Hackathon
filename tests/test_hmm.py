import pytest
import numpy as np

from src.data.loader import TripLoader
from src.features.candidates import CandidateGenerator
from src.map_matching.hmm import MapMatchingHMM
from src.map_matching.viterbi import viterbi_decode
from src.models.evaluate import compute_route_continuity


def test_viterbi_decoding():
    loader = TripLoader("data")
    trips = loader.list_trips("student_test")
    trip = loader.load_trip(trips[0], split="student_test")

    sample_track = trip["track"][:20]
    cg = CandidateGenerator(search_radius_m=150.0, top_k=5)
    candidates = cg.generate_candidates(sample_track, trip["arcs"], trip["nodes"])

    hmm = MapMatchingHMM(sigma_z=20.0, beta=30.0)
    matched_arcs, projected_coords = viterbi_decode(
        sample_track, candidates, trip["arcs"], trip["nodes"], hmm=hmm
    )

    assert len(matched_arcs) == len(sample_track)
    assert len(projected_coords) == len(sample_track)

    continuity = compute_route_continuity(matched_arcs, trip["arcs"])
    assert 0.0 <= continuity <= 1.0

