import pytest
import numpy as np
from src.data.loader import TripLoader
from src.map_matching.graph import RoadGraph


def test_road_graph_construction():
    loader = TripLoader("data")
    trips = loader.list_trips("student_test")
    trip = loader.load_trip(trips[0], split="student_test")

    graph = RoadGraph(trip["arcs"], trip["nodes"])
    assert graph.num_arcs == len(trip["arcs"])
    assert graph.num_nodes == len(trip["nodes"])

    # Test adjacency for identical arc
    assert graph.are_arcs_adjacent(0, 0) is True
    assert graph.are_arcs_incident(0, 0) is True

    # Test boundary / invalid indices
    assert graph.are_arcs_adjacent(-1, 0) is False
    assert graph.are_arcs_incident(0, 99999999) is False

