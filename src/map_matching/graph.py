"""
Road network graph representation and shortest path calculations for PathSnap.
"""

from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from src.data.preprocessing import haversine_distance


class RoadGraph:
    """
    Graph representation of the local road network.
    Stores topological connectivity and arc segment geometries.
    """

    def __init__(self, arcs: np.ndarray, nodes: np.ndarray):
        self.arcs = arcs
        self.nodes = nodes
        self.num_arcs = len(arcs)
        self.num_nodes = len(nodes)

        # Adjacency structures
        # node_to_outgoing_arcs[node_id] = list of arc_ids
        # node_to_incoming_arcs[node_id] = list of arc_ids
        self.outgoing_arcs: Dict[int, List[int]] = {}
        self.incoming_arcs: Dict[int, List[int]] = {}

        for arc_id in range(self.num_arcs):
            u = int(arcs[arc_id, 0])
            v = int(arcs[arc_id, 1])
            self.outgoing_arcs.setdefault(u, []).append(arc_id)
            self.incoming_arcs.setdefault(v, []).append(arc_id)

    def are_arcs_adjacent(self, arc1: int, arc2: int) -> bool:
        """
        Check if arc1 connects directly to arc2 (arc1.to_node == arc2.from_node).
        """
        if arc1 == arc2:
            return True
        if arc1 < 0 or arc1 >= self.num_arcs or arc2 < 0 or arc2 >= self.num_arcs:
            return False
        return bool(self.arcs[arc1, 1] == self.arcs[arc2, 0])

    def are_arcs_incident(self, arc1: int, arc2: int) -> bool:
        """
        Check if arc1 and arc2 share any intersection node.
        """
        if arc1 == arc2:
            return True
        if arc1 < 0 or arc1 >= self.num_arcs or arc2 < 0 or arc2 >= self.num_arcs:
            return False
        u1, v1 = self.arcs[arc1, 0], self.arcs[arc1, 1]
        u2, v2 = self.arcs[arc2, 0], self.arcs[arc2, 1]
        return bool(v1 == u2 or v1 == v2 or u1 == u2 or u1 == v2)

    def get_arc_geometry(self, arc_id: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get (from_coord, to_coord) for an arc in (lat, lon).
        """
        u = self.arcs[arc_id, 0]
        v = self.arcs[arc_id, 1]
        u_coord = (float(self.nodes[u, 1]), float(self.nodes[u, 0]))  # (lat, lon)
        v_coord = (float(self.nodes[v, 1]), float(self.nodes[v, 0]))
        return u_coord, v_coord

