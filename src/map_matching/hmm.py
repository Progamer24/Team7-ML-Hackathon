"""
Hidden Markov Model (HMM) formulation for map matching in PathSnap.
Models emission probabilities and spatial-topological transition probabilities.
"""

from typing import Dict, List, Optional
import numpy as np
from src.data.preprocessing import haversine_distance


class MapMatchingHMM:
    """
    HMM for Map Matching.
    - Hidden States: Candidate road arcs at each GPS epoch.
    - Observations: Noisy GPS coordinates.
    - Emissions: Gaussian distance likelihood + ML candidate match confidence.
    - Transitions: Topological network connectivity and distance consistency.
    """

    def __init__(
        self,
        sigma_z: float = 20.0,
        beta: float = 30.0,
        unconnected_penalty: float = 1e-4,
    ):
        self.sigma_z = sigma_z
        self.beta = beta
        self.unconnected_penalty = unconnected_penalty

    def compute_emission_prob(
        self,
        candidate: Dict,
        classifier_prob: Optional[float] = None,
    ) -> float:
        """
        Emission likelihood P(z_t | s_t).
        Combines Gaussian distance sensor model with classifier probability.
        """
        dist = candidate["perp_dist"]
        # Gaussian emission probability
        p_dist = np.exp(-0.5 * (dist / self.sigma_z) ** 2) / (np.sqrt(2 * np.pi) * self.sigma_z)

        if classifier_prob is not None:
            # Blend classifier confidence with distance prior
            p_emit = 0.5 * p_dist + 0.5 * max(1e-4, classifier_prob)
        else:
            p_emit = p_dist

        return float(max(1e-9, p_emit))

    def compute_transition_prob(
        self,
        cand_prev: Dict,
        cand_curr: Dict,
        dist_gps: float,
        arcs: np.ndarray,
        nodes: np.ndarray,
    ) -> float:
        """
        Transition probability P(s_t | s_{t-1}).
        Evaluates topological connection and route length consistency.
        """
        arc_prev = cand_prev["arc_id"]
        arc_curr = cand_curr["arc_id"]

        # Case 1: Same road segment
        if arc_prev == arc_curr:
            delta_proj = abs(cand_curr["proj_ratio"] - cand_prev["proj_ratio"]) * cand_curr["arc_len"]
            diff = abs(delta_proj - dist_gps)
            return float(np.exp(-diff / self.beta) / self.beta)

        # Case 2: Directly connected arcs (to_node == from_node)
        prev_to = cand_prev["to_node"]
        curr_from = cand_curr["from_node"]
        if prev_to == curr_from:
            # Distance from prev proj point to end of arc_prev + start of arc_curr to curr proj point
            remain_prev = (1.0 - cand_prev["proj_ratio"]) * cand_prev["arc_len"]
            into_curr = cand_curr["proj_ratio"] * cand_curr["arc_len"]
            route_dist = remain_prev + into_curr
            diff = abs(route_dist - dist_gps)
            return float(np.exp(-diff / self.beta) / self.beta)

        # Case 3: Connected in reverse / shared junction
        prev_from = cand_prev["from_node"]
        curr_to = cand_curr["to_node"]
        if prev_to == curr_to or prev_from == curr_from:
            return float(0.2 * np.exp(-dist_gps / self.beta) / self.beta)

        # Case 4: Disconnected or multi-hop transition
        return float(self.unconnected_penalty)

