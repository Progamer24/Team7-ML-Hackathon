"""
Reproducible HMM and Viterbi Sequential Route Reconstruction Experiments for PathSnap.
Compares Classifier-Only predictions vs. Classifier + HMM Viterbi decoding.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.loader import TripLoader
from src.features.candidates import CandidateGenerator
from src.features.engineering import FeatureExtractor
from src.models.train import load_model
from src.models.evaluate import compute_route_continuity
from src.map_matching.hmm import MapMatchingHMM
from src.map_matching.viterbi import viterbi_decode
from src.visualization.maps import create_folium_map, save_folium_map


def run_hmm_experiments():
    print("=" * 60)
    print("PATHSNAP: HMM & VITERBI ROUTE RECONSTRUCTION EXPERIMENTS")
    print("=" * 60)

    loader = TripLoader("data")
    test_trips = loader.list_trips("student_test")
    cg = CandidateGenerator(search_radius_m=120.0, top_k=6)
    fe = FeatureExtractor()
    hmm = MapMatchingHMM(sigma_z=20.0, beta=35.0)

    model = load_model("artifacts/model.joblib")

    experiment_results = []
    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    for tid in test_trips[:5]:
        trip = loader.load_trip(tid, split="student_test")
        track = trip["track"][:250]
        arcs = trip["arcs"]
        nodes = trip["nodes"]
        route = trip["route"]

        cands = cg.generate_candidates(track, arcs, nodes)
        df_feat = fe.extract_features_for_trip(track, arcs, nodes, cands)
        X = df_feat[fe.FEATURE_NAMES].values
        probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
        df_feat["prob"] = probs

        # Classifier-only greedy matching
        greedy_arcs = []
        for pt_idx in range(len(track)):
            sub = df_feat[df_feat["point_idx"] == pt_idx]
            if len(sub) > 0:
                best_arc = sub.loc[sub["prob"].idxmax()]["arc_id"]
                greedy_arcs.append(int(best_arc))
            else:
                greedy_arcs.append(-1)

        # HMM Viterbi matching
        probs_by_point = [
            df_feat[df_feat["point_idx"] == pt_idx]["prob"].tolist()
            for pt_idx in range(len(track))
        ]
        viterbi_arcs, proj_coords = viterbi_decode(
            track, cands, arcs, nodes, hmm=hmm, classifier_probs_by_point=probs_by_point
        )

        route_set = set(route) if route is not None else set()
        acc_greedy = sum(1 for a in greedy_arcs if a in route_set) / max(1, len(track))
        acc_viterbi = sum(1 for a in viterbi_arcs if a in route_set) / max(1, len(track))

        cont_greedy = compute_route_continuity(greedy_arcs, arcs)
        cont_viterbi = compute_route_continuity(viterbi_arcs, arcs)

        experiment_results.append({
            "trip_id": tid,
            "observations": len(track),
            "acc_greedy": round(acc_greedy, 4),
            "acc_viterbi": round(acc_viterbi, 4),
            "continuity_greedy": round(cont_greedy, 4),
            "continuity_viterbi": round(cont_viterbi, 4),
        })

    df_exp = pd.DataFrame(experiment_results)
    df_exp.to_csv("results/evaluation_results.csv", index=False)
    print("\nClassifier-Only vs. Classifier + HMM Results:")
    print(df_exp)

    # Export Folium Map for first test trip
    sample_trip = loader.load_trip(test_trips[0], split="student_test")
    ref_coords = []
    if sample_trip["route"] is not None:
        for a_id in sample_trip["route"]:
            if a_id < len(sample_trip["arcs"]):
                u, v = sample_trip["arcs"][a_id, 0], sample_trip["arcs"][a_id, 1]
                # nodes: [lon, lat] -> map needs [lat, lon]
                ref_coords.append([float(sample_trip["nodes"][u, 1]), float(sample_trip["nodes"][u, 0])])
                ref_coords.append([float(sample_trip["nodes"][v, 1]), float(sample_trip["nodes"][v, 0])])

    fmap = create_folium_map(
        track=sample_trip["track"][:250],
        ground_truth_coords=ref_coords,
        trip_id=test_trips[0],
    )
    save_folium_map(fmap, plots_dir / "hmm_route_reconstruction.html")
    print("\nSaved reconstructed route map to results/plots/hmm_route_reconstruction.html")


if __name__ == "__main__":
    run_hmm_experiments()
