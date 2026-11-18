"""
Reproducible Exploratory Data Analysis (EDA) Pipeline for PathSnap.
Analyzes dataset inventory, coordinate validity, sampling intervals, and road network topology.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.loader import TripLoader
from src.data.preprocessing import haversine_distance, compute_bearing
from src.visualization.maps import create_folium_map, save_folium_map


def run_eda():
    print("=" * 60)
    print("PATHSNAP: EXPLORATORY DATA ANALYSIS PIPELINE")
    print("=" * 60)

    loader = TripLoader("data")
    train_trips = loader.list_trips("train")
    test_trips = loader.list_trips("student_test")

    print(f"Total Training Trips: {len(train_trips)}")
    print(f"Total Student-Testing Trips: {len(test_trips)}")

    summary_records = []
    for split, trips in [("train", train_trips[:15]), ("student_test", test_trips)]:
        for tid in trips:
            trip = loader.load_trip(tid, split=split)
            val = loader.validate_trip(trip)
            summary_records.append({
                "split": split,
                "trip_id": tid,
                **val
            })

    df_summary = pd.DataFrame(summary_records)
    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_summary.to_csv("results/dataset_inventory.csv", index=False)
    print("\nDataset Summary Statistics:")
    print(df_summary[["split", "trip_id", "num_observations", "num_nodes", "num_arcs", "nominal_1hz_pct"]].head(10))

    # Sample detailed trajectory analysis
    sample_trip = loader.load_trip(test_trips[0], split="student_test")
    track = sample_trip["track"]
    dts = np.diff(track[:, 2])
    speeds = []
    for i in range(len(track) - 1):
        dist = haversine_distance(track[i, 1], track[i, 0], track[i+1, 1], track[i+1, 0])
        dt = max(1e-3, dts[i])
        speeds.append(dist / dt)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(dts, bins=25, color="#3B82F6", edgecolor="black")
    plt.title("GPS Sampling Intervals (Δt)")
    plt.xlabel("Seconds")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(speeds, bins=25, color="#10B981", edgecolor="black")
    plt.title("Instantaneous Speeds (m/s)")
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plots_dir / "eda_distributions.png", dpi=150)
    plt.close()
    print("\nSaved distribution plots to results/plots/eda_distributions.png")

    # Generate sample Folium map
    fmap = create_folium_map(
        track=sample_trip["track"],
        trip_id=test_trips[0]
    )
    save_folium_map(fmap, plots_dir / "eda_sample_map.html")
    print("Saved interactive Folium map to results/plots/eda_sample_map.html")


if __name__ == "__main__":
    run_eda()

