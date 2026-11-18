# PathSnap Dataset Documentation

This directory contains the spatial GPS trajectories and road network topological data used for map-matching and route reconstruction in the PathSnap project.

---

## Directory Organization

```text
data/
├── train/              # 80 training trips (IDs: 00000001 to 00000099)
├── student_test/       # 10 student-test trips (IDs: 00000000, 00000007, 00000009, etc.)
└── README.md           # Dataset specification and schema documentation
```

---

## Trip File Formats

Each trip is encapsulated within its own subfolder named `<trip_id>/` and contains four primary files:

### 1. `<trip_id>.arcs` (Road Network Segments)
- **Description**: Defines directed edges (road segments) connecting intersection nodes in the local road network.
- **Format**: Plain text, whitespace/tab-separated values, headerless.
- **Columns**:
  - Column 0: `from_node_id` (Integer) - Source intersection node ID.
  - Column 1: `to_node_id` (Integer) - Destination intersection node ID.
- **Note**: The **0-based row index** in this file serves as the unique identifier `arc_id` for each road segment.

### 2. `<trip_id>.nodes` (Road Network Intersections)
- **Description**: Geospatial coordinates of intersection nodes in the local road network.
- **Format**: Plain text, whitespace/tab-separated values, headerless.
- **Columns**:
  - Column 0: `latitude` (Float) - WGS84 latitude in decimal degrees.
  - Column 1: `longitude` (Float) - WGS84 longitude in decimal degrees.
- **Note**: The **0-based line number** corresponds directly to `node_id`.

### 3. `<trip_id>.track` (Noisy GPS Observations)
- **Description**: Chronological sequence of noisy GPS trajectory observations recorded from the moving vehicle.
- **Format**: Plain text, whitespace/tab-separated values, headerless.
- **Columns**:
  - Column 0: `latitude` (Float) - Observed GPS latitude in decimal degrees.
  - Column 1: `longitude` (Float) - Observed GPS longitude in decimal degrees.
  - Column 2: `timestamp` (Float) - Elapsed observation time in seconds (or relative timestamp).

### 4. `<trip_id>.route` (Reference Ground-Truth Route)
- **Description**: Chronological sequence of ground-truth `arc_id` values traversed by the vehicle.
- **Format**: Plain text, single integer column, headerless.
- **Values**: Traversed `arc_id` sequence.
- **Note**: Used for model training and validation. For final evaluation endpoints, ground truth remains server-side.

---

## Dataset Partitions

| Partition | Trip Count | Purpose |
| :--- | :--- | :--- |
| `train/` | 80 trips | Feature extraction, model training, cross-validation, and Bayesian hyperparameter tuning. |
| `student_test/` | 10 trips | Out-of-sample evaluation, observation-level Top-1 accuracy, and HMM route continuity benchmarking. |

---

## Coordinate Reference System & Scale
- Coordinates are expressed in decimal degrees (WGS84 ellipsoid).
- Distance metrics are computed using the Haversine great-circle formula and Euclidean orthogonal projections in metric space ($1^\circ \text{ latitude} \approx 111{,}000\text{ m}$).

