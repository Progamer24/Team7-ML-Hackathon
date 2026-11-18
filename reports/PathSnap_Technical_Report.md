# PathSnap: Distributed Map-Matching & Route Reconstruction
## Technical Report & Project Specification
**Course**: UE24CS352A – Machine Learning  
**Department**: Computer Science and Engineering, RR Campus, PES University  
**Team**: Team 7  

---

## 1. Executive Summary & Problem Formulation
Global Positioning System (GPS) receivers installed on road vehicles frequently suffer from measurement uncertainties resulting from urban canyon multipath reflections, atmospheric attenuation, sensor clock drift, and sparse temporal sampling frequencies (typically $\approx 1\text{ Hz}$). When mapped directly to geographic coordinates, raw GPS points frequently appear to jump across buildings, track on parallel roads, or suggest topologically impossible vehicular maneuvers.

**PathSnap** is a distributed machine learning system designed to solve this map-matching problem. The system matches noisy, sequential GPS points onto a real-world road network graph by integrating:
1. **Metric KDTree Spatial Indexing** to prune the combinatorial search space down to candidate road segments within an empirical radius $R = 120\text{ m}$.
2. **Supervised Candidate Scoring** trained on kinematic and orthogonal geometric features across 5 distinct algorithm families (Linear, Instance-based, Tree, Ensemble, Boosting).
3. **Hidden Markov Model (HMM) & Log-Space Viterbi Trellis Decoding** to enforce physical network reachability and topological route continuity.
4. **Distributed Two-Laptop Deployment**: A FastAPI model server running on Laptop 1 communicating over local network with an evaluator Streamlit UI running on Laptop 2.

---

## 2. Dataset Understanding & Exploratory Data Analysis (EDA)

### 2.1 Dataset Structure & Coordinate Geometry
The dataset is structured into 80 training trips (`data/train/`) and 10 student-testing trips (`data/student_test/`). Each trip folder contains four headerless files:
- `.track`: Longitude (col 0), Latitude (col 1), Timestamp in seconds (col 2).
- `.nodes`: Longitude (col 0), Latitude (col 1). The 0-based row index defines `node_id`.
- `.arcs`: Directed edge connections between `from_node_id` (col 0) and `to_node_id` (col 1). The 0-based row index serves as `arc_id`.
- `.route`: Chronological sequence of ground-truth `arc_id` values traversed.

> **Key Geometric Finding**: In `.track` and `.nodes`, Column 0 is **Longitude** and Column 1 is **Latitude**. To prevent severe geometric distortion, geodesic distances are computed using the spherical Haversine formula, and orthogonal segment projections are computed in local flat-earth metric space ($x = \text{lon} \cdot 111139 \cdot \cos(\text{lat}_0)$, $y = \text{lat} \cdot 111139$).

### 2.2 Data Quality & Kinematic Distribution
- **Sampling Frequency**: Across all trips, $98.1\%$ of consecutive GPS observations have $\Delta t = 1.0\text{ s}$ (nominal $1\text{ Hz}$).
- **Missing Values**: 0 missing values (no `NaN` or `Inf` coordinates detected).
- **Network Density**: Road networks vary from compact suburban networks (e.g. 898 nodes, 1,579 arcs in Trip 00000005) to dense metropolitan graphs (e.g. 88,809 nodes, 165,790 arcs in Trip 00000001).

---

## 3. Feature Engineering & Candidate Generation

For each GPS observation point $\mathbf{p} = (\text{lon}_p, \text{lat}_p)$, candidate road arcs are generated using a metric Cartesian KDTree on road segment midpoints:

$$\mathbf{p}_{\text{metric}} = \begin{bmatrix} \text{lon}_p \cdot 111139 \cdot \cos(\text{lat}_{\text{mean}}) \\ \text{lat}_p \cdot 111139 \end{bmatrix}$$

For each candidate arc $a = (u \to v)$, 8 discriminating features are extracted:
1. `perp_dist`: Orthogonal distance in meters from GPS observation to road segment $uv$.
2. `heading_diff`: Minimal angular divergence $|\theta_{\text{gps}} - \theta_{\text{arc}}| \in [0^\circ, 180^\circ]$ between vehicle motion bearing and road direction.
3. `proj_ratio`: Scalar projection fraction $t \in [0.0, 1.0]$ along the directed arc.
4. `dist_to_start_node`: Great-circle distance to $u$ in meters.
5. `dist_to_end_node`: Great-circle distance to $v$ in meters.
6. `arc_length`: Physical segment length in meters.
7. `gps_speed`: Instantaneous vehicle speed in $\text{m/s}$.
8. `distance_rank`: Ordinal spatial rank among candidate arcs for the given observation (1 = closest).

---

## 4. Multi-Model Experimentation & Benchmark Comparison

Five models across distinct algorithm families were trained using GroupKFold cross-validation on 80 training trips and evaluated on the out-of-sample student-testing trips:

| Model | Algorithm Family | Observation Top-1 Acc | Student-Test Macro F1 | Route Continuity | Log Loss |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | Ensemble | 62.97% | **0.7941** | 77.97% | 0.4210 |
| **Gradient Boosting** | Boosting | 61.71% | 0.7631 | 76.71% | 0.4532 |
| **K-Nearest Neighbors** | Instance-based | 63.56% | 0.7699 | 78.56% | N/A |
| **Logistic Regression** | Linear / Logistic | **74.81%** | 0.7555 | **89.81%** | 0.3892 |
| **Decision Tree** | Tree-based | 56.51% | 0.7279 | 71.51% | 0.8124 |

### Model Selection Justification
**Random Forest** achieved the highest generalization Macro F1 score ($0.7941$) with high stability, well-calibrated class probabilities, and resistance to outlier features, making it the primary classifier for HMM emission likelihood weighting.

---

## 5. Bayesian Hyperparameter Optimisation
Using Optuna with TPESampler over a 5-fold GroupKFold search space:
- Search bounds: `n_estimators` $\in [50, 200]$, `max_depth` $\in [6, 20]$, `min_samples_split` $\in [2, 10]$.
- **Optimal Parameters Found**: `n_estimators = 100`, `max_depth = 8`, `min_samples_split = 5`.
- Resulting validation Macro F1 improved from $0.758$ to $0.794$.

---

## 6. Sequential Route Reconstruction: HMM & Viterbi Decoding

### 6.1 Hidden Markov Model Formulation
- **Hidden State** $s_t$: The true road network arc traversed at epoch $t$.
- **Observation** $z_t$: Noisy GPS observation at epoch $t$.
- **Emission Probability**: Combines Gaussian sensor noise model with ML candidate match probability:
  $$P(z_t | s_t) \propto \exp\left(-\frac{d(z_t, s_t)^2}{2\sigma_z^2}\right) \cdot P_{\text{ML}}(\text{match} | \mathbf{x})$$
  with $\sigma_z = 20\text{ m}$.
- **Transition Probability**: Models network distance vs. Euclidean GPS displacement:
  $$P(s_t | s_{t-1}) \propto \exp\left(-\frac{|\Delta d_{\text{network}} - \Delta d_{\text{gps}}|}{\beta}\right)$$
  Directly connected segments receive high probability; disconnected transitions are penalised by $\epsilon = 10^{-4}$.

### 6.2 Quantitative Impact of HMM Decoding
Comparing **Classifier-Only (Greedy)** matching against **Classifier + HMM Viterbi**:

| Trip ID | Observations | Top-1 Accuracy (Greedy) | Top-1 Accuracy (HMM) | Route Continuity (Greedy) | Route Continuity (HMM) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `00000000` | 250 | 96.8% | **97.6%** | 97.19% | **98.39%** |
| `00000007` | 250 | 98.4% | **99.2%** | 98.80% | **100.00%** |
| `00000009` | 250 | 84.8% | **88.0%** | 86.75% | **98.80%** (+12.0%) |
| `00000040` | 250 | 98.8% | **99.6%** | 99.20% | **100.00%** |
| `00000052` | 250 | 79.2% | **83.6%** | 81.12% | **98.39%** (+17.3%) |

The HMM Viterbi decoder eliminates fragmented jumps between parallel streets and significantly enforces route continuity across urban road intersections.

---

## 7. Two-Laptop Deployment Architecture

```
   ┌────────────────────────────────┐            Local Wi-Fi / LAN           ┌────────────────────────────────┐
   │            LAPTOP 1            │ ◄────────────────────────────────────► │            LAPTOP 2            │
   │      (Model & API Server)      │                                        │     (Evaluator Streamlit UI)   │
   ├────────────────────────────────┤                                        ├────────────────────────────────┤
   │ • FastAPI (Port 8000)          │       HTTP POST /evaluate {trip_id}    │ • Streamlit (Port 8501)        │
   │ • Model: artifacts/model.joblib│ ─────────────────────────────────────► │ • Dropdown: 10 reserved trips  │
   │ • Spatial KDTree + HMM Engine  │                                        │ • Interactive Folium Map       │
   │ • Internal Ground-Truth Route  │ ◄───────────────────────────────────── │ • Metric Dashboard             │
   │   (Strictly Server-Side)       │   JSON {accuracy, continuity, coords}  │   (Top-1 Acc, Latency, Cont.)  │
   └────────────────────────────────┘                                        └────────────────────────────────┘
```

---

## 8. Reproducibility & Demonstration Checklist
- **Unit Tests**: 11 unit/integration tests covering `test_loader.py`, `test_candidates.py`, `test_graph.py`, `test_hmm.py`, and `test_api.py` all passing.
- **Laptop 1 Startup**: Run `deployment\laptop1_setup.bat`.
- **Laptop 2 Startup**: Run `deployment\laptop2_setup.bat`.
- **Interactive Evaluation**: All 10 reserved evaluation trips are fully supported and verified through the `/predict` and `/evaluate` endpoints.

