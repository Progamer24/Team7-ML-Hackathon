# PathSnap: Distributed Map-Matching & Route Reconstruction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)

**PathSnap** is an end-to-end distributed machine learning system that reconstructs realistic vehicle trajectories by snapping noisy, low-frequency GPS observations onto real-world road networks. The system integrates spatial candidate indexing, supervised feature learning, and Hidden Markov Model (HMM) Viterbi trellis decoding into a distributed two-laptop architecture.

---

## Repository Structure

```text
PathSnap/
│
├── data/
│   ├── train/                    # 80 training trips (GPS tracks, road networks, route labels)
│   ├── student_test/             # 10 reserved evaluation trips
│   └── README.md                 # Detailed dataset documentation & schema
│
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis & network visualization
│   ├── 02_model_training.ipynb   # Multi-model benchmarking & Bayesian tuning
│   └── 03_hmm_experiments.ipynb  # HMM & Viterbi sequential route reconstruction
│
├── src/
│   ├── data/
│   │   ├── loader.py             # Parser for .arcs, .nodes, .track, .route files
│   │   └── preprocessing.py      # Haversine distance, bearings, projections, GPS cleaning
│   │
│   ├── features/
│   │   ├── candidates.py         # Metric KDTree spatial candidate arc generation
│   │   └── engineering.py        # Orthogonal distances, heading diffs, kinematic features
│   │
│   ├── models/
│   │   ├── train.py              # Scikit-learn pipelines (LR, KNN, DT, RF, GBM, MLP)
│   │   ├── evaluate.py           # Top-1 observation accuracy, miss rate, continuity, log loss
│   │   └── tune.py               # Bayesian optimization with GroupKFold
│   │
│   ├── map_matching/
│   │   ├── hmm.py                # Emission & topological transition probability models
│   │   └── viterbi.py            # Log-space dynamic programming Viterbi decoder
│   │
│   ├── api/
│   │   └── main.py               # FastAPI inference backend (Laptop 1)
│   │
│   └── visualization/
│       └── maps.py               # Multi-layer Folium map generation with layer controls
│
├── artifacts/
│   ├── model.joblib              # Serialized trained model pipeline
│   └── config.json               # Spatial, model, HMM, and deployment parameters
│
├── results/
│   ├── model_comparison.csv      # Empirical benchmark table across algorithm families
│   ├── tuning_results.csv        # Hyperparameter search trials and scores
│   └── plots/                    # Exported evaluation charts and map visualizers
│
├── ui/
│   └── app.py                    # Streamlit Evaluator Interface (Laptop 2)
│
├── tests/
│   ├── test_loader.py            # Unit tests for data loading and geometric routines
│   ├── test_candidates.py        # Tests for spatial indexing & feature matrices
│   └── test_hmm.py               # Tests for HMM transitions & Viterbi decoding
│
├── requirements.txt              # Core project dependencies
├── README.md                     # Project documentation & guidelines
└── .gitignore                    # Version control ignore rules
```

---

## Methodological Pipeline

```
Raw GPS Points (track) + Road Network (arcs, nodes)
         │
         ▼
[Spatial Candidate Arc Generation (KDTree)]
         │
         ▼
[Feature Extraction: Perpendicular Distance, Bearing Diff, Projection Ratio, Speed]
         │
         ▼
[Supervised Classifier: P(match | features)]
         │
         ▼
[Hidden Markov Model + Viterbi Decoding]
   ├── Emission: P(z_t | s_t) ∝ P_ML(match) · exp(-d² / 2σ²)
   └── Transition: P(s_t | s_{t-1}) ∝ exp(-|Δd_net - Δd_gps| / β)
         │
         ▼
Optimal Reconstructed Vehicle Trajectory
```

---

## Two-Laptop Distributed Deployment Architecture

PathSnap is designed for real-time distributed evaluation across two separate network nodes:

| Host | Component | Description & Responsibilities |
| :--- | :--- | :--- |
| **Laptop 1** | **Model & API Server** (`src/api/main.py`) | Hosts the trained ML classifier, HMM/Viterbi engine, road network graph, and server-side evaluation. Ground-truth route labels remain strictly on this server. |
| **Laptop 2** | **Evaluator UI** (`ui/app.py`) | Evaluator-facing Streamlit application. Allows picking any of the 10 student-test trips, issuing evaluation requests to Laptop 1, and rendering interactive multi-layer Folium maps. |

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+
- `pip` package manager

### 2. Environment Setup
```bash
# Clone the repository
git clone <repo_url>
cd PathSnap

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Execution Guide

### Running Unit Tests
```bash
pytest tests/
```

### Starting Laptop 1 (Model API Server)
```bash
# Run FastAPI server on port 8000
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Documentation (Swagger UI): `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Starting Laptop 2 (Evaluator UI)
```bash
# Run Streamlit interface
streamlit run ui/app.py --server.port 8501
```
In the UI sidebar:
1. Enter the IP address and port of Laptop 1 (e.g., `http://192.168.1.105:8000`).
2. Select any reserved evaluation trip (e.g. `00000000`, `00000007`, `00000040`).
3. Click **Run Evaluation** to display the reconstructed path and ground truth comparison.

---

## Model Benchmark Results

Empirical results from 5 distinct algorithm families evaluated on out-of-sample student test trajectories:

| Model | Algorithm Family | Observation Top-1 Acc | Student-Test Macro F1 | Route Continuity |
| :--- | :--- | :---: | :---: | :---: |
| **Logistic Regression** | Linear / Logistic | **74.81%** | 0.7555 | **89.81%** |
| **Random Forest** | Ensemble | 62.97% | **0.7941** | 77.97% |
| **Gradient Boosting** | Boosting | 61.71% | 0.7631 | 76.71% |
| **K-Nearest Neighbors** | Instance-based | 63.56% | 0.7699 | 78.56% |
| **Decision Tree** | Tree-based | 56.51% | 0.7279 | 71.51% |

---

## Evaluation Metrics Summary
- **Observation-Level Top-1 Accuracy**: Proportion of GPS observations where the highest-scoring candidate is the ground-truth traversed road segment.
- **Route Continuity**: Fraction of consecutive reconstructed road segments that are topologically adjacent in the road network graph.
- **Candidate Miss Rate**: Cases where the true road segment is absent from the spatially indexed candidate set.

