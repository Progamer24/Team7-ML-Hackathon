"""
Streamlit Evaluator Interface for PathSnap (Laptop 2 Deployment).
Communicates with Laptop 1 API to fetch predictions, run evaluation, and visualize routes with Folium.
Includes multi-layer toggling, model diagnostic metrics, and multi-model benchmark analytics.
"""

from pathlib import Path
import json
import requests
import streamlit as st
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="PathSnap: Map-Matching & Route Reconstruction",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-badge {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">PathSnap: Map-Matching & Route Reconstruction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">UE24CS352A Machine Learning | Team 7 Distributed Evaluator Dashboard</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# SIDEBAR CONFIGURATION
# ----------------------------------------------------------------------------
st.sidebar.header("Network & Connection")
api_base_url = st.sidebar.text_input(
    "Laptop 1 API Base URL",
    value="http://localhost:8000",
    help="Address of Laptop 1 running the FastAPI server (e.g., http://192.168.1.100:8000)",
)

# Check API Health & Fetch Model Info
api_online = False
model_info = {}
try:
    health_resp = requests.get(f"{api_base_url}/health", timeout=2)
    if health_resp.status_code == 200:
        api_online = True
        st.sidebar.success("API Status: Connected (Online)")
    else:
        st.sidebar.warning(f"API Warning: HTTP {health_resp.status_code}")
except Exception:
    st.sidebar.error("API Status: Disconnected (Check Laptop 1 connection)")

# Try fetching dynamic model info from API or local fallback
if api_online:
    try:
        info_resp = requests.get(f"{api_base_url}/model_info", timeout=2)
        if info_resp.status_code == 200:
            model_info = info_resp.json()
    except Exception:
        pass

# Fallback model info from local artifacts if running co-located
if not model_info:
    local_cfg = Path("artifacts/config.json")
    if local_cfg.exists():
        try:
            with open(local_cfg, "r") as f:
                model_info = {"configuration": json.load(f), "selected_model": "RandomForestClassifier (Tuned)"}
        except Exception:
            pass

# Trip Partition Selection
st.sidebar.markdown("---")
st.sidebar.header("Trip Selection")

eval_trips = [
    "00000015", "00000017", "00000023", "00000030", "00000049",
    "00000059", "00000067", "00000073", "00000087", "00000094",
]
student_trips = [
    "00000000", "00000007", "00000009", "00000040", "00000052",
    "00000056", "00000065", "00000076", "00000077", "00000091",
]

if api_online:
    try:
        trips_resp = requests.get(f"{api_base_url}/trips", timeout=2)
        if trips_resp.status_code == 200:
            data = trips_resp.json()
            if data.get("evaluator_trips"):
                eval_trips = data["evaluator_trips"]
            if data.get("student_trips"):
                student_trips = data["student_trips"]
    except Exception:
        pass

partition = st.sidebar.radio("Dataset Partition", ["Evaluator Test (Official)", "Student Test (Dev)"], index=0)
available_trips = eval_trips if "Evaluator" in partition else student_trips
selected_trip = st.sidebar.selectbox("Select Evaluation Trip", available_trips)

col_btn1, col_btn2 = st.sidebar.columns(2)
run_predict = col_btn1.button("Run Prediction", use_container_width=True)
run_evaluate = col_btn2.button("Run Evaluation", type="primary", use_container_width=True)

# ----------------------------------------------------------------------------
# SIDEBAR LAYER TOGGLES
# ----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🎛️ Map Route Layers")
toggle_gps = st.sidebar.checkbox("🔵 Raw GPS Observations", value=True, help="Dashed blue polyline representing noisy raw GPS sensor fixes.")
toggle_pred = st.sidebar.checkbox("🔴 PathSnap Predicted Route", value=True, help="Solid red line representing the HMM + Viterbi reconstructed road route.")
toggle_ref = st.sidebar.checkbox("🟢 Ground Truth Route", value=True, help="Solid green line representing the true reference route traversed.")
toggle_markers = st.sidebar.checkbox("📍 Start / End Markers", value=True, help="Green start play icon and red end stop icon.")

# ----------------------------------------------------------------------------
# TOP MODEL DIAGNOSTIC BAR
# ----------------------------------------------------------------------------
with st.expander("ℹ️ Deployed Model Architecture & Hyperparameters", expanded=False):
    cfg_data = model_info.get("configuration", {})
    m_cfg = cfg_data.get("model_configuration", {})
    s_cfg = cfg_data.get("spatial_parameters", {})
    h_cfg = cfg_data.get("hmm_parameters", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Production Model**")
        st.write(f"Algorithm: **{m_cfg.get('selected_model', 'RandomForestClassifier')}**")
        st.write(f"Family: **{m_cfg.get('algorithm_family', 'Ensemble')}**")
        st.write(f"Best CV Score: **{m_cfg.get('best_cv_score', 0.8447)}**")
    with c2:
        st.markdown("**Tuned Hyperparameters**")
        st.write(f"n_estimators: **{m_cfg.get('n_estimators', 100)}**")
        st.write(f"max_depth: **{m_cfg.get('max_depth', 11)}**")
        st.write(f"min_samples_split: **{m_cfg.get('min_samples_split', 8)}**")
        st.write(f"class_weight: **{m_cfg.get('class_weight', 'balanced_subsample')}**")
    with c3:
        st.markdown("**Spatial Candidates**")
        st.write(f"Search Radius: **{s_cfg.get('search_radius_m', 120.0)} m**")
        st.write(f"Top-K Candidates: **{s_cfg.get('top_k_candidates', 8)}**")
        st.write(f"Spatial Index: **cKDTree ($O(\\log N)$)**")
    with c4:
        st.markdown("**HMM Sequential Engine**")
        st.write(f"Sensor Noise ($\\sigma_z$): **{h_cfg.get('sigma_z', 20.0)} m**")
        st.write(f"Transition Rate ($\\beta$): **{h_cfg.get('beta', 35.0)} m**")
        st.write(f"Decoding: **Log-space Viterbi Trellis**")

# ----------------------------------------------------------------------------
# MAIN CONTENT TABS
# ----------------------------------------------------------------------------
tab_map, tab_benchmark, tab_tuning, tab_evaluator_set = st.tabs([
    "🗺️ Interactive Map & Evaluation",
    "🏆 Multi-Model Benchmark (6 Families)",
    "📈 Bayesian Optimization Trials",
    "🎯 Official Evaluator Test Set"
])

# ----------------------------------------------------------------------------
# TAB 1: INTERACTIVE MAP & EVALUATION
# ----------------------------------------------------------------------------
with tab_map:
    # State handling for map rendering
    if "latest_result" not in st.session_state:
        st.session_state["latest_result"] = None
    if "latest_trip" not in st.session_state:
        st.session_state["latest_trip"] = None
    if "latest_mode" not in st.session_state:
        st.session_state["latest_mode"] = None

    if run_evaluate or run_predict:
        endpoint = "/evaluate" if run_evaluate else "/predict"
        mode_label = "Evaluation" if run_evaluate else "Prediction"
        with st.spinner(f"Querying Laptop 1 API ({endpoint}) for trip {selected_trip}..."):
            try:
                resp = requests.post(
                    f"{api_base_url}{endpoint}",
                    json={"trip_id": selected_trip},
                    timeout=30,
                )
                if resp.status_code == 200:
                    st.session_state["latest_result"] = resp.json()
                    st.session_state["latest_trip"] = selected_trip
                    st.session_state["latest_mode"] = mode_label
                else:
                    st.error(f"API Error ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(f"Failed to communicate with Laptop 1 server: {e}")

    # Render Result if available
    res = st.session_state.get("latest_result")
    curr_trip = st.session_state.get("latest_trip", selected_trip)
    curr_mode = st.session_state.get("latest_mode", "Evaluation")

    if res:
        st.success(f"Trip **{curr_trip}** successfully processed via **{curr_mode}** in {res.get('inference_time_ms', 0):.1f} ms!")

        # Key Metrics Row
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("GPS Observations", res.get("num_observations", 0))
        with m2:
            if "observation_top1_accuracy" in res:
                st.metric("Top-1 Accuracy", f"{res['observation_top1_accuracy'] * 100:.1f}%")
            else:
                st.metric("Mode", "Prediction (Unlabeled)")
        with m3:
            st.metric("Route Continuity", f"{res.get('route_continuity', 0) * 100:.1f}%")
        with m4:
            if "candidate_miss_rate" in res:
                st.metric("Candidate Miss Rate", f"{res['candidate_miss_rate'] * 100:.1f}%")
            else:
                st.metric("Matched Arcs", len(res.get("predicted_arcs", [])))
        with m5:
            st.metric("Latency", f"{res.get('inference_time_ms', 0):.1f} ms", f"{res.get('inference_time_ms', 0)/max(1, res.get('num_observations', 1)):.3f} ms/pt")

        # Map Construction with Dynamic Toggles
        raw_gps = np.array(res.get("raw_gps", []))
        pred_coords = res.get("predicted_coords", [])
        ref_coords = res.get("reference_route_coords", None)

        if len(raw_gps) > 0:
            import folium
            center_lat = float(np.mean(raw_gps[:, 0]))
            center_lon = float(np.mean(raw_gps[:, 1]))

            fmap = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")

            # 1. Raw GPS Observation Layer
            if toggle_gps:
                gps_layer = folium.FeatureGroup(name="Observed GPS Trajectory", show=True)
                folium.PolyLine(
                    locations=raw_gps.tolist(),
                    color="#1E40AF",
                    weight=2.5,
                    dash_array="6, 8",
                    opacity=0.7,
                    tooltip="Raw GPS Observation Sequence",
                ).add_to(gps_layer)
                gps_layer.add_to(fmap)

            # 2. Start & End Milestone Markers
            if toggle_markers:
                marker_layer = folium.FeatureGroup(name="Start / End Milestones", show=True)
                folium.Marker(
                    raw_gps[0].tolist(),
                    popup=f"Start GPS (Trip {curr_trip})",
                    tooltip="Start Point",
                    icon=folium.Icon(color="green", icon="play"),
                ).add_to(marker_layer)
                folium.Marker(
                    raw_gps[-1].tolist(),
                    popup=f"End GPS (Trip {curr_trip})",
                    tooltip="End Point",
                    icon=folium.Icon(color="red", icon="stop"),
                ).add_to(marker_layer)
                marker_layer.add_to(fmap)

            # 3. PathSnap Predicted Route Layer
            if toggle_pred and pred_coords:
                pred_layer = folium.FeatureGroup(name="PathSnap Predicted Route", show=True)
                folium.PolyLine(
                    locations=pred_coords,
                    color="#DC2626",
                    weight=4.0,
                    opacity=0.9,
                    tooltip="PathSnap HMM Viterbi Matched Route",
                ).add_to(pred_layer)
                pred_layer.add_to(fmap)

            # 4. Ground Truth Reference Route Layer
            if toggle_ref and ref_coords:
                ref_layer = folium.FeatureGroup(name="Reference Route (Ground Truth)", show=True)
                folium.PolyLine(
                    locations=ref_coords,
                    color="#16A34A",
                    weight=3.0,
                    opacity=0.8,
                    tooltip="Ground Truth Traversing Route",
                ).add_to(ref_layer)
                ref_layer.add_to(fmap)

            # Built-in Folium layer control inside map
            folium.LayerControl(collapsed=False).add_to(fmap)

            # Render in Streamlit
            try:
                from streamlit_folium import folium_static
                folium_static(fmap, width=1100, height=540)
            except ImportError:
                st.components.v1.html(fmap._repr_html_(), height=540)
    else:
        st.info("👈 Select any reserved trip in the sidebar and click **'Run Evaluation'** or **'Run Prediction'** to generate the interactive route map.")

# ----------------------------------------------------------------------------
# TAB 2: MULTI-MODEL BENCHMARK (6 ALGORITHM FAMILIES)
# ----------------------------------------------------------------------------
with tab_benchmark:
    st.markdown("### Benchmarking Across 6 Algorithm Families")
    st.markdown("Candidate scoring models evaluated on candidate pairs ($1:5$ class ratio) using Group-K-Fold cross-validation:")

    comp_file = Path("results/model_comparison.csv")
    if comp_file.exists():
        df_comp = pd.read_csv(comp_file)
        st.dataframe(
            df_comp.style.format({
                "CV Macro F1-score": "{:.4f}",
                "Student-Test Macro F1": "{:.4f}",
                "Top-1 Accuracy": "{:.2%}",
                "Route Continuity": "{:.2%}",
                "Log Loss": "{:.4f}",
                "Inference Time (ms/obs)": "{:.3f}",
            }).highlight_max(subset=["Top-1 Accuracy", "Route Continuity", "CV Macro F1-score"], color="#D1FAE5"),
            use_container_width=True,
        )

        b_c1, b_c2 = st.columns(2)
        plot_acc = Path("results/plots/model_comparison_accuracy.png")
        plot_metrics = Path("results/plots/metrics_comparison.png")
        if plot_acc.exists():
            b_c1.image(str(plot_acc), caption="Observation Top-1 Accuracy across Algorithm Families", use_container_width=True)
        if plot_metrics.exists():
            b_c2.image(str(plot_metrics), caption="Performance Metrics Comparison Across Families", use_container_width=True)
    else:
        st.info("Run `python refresh_all.py` on Laptop 1 to generate benchmark comparison data.")

# ----------------------------------------------------------------------------
# TAB 3: BAYESIAN OPTIMIZATION TRIALS
# ----------------------------------------------------------------------------
with tab_tuning:
    st.markdown("### Optuna Bayesian Hyperparameter Optimization")
    st.markdown("Iterative parameter search maximizing blended objective: $0.6 \\times \\text{Macro } F_1 + 0.4 \\times \\text{Accuracy}$:")

    tune_file = Path("results/tuning_results.csv")
    conv_plot = Path("results/plots/tuning_convergence.png")

    if tune_file.exists():
        df_tune = pd.read_csv(tune_file)
        t_col1, t_col2 = st.columns([3, 2])
        with t_col1:
            st.markdown(f"**Optimization Trials Completed**: {len(df_tune)}")
            st.dataframe(
                df_tune.sort_values(by="blended_objective", ascending=False).head(10).style.highlight_max(subset=["blended_objective"], color="#FEF3C7"),
                use_container_width=True,
            )
        with t_col2:
            if conv_plot.exists():
                st.image(str(conv_plot), caption="Optuna TPESampler Convergence Curve", use_container_width=True)
            else:
                st.line_chart(df_tune[["mean_macro_f1", "mean_accuracy", "blended_objective"]])
    else:
        st.info("Tuning results not found. Run Bayesian optimization in `02_model_training.ipynb`.")

# ----------------------------------------------------------------------------
# TAB 4: OFFICIAL EVALUATOR TEST SET LEADERBOARD
# ----------------------------------------------------------------------------
with tab_evaluator_set:
    st.markdown("### Official Evaluator Test Partition (10 Reserved Trips)")
    st.markdown("Performance comparison between Classifier-Only Greedy Matching and HMM Viterbi Sequential Decoding:")

    eval_res_file = Path("results/evaluator_test_results.csv")
    if eval_res_file.exists():
        df_eval_set = pd.read_csv(eval_res_file)

        avg_acc_g = df_eval_set["acc_greedy"].mean()
        avg_acc_v = df_eval_set["acc_viterbi"].mean()
        avg_cont_g = df_eval_set["continuity_greedy"].mean()
        avg_cont_v = df_eval_set["continuity_viterbi"].mean()

        es1, es2, es3, es4 = st.columns(4)
        es1.metric("Avg Greedy Accuracy", f"{avg_acc_g * 100:.2f}%")
        es2.metric("Avg Viterbi Accuracy", f"{avg_acc_v * 100:.2f}%", f"+{(avg_acc_v - avg_acc_g) * 100:.2f}%")
        es3.metric("Avg Greedy Continuity", f"{avg_cont_g * 100:.2f}%")
        es4.metric("Avg Viterbi Continuity", f"{avg_cont_v * 100:.2f}%", f"+{(avg_cont_v - avg_cont_g) * 100:.2f}%")

        st.dataframe(
            df_eval_set.style.format({
                "acc_greedy": "{:.2%}",
                "acc_viterbi": "{:.2%}",
                "continuity_greedy": "{:.2%}",
                "continuity_viterbi": "{:.2%}",
            }).highlight_max(subset=["acc_viterbi", "continuity_viterbi"], color="#D1FAE5"),
            use_container_width=True,
        )
    else:
        st.info("Official evaluator test results file `results/evaluator_test_results.csv` not found.")
