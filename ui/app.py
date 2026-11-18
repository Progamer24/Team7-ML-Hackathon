"""
Streamlit Evaluator Interface for PathSnap (Laptop 2 Deployment).
Communicates with Laptop 1 API to fetch predictions, run evaluation, and visualize routes with Folium.
"""

import requests
import streamlit as st
import numpy as np

# Page configuration
st.set_page_config(
    page_title="PathSnap: Map-Matching & Route Reconstruction",
    page_icon="🗺️",
    layout="wide",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #3B82F6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">PathSnap: Map-Matching & Route Reconstruction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">UE24CS352A Machine Learning | Distributed Evaluator Interface</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("Configuration & Network")
api_base_url = st.sidebar.text_input(
    "Laptop 1 API Base URL",
    value="http://localhost:8000",
    help="Address of Laptop 1 running the FastAPI server (e.g., http://192.168.1.100:8000)",
)

# Check API Health
try:
    health_resp = requests.get(f"{api_base_url}/health", timeout=3)
    if health_resp.status_code == 200:
        st.sidebar.success("API Status: Connected (Online)")
    else:
        st.sidebar.warning(f"API Warning: Status {health_resp.status_code}")
except Exception:
    st.sidebar.error("API Status: Disconnected (Check Laptop 1 connection)")

# Fetch Student Test Trips
fallback_trips = [
    "00000000", "00000007", "00000009", "00000040", "00000052",
    "00000056", "00000065", "00000076", "00000077", "00000091",
]

available_trips = fallback_trips
try:
    trips_resp = requests.get(f"{api_base_url}/trips", timeout=3)
    if trips_resp.status_code == 200:
        available_trips = trips_resp.json().get("available_trips", fallback_trips)
except Exception:
    pass

selected_trip = st.sidebar.selectbox("Select Reserved Evaluation Trip", available_trips)

col_btn1, col_btn2 = st.sidebar.columns(2)
run_predict = col_btn1.button("Run Prediction", use_container_width=True)
run_evaluate = col_btn2.button("Run Evaluation", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Route Layer Legend")
st.sidebar.markdown("- 🔵 **Dashed Blue**: Raw GPS Observations")
st.sidebar.markdown("- 🔴 **Red Line**: PathSnap Predicted Trajectory")
st.sidebar.markdown("- 🟢 **Green Line**: Ground Truth Reference Route")

# Main Content Area
if run_evaluate or run_predict:
    endpoint = "/evaluate" if run_evaluate else "/predict"
    with st.spinner(f"Sending request to Laptop 1 API ({endpoint})..."):
        try:
            resp = requests.post(
                f"{api_base_url}{endpoint}",
                json={"trip_id": selected_trip},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"Trip {selected_trip} processed successfully in {data.get('inference_time_ms', 0)} ms!")

                # Display Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("GPS Observations", data.get("num_observations", 0))
                with m2:
                    if "observation_top1_accuracy" in data:
                        st.metric("Top-1 Accuracy", f"{data['observation_top1_accuracy'] * 100:.1f}%")
                    else:
                        st.metric("Inference Latency", f"{data.get('inference_time_ms', 0)} ms")
                with m3:
                    st.metric("Route Continuity", f"{data.get('route_continuity', 0) * 100:.1f}%")
                with m4:
                    if "candidate_miss_rate" in data:
                        st.metric("Candidate Miss Rate", f"{data['candidate_miss_rate'] * 100:.1f}%")
                    else:
                        st.metric("Status", data.get("status", "OK"))

                # Create Folium Map
                raw_gps = np.array(data.get("raw_gps", []))
                pred_coords = data.get("predicted_coords", [])
                ref_coords = data.get("reference_route_coords", None)

                if len(raw_gps) > 0:
                    import folium
                    center_lat = float(np.mean(raw_gps[:, 0]))
                    center_lon = float(np.mean(raw_gps[:, 1]))

                    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")

                    # Layer 1: Raw GPS
                    gps_layer = folium.FeatureGroup(name="Observed GPS Trajectory")
                    folium.PolyLine(
                        locations=raw_gps.tolist(),
                        color="#1E40AF",
                        weight=2.5,
                        dash_array="6, 8",
                        opacity=0.7,
                    ).add_to(gps_layer)

                    folium.Marker(
                        raw_gps[0].tolist(),
                        popup=f"Start GPS ({selected_trip})",
                        icon=folium.Icon(color="green", icon="play"),
                    ).add_to(gps_layer)
                    folium.Marker(
                        raw_gps[-1].tolist(),
                        popup=f"End GPS ({selected_trip})",
                        icon=folium.Icon(color="red", icon="stop"),
                    ).add_to(gps_layer)
                    gps_layer.add_to(fmap)

                    # Layer 2: Predicted Route
                    if pred_coords:
                        pred_layer = folium.FeatureGroup(name="PathSnap Predicted Route")
                        folium.PolyLine(
                            locations=pred_coords,
                            color="#DC2626",
                            weight=4,
                            opacity=0.9,
                        ).add_to(pred_layer)
                        pred_layer.add_to(fmap)

                    # Layer 3: Ground Truth Route (if evaluation endpoint)
                    if ref_coords:
                        ref_layer = folium.FeatureGroup(name="Reference Route (Ground Truth)")
                        folium.PolyLine(
                            locations=ref_coords,
                            color="#16A34A",
                            weight=3,
                            opacity=0.8,
                        ).add_to(ref_layer)
                        ref_layer.add_to(fmap)

                    folium.LayerControl(collapsed=False).add_to(fmap)

                    # Render Map in Streamlit
                    try:
                        from streamlit_folium import folium_static
                        folium_static(fmap, width=1100, height=550)
                    except ImportError:
                        # Fallback to direct HTML rendering
                        st.components.v1.html(fmap._repr_html_(), height=550)
            else:
                st.error(f"API Error ({resp.status_code}): {resp.text}")
        except Exception as e:
            st.error(f"Failed to communicate with API server: {e}")
else:
    st.info("Select one of the 10 reserved evaluation trips in the sidebar and click 'Run Evaluation' or 'Run Prediction'.")

