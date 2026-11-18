"""
FastAPI Backend Application for PathSnap (Laptop 1 Deployment).
Handles map-matching inference, HMM route reconstruction, and server-side evaluation.

Coordinate convention:
- Internal raw files: [longitude, latitude, ...]
- Map rendering outputs: [latitude, longitude]
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data.loader import TripLoader
from src.features.candidates import CandidateGenerator
from src.features.engineering import FeatureExtractor
from src.models.train import load_model
from src.models.evaluate import compute_route_continuity
from src.map_matching.hmm import MapMatchingHMM
from src.map_matching.viterbi import viterbi_decode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PathSnap-API] %(message)s",
)
logger = logging.getLogger("PathSnapAPI")

app = FastAPI(
    title="PathSnap Map-Matching API",
    description="Distributed ML and HMM-based route reconstruction service for noisy GPS trajectories.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"

loader = TripLoader(DATA_DIR)
candidate_generator = CandidateGenerator(search_radius_m=120.0, top_k=8)
feature_extractor = FeatureExtractor()
hmm = MapMatchingHMM(sigma_z=20.0, beta=35.0)

model_pipeline = None


def get_model():
    """Lazy load or cached access to the trained model artifact."""
    global model_pipeline
    if model_pipeline is None:
        if MODEL_PATH.exists():
            try:
                model_pipeline = load_model(MODEL_PATH)
                logger.info("Successfully loaded trained model pipeline from %s", MODEL_PATH)
            except Exception as e:
                logger.warning("Could not load model artifact (%s). Using geometric fallback.", e)
        else:
            logger.warning("Model artifact not found at %s. Operating in geometric mode.", MODEL_PATH)
    return model_pipeline


class TripRequest(BaseModel):
    trip_id: str = Field(..., json_schema_extra={"example": "00000000"}, description="Evaluation Trip Identifier")


class PredictResponse(BaseModel):
    trip_id: str
    status: str
    num_observations: int
    predicted_arcs: List[int]
    predicted_coords: List[List[float]]
    raw_gps: List[List[float]]
    inference_time_ms: float
    route_continuity: float


class EvaluateResponse(BaseModel):
    trip_id: str
    num_observations: int
    observation_top1_accuracy: float
    route_continuity: float
    candidate_miss_rate: float
    inference_time_ms: float
    raw_gps: List[List[float]]
    predicted_coords: List[List[float]]
    reference_route_coords: List[List[float]]


@app.get("/health")
def health_check():
    """Health check endpoint to verify server status."""
    logger.info("Health check ping received.")
    return {
        "status": "healthy",
        "service": "PathSnap-API",
        "model_loaded": MODEL_PATH.exists(),
    }


@app.get("/trips")
def list_student_trips():
    """List available student test trips for evaluation."""
    trips = loader.list_trips("student_test")
    logger.info("Listing student test trips: found %d trips", len(trips))
    return {"available_trips": trips}


@app.post("/predict", response_model=PredictResponse)
def predict_route(req: TripRequest):
    """
    Run map-matching inference on a given trip.
    Applies candidate generation, ML scoring, and Viterbi HMM sequence decoding.
    """
    start_time = time.perf_counter()
    logger.info("Received /predict request for trip: %s", req.trip_id)

    try:
        trip_data = loader.load_trip(req.trip_id)
    except FileNotFoundError:
        logger.error("Trip %s not found in repository data directory.", req.trip_id)
        raise HTTPException(status_code=404, detail=f"Trip {req.trip_id} not found.")

    track = trip_data["track"]
    arcs = trip_data["arcs"]
    nodes = trip_data["nodes"]

    # 1. Candidate generation
    candidates_by_point = candidate_generator.generate_candidates(track, arcs, nodes)

    # 2. Extract features and compute candidate classifier probabilities
    pipe = get_model()
    classifier_probs_by_point = []

    if pipe is not None:
        df_feat = feature_extractor.extract_features_for_trip(
            track, arcs, nodes, candidates_by_point, route=None
        )
        if len(df_feat) > 0:
            X = df_feat[FeatureExtractor.FEATURE_NAMES].values
            probs = pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") else pipe.predict(X)
            df_feat["prob"] = probs
            for pt_idx in range(len(track)):
                pt_probs = df_feat[df_feat["point_idx"] == pt_idx]["prob"].tolist()
                classifier_probs_by_point.append(pt_probs)
        else:
            classifier_probs_by_point = None
    else:
        classifier_probs_by_point = None

    # 3. Viterbi HMM Decoding
    matched_arcs, projected_coords = viterbi_decode(
        track, candidates_by_point, arcs, nodes, hmm=hmm, classifier_probs_by_point=classifier_probs_by_point
    )

    continuity = compute_route_continuity(matched_arcs, arcs)
    inference_time = (time.perf_counter() - start_time) * 1000.0

    logger.info(
        "Trip %s processed: %d points | Continuity: %.3f | Latency: %.2f ms",
        req.trip_id,
        len(track),
        continuity,
        inference_time,
    )

    # Convert raw track from [lon, lat, timestamp] to [lat, lon] for Folium
    raw_gps_latlon = [[float(p[1]), float(p[0])] for p in track]

    return PredictResponse(
        trip_id=req.trip_id,
        status="success",
        num_observations=len(track),
        predicted_arcs=matched_arcs,
        predicted_coords=[[float(c[0]), float(c[1])] for c in projected_coords],
        raw_gps=raw_gps_latlon,
        inference_time_ms=round(inference_time, 2),
        route_continuity=round(continuity, 4),
    )


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate_trip(req: TripRequest):
    """
    Perform server-side evaluation against internal ground truth.
    Ground-truth labels remain strictly protected on server.
    """
    start_time = time.perf_counter()
    logger.info("Received /evaluate request for trip: %s", req.trip_id)

    try:
        trip_data = loader.load_trip(req.trip_id)
    except FileNotFoundError:
        logger.error("Trip %s not found.", req.trip_id)
        raise HTTPException(status_code=404, detail=f"Trip {req.trip_id} not found.")

    track = trip_data["track"]
    arcs = trip_data["arcs"]
    nodes = trip_data["nodes"]
    route = trip_data["route"]

    if route is None or len(route) == 0:
        raise HTTPException(status_code=400, detail=f"No reference route for trip {req.trip_id}.")

    route_set = set(route)

    # Candidate generation & ML probability inference
    candidates_by_point = candidate_generator.generate_candidates(track, arcs, nodes)
    pipe = get_model()
    classifier_probs = []

    if pipe is not None:
        df_feat = feature_extractor.extract_features_for_trip(track, arcs, nodes, candidates_by_point)
        if len(df_feat) > 0:
            X = df_feat[FeatureExtractor.FEATURE_NAMES].values
            probs = pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") else pipe.predict(X)
            df_feat["prob"] = probs
            for pt_idx in range(len(track)):
                classifier_probs.append(df_feat[df_feat["point_idx"] == pt_idx]["prob"].tolist())

    matched_arcs, projected_coords = viterbi_decode(
        track, candidates_by_point, arcs, nodes, hmm=hmm, classifier_probs_by_point=classifier_probs
    )

    correct = sum(1 for a in matched_arcs if a in route_set)
    accuracy = correct / len(matched_arcs) if len(matched_arcs) > 0 else 0.0

    missed_cands = 0
    for cands in candidates_by_point:
        if not any(c["arc_id"] in route_set for c in cands):
            missed_cands += 1
    miss_rate = missed_cands / len(track) if len(track) > 0 else 0.0

    continuity = compute_route_continuity(matched_arcs, arcs)
    inference_time = (time.perf_counter() - start_time) * 1000.0

    # Build reference route coordinates in [lat, lon]
    ref_coords = []
    for arc_id in route:
        if arc_id < len(arcs):
            u, v = arcs[arc_id, 0], arcs[arc_id, 1]
            ref_coords.append([float(nodes[u, 1]), float(nodes[u, 0])])
            ref_coords.append([float(nodes[v, 1]), float(nodes[v, 0])])

    raw_gps_latlon = [[float(p[1]), float(p[0])] for p in track]

    logger.info(
        "Evaluation completed for %s: Top-1 Acc: %.3f | Continuity: %.3f | Miss Rate: %.3f",
        req.trip_id,
        accuracy,
        continuity,
        miss_rate,
    )

    return EvaluateResponse(
        trip_id=req.trip_id,
        num_observations=len(track),
        observation_top1_accuracy=round(accuracy, 4),
        route_continuity=round(continuity, 4),
        candidate_miss_rate=round(miss_rate, 4),
        inference_time_ms=round(inference_time, 2),
        raw_gps=raw_gps_latlon,
        predicted_coords=[[float(c[0]), float(c[1])] for c in projected_coords],
        reference_route_coords=ref_coords,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
